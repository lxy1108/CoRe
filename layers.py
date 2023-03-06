import torch.nn as nn
import torch

class ConvolutionBlock(torch.nn.Module):
    """
    Causal convolution block, composed sequentially of two causal convolutions
    (with leaky ReLU activation functions), and a parallel residual connection.
    Takes as input a three-dimensional tensor (`B`, `C`, `L`) where `B` is the
    batch size, `C` is the number of input channels, and `L` is the length of
    the input. Outputs a three-dimensional tensor (`B`, `C`, `L`).
    @param in_channels Number of input channels.
    @param out_channels Number of output channels.
    @param kernel_size Kernel size of the applied non-residual convolutions.
    @param dilation Dilation parameter of non-residual convolutions.
    @param final Disables, if True, the last activation function.
    """
    def __init__(self, channels, kernel_size, dilation):
        super(ConvolutionBlock, self).__init__()

        # Computes left padding so that the applied convolutions are causal

        # First causal convolution
        conv1 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            channels, channels, kernel_size,
            padding="same", dilation=dilation
        ))
        # The truncation makes the convolution causal
        relu1 = torch.nn.LeakyReLU()

        # Second causal convolution
        conv2 = torch.nn.utils.weight_norm(torch.nn.Conv1d(
            channels, channels, kernel_size,
            padding="same", dilation=dilation
        ))
        relu2 = torch.nn.LeakyReLU()

        # Causal network
        # self.causal = torch.nn.Sequential(
        #     conv1, chomp1, relu1, conv2, chomp2, relu2
        # )
        
        self.cnn = torch.nn.Sequential(
            conv1, relu1, conv2, relu2
        )

        # Residual connection
        self.upordownsample = torch.nn.Conv1d(
            channels, channels, 1
        ) if channels != channels else None

        # Final activation function
        # self.relu = torch.nn.LeakyReLU()

    def forward(self, x):
        out_causal = self.cnn(x)
        res = x if self.upordownsample is None else self.upordownsample(x)
        return out_causal + res
        # return self.relu(out_causal + res)

class TransformerCNNLayer(nn.Module):
    def __init__(self, d_model, kernel_size, nhead, dilation = 1):
        super(TransformerCNNLayer, self).__init__()
        self.transformer = nn.TransformerEncoderLayer(d_model, nhead, batch_first = True)
        self.conv = ConvolutionBlock(d_model, kernel_size, dilation)

    def forward(self, x):
        b, n, d, s = x.shape
        trans_intput = torch.permute(x, (0, 3, 1, 2)).reshape(b * s, n, d)
        trans_output = self.transformer(trans_intput).reshape(b, s, n, d)
        conv_input = torch.permute(trans_output, (0, 2, 3, 1)).reshape(b * n, d, s)
        conv_output = self.conv(conv_input).reshape(b, n, d, s)
        return conv_output

class TransformerCNN(nn.Module):
    def __init__(self, d_model, kernel_size, nhead, nlayers, dilation=False):
        super(TransformerCNN, self).__init__()
        self.encoder = torch.nn.Sequential(*[TransformerCNNLayer(d_model, kernel_size, nhead, pow(2, i) if dilation else 1) for i in range(nlayers)])

    def forward(self, x):
        return self.encoder(x)



class DecoderLayer(nn.Module):
    def __init__(self, d_model, kernel_size, dilation):
        super(DecoderLayer, self).__init__()
        self.convx = ConvolutionBlock(d_model, kernel_size, dilation)
        self.convy = ConvolutionBlock(d_model, kernel_size, dilation)
        self.fc = nn.Linear(3 * d_model, d_model)

    def forward(self, emb, query_x, query_y):
        b, q, d = query_y.shape
        emb_input = emb.unsqueeze(1).repeat(1, q, 1)
        query_y_input = torch.permute(self.convy(torch.permute(query_y, (0, 2, 1))), (0, 2, 1))
        query_x_input = torch.permute(self.convx(torch.permute(query_x, (0, 2, 1))), (0, 2, 1))
        res = self.fc(torch.cat([emb_input, query_x_input, query_y_input], dim=-1))
        return query_x_input, res + query_y



class MyDecoder(nn.Module):
    def __init__(self, d_model, kernel_size, nlayer, dilation=False):
        super(MyDecoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, kernel_size, pow(2, i) if dilation else 1) for i in range(nlayer)])

    def forward(self, emb, query_x, query_y):
        for layer in self.layers:
            query_x, query_y = layer(emb, query_x, query_y)
        return query_y

