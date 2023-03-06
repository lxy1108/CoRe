from typing import Optional, Tuple, Any
import torch
import torch.nn as nn
from torch.nn import functional as F
from pytorch_lightning import LightningModule

# from layers.Embed import PositionalEmbedding
# from layers.SelfAttention_Family import SampledAttention, FullAttention, AttentionLayer
# from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
# from layers.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
# from layers.Corr_EncDec import CorrEncoder, CorrDecoder
from layers import TransformerCNN, MyDecoder


class Encoder(nn.Module):
    def __init__(self, endo_dim: int, exo_dim: int, enc_dim: int, kernel_size: int, nlayer: int, dilation: bool):
        super(Encoder, self).__init__()
        self.enc_dim = enc_dim
        self.conv = nn.Conv1d(1, enc_dim, kernel_size, padding="same")
        self.vec = nn.Parameter(torch.randn(1, 1, 1, enc_dim))
        self.series_emb = nn.Embedding(endo_dim + exo_dim + 1, enc_dim)
        self.type_emb = nn.Embedding(3, enc_dim)
        self.encoder = TransformerCNN(enc_dim, kernel_size, 1, nlayer, dilation)

    def forward(self, ref_x: torch.Tensor, ref_y: torch.Tensor) -> torch.Tensor:
        b, n, r = ref_x.shape
        _, m, _ = ref_y.shape
        device = ref_x.device
        val_emb = self.conv(torch.cat([ref_y, ref_x], dim=1).reshape(b * (m + n), 1, r)).reshape(b, m + n, -1, r)
        val_emb = torch.cat([self.vec.expand((b, 1, r, self.enc_dim)), torch.permute(val_emb, (0, 1, 3, 2))], dim=1) #b*(n+1)*r e
        series_emb = self.series_emb(torch.arange(m + n + 1).unsqueeze(1).to(device)).reshape(1, m + n + 1, 1, -1).expand_as(val_emb)
        type_emb = self.type_emb(torch.Tensor([0] + [1] * m + [2] * n).long().unsqueeze(1).to(device)).reshape(1, m + n + 1, 1, -1).expand_as(val_emb)
        input_emb = val_emb + series_emb + type_emb#b*(n+1)*r*e'
        input_emb = torch.permute(input_emb, (0, 1, 3, 2))#b*(1+m+n)*e*r
        return self.encoder(input_emb)[:, 0, :, :].mean(-1)



class Decoder(nn.Module):
    def __init__(self, endo_dim: int, back_len: int, max_exo_num: int, enc_dim: int, kernel_size: int, nlayer: int, dilation):
        super(Decoder, self).__init__()
        self.back_len = back_len
        self.nlayer = nlayer
        self.xconv = nn.Conv1d(max_exo_num, enc_dim, kernel_size, padding="same")
        self.yconv = nn.Conv1d(endo_dim, enc_dim, kernel_size, padding="same")
        self.decoder = MyDecoder(enc_dim, kernel_size, nlayer, dilation)
        self.output_linear = nn.Linear(enc_dim, endo_dim)
        self.lstm1 = nn.LSTM(endo_dim, enc_dim, 3, batch_first=True)
        self.lstm2 = nn.LSTM(enc_dim, enc_dim, 3, batch_first=True)

    def forward(self, emb: torch.Tensor, query_x: torch.Tensor, query_y: torch.Tensor) -> torch.Tensor:
        b, n, q = query_y.shape
        device = query_x.device
        val_emb_x = torch.permute(self.xconv(query_x), (0, 2, 1))
        # val_emb_x = self.xconv(query_x.reshape(b * n, 1, q))
        # val_emb_x = torch.permute(val_emb_x, (0, 2, 1)) #b*(n)*q*e
        val_emb_y = torch.permute(self.yconv(query_y), (0, 2, 1))
        pred_y = self.decoder(emb, val_emb_x, val_emb_y)
        back_y, (h_0, c_0) = self.lstm1(torch.permute(query_y[:, :, :self.back_len], (0, 2, 1)))
        output, _ = self.lstm2(pred_y[:, self.back_len:, :], (h_0, c_0))
        rs =  self.output_linear(output)
        return torch.permute(rs, (0, 2, 1))#.squeeze()




class TSPredictor(LightningModule):

    def __init__(self, endo_dim: int,
                       exo_num: int, 
                       enc_dim: int, 
                       rlen: int,
                       qlen: int,
                       back_len: int,
                       kernel_size: int, 
                       nlayer: int,
                       sample_range: int,
                       sample_num: int,
                       decomp_trend: bool,
                       dilation: bool = False,
                       lr: float = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.enc_dim = enc_dim
        self.back_len = back_len
        self.rlen = rlen
        self.sample_range = sample_range
        self.sample_stride = sample_num
        self.decomp_trend = decomp_trend
        self.encoder = Encoder(endo_dim, exo_num, enc_dim, kernel_size, nlayer, dilation)
        self.decoder = Decoder(endo_dim, back_len, exo_num, enc_dim, kernel_size, nlayer, dilation)
        self.emb_list, self.emb_rs = [], None

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, y)



    def training_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int):
        # ref_x, ref_y, ref_x_pos, ref_y_pos, query_x, query_y = batch # b*n*r, b*r, b*n*q, b*r
        ref_x, ref_y, query_x, query_y = batch # b*n*r, b*r, b*n*q, b*r
        ref_x, ref_y, query_x, query_y = ref_x.float(), ref_y.float(), query_x.float(), query_y.float()
        emb = self.forward(ref_x, ref_y)
        # print([8]*50)
        # emb = self.emb_list[ins_id, :]
        # emb_pos = self.forward(ref_x_pos, ref_y_pos.unsqueeze(1))
        # pred_y = self.decoder(emb, query_x, query_y_mask.unsqueeze(1))
        # pred_y = self.decoder(emb, query_x)
        # mse_loss = F.mse_loss(pred_y, query_y)# + F.mse_loss(pred_y_pos, query_y)
        query_y_decode = query_y.clone()
        query_y_decode[..., self.back_len:] = torch.mean(query_y[..., :self.back_len], dim=-1, keepdims=True).expand_as(query_y[..., self.back_len:]) 
        pred_y = self.decoder(emb, query_x, query_y_decode)
        # mse_loss = F.mse_loss(pred_y.squeeze(), query_y.squeeze()[..., self.back_len:])
        # emb = torch.randn(emb.shape).to(emb.device)
        # pred_y = self.decoder(emb, query_x, query_y[..., :self.back_len])
        # infonce_loss = self.infonce(emb, emb_pos)
        if self.decomp_trend:
            pred_seasonal, pred_trend = torch.split(pred_y.reshape(pred_y.shape[0], -1, 2, pred_y.shape[-1]), 1, 2)
            label_seasonal, label_trend = torch.split(query_y[..., self.back_len:].reshape(pred_y.shape[0], -1, 2, pred_y.shape[-1]), 1, 2)
            # param = min((50 + batch_idx // 100), 500)
            # weight = np.concatenate([np.ones(192),np.exp(-np.arange(pred_seasonal.shape[-1]-192)/param)])
            # weight = torch.from_numpy(weight).float().to(ref_x.device)[None, None, :]
            seasonal_mse = (F.mse_loss(pred_seasonal, label_seasonal, reduction='none')).mean()
            trend_mse = (F.mse_loss(pred_trend, label_trend, reduction='none')).mean()
            total_loss = F.mse_loss(pred_seasonal + pred_trend, label_seasonal + label_trend)
            self.log("seasonal_mse_loss", seasonal_mse, prog_bar=True)
            self.log("trend_mse_loss", trend_mse, prog_bar=True)
            self.log("train_mse_loss", total_loss, prog_bar=True)
            mse_loss = (seasonal_mse+trend_mse) * 0.5
            # mse_loss = (seasonal_mse1+trend_mse1) * 0.5
        else:
            mse_loss = F.mse_loss(pred_y, query_y[..., self.back_len:])
            self.log("train_mse_loss", mse_loss, prog_bar=True)
        # self.log("encoder_loss", en_loss, prog_bar=True)
        # self.log("exo_num", ref_x.shape[1], prog_bar=True)
        return mse_loss.mean()# + infonce_loss

    # def on_validation_start(self) -> None:
    #     self.emb_list = []

    def validation_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx % 2 == 0:
            # if batch_idx == 0:
            #     self.emb_list = []
            ref_x, ref_y = batch
            emb = self.forward(ref_x, ref_y)
            return emb
            # self.emb_list.append(emb)
        else:
            # if batch_idx == 0:
            #     emb = torch.cat(self.emb_list, 0)
            #     self.emb_rs = emb.reshape(batch[-1][0].item(), -1, emb.shape[-1])
            # query_x, query_y, ins_id, index, group_num = batch # b*n*r, b*r, b*n*q, b*r
            # emb_start_idx = index // self.sample_stride
            # emb_idx = emb_start_idx.unsqueeze(-1).repeat(1, self.sample_range // self.sample_stride) + torch.arange(self.sample_range // self.sample_stride).to(query_x.device)
            # emb_idx = torch.minimum(emb_idx, torch.ones_like(emb_idx).to(query_x.device) * (self.emb_rs.shape[1] - 1))
            # emb_idx = emb_idx.unsqueeze(-1).repeat(1, 1, self.emb_rs.shape[-1])
            # emb = self.emb_rs[ins_id, ...]
            # emb = torch.gather(emb, 1, emb_idx).mean(1)
            if self.emb_rs == None:
                emb = torch.cat(self.emb_list, 0)
                self.emb_rs = emb.reshape(-1, self.sample_stride, emb.shape[-1]).mean(1)
            query_x, query_y, ins_id = batch # b*n*r, b*r, b*n*q, b*r
            emb = self.emb_rs[ins_id, :].to(query_x.device)
            # emb = torch.randn(emb.shape).to(emb.device)
            query_y_decode = query_y.clone()
            query_y_decode[..., self.back_len:] = torch.mean(query_y[..., :self.back_len], dim=-1, keepdims=True).expand_as(query_y[..., self.back_len:]) 
            pred_y = self.decoder(emb, query_x, query_y_decode)
            # mse_loss = F.mse_loss(pred_y.squeeze(), query_y.squeeze()[..., self.back_len:])
            # pred_y = self.decoder(emb, query_x, query_y[..., :self.back_len])
            if self.decomp_trend:
                pred_seasonal, pred_trend = torch.split(pred_y.reshape(pred_y.shape[0], -1, 2, pred_y.shape[-1]), 1, 2)
                label_seasonal, label_trend = torch.split(query_y[..., self.back_len:].reshape(pred_y.shape[0], -1, 2, pred_y.shape[-1]), 1, 2)
                seasonal_mse = F.mse_loss(pred_seasonal, label_seasonal)
                trend_mse = F.mse_loss(pred_trend, label_trend)
                total_loss = F.mse_loss(pred_seasonal + pred_trend, label_seasonal + label_trend)
                self.log("seasonal_loss", seasonal_mse, prog_bar=True)
                self.log("trend_loss", trend_mse, prog_bar=True)
                self.log("val_loss", total_loss, prog_bar=True)
            else:
                mse_loss = F.mse_loss(pred_y, query_y[..., self.back_len:])
                self.log("val_loss", mse_loss, prog_bar=True)
    
    def validation_step_end(self, val_step_outputs):
        if val_step_outputs is not None:
            self.emb_list.append(val_step_outputs)

    def validation_epoch_end(self, outputs):
        self.emb_list = []
        self.emb_rs = None


    def on_test_start(self) -> None:
        self.emb_list = []

    def test_step(self, batch, batch_idx, dataloader_idx):
        if dataloader_idx % 2 == 0:
            if batch_idx == 0:
                self.emb_list = []
            ref_x, ref_y = batch
            emb = self.forward(ref_x, ref_y)
            self.emb_list.append(emb)
        else:
            # if batch_idx == 0:
            #     emb = torch.cat(self.emb_list, 0)
            #     self.emb_rs = emb.reshape(batch[-1][0].item(), -1, emb.shape[-1])
            # query_x, query_y, ins_id, index, group_num = batch # b*n*r, b*r, b*n*q, b*r
            # emb_start_idx = index // self.sample_stride
            # emb_idx = emb_start_idx.unsqueeze(-1).repeat(1, self.sample_range // self.sample_stride) + torch.arange(self.sample_range // self.sample_stride).to(query_x.device)
            # emb_idx = torch.minimum(emb_idx, torch.ones_like(emb_idx).to(query_x.device) * (self.emb_rs.shape[1] - 1))
            # emb_idx = emb_idx.unsqueeze(-1).repeat(1, 1, self.emb_rs.shape[-1])
            # emb = self.emb_rs[ins_id, ...]
            # emb = torch.gather(emb, 1, emb_idx).mean(1)
            if batch_idx == 0:
                emb = torch.cat(self.emb_list, 0)
                self.emb_rs = emb.reshape(-1, self.sample_stride, emb.shape[-1]).mean(1)
                # data = emb.reshape(-1, self.sample_stride, emb.shape[-1]).detach().cpu().numpy()
                # print(data.shape)
                # np.save("vec.npy", data)
            query_x, query_y, ins_id = batch # b*n*r, b*r, b*n*q, b*r
            emb = self.emb_rs[ins_id, :]
            # emb = torch.randn(emb.shape).to(emb.device)
            query_y_decode = query_y.clone()
            query_y_decode[..., self.back_len:] = torch.mean(query_y[..., :self.back_len], dim=-1, keepdims=True).expand_as(query_y[..., self.back_len:]) 
            pred_y = self.decoder(emb, query_x, query_y_decode)
            # mse_loss = F.mse_loss(pred_y[..., self.back_len:], query_y.squeeze()[..., self.back_len:])
            # mae_loss = F.l1_loss(pred_y[..., self.back_len:], query_y.squeeze()[..., self.back_len:])
            # pred_y = self.decoder(emb, query_x, query_y[..., :self.back_len])
            if self.decomp_trend:
                pred_seasonal, pred_trend = torch.split(pred_y.reshape(pred_y.shape[0], -1, 2, pred_y.shape[-1]), 1, 2)
                label_seasonal, label_trend = torch.split(query_y[..., self.back_len:].reshape(pred_y.shape[0], -1, 2, pred_y.shape[-1]), 1, 2)
                seasonal_mse = F.mse_loss(pred_seasonal, label_seasonal)
                trend_mse = F.mse_loss(pred_trend, label_trend)
                mse_loss = F.mse_loss(pred_seasonal + pred_trend, label_seasonal + label_trend)
                mae_loss = F.l1_loss(pred_seasonal + pred_trend, label_seasonal + label_trend)
                self.log("seasonal_loss", seasonal_mse, prog_bar=True)
                self.log("trend_loss", trend_mse, prog_bar=True)
                self.log("test_mse_loss", mse_loss, prog_bar=True)
                self.log("test_mae_loss", mae_loss, prog_bar=True)
            else:
                mse_loss = F.mse_loss(pred_y, query_y[..., self.back_len:])
                mae_loss = F.l1_loss(pred_y, query_y[..., self.back_len:])
                self.log("test_mse_loss", mse_loss, prog_bar=True)
                self.log("test_mae_loss", mae_loss, prog_bar=True)

    def on_predict_start(self) -> None:
        self.emb_list = []

    def predict_step(self, batch: Tuple[torch.Tensor, ...], batch_idx: int, dataloader_idx: Optional[int] = None):
        if dataloader_idx % 2 == 0:
            if batch_idx == 0:
                self.emb_list = []
            ref_x, ref_y = batch
            emb = self.forward(ref_x, ref_y)
            self.emb_list.append(emb)
        else:
            if batch_idx == 0:
                emb = torch.cat(self.emb_list, 0)
                self.emb_rs = emb.reshape(batch[-1][0].item(), -1, emb.shape[-1])
            query_x, query_y, ins_id, index, group_num = batch # b*n*r, b*r, b*n*q, b*r
            emb_start_idx = index // self.sample_stride
            emb_idx = emb_start_idx.unsqueeze(-1).repeat(1, self.sample_range // self.sample_stride) + torch.arange(self.sample_range // self.sample_stride).to(query_x.device)
            emb_idx = torch.minimum(emb_idx, torch.ones_like(emb_idx).to(query_x.device) * (self.emb_rs.shape[1] - 1))
            emb_idx = emb_idx.unsqueeze(-1).repeat(1, 1, self.emb_rs.shape[-1])
            emb = self.emb_rs[ins_id, ...]
            emb = torch.gather(emb, 1, emb_idx).mean(1)
            query_y_decode = query_y.clone()
            query_y_decode[..., self.back_len:] = torch.mean(query_y[..., :self.back_len], dim=-1, keepdims=True).expand_as(query_y[..., self.back_len:]) 
            pred_y = self.decoder(emb, query_x, query_y_decode)
            mse_loss = F.mse_loss(pred_y[..., self.back_len:], query_y.squeeze()[..., self.back_len:], reduction="none")
            mae_loss = F.l1_loss(pred_y[..., self.back_len:], query_y.squeeze()[..., self.back_len:], reduction="none")
            # pred_y = self.decoder(emb, query_x, query_y[..., :self.back_len])
            # mse_loss = F.mse_loss(pred_y, query_y.squeeze()[..., self.back_len:], reduction="none")
            # mae_loss = F.l1_loss(pred_y, query_y.squeeze()[..., self.back_len:], reduction="none")
            return pred_y, query_y.squeeze(), mse_loss, mae_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return optimizer
