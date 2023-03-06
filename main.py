
import argparse
import os
import sys

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from data_attr import dataset_dict
from predict_model import TSPredictor
from datamodule import TSTrainDataModule

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
CHECKPOINT_PATH = "saved_models"
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def train(args):
    data = TSTrainDataModule(args.dataset, args.moving_avg, args.exo_num, args.covariate, args.rlen, 
                args.qlen, args.blen, args.sample_range, args.sample_num, args.batch_size, args.decomp)

    endo_dim = dataset_dict[args.dataset].dimension * (2 if args.decomp else 1)     
    model = TSPredictor(endo_dim, args.exo_num, args.enc_dim, args.rlen, args.qlen, args.blen, 
                args.kernel_size, args.nlayer, args.sample_range, args.sample_num, 
                args.decomp, args.dilation, args.lr)  

    save_bests = [] 
    save_bests.append(ModelCheckpoint(
                                        monitor='val_loss/dataloader_idx_1',
                                        filename='checkpoint-{epoch}-{step}-{val_loss/dataloader_idx_1:.2f}'
                                    ))
    save_bests.append(EarlyStopping('val_loss/dataloader_idx_1', strict=False, patience=args.patience))    

    use_gpu = torch.cuda.is_available()
    gpu_num = torch.cuda.device_count()
    model_name = "{}_pl{}_lr{}_bsz{}".format(args.dataset, args.qlen, args.lr, args.batch_size)
    trainer = pl.Trainer(default_root_dir = os.path.join(CHECKPOINT_PATH, model_name),
                            accelerator="gpu" if use_gpu else "cpu", 
                            devices=gpu_num if use_gpu else None, 
                            strategy="dp" if gpu_num > 1 else None,
                            max_epochs = args.epochs,
                            val_check_interval = args.val_interval if args.val_interval < 1 else None,
                            check_val_every_n_epoch = int(args.val_intervalval_interval) if args.val_interval > 1 else 1,
                            log_every_n_steps = 2,
                            callbacks = save_bests,
                            logger=False,
                            num_sanity_val_steps=0
                            )
    trainer.fit(model, datamodule=data)


def test(args):
    data = TSTrainDataModule(args.dataset, args.moving_avg, args.exo_num, args.covariate, args.rlen, 
                args.qlen, args.blen, args.sample_range, args.sample_num, args.batch_size, args.decomp)
    model = TSPredictor.load_from_checkpoint(args.ckpt_path)
    trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0,
                        log_every_n_steps = 2,
                        )
    trainer.test(model, datamodule=data)

def predict(args):
    data = TSTrainDataModule(args.dataset, args.moving_avg, args.exo_num, args.covariate, args.rlen, 
                args.qlen, args.blen, args.sample_range, args.sample_num, args.batch_size, args.decomp)
    model = TSPredictor.load_from_checkpoint(args.ckpt_path)
    trainer = pl.Trainer(gpus = 1 if torch.cuda.is_available() else 0)
    return trainer.predict(model, datamodule=data)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CoRe: Transferable Long-range Time Series Forecasting Enhanced by Covariates-Guided Representation')
    
    parser.add_argument('--mode', type=int, default=0, help='0 for training, 1 for validation, 2 for prediction')
    # model parameters
    parser.add_argument('--exo_num', type=int, default=4, help='number of auxiliary covariates')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--covariate', type=str, default='frequency', help='type of covariates')
    parser.add_argument('--enc_dim', type=int, default=64, help='dimension of model')
    parser.add_argument('--rlen', type=int, default=384, help='length of historical window')
    parser.add_argument('--qlen', type=int, default=96, help='length of prediction window')
    parser.add_argument('--blen', type=int, default=96, help='length of lookback window')
    parser.add_argument('--kernel_size', type=int, default=3, help='size of convolution kernel')
    parser.add_argument('--dilation', action='store_true', default=False, help='use dilation in convoluion layer')
    parser.add_argument('--nlayer', type=int, default=3, help='number of model layers')
    parser.add_argument('--sample_range', type=int, default=1000, help='sample range of historical window for evaluation')
    parser.add_argument('--sample_num', type=int, default=8, help='sample number of historical window for evaluation')
    parser.add_argument('--decomp', action='store_true', default=True, help='decomp data to trend and seasonal components')

    # train parameters
    parser.add_argument('--val_interval', type=float, default=0.002, help='interval of validation')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size')
    parser.add_argument('--epochs', type=int, default=10, help='number of training epochs')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--dataset', type=str, default='traffic', help='name of dataset')

    #test parameters
    parser.add_argument('--ckpt_path', type=str, default=None, help='relative checkpoint path')

    args = parser.parse_args()

    if args.mode == 0:
        train(args)
    elif args.mode == 1:
        test(args)
    elif args.mode == 2:
        predict(args)
