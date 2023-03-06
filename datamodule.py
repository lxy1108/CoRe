import random
from typing import List, Optional, Tuple, Sequence

import numpy as np

import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl

from custom_dataset import PredictTrainDataset, PredictQueryDataset, PredictRefDataset, prepare_dataset
from data_attr import dataset_dict

class TSTrainDataModule(pl.LightningDataModule):
    def __init__(self, dataset: List,
                 moving_avg: int,
                 exo_num: int,
                 covariate: str,
                 rlen: int,
                 qlen: int,
                 blen: int,
                 sample_range: int,
                 sample_stride: int,
                 batch_size: int = 256,
                 decomp_trend: bool=True):
        super().__init__()
        self.name = dataset
        self.moving_avg = moving_avg
        self.exo_num = exo_num
        self.covariate = covariate
        self.rlen = rlen
        self.qlen = qlen
        self.blen = blen
        self.sample_range = sample_range
        self.sample_stride = sample_stride
        self.batch_size = batch_size
        self.decomp_trend = decomp_trend


    def train_dataloader(self) -> DataLoader:
        d, e = prepare_dataset(dataset_dict[self.name], moving_avg=self.moving_avg, exo_num=self.exo_num, covariate=self.covariate, decomp_trend=self.decomp_trend)
        train_dataset = PredictTrainDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, self.decomp_trend)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self) -> Sequence[DataLoader]:
        dataloaders = []
        d, e = prepare_dataset(dataset_dict[self.name], moving_avg=self.moving_avg, exo_num=self.exo_num, covariate=self.covariate, decomp_trend=self.decomp_trend)
        dataloaders.append(DataLoader(PredictRefDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, self.sample_stride, "valid", self.decomp_trend), batch_size=self.batch_size*8, shuffle=False))
        dataloaders.append(DataLoader(PredictQueryDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, "valid", self.decomp_trend), batch_size=self.batch_size*8, shuffle=False))
        return dataloaders

    def test_dataloader(self) -> Sequence[DataLoader]:
        dataloaders = []
        d, e = prepare_dataset(dataset_dict[self.name], moving_avg=self.moving_avg, exo_num=self.exo_num, covariate=self.covariate, decomp_trend=self.decomp_trend)
        dataloaders.append(DataLoader(PredictRefDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, self.sample_stride, "test",  self.decomp_trend), batch_size=self.batch_size*8, shuffle=False))
        dataloaders.append(DataLoader(PredictQueryDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, "test",  self.decomp_trend), batch_size=self.batch_size*8, shuffle=False))
        return dataloaders

    def predict_dataloader(self) -> Sequence[DataLoader]:
        dataloaders = []
        d, e = prepare_dataset(dataset_dict[self.name], moving_avg=self.moving_avg, exo_num=self.exo_num, covariate=self.covariate, decomp_trend=self.decomp_trend)
        dataloaders.append(DataLoader(PredictRefDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, self.sample_stride, "test",  self.decomp_trend), batch_size=self.batch_size*8, shuffle=False))
        dataloaders.append(DataLoader(PredictQueryDataset(self.name, self.rlen, self.qlen, self.blen, d, e, self.sample_range, "test", self.decomp_trend), batch_size=self.batch_size*8, shuffle=False))
        return dataloaders


