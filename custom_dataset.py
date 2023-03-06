from logging import raiseExceptions
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
import matplotlib.pyplot as plt
import scipy.signal
from data_attr import DatasetAttr, dataset_dict
import random

def sliding_window_view(x, window_shape, axis=None, *,
                        subok=False, writeable=False):
    window_shape = (tuple(window_shape)
                    if np.iterable(window_shape)
                    else (window_shape,))
    # first convert input to array, possibly keeping subclass
    x = np.array(x, copy=False, subok=subok)

    window_shape_array = np.array(window_shape)
    if np.any(window_shape_array < 0):
        raise ValueError('`window_shape` cannot contain negative values')

    axis = (axis,)

    out_strides = x.strides + tuple(x.strides[ax] for ax in axis)

    # note: same axis can be windowed repeatedly
    x_shape_trimmed = list(x.shape)
    for ax, dim in zip(axis, window_shape):
        if x_shape_trimmed[ax] < dim:
            raise ValueError(
                'window shape cannot be larger than input array shape')
        x_shape_trimmed[ax] -= dim - 1
    out_shape = tuple(x_shape_trimmed) + window_shape
    return np.lib.stride_tricks.as_strided(x, strides=out_strides, shape=out_shape,
                      subok=subok, writeable=writeable)


def decomp(x: np.ndarray, kernel_size: int) -> np.ndarray:
    front = np.tile(x[:, 0:1], (1, (kernel_size - 1) // 2))
    end = np.tile(x[:, -1:], (1, (kernel_size - 1) // 2))
    x_pad = np.concatenate([front, x, end], axis=1)
    trend = sliding_window_view(x_pad, kernel_size, axis=1).mean(-1)
    seasonal = x - trend
    return seasonal, trend

def gen_exo(data:np.ndarray, group_size: int, kernel_size: int, exo_num: int) -> np.ndarray:
    group_num = data.shape[0] // group_size
    x, _ = decomp(data, kernel_size)
    # x = data
    x_ft = np.fft.rfft(x, axis=-1)
    amp = np.abs(x_ft)
    fre = np.fft.fftfreq(x.shape[-1])[:x_ft.shape[-1]]
    top_fre_list = []
    for i in range(group_num):
        peaks, props = scipy.signal.find_peaks(amp[i * group_size: (i + 1) *group_size].sum(0), distance=200, height=0)
        heights = props["peak_heights"]
        top_heights_ind = np.argpartition(heights, -exo_num)[-exo_num:]
        top_ind = peaks[top_heights_ind]
        top_fre_list.append(fre[top_ind])
    top_fre_list = np.stack(top_fre_list, axis=0)
    t = np.tile(np.arange(x.shape[1])[np.newaxis, :], (group_num, 1))
    print(1 / top_fre_list)
    exo = [np.sin(2 * np.pi * t * top_fre_list[:, i: i + 1]) for i in range(exo_num)]
    # exo = [np.sin(2 * np.pi * t) for i in range(exo_num)]
    return np.stack(exo, axis=1).squeeze()


def prepare_dataset(data_attr: DatasetAttr, 
                    moving_avg=25,
                    exo_num=4,
                    features='M', 
                    target='OT', 
                    covariate='calendar',
                    scale=True, 
                    decomp_trend = True):
        scaler = StandardScaler()
        df_raw = pd.read_csv(data_attr.path)

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [target]]

        if features == 'M' or features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif features == 'S':
            df_data = df_raw[[target]]
        else:
            raise ValueError(f"Unexpected features : {features}")
        
        # data = decomp(df_data.values.T, 95)
        data = df_data.values.T
        
        if scale:
            if "ETTh" in data_attr.path.stem:
                num_train = 12 * 30 * 24
            elif "ETTm" in data_attr.path.stem:
                num_train = 12 * 30 * 24 * 4
            else:
                num_train = int(len(df_raw) * 0.7)
            scaler.fit(data[:, 0:num_train].T)
            data = scaler.transform(data.T).T

        # if mode == "train":
        #     data, _ = decomp(data, 75*3)

        # seasonal0, trend0 = decomp(data, 25)
        # seasonal1, trend1 = decomp(trend0, 75)
        # data = seasonal0 + seasonal1

        if covariate == 'calendar':
            df_stamp = df_raw[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1) / 12
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1) / 31
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1) / 7
            if data_attr.freq in ["h", "m"]:
                df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1) / 24
            if data_attr.freq == "m":
                df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1) / 60
                # df_stamp['minute'] = df_stamp.minute.map(lambda x: x // int(freq[:-1]))
            data_stamp = df_stamp.drop(['date'], 1).values.T
        elif covariate == 'frequency':
            data_stamp = gen_exo(data, data_attr.dimension, moving_avg, exo_num)
        else:
            raise ValueError('invalid covariate: {}'.format(covariate))
        if decomp_trend:
            seasonal, trend = decomp(data, moving_avg)
            data = np.stack([seasonal, trend], axis=1).reshape(-1, data.shape[-1])
        return torch.from_numpy(data).float(), torch.from_numpy(data_stamp).float()

class PredictTrainDataset(TensorDataset):
    def __init__(self, name, rlen, qlen, blen, data, data_stamp, sample_range, decomp_trend=True):
        self.name = name
        self.rlen = rlen
        self.qlen = qlen
        self.blen = blen
        self.sample_range = sample_range
        # if generalize == True:
        #     tstart, tend = 0, data.shape[1]
        # else:
        if name.startswith("etth"):
            num_train = 12 * 30 * 24
        elif name.startswith("ettm"):
            num_train = 12 * 30 * 24 * 4
        else:
            num_train = int(data.shape[1] * 0.7)
        tstart, tend = 0, num_train
        self.data = data[..., tstart: tend]
        self.data_stamp = data_stamp[..., tstart: tend]

        self.group_size = dataset_dict[self.name].dimension * (2 if decomp_trend else 1)
        self.group_num = self.data.shape[0] // self.group_size
        self.temporal_len = self.data.shape[1] - self.rlen - self.qlen + 1


    def __getitem__(self, index):
        ins_begin, ref_begin_max = index // self.temporal_len * self.group_size, index % self.temporal_len
        ins_end = ins_begin + self.group_size
        ref_begin_min = max(ref_begin_max - self.sample_range, 0)
        ref_begin = random.randint(ref_begin_min, ref_begin_max)
        ref_end = ref_begin + self.rlen
        query_begin = ref_begin_max + self.rlen - self.blen
        query_end = query_begin + self.blen + self.qlen

        ref_y = self.data[ins_begin: ins_end, ref_begin: ref_end]
        query_y = self.data[ins_begin: ins_end, query_begin: query_end]
        if len(self.data_stamp.shape) == 2:
            ref_x = self.data_stamp[:, ref_begin: ref_end]
            query_x = self.data_stamp[:, query_begin: query_end]
        else:
            ref_x = self.data_stamp[ins_begin // self.group_size, :, ref_begin: ref_end]
            query_x = self.data_stamp[ins_begin // self.group_size, :, query_begin: query_end]

        # print(ref_x.shape, ref_y.shape, query_x.shape, query_y.shape)
        return ref_x, ref_y, query_x, query_y

    def __len__(self):
        return self.temporal_len * self.group_num


class PredictQueryDataset(TensorDataset):
    def __init__(self, name, rlen, qlen, blen, data, data_stamp, sample_range, mode="valid", decomp_trend=True):
        self.name = name
        self.rlen = rlen
        self.qlen = qlen
        self.blen = blen
        self.sample_range = sample_range
        self.mode = mode
        # if generalize == True:
        #     tstart, tend = sample_range, data.shape[1]
        # else:
        if name.startswith("etth"):
            num_train = 12 * 30 * 24
            num_test = 4 * 30 * 24
            num_vali = 4 * 30 * 24
        elif name.startswith("ettm"):
            num_train = 12 * 30 * 24 * 4
            num_vali = 4 * 30 * 24 * 4
            num_test = 4 * 30 * 24 * 4
        else:
            num_train = int(data.shape[1] * 0.7)
            num_test = int(data.shape[1] * 0.2)
            num_vali = data.shape[1] - num_train - num_test
        if mode == "valid":
            tstart, tend = num_train - blen, num_train + num_vali
        elif mode == "test":
            tstart, tend = num_train + num_vali - blen, num_train + num_vali + num_test
        else:
            raise ValueError("invalid mode: {}".format(mode))
        self.data = data[..., tstart: tend]
        self.data_stamp = data_stamp[..., tstart: tend]

        self.group_size = dataset_dict[self.name].dimension * (2 if decomp_trend else 1)
        self.group_num = self.data.shape[0] // self.group_size
        self.temporal_len = self.data.shape[1] - self.blen - self.qlen + 1


    def __getitem__(self, index):
        ins_begin, query_begin = index // self.temporal_len * self.group_size, index % self.temporal_len
        ins_end = ins_begin + self.group_size
        query_end = query_begin + self.blen + self.qlen

        query_y = self.data[ins_begin: ins_end, query_begin: query_end]
        if len(self.data_stamp.shape) == 2:
            query_x = self.data_stamp[:, query_begin: query_end]
        else:
            query_x = self.data_stamp[ins_begin // self.group_size, :, query_begin: query_end]
        return query_x, query_y, ins_begin // self.group_size

    def __len__(self):
        return self.temporal_len * self.group_num

class PredictRefDataset(TensorDataset):
    def __init__(self, name, rlen, qlen, blen, data, data_stamp, sample_range, sample_num, mode="valid", decomp_trend=True):
        self.name = name
        self.rlen = rlen
        self.qlen = qlen
        self.blen = blen
        self.sample_range = sample_range
        self.sample_num = sample_num
        self.sample_stride = (sample_range - rlen) // sample_num

        if name.startswith("etth"):
            num_train = 12 * 30 * 24
            num_test = 4 * 30 * 24
            num_vali = 4 * 30 * 24
        elif name.startswith("ettm"):
            num_train = 12 * 30 * 24 * 4
            num_vali = 4 * 30 * 24 * 4
            num_test = 4 * 30 * 24 * 4
        else:
            num_train = int(data.shape[1] * 0.7)
            num_test = int(data.shape[1] * 0.2)
            num_vali = data.shape[1] - num_train - num_test
        if mode == "valid":
            tend = num_train
        elif mode == "test":
            tend = num_train + num_vali
        else:
            raise ValueError("invalid mode: {}".format(mode))
        self.data = data[..., tend - sample_range : tend]
        self.data_stamp = data_stamp[..., tend - sample_range : tend]
        
        self.group_size = dataset_dict[self.name].dimension * (2 if decomp_trend else 1)
        self.group_num = self.data.shape[0] // self.group_size


    def __getitem__(self, index):
        ins_begin, sample_id = index // self.sample_num * self.group_size, index % self.sample_num
        ins_end = ins_begin + self.group_size
        sam_begin = sample_id * self.sample_stride

        ref_y = self.data[ins_begin: ins_end, sam_begin: sam_begin + self.rlen]
        if len(self.data_stamp.shape) == 2:
            ref_x = self.data_stamp[:, sam_begin: sam_begin + self.rlen]
        else:
            ref_x = self.data_stamp[ins_begin // self.group_size, :, sam_begin: sam_begin + self.rlen]

        return ref_x, ref_y

    def __len__(self):
        return self.sample_num * self.group_num