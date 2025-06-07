# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
The dataloader for loading .ts data.
Refer to https://github.com/gzerveas/mvts_transformer.
"""
import logging
from typing import Optional, Union

import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from romae.utils import gen_mask
from .load_from_tsfile_to_dataframe import load_from_tsfile_to_dataframe
from .base import BaseDataset
import numpy as np

logger = logging.getLogger(__name__)


class TSDataset(BaseDataset):
    """
    A dataset for time series forecasting.
    """

    def __init__(
            self,
            prefix: str = "data/Multivariate_ts",
            name: str = "Heartbeat",
            max_seq_len: int = 0,
            dataset: str = "train",
            dataset_split_ratio: float = 0.,
            task: str = "classification",
            preprocessor = None,
    ):
        super().__init__(preprocessor=preprocessor)

        if max_seq_len == 0:
            train_df, _ = load_from_tsfile_to_dataframe(f"{prefix}/{name}/{name}_TRAIN.ts")
            train_lengths = train_df.applymap(len).values
            test_df, _ = load_from_tsfile_to_dataframe(f"{prefix}/{name}/{name}_TEST.ts")
            test_lengths = test_df.applymap(len).values
            self._max_seq_len = int(max(np.max(train_lengths), np.max(test_lengths)))
        else:
            self._max_seq_len = max_seq_len

        original_dataset = dataset
        if dataset == 'valid':
            original_dataset = 'train'
        df, labels = load_from_tsfile_to_dataframe(f"{prefix}/{name}/{name}_{original_dataset.upper()}.ts")

        if task == "classification":
            labels = pd.Series(labels, dtype="category")
            self.class_names = labels.cat.categories
            labels_df = pd.DataFrame(labels.cat.codes, dtype=np.int64)
        elif task == "regression":
            labels_df = pd.DataFrame(labels, dtype=np.float32)
        else:
            raise ValueError(f"Unknown task: {task}")

        assert len(df) == len(labels)
        self.datasize = len(df)
        lengths = df.applymap(len).values

        df = pd.concat((
            pd.DataFrame({col: df.loc[row, col] for col in df.columns})
            .reset_index(drop=True)
            .set_index(pd.Series(np.max(lengths[row, :]) * [row]))
            for row in range(df.shape[0])
        ),
            axis=0,
        )

        def interpolate_missing(y):
            """
            Replaces NaN values in pd.Series `y` using linear interpolation
            """
            if y.isna().any():
                y = y.interpolate(method="linear", limit_direction="both")
            return y

        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)
        # Fill remaining NaNs with 0. This will happen if all values in a column are NaN
        df = df.fillna(0)

        feature = []
        for i in range(self.datasize):
            nowdf = df.loc[i]
            _seq_len = len(nowdf)
            newdf = pd.DataFrame(index=[i for num in range(self._max_seq_len - _seq_len)],
                                 columns=nowdf.columns).fillna(0)
            feature.append(pd.concat([newdf, nowdf]))

        self.feature = pd.concat(feature)
        self.label = labels_df

        assert dataset_split_ratio != 0 or dataset != 'valid', 'valid dataset must have dataset_split_ratio'

        if dataset in ['train', 'valid'] and dataset_split_ratio > 0:
            data_index = np.arange(self.datasize)
            label_np = self.label.values
            if task == "classification":
                stratify = label_np
            elif task == "regression":
                stratify = None
            ind_x_train, ind_x_valid, _, _ = train_test_split(data_index, label_np,
                                                              test_size=dataset_split_ratio, stratify=stratify,
                                                              shuffle=True, random_state=42)
            if dataset == 'train':
                ind_x = ind_x_train
            elif dataset == 'valid':
                ind_x =  ind_x_valid
            self.datasize = len(ind_x)
            self.feature = self.feature.loc[ind_x]
            self.label = self.label.loc[ind_x]
            rename_dict = {v: k for k, v in enumerate(ind_x)}
            self.feature = self.feature.rename(index=rename_dict)
            self.label = self.label.rename(index=rename_dict)

        self.x_shape = self.initialize_transformations_pandas(self.feature, self.preprocessor.x_transformation)
        self.y_shape = self.initialize_transformations_pandas(self.label, self.preprocessor.y_transformation)

        if not self.preprocessor.online_preprocess:
            self.feature = self.perform_transformations_pandas(self.feature, self.preprocessor.x_transformation)
            self.label = self.perform_transformations_pandas(self.label, self.preprocessor.y_transformation)

        self.feature = self.feature.astype(np.float32)

    def initialize_transformations_pandas(self, df, transformations):
        batch_feature = []
        for i in range(self.datasize):
            tmp_feature = df.loc[i].to_numpy()
            batch_feature.append(tmp_feature.reshape((1, *tmp_feature.shape)))
        feature_numpy = np.concatenate(batch_feature, axis=0)
        shape = self.initialize_transformations(feature_numpy, transformations,
                                                meta=None, reshape_param=df.loc[0].to_numpy().shape)
        return shape

    def perform_transformations_pandas(self, df, transformations):
        if transformations is None:
            return df
        for transform in transformations:
            batch_transform_data = []
            for i in range(self.datasize):
                transformed_data = transform.transform(df.loc[i].values)
                transformed_dataframe = pd.DataFrame(index=[i] * len(transformed_data),
                                                     data=transformed_data,
                                                     columns=np.arange(transformed_data.shape[1]))
                batch_transform_data.append(transformed_dataframe)
            df = pd.concat(batch_transform_data)
        return df

    @property
    def num_classes(self):
        try:
            return len(self.class_names)
        except AttributeError:
            return 1

    @property
    def num_variables(self):
        return self.x_shape[-1]

    @property
    def max_seq_len(self):
        return self.x_shape[-2]

    def __len__(self):
        return self.datasize

    def __getitem__(self, idx):
        feature = self.feature.loc[idx].to_numpy()
        label = self.label.loc[idx].to_numpy()

        if self.preprocessor.online_preprocess:
            feature = self.perform_transformations(feature,
                                                   self.preprocessor.x_transformation,
                                                   self.feature.loc[idx].to_numpy().shape[1:])
            label = self.perform_transformations(label,
                                                 self.preprocessor.y_transformation,
                                                 self.label.loc[idx].to_numpy().shape[1:])
        return feature, label

    def get_values(self):
        return self.feature.to_numpy(), self.label.to_numpy()


class MaskTSDataset(TSDataset):
    """
    A dataset for time series forecasting.
    """

    def __init__(
            self,
            prefix: str = "data/Multivariate_ts",
            name: str = "Heartbeat",
            max_seq_len: int = 0,
            dataset: str = "train",
            dataset_split_ratio: float = 0.,
            task: str = "classification",
            preprocessor = None,
            mask_ratio: Optional[float] = 0.3,
    ):
        super().__init__(prefix, name, max_seq_len, dataset, dataset_split_ratio, task, preprocessor)

        self.mask_ratio = mask_ratio

        for i in range(self.datasize):
            _length = self._max_seq_len - self.feature.loc[i]['dim_0'].isnull().sum()
            removed_points = np.random.choice(_length, int(_length * self.mask_ratio),
                                              replace=False) + self.feature.loc[i]['dim_0'].isnull().sum()
            for idx in removed_points:
                self.feature.loc[i].iloc[idx] = np.nan


class UEADataset(Dataset):
    """
    Adapts MaskTSDataset for use with RoMAE
    """
    def __init__(self,
        prefix: str = "data/Multivariate_ts",
        name: str = "Heartbeat",
        max_seq_len: int = 0,
        dataset: str = "train",
        dataset_split_ratio: float = 0.,
        task: str = "classification",
        preprocessor = None,
        mask_ratio: Optional[float] = 0.3,
        pretrain_mask_ratio: float = 0.):
        self.inner_ds = MaskTSDataset(
            prefix=prefix,
            name=name,
            max_seq_len=max_seq_len,
            dataset=dataset,
            dataset_split_ratio=dataset_split_ratio,
            task=task,
            preprocessor=preprocessor,
            mask_ratio=mask_ratio,
        )
        self.pretrain_mask_ratio = pretrain_mask_ratio

    def __len__(self):
        return len(self.inner_ds)

    def __getitem__(self, idx):
        value, label = self.inner_ds[idx]
        value, label = torch.tensor(value, dtype=torch.float32), label
        positions = torch.arange(value.shape[0])
        nan_mask = value[:, 0].isnan()
        value, positions = value[~nan_mask], positions[~nan_mask]
        mask = gen_mask(self.pretrain_mask_ratio,
                        torch.zeros_like(positions, dtype=torch.bool)[None, ...],
                        single=True).squeeze()
        return {
            "values": value[..., None, None],
            "label": label[0],
            "positions": positions[None, ...],
            "mask": mask
        }
