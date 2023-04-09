import numpy as np
import openml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
import logging
from dataclasses import dataclass
import typing as ty
import warnings
from copy import deepcopy
import sklearn.preprocessing
import torch


# for downloading and parsing OpenML data

def get_categories_full_cat_data(full_cat_data_for_encoder):
    return (
        None
        if full_cat_data_for_encoder is None
        else [
            len(set(full_cat_data_for_encoder.values[:, i]))
            for i in range(full_cat_data_for_encoder.shape[1])
        ]
    )



def get_data_openml(dataset_id):
    dataset = openml.datasets.get_dataset(dataset_id)
    data, targets, categorical_indicator, attribute_names = dataset.get_data(dataset_format="dataframe",
                                                                                 target=dataset.default_target_attribute)
    categorical_columns = list(data.columns[np.array(categorical_indicator)])
    numerical_columns = list(data.columns[~np.array(categorical_indicator)])
    return data, targets, categorical_columns, numerical_columns


def get_data_locally(name):
    cat_path = '../../../data/{}/C_{}.npy'.format(name, 'train')
    if os.path.exists(cat_path):
        categorical_array = np.vstack([np.load('../../../data/{}/C_{}.npy'.format(name, part)) for part in ['train', 'val', 'test']])
        categorical_columns = ['cat_{}'.format(i) for i in range(categorical_array.shape[1])]
        cat_df = pd.DataFrame(categorical_array, columns=categorical_columns)
    else:
        cat_df = pd.DataFrame()
        categorical_columns = []

    num_path = '../../../data/{}/N_{}.npy'.format(name, 'train')
    if os.path.exists(num_path):
        numerical_array = np.vstack([np.load('../../../data/{}/N_{}.npy'.format(name, part)) for part in ['train', 'val', 'test']])
        numerical_columns = ['num_{}'.format(i) for i in range(numerical_array.shape[1])]
        num_df =pd.DataFrame(numerical_array, columns=numerical_columns)
    else:
        num_df = pd.DataFrame()
        numerical_columns = []

    data = pd.concat([num_df, cat_df], axis = 1)
    assert data.shape[0] == num_df.shape[0]

    targets = np.concatenate([np.load('../../../data/{}/y_{}.npy'.format(name, part)) for part in ['train', 'val', 'test']])
    targets = pd.Series(targets, name = 'target')
    return data, targets, categorical_columns, numerical_columns


def get_balanced_idx(y):
    """ Get the ids required to class balanced dataset X,y by equal numbers for
    each value of y
    """
    assert y.shape in ((y.shape[0], 1), (y.shape[0], )), "y must be a vector"
    y = y.flatten()
    unique_y = np.unique(y)
    n_classes = len(unique_y)
    n_samples_per_class = np.min([np.sum(y == unique_y[i]) for i in range(n_classes)])
    idx = []
    for i in range(n_classes):
        idx.append(np.random.choice(np.where(y == unique_y[i])[0], n_samples_per_class, replace=False))
    idx = np.concatenate(idx)
    return idx

def get_data(dataset_id, source, task, datasplit=[.65, .15, .2], seed = 0, balanced=False):
    # TODO: we want to change strategy for seeding

    if source == 'openml':
        data, targets, categorical_columns, numerical_columns = get_data_openml(dataset_id)
    elif source == 'local':
        data, targets, categorical_columns, numerical_columns = get_data_locally(dataset_id)

    np.random.seed(seed)
    # Fixes some bugs in openml datasets
    if targets.dtype.name == "category":
        targets = targets.apply(str).astype('object')

    for col in categorical_columns:
        data[col] = data[col].apply(str).astype("object")

    # reindex and find NaNs/Missing values in categorical columns
    data, targets = data.reset_index(drop=True), targets.reset_index(drop=True)
    data[categorical_columns] = data[categorical_columns].fillna("___null___")
    if balanced:
        assert len(targets.values.shape)==1, "Balanced sampling only works for 1D targets"
        idx = get_balanced_idx(targets.values)
        data, targets = data.iloc[idx].reset_index(drop=True), targets.iloc[idx].reset_index(drop=True)


    # split data into train/val/test
    # TODO add stratified sampling and possible deterministic sampling (see sk_learn)
    data["Set"] = np.random.choice(["train", "valid", "test"], p=datasplit, size=(data.shape[0],))
    train_indices = data[data.Set == "train"].index
    valid_indices = data[data.Set == "valid"].index
    test_indices = data[data.Set == "test"].index
    data = data.drop(columns=['Set'])

    if task != 'regression':
        l_enc = LabelEncoder()
        targets = l_enc.fit_transform(targets)
    else:
        targets = targets.to_numpy()

    data_cat_train = data[categorical_columns].values[train_indices]
    data_num_train = data[numerical_columns].values[train_indices]
    targets_train = targets[train_indices]

    data_cat_val = data[categorical_columns].values[valid_indices]
    data_num_val = data[numerical_columns].values[valid_indices]
    targets_val = targets[valid_indices]

    data_cat_test = data[categorical_columns].values[test_indices]
    data_num_test = data[numerical_columns].values[test_indices]
    targets_test = targets[test_indices]

    info = {"name": dataset_id,
            "task_type": task,
            "n_num_features": len(numerical_columns),
            "n_cat_features": len(categorical_columns),
            "train_size": len(train_indices),
            "val_size": len(valid_indices),
            "test_size": len(test_indices)}

    if task == "multiclass":
        info["n_classes"] = len(set(targets))
    if task == "binclass":
        info["n_classes"] = 1
    if task == "regression":
        info["n_classes"] = 1

    if len(numerical_columns) > 0:
        numerical_data = {"train": data_num_train, "val": data_num_val, "test": data_num_test}
    else:
        numerical_data = None

    if len(categorical_columns) > 0:
        categorical_data = {"train": data_cat_train, "val": data_cat_val, "test": data_cat_test}
    else:
        categorical_data = None

    targets = {"train": targets_train, "val": targets_val, "test": targets_test}

    if len(categorical_columns) > 0:
        full_cat_data_for_encoder = data[categorical_columns]
    else:
        full_cat_data_for_encoder = None

    return numerical_data, categorical_data, targets, info, full_cat_data_for_encoder



ArrayDict = ty.Dict[str, np.ndarray]


@dataclass
class TabularDataset:
    x_num: ty.Optional[ArrayDict]
    x_cat: ty.Optional[ArrayDict]
    y: ArrayDict
    info: ty.Dict[str, ty.Any]
    normalization: ty.Optional[str]
    cat_policy: str
    seed: int
    full_cat_data_for_encoder: ty.Optional[pd.DataFrame]
    y_policy: ty.Optional[str] = None

    @property
    def is_binclass(self) -> bool:
        return self.info['task_type'] == "binclass"

    @property
    def is_multiclass(self) -> bool:
        return self.info['task_type'] == "multiclass"

    @property
    def is_regression(self) -> bool:
        return self.info['task_type'] == "regression"

    @property
    def n_num_features(self) -> int:
        return self.info["n_num_features"]

    @property
    def n_cat_features(self) -> int:
        return self.info["n_cat_features"]

    @property
    def n_features(self) -> int:
        return self.n_num_features + self.n_cat_features

    @property
    def n_classes(self) -> int:
        return self.info["n_classes"]

    @property
    def parts(self) -> int:
        return self.x_num.keys() if self.x_num is not None else self.x_cat.keys()

    def size(self, part: str) -> int:
        x = self.x_num if self.x_num is not None else self.x_cat
        assert x is not None
        return len(x[part])



    def normalize(self, x_num, noise=1e-3):
        x_num_train = x_num['train'].copy()
        if self.normalization == 'standard':
            normalizer = sklearn.preprocessing.StandardScaler()
        elif self.normalization == 'quantile':
            normalizer = sklearn.preprocessing.QuantileTransformer(
                output_distribution='normal',
                n_quantiles=max(min(x_num['train'].shape[0] // 30, 1000), 10),
                subsample=1e9,
                random_state=self.seed,
            )
            if noise:
                stds = np.std(x_num_train, axis=0, keepdims=True)
                noise_std = noise / np.maximum(stds, noise)
                x_num_train += noise_std * np.random.default_rng(self.seed).standard_normal(x_num_train.shape)
        else:
            raise ValueError('Unknown Normalization')
        normalizer.fit(x_num_train)
        return {k: normalizer.transform(v) for k, v in x_num.items()}

    def handle_missing_values_numerical_features(self, x_num):
        # TODO: handle num_nan_masks for SAINT
        # num_nan_masks_int = {k: (~np.isnan(v)).astype(int) for k, v in x_num.items()}
        num_nan_masks = {k: np.isnan(v) for k, v in x_num.items()}
        if any(x.any() for x in num_nan_masks.values()):

            # TODO check if we need self.x_num here
            num_new_values = np.nanmean(self.x_num['train'], axis=0)
            for k, v in x_num.items():
                num_nan_indices = np.where(num_nan_masks[k])
                v[num_nan_indices] = np.take(num_new_values, num_nan_indices[1])
        return x_num

    def encode_categorical_features(self, x_cat):
        encoder = sklearn.preprocessing.OrdinalEncoder(handle_unknown='error', dtype='int64')
        encoder.fit(self.full_cat_data_for_encoder.values)
        x_cat = {k: encoder.transform(v) for k, v in x_cat.items()}
        return x_cat

    def transform_categorical_features_to_ohe(self, x_cat):
        ohe = sklearn.preprocessing.OneHotEncoder(handle_unknown='ignore', sparse=False, dtype='float32')
        ohe.fit(self.full_cat_data_for_encoder.astype('str'))
        x_cat = {k: ohe.transform(v.astype('str')) for k, v in x_cat.items()}
        return x_cat

    def concatenate_data(self, x_cat, x_num):
        if self.cat_policy == 'indices':
            result = [x_num, x_cat]

        elif self.cat_policy == 'ohe':
            # TODO: handle output for models that need ohe
            raise ValueError('Not implemented')
        return result

    def preprocess_data(self) -> ty.Union[ArrayDict, ty.Tuple[ArrayDict, ArrayDict]]:
        # TODO: seed (?)
        logging.info('Building Dataset')
        # TODO: figure out if we really need a copy of data or if we can preprocess it in place
        if self.x_num:
            x_num = deepcopy(self.x_num)
            x_num = self.handle_missing_values_numerical_features(x_num)
            if self.normalization:
                x_num = self.normalize(x_num)
        else:
            # if x_num is None replace with empty tensor for dataloader
            x_num = {part: torch.empty(self.size(part),0) for part in self.parts}

        # if there are no categorical features, return only numerical features
        if self.cat_policy == 'drop' or not self.x_cat:
            assert x_num is not None
            x_num = to_tensors(x_num)
            # if x_cat is None replace with empty tensor for dataloader
            x_cat = {part: torch.empty(self.size(part),0) for part in self.parts}
            return [x_num, x_cat]

        x_cat = deepcopy(self.x_cat)
        # x_cat_nan_masks = {k: v == '___null___' for k, v in x_cat.items()}
        x_cat = self.encode_categorical_features(x_cat)

        x_cat, x_num = to_tensors(x_cat), to_tensors(x_num)
        result = self.concatenate_data(x_cat, x_num)

        return result

    def build_y(self) -> ty.Tuple[ArrayDict, ty.Optional[ty.Dict[str, ty.Any]]]:
        if self.is_regression:
            assert self.y_policy == 'mean_std'
        y = deepcopy(self.y)
        if self.y_policy:
            if not self.is_regression:
                warnings.warn('y_policy is not None, but the task is NOT regression')
                info = None
            elif self.y_policy == 'mean_std':
                mean, std = self.y['train'].mean(), self.y['train'].std()
                y = {k: (v - mean) / std for k, v in y.items()}
                info = {'policy': self.y_policy, 'mean': mean, 'std': std}
            else:
                raise ValueError('Unknown y policy')
        else:
            info = None

        y = to_tensors(y)
        if self.is_regression or self.is_binclass:
            y = {part: y[part].float() for part in self.parts}
        return y, info


def to_tensors(data: ArrayDict) -> ty.Dict[str, torch.Tensor]:
    return {k: torch.as_tensor(v) for k, v in data.items()}
