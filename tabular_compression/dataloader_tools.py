import logging
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from dataset_tools import get_data, get_categories_full_cat_data, TabularDataset


# Ignore statements for pylint:
#     Too many branches (R0912), Too many statements (R0915), No member (E1101),
#     Not callable (E1102), Invalid name (C0103), No exception (W0702),
#     Too many local variables (R0914), Missing docstring (C0116, C0115).
# pylint: disable=R0912, R0915, E1101, E1102, C0103, W0702, R0914, C0116, C0115



def get_dataloaders(cfg, train_batch_size, test_batch_size,balanced=True):
    x_numerical, x_categorical, y, info, full_cat_data_for_encoder = get_data(dataset_id=cfg['name'],
                                                                              source=cfg['source'],
                                                                              task=cfg['task'],
                                                                              datasplit=[.8, 0.1, 0.1],
                                                                              balanced=balanced)
    print("dataset size", y['train'].shape)
    
    # TODO: seed! Also, change cat_policy for xgboost
    dataset = TabularDataset(x_numerical, x_categorical, y, info, normalization=cfg['normalization'],
                             cat_policy="indices",
                             seed=0,
                             full_cat_data_for_encoder=full_cat_data_for_encoder,
                             y_policy=cfg['y_policy'])

    X = dataset.preprocess_data()
    Y, y_info = dataset.build_y()
    unique_categories = get_categories_full_cat_data(full_cat_data_for_encoder)
    n_numerical = dataset.n_num_features
    n_categorical = dataset.n_cat_features
    if cfg['task'] == 'binclass':
        n_classes = 2
        # convert it into multi-class labels
        for split in y.keys():
            y[split] = y[split].astype(int)
    elif cfg['task'] == 'multiclass':
        n_classes = dataset.n_classes
    else:
        raise NotImplementedError
    #n_classes = dataset.n_classes

    logging.info(f"Task: {cfg['task']}, Dataset: {cfg['name']}, n_numerical: {n_numerical}, "
                 f"n_categorical: {n_categorical}, n_classes: {n_classes}, n_train_samples: {dataset.size('train')}, "
                 f"n_val_samples: {dataset.size('val')}, n_test_samples: {dataset.size('test')}")

    trainset = TensorDataset(X[0]["train"], X[1]["train"], Y["train"])
    valset = TensorDataset(X[0]["val"], X[1]["val"], Y["val"])
    testset = TensorDataset(X[0]["test"], X[1]["test"], Y["test"])

    trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    valloader = DataLoader(valset, batch_size=train_batch_size, shuffle=True, drop_last=True)
    testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, drop_last=False)

    loaders = {"train": trainloader, "val": valloader, "test": testloader}
    return loaders, unique_categories, n_numerical, n_classes

# def get_balanced_dataloaders(cfg, train_batch_size, test_batch_size):
#     x_numerical, x_categorical, y, info, full_cat_data_for_encoder = get_data(dataset_id=cfg['name'],
#                                                                               source=cfg['source'],
#                                                                               task=cfg['task'],
#                                                                               datasplit=[.8, 0.1, 0.1])
#     bids = get_balanced_idx(y)
#     # TODO: seed! Also, change cat_policy for xgboost
#     dataset = TabularDataset(x_numerical[bids], x_categorical[bids], y[bids], info, normalization=cfg['normalization'],
#                              cat_policy="indices",
#                              seed=0,
#                              full_cat_data_for_encoder=full_cat_data_for_encoder,
#                              y_policy=cfg['y_policy'])

#     X = dataset.preprocess_data()
#     Y, y_info = dataset.build_y()
#     unique_categories = get_categories_full_cat_data(full_cat_data_for_encoder)
#     n_numerical = dataset.n_num_features
#     n_categorical = dataset.n_cat_features
#     n_classes = dataset.n_classes

#     logging.info(f"Task: {cfg['task']}, Dataset: {cfg['name']}, n_numerical: {n_numerical}, "
#                  f"n_categorical: {n_categorical}, n_classes: {n_classes}, n_train_samples: {dataset.size('train')}, "
#                  f"n_val_samples: {dataset.size('val')}, n_test_samples: {dataset.size('test')}")

#     trainset = TensorDataset(X[0]["train"], X[1]["train"], Y["train"])
#     valset = TensorDataset(X[0]["val"], X[1]["val"], Y["val"])
#     testset = TensorDataset(X[0]["test"], X[1]["test"], Y["test"])

#     trainloader = DataLoader(trainset, batch_size=train_batch_size, shuffle=True, drop_last=True)
#     valloader = DataLoader(valset, batch_size=train_batch_size, shuffle=True, drop_last=True)
#     testloader = DataLoader(testset, batch_size=test_batch_size, shuffle=True, drop_last=False)

#     loaders = {"train": trainloader, "val": valloader, "test": testloader}
#     return loaders, unique_categories, n_numerical, n_classes

def get_dataset_dict(name):
    ## Below is all OpenML classification datasets (de-duplicated) with at least 100k samples, <10 features, and no missing values,
    ## Excluded one dataset because it was actually just a text classification dataset.
    if name == 'SEA(50000)': #3 features all numerical, 2 classes imbalanced
        return {'name': 162, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'Agrawal1': #6 numerical features 3 categorical, 2 class imbalanced
        return {'name': 1235, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'Stagger1': #3 categorical features, 2 class imbalanced
        return {'name': 1236, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'airlines': #3 numerical features 4 numerical features, 2 class roughly balanced
        return {'name': 1169, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(glass,nominal,137781)': # 9 features all categorical, 7 classes imbalanced
        return {'name': 133, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'walking-activity': # 4 features all numerical, 22 classes imbalanced
        return {'name': 1509, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(breast-cancer,nominal,1000000)': # 9 features all categorical, 2 classes imbalanced
        return {'name': 77, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'skin-segmentation': # 3 features all numerical, 2 classes imbalanced
        return {'name': 1502, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'Idpa': #5 numerical features 2 categorical, 11 classes imbalanced
        return {'name': 1483, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'Click_prediction_small': #9 features all numerical, 2 classes super imbalanced
        return {'name': 1216, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'seattlecrime6': #2 numerical features, 5 categorical features, 141 classes imbalanced, some missing features but no missing labels
        return {'name': 41960, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'sf-police-incidents': #1 numerical features, 5 categorical features, 2 classes balanced
        return {'name': 42344, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
        
    ## Below are some additional datasets with more features in case we need them
    elif name == 'aloi': #128 features all numerical, 1k classes roughly balanced
        return {'name': 1592, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'covertype': #54 features all numerical, roughly balanced
        return {'name': 293, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'SantanderCustomerSatisfaction': #200 numerical features 1 categorical, imbalanced
        return {'name': 42435, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'poker-hand': #10 features all numerical, dominated by two classes
        return {'name': 1567, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(audiology,1000,1)': #69 features all categorical, 24 classes dominated by 2
        return {'name': 1387, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(autos,1000,1)': #15 numerical features 10 categorical, 7 classes imbalanced
        return {'name': 1393, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(anneal,1000,1)': #6 numerical features 32 categorical, 6 classes imbalanced
        return {'name': 1351, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(lymph,1000,1)': #3 numerical features 15 categorical, 4 classes dominated by 2
        return {'name': 1402, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(vote)': #16 features all categorical, 2 classes imbalanced
        return {'name': 143, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(solar-flare)': #12 features all categorical, 3 classes very very imbalanced
        return {'name': 1179, 'source': 'openml', 'task': 'multiclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(labor)': #8 numerical features 8 categorical, 2 classes imbalanced
        return {'name': 246, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
    elif name == 'BNG(SPECTF)': #44 features all numerical, 2 classes imbalanced
        return {'name': 1212, 'source': 'openml', 'task': 'binclass', 'normalization': 'standard', 'y_policy': None}
