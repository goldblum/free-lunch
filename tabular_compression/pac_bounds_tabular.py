import torch
from torch.utils.data import DataLoader
from torch.optim import SGD,Adam
from oil.utils.utils import LoaderTo, cosLr, islice, export, imap
from oil.tuning.study import train_trial
#from oil.datasetup.datasets import split_dataset,CIFAR100,CIFAR10
from pactl.data import get_dataset,get_data_dir
#from oil.architectures.img_classifiers import layer13s
from oil.utils.parallel import try_multigpu_parallelize
from oil.tuning.args import argupdated_config
from oil.model_trainers.classifier import Classifier
from functools import partial

import torch.nn as nn
import torch.nn.functional as F
from pactl.nn.projectors import LazyRandom,IDModule,RoundedKron, FixedNumpySeed, FixedPytorchSeed,RoundedDoubleKron
from pactl.nn.projectors import FiLMLazyRandom,CombinedRDKronFiLM,RoundedDoubleKronQR
from oil.datasetup.datasets import augLayers,EasyIMGDataset
#import torchvision.datasets as datasets
import timm
from timm.data.transforms import RandomResizedCropAndInterpolation
import torchvision.transforms as transforms
import copy
from oil.tuning.study import Study
from oil.tuning.args import argupdated_config
import warnings
from pactl.nn import resnet20,layer13s
from pactl.nn.small_cnn import Expression
import pactl
import pandas as pd
#from pactl.bounds.get_bound_from_checkpoint import evaluate_idmodel,auto_eval
from oil.datasetup.augLayers import RandomTranslate,RandomHorizontalFlip
from pactl.bounds.get_bound_from_chk_v2 import evaluate_idmodel#,auto_eval
import numpy as np
from dataloader_tools import get_dataloaders, get_dataset_dict
import torch

def one_hot_embed_categorical(cat_feature, n_unique):
    """
    Embed categorical features with one-hot encoding
    """
    return F.one_hot(cat_feature, num_classes=n_unique).float()

def oh_embed_all(cat_feature, unique_categories):
    """
    Embed categorical features with one-hot encoding
    """
    all_ohs = [one_hot_embed_categorical(cat_feature[:,i], unique_categories[i]) \
         for i in range(cat_feature.shape[1])]
    return torch.cat(all_ohs, dim=-1)

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=100, n_hidden=2, activation=nn.ReLU()):
        super().__init__()
        self.num_classes = out_dim
        self.net = nn.Sequential(
            *[nn.Sequential(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), activation)
                for i in range(n_hidden)],
            nn.Linear(hidden_dim, out_dim)
        )
    def forward(self, x):
        return self.net(x.mean(-1).mean(-1))

def preprocess_mb(mb, unique_categories):
    """ Converts tabular features into small images"""
    x_n,x_c,y = mb
    if unique_categories:
        x_c = oh_embed_all(x_c, unique_categories).to(x_n.device)
        x = torch.cat([x_n.float(),x_c],-1)
    else:
        x = x_n.float()
    d = x.shape[-1]
    # round d up to nearest square
    h = int(np.ceil(np.sqrt(d)))
    z = torch.zeros(x.shape[0],h*h,dtype=x.dtype,device=x.device)
    z[:,:d] = x
    x = z.view(x.shape[0],1,h,h)
    x = nn.Upsample(scale_factor=2)(x)
    #x = (x[:,:,None,None]+ torch.zeros(x.shape[0],x.shape[-1],4,4).to(x.device)).float()
    return x, y.long()

def makeTrainer(*,num_epochs=20,data_dir=None,projector=RoundedDoubleKronQR,k=24,
                bs=512,lr=1e-4,optim=Adam,device='cuda',trainer=Classifier,expt='',
                opt_config={},d=1000,seed=137,dataset='airlines',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':1.,'timeFrac':1/4}}):

    dataset_dict = get_dataset_dict(dataset)
    loaders, unique_categories, n_numerical, n_classes = get_dataloaders(dataset_dict, bs, bs)
    device = torch.device(device)
    with FixedPytorchSeed(seed):
        cin = n_numerical+(sum(unique_categories) if unique_categories else 0)
        model = layer13s(in_chans=1,num_classes=n_classes,base_width=k).to(device)
        #model = MLP(cin, n_classes, hidden_dim=k*8, n_hidden=2).to(device)
        model = IDModule(model,projector,d).to(device)

    preprocessed_dl =lambda dl: imap(lambda mb: preprocess_mb(mb, unique_categories), dl)
    dataloaders = {k:LoaderTo(preprocessed_dl(v),device) for k,v in loaders.items()}
    dataloaders['Train'] = islice(dataloaders['train'],1+len(dataloaders['train'])//10)
    opt_constr = partial(optim, lr=lr, **opt_config)
    return trainer(model,dataloaders,opt_constr,**trainer_config)


import warnings
from oil.tuning.study import train_trial
from oil.tuning.args import argupdated_config


import os
import pandas as pd
import time

def Trial(cfg,i=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if i is not None:
            orig_suffix = cfg.setdefault('trainer_config',{}).get('log_suffix','')
            cfg['trainer_config']['log_suffix'] = os.path.join(orig_suffix,f'trial{i}/')
        trainer = makeTrainer(**cfg)
        
        # sub_params = sum([param.numel() for param in trainer.model.trainable_initparams])
        # params = sum([param.numel() for param in trainer.model.trainable_initparams])
        start = time.time()
        trainer.train(cfg['num_epochs'])
        df_out  = trainer.ckpt['outcome']
        df_out['time'] = time.time()-start
        if isinstance(trainer.model, IDModule):
            df_out['subspace_params']=trainer.model.d
            df_out['base_params']=trainer.model.D
            df_out['num_classes']=trainer.model._forward_net[0].num_classes
            df_out['ds_size'] = len(trainer.dataloaders['train'].dataset)
        try:
            pd.set_option('display.max_columns', 500)
            pd.set_option('display.width', 1000)
            df2 = pd.Series(evaluate_idmodel(trainer.model,trainer.dataloaders['train'],
            trainer.dataloaders['test'],lr=cfg['lr'],epochs=cfg['num_epochs']//4,use_kmeans=False,levels=9,misc_extra_bits=3))
            if isinstance(df_out,pd.DataFrame):
                df_out = df_out.iloc[0]
            df_out = df_out.append(df2)
            df_out = pd.DataFrame(df_out).T
            print(df_out)
        except Exception as e:
            print('failed to evaluate due to ',e)
        # combine the two
        torch.cuda.empty_cache()
        del trainer
        return cfg,df_out

simpleTrial = train_trial(makeTrainer)
if __name__=='__main__':
    with warnings.catch_warnings():
        cfg_spec = makeTrainer.__kwdefaults__
        warnings.simplefilter("ignore", category=UserWarning)
        cfg_spec = copy.deepcopy(cfg_spec)
        cfg_spec.update({
            'd':[250,500,100,2000],
            'study_name':'test_several_long_smalld','num_epochs':40,
            'dataset':['Agrawal1','airlines','Click_prediction_small','walking-activity',
            'BNG(labor)','BNG(SPECTF)','BNG(vote)'],#,'BNG(autos,1000,1)','covertype']
        })
        cfg_spec = argupdated_config(cfg_spec)
        name = cfg_spec.pop('study_name')
        thestudy = Study(Trial,cfg_spec,study_name=name,
                base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
        thestudy.run(ordered=True)
        print(thestudy.covariates())
        print(thestudy.outcomes)
