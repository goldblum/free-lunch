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
from oil.datasetup.datasets import CIFAR10, split_dataset
from pactl.data import get_dataset,get_data_dir

def makeTrainer(*,num_epochs=50,data_dir=None,projector=CombinedRDKronFiLM,k=16,
                bs=50,lr=.1,optim=SGD,device='cuda',trainer=Classifier,expt='',
                opt_config={},d=5000,seed=137,dataset='cifar10',
                trainer_config={'log_dir':None,'log_args':{'minPeriod':1.,'timeFrac':1/4}}):
    # datasets = {}
    # datasets['train'] = dataset(f'~/datasets/{dataset}/',train=True)
    # datasets['test'] = dataset(f'~/datasets/{dataset}/', train=False)
    trainset, testset=  get_dataset(dataset,root=get_data_dir(None),aug=False)
    # why do we have to set these again?
    trainset.class_weights=None
    trainset.ignored_index=-1
    testset.class_weights=None
    testset.ignored_index=-1
    datasets = {'train':trainset,'test':testset}
    device = torch.device(device)
    with FixedPytorchSeed(seed):
        model = layer13s(in_chans=datasets['train'].num_inputs,num_classes=datasets['train'].num_classes,base_width=k).to(device)
        model = IDModule(model,projector,d).to(device)

    dataloaders = {k:LoaderTo(DataLoader(v,batch_size=bs,shuffle=(k=='train'),
                num_workers=0,pin_memory=False),device) for k,v in datasets.items()}
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
            'd':[2000,3000,5000],
            'study_name':'img_label_compression','num_epochs':80,
            'dataset':['fmnist','svhn','cifar10','cifar100'],#,'BNG(autos,1000,1)','covertype']
        })
        cfg_spec = argupdated_config(cfg_spec)
        name = cfg_spec.pop('study_name')
        thestudy = Study(Trial,cfg_spec,study_name=name,
                base_log_dir=cfg_spec['trainer_config'].get('log_dir',None))
        thestudy.run(ordered=True)
        print(thestudy.covariates())
        print(thestudy.outcomes)