import os
from utils import load_pickle, get_mdl, flatten
from mingpt.utils import set_seed
import statistics
import random
from scipy.stats import ttest_ind

load_dir = './saved_dicts/'
save_dir = './saved_plots/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
sequence_length = 100
models = ['gpt2','gpt2-medium','gpt2-large']
seeds = range(10)
sample = True
pretrained = False


set_seed(0)
num_samples = 100000
chars = ['0','1']
uniform_samples = [get_mdl(''.join(random.choice(chars) for i in range(sequence_length))) for j in range(num_samples)]
uniform_mean = statistics.mean(uniform_samples)
uniform_std = statistics.stdev(uniform_samples)
print('uniform_mean', uniform_mean)
print('uniform_std', uniform_std)


mean_dict = {}
std_dict = {}
samples_dict = {}

ttest_dict = {}

for model in models:
    if pretrained:
        filenames = [os.path.join(load_dir, 'longsamples_'+model+'_pretrained_length'+str(sequence_length)+'_seed'+str(seed)+'.pickle') for seed in seeds]
    else:
        filenames = [os.path.join(load_dir, 'longsamples_'+model+'_initialized_length'+str(sequence_length)+'_sample_'+str(sample)+'_seed'+str(seed)+'.pickle') for seed in seeds]
    mdl_list = flatten([load_pickle(filename) for filename in filenames])
    ttest_dict[model] = ttest_ind(mdl_list, uniform_samples, equal_var = False, alternative='less')
    mean_dict[model] = statistics.mean(mdl_list)    
    std_dict[model] = statistics.stdev(mdl_list)
    samples_dict[model] = len(mdl_list)
    
    
print('means', mean_dict)
print('stds', std_dict)  
print('num_samples', samples_dict)  
print('t-tests', ttest_dict)

