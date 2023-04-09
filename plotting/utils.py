from collections import OrderedDict
import itertools
import math
import statistics
import torch.nn as nn
import torch
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

def sequences_by_mdl(sequence_length=10):
    programs = []
    for length in range(1, sequence_length+1):
        programs_length = [''.join(seq) for seq in itertools.product('01', repeat=length)]
        programs += programs_length
    sequences = OrderedDict()
    for program in programs:
        multiply = int(sequence_length/len(program) + 1)
        sequence = (multiply*program)[:sequence_length]
        if sequence not in sequences.keys():
            sequences[sequence]=len(program)
    return sequences

def get_mdl(sequence = '0'):
    if isinstance(sequence, list):
        sequence = ''.join([str(item) for item in sequence])
    for length in range(1,len(sequence)+1):
        program = sequence[0:length]
        multiply = int(len(sequence)/length + 1)
        if (multiply*program)[:len(sequence)]==sequence: return length
    raise Exception('No valid program found')
    
def kl_div(prob_dict1, prob_dict2):
    assert prob_dict1.keys() == prob_dict2.keys()
    return sum([prob_dict1[key]*math.log(prob_dict1[key]/prob_dict2[key]) for key in prob_dict1.keys()])
    
def universal_prior(seq_length=10):
    sequences = sequences_by_mdl(seq_length)
    normalizer = sum([(0.5)**sequences[key] for key in sequences.keys()])
    for key in sequences.keys():
        sequences[key] = (0.5)**sequences[key]/normalizer    
    return sequences
    
def universal_prior_summable(seq_length=10):
    '''makes it so that sequence would converge if extended to infinity by increasing exponent'''
    sequences = sequences_by_mdl(seq_length)
    normalizer = sum([(0.5)**(2*sequences[key]) for key in sequences.keys()])
    for key in sequences.keys():
        sequences[key] = (0.5)**(2*sequences[key])/normalizer    
    return sequences
    
def uniform_prior(seq_length=10):
    sequences = sequences_by_mdl(seq_length)
    normalizer = len(sequences.keys())
    for key in sequences.keys():
        sequences[key] = 1/normalizer
    return sequences
    
def get_corrects(probs_dict):
    corrects_dict = {}
    for key in probs_dict.keys():
        prefix = key[:-1]
        dist = [v for k,v in probs_dict.items() if k.startswith(prefix)]
        corrects_dict[key] = 1 if probs_dict[key] == max(dist) else 0
    return corrects_dict
    
def get_acc_by_mdl(corrects_dict, sequence_length):
    all_sequences = sequences_by_mdl(sequence_length = sequence_length)
    accs = {}
    for length in range(1,sequence_length+1):
        keys = [k for k,v in all_sequences.items() if v == length]
        accs[length] = statistics.mean([corrects_dict[key] for key in keys])
    return accs
    
def get_prob_by_mdl(probs_dict, sequence_length):
    '''compute the average generation probability by mdl'''
    all_sequences = sequences_by_mdl(sequence_length = sequence_length)
    prob_by_mdl = {}
    for length in range(1,sequence_length+1):
        keys = [k for k,v in all_sequences.items() if v == length]
        prob_by_mdl[length] = statistics.mean([probs_dict[key] for key in keys])
    return prob_by_mdl
    
def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
        
def save_pickle(f, filename):
    with open(filename, 'wb') as handle:
        pickle.dump(f, handle)

def combine_counts_dicts(dicts):
    '''input is list of dicts.  all elements have the same keys, and values are counts'''
    '''outputs dict of probabilities'''
    total_dict = {}
    total_seqs = 0
    for key in dicts[0].keys():
        total_dict[key] = sum([d[key] for d in dicts])
        total_seqs += total_dict[key]
    for key in total_dict.keys():
        total_dict[key] = total_dict[key]/total_seqs
    return total_dict
    
def flatten(l, ltypes=(list, tuple)):
    ltype = type(l)
    l = list(l)
    i = 0
    while i < len(l):
        while isinstance(l[i], ltypes):
            if not l[i]:
                l.pop(i)
                i -= 1
                break
            else:
                l[i:i + 1] = l[i]
        i += 1
    return ltype(l)


def line_plot(x, y_dict, x_label, y_label, title=None, save_path=None, logscale = True, styles = None,
    figsize=(5.5,2.9),**kwargs):
    sns.set(style='whitegrid')
    sns.set_context("paper", font_scale=1.75)
    fig, ax = plt.subplots(figsize=figsize, layout='constrained')
    for key in y_dict.keys():
        y = y_dict[key]
        ax.plot(x, y, styles[key] if styles is not None else '', label=key, **kwargs)
    ax.set_xlabel(x_label)  # Add an x-label to the axes.
    ax.set_ylabel(y_label)  # Add a y-label to the axes.
    if title is not None:
        ax.set_title(title)  # Add a title to the axes.
    ax.legend();  # Add a legend.
    if logscale:
        plt.yscale('log')
    plt.savefig(save_path) 
    plt.savefig(save_path.replace('pdf', 'png'))
    plt.show()