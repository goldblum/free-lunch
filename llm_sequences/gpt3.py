import itertools
from utils import save_pickle
import json
import os
import random
import argparse
from trees import load_complexity_dict
from mingpt.utils import set_seed

def logit_bias_dict(tokenizer):
    dictionary = {}
    for idx in range(50257):
        if idx in tokenizer.token_dict.values():
            dictionary[idx] = 100
    return dictionary

class SequenceSampler:
    def __init__(self, max_mdl, sequence_length, binary_output = False):
        self.mdl_dict = load_complexity_dict(max_mdl, sequence_length, binary_output = binary_output)
    def samples(self, mdl, num_samples):
        if len(self.mdl_dict[mdl]) > num_samples:
            return random.sample(self.mdl_dict[mdl], num_samples)
        else:
            return self.mdl_dict[mdl]

def model_id(name='ada'):
    ids = {'ada':'text-ada-001', 'curie': 'text-curie-001', 'babbage':'text-babbage-001', 'davinci':'text-davinci-002'}
    return ids[name]

class Tokenizer:
    def __init__(self, space=True):
        if space:
            self.token_dict = {0:657,1:352,2:362,3:513,4:604,5:642,6:718,7:767,8:807,9:860}
        else:
            self.token_dict = {0:15,1:16,2:17,3:18,4:19,5:20,6:21,7:22,8:23,9:24}
        self.token_invdict = {v:k for k, v in self.token_dict.items()}
        self.delimit = 11
        self.eot = 50256
    def tokenize_num(self, num):
        '''takes in number and tokenizes'''
        return [self.token_dict[int(d)] for d in str(num)]+[self.delimit]
    def tokenize_seq(self, seq):
        '''takes in list of numbers and tokenizes'''
        return [self.eot]+sum([self.tokenize_num(el) for el in seq],[])[:-1]
    def untokenize_seq(self, tokenized):
        '''takes in tokenized list of numbers and untokenizes'''
        dec = [self.token_invdict[k] if k in self.token_invdict.keys() else k for k in tokenized] #numbers are now decimal digits
        buff = ''
        seq_str = ''
        for el in dec:
            if el==self.delimit or el==self.eot:
                seq_str+=buff+','
                buff = ''
            else:
                buff+= str(el)
        if seq_str[0]==',': seq_str = seq_str[1:]
        seq_str='['+seq_str+buff+']'
        return json.loads(seq_str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence probabilities from gpt3')
    parser.add_argument('--model', default='ada', type=str, help='model size')
    parser.add_argument('--seqs_per_mdl', default=100, type=int, help='how many sequences per program length')
    parser.add_argument('--max_mdl', default=7, type=int, help='how high complexity to go')
    parser.add_argument('--sequence_length', default=30, type=int, help='how long are the integer sequences?')
    parser.add_argument('--prompt_length', default=200, type=int, help='prompt length')
    parser.add_argument('--save_dir', default='./saved_dicts', type=str, help='path to save probs_dict')
    parser.add_argument('--seed', default=0, type=int, help='seed')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--logit_bias', action='store_true', help='restrict probabilities to decimal digit tokens')
    parser.add_argument('--api_key', default='', type=str, help='openai API key')
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)

    if not args.debug:
        import openai
        openai.api_key = args.api_key

    model = model_id(args.model)
    tokenizer = Tokenizer(space=True)
    sampler = SequenceSampler(args.max_mdl,args.sequence_length,binary_output=False)
    logit_bias = logit_bias_dict(tokenizer)

    save_dict = {}
    for mdl in range(args.max_mdl+1):
        save_dict[mdl] = {}
        samples = sampler.samples(mdl, args.seqs_per_mdl)
        print('samples', samples)
        for idx in range(len(samples)):
            sample = samples[idx]
            tokenized = tokenizer.tokenize_seq(sample)
            prompt = tokenized[:args.prompt_length]
            print('clipped', prompt)
            if not args.debug:
                if args.logit_bias:
                    response = openai.Completion.create(model=model, prompt=prompt, logprobs=0, logit_bias=logit_bias, max_tokens=0, echo=True)
                else:
                    response = openai.Completion.create(model=model, prompt=prompt, logprobs=0, max_tokens=0, echo=True)
                print('response:', response)
                save_dict[mdl][str(sample)] = response
    print('save dict', save_dict)

    if not args.debug:
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)
        filename = os.path.join(args.save_dir, args.model+'_max_mdl'+str(args.max_mdl)+'_length'+str(args.sequence_length)+'_prompt'+str(args.prompt_length)+'_'+str(args.seqs_per_mdl)+'seqs_logit_bias_'+str(args.logit_bias)+'.pickle')
        save_pickle(save_dict, filename)

