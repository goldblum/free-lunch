'''Saves out dictionary of counts assigned to all sequences using randomly initialized LMs'''

import torch
import random
import os
from mingpt.model import GPT
import argparse
from mingpt.utils import set_seed
from utils import reset_all_weights, sequences_by_mdl
import pickle

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence prediction from pre-trained language models')
    parser.add_argument('--num_seqs', default=20, type=int, help='how many sequences to generate')
    parser.add_argument('--sequence_length', default=4, type=int, help='how long sequences to generate')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='gpt2', type=str, help='model size')
    parser.add_argument('--temperature', default=1.0, type=float, help='logit temperature scaling')
    parser.add_argument('--save_dir', default='./saved_dicts', type=str, help='path to save probs_dict')
    parser.add_argument('--sample', action='store_false', help='use argmax rather than randomly sampling sequences if not. generates lower complexity')
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    
    all_sequences = sequences_by_mdl(sequence_length = args.sequence_length)
    counts_dict = {key:0 for key in all_sequences.keys()}
    
    new_tokens = args.sequence_length-1
    vocab_size = 2 #num distinct tokens
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model_config = GPT.get_default_config()
    model_config.model_type = args.model
    model_config.vocab_size = vocab_size
    model_config.block_size = 1 #length of block to stride across during training, which we aren't doing
    model = GPT(model_config).to(device)
    model.eval();

    for i in range(args.num_seqs):
        if i % 1000 == 0: print('seq', i)
        reset_all_weights(model)
        inputs = torch.tensor([random.randint(0,1)]).unsqueeze(0).to(device)
        seq = model.generate(inputs, max_new_tokens=new_tokens, temperature=args.temperature, do_sample=args.sample).squeeze().cpu().tolist()
        seq = ''.join(str(e) for e in seq)
        counts_dict[seq] += 1

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, 'counts_'+args.model+'_initialized_length'+str(args.sequence_length)+'_sample_'+str(args.sample)+'_seed'+str(args.seed)+'.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(counts_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to:', filename)


