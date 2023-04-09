'''Saves out dictionary of counts assigned to all sequences using randomly initialized LMs'''

import torch
import random
import os
from mingpt.model import GPT
from mingpt.bpe import BPETokenizer
import argparse
from mingpt.utils import set_seed
from utils import reset_all_weights, get_mdl
import pickle
import torch.nn.functional as F

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence prediction from pre-trained language models')
    parser.add_argument('--num_seqs', default=10, type=int, help='how many sequences to generate')
    parser.add_argument('--sequence_length', default=5, type=int, help='how long sequences to generate')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    parser.add_argument('--model', default='gpt2', type=str, help='model size')
    parser.add_argument('--temperature', default=1.0, type=float, help='logit temperature scaling')
    parser.add_argument('--save_dir', default='./saved_dicts', type=str, help='path to save probs_dict')
    args = parser.parse_args()
    print(args)

    set_seed(args.seed)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT.from_pretrained(args.model)
    model.to(device)
    model.eval()
    tokenizer = BPETokenizer()

    token0 = tokenizer('0')
    token1 = tokenizer('1')
    
    x = torch.tensor([[tokenizer.encoder.encoder['<|endoftext|>']]], dtype=torch.long).to(device)

    mdl_list = []

    for i in range(args.num_seqs):
        if i % 1000 == 0: print('seq', i)
        idx = x.expand(1, -1)
        last_prob = 0
        for _ in range(args.sequence_length):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= model.block_size else idx[:, -model.block_size:]
            # forward the model to get the logits for the index in the sequence
            with torch.no_grad():
                logits, _ = model(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / args.temperature
            
            inf_tens = torch.zeros_like(logits)-float('Inf')
            inf_tens[0,token0] = logits[0,token0]
            inf_tens[0,token1] = logits[0,token1]
            logits = inf_tens
            probs = F.softmax(logits, dim=-1)
            
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)

            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)
            
            #last_prob = probs[0, idx_next]
            #print('last_prob', last_prob)
        
        seq = tokenizer.decode(idx.cpu().squeeze())
        seq = seq.replace('<|endoftext|>', '')
        mdl_list.append(get_mdl(seq))
        
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, 'longsamples_'+args.model+'_pretrained_length'+str(args.sequence_length)+'_seed'+str(args.seed)+'.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(mdl_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to:', filename)

