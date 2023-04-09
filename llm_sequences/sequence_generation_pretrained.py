'''Saves out dictionary of probabilities assigned to all sequences using pretrained huggingface LMs'''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from mingpt.model import GPT
from mingpt.utils import set_seed
from mingpt.bpe import BPETokenizer
import torch.nn.functional as F
import argparse
import random
from utils import sequences_by_mdl, get_corrects, get_acc_by_mdl
import pickle
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sequence prediction from pre-trained language models')
    parser.add_argument('--model', default='gpt2', type=str, help='model size')
    parser.add_argument('--sequence_length', default=4, type=int, help='how long are the sequences?')
    parser.add_argument('--temperature', default=1.0, type=float, help='logit temperature scaling')
    parser.add_argument('--save_dir', default='./saved_dicts', type=str, help='path to save probs_dict')
    args = parser.parse_args()
    print(args)

    model_type = args.model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT.from_pretrained(model_type)
    model.to(device)
    model.eval()
    tokenizer = BPETokenizer()
    
    all_sequences = sequences_by_mdl(sequence_length = args.sequence_length)
    print('Number of generated sequences:', len(all_sequences.keys()))
    probs_dict = {}
    
    total_seqs = 0
    total_prob = 0
    
    for sequence in all_sequences.keys():
    
        if total_seqs % 100 == 0:
            print('Sequence ' + str(total_seqs) + ' out of ' + str(2**args.sequence_length))
    
        # convert sequence to tensor of token indices
        tokens = []
        for element in sequence:
            tokens.append(tokenizer(str(element)))
        x = torch.cat(tokens)[:,0].unsqueeze(0).to(device)
        inputs = x[:,:-1].contiguous()
        targets = x[:,1:].contiguous()

        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = inputs if inputs.size(1) <= model.block_size else inputs[:, -model.block_size:]
        # forward the model to get the logits for the index in the sequence
        with torch.no_grad():
            logits, _ = model(idx_cond)
        logits = logits/args.temperature
        
        token0 = tokenizer('0')
        token1 = tokenizer('1')
        inf_tens = torch.zeros_like(logits)-float('Inf')
        inf_tens[:,:,token0] = logits[:,:,token0]
        inf_tens[:,:,token1] = logits[:,:,token1]
        logits = inf_tens.double()
        
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction = 'none')
        loss = loss.view(targets.shape)
        
        prepend = torch.tensor([0.5]).expand(loss.shape[0]).unsqueeze(-1).double()
        probs = torch.cat([prepend, torch.exp(-1.0*loss).cpu()], dim=1)
        seq_prob = torch.prod(probs, 1).item()
        probs_dict[sequence] = seq_prob
        
        total_prob+=seq_prob
        total_seqs+=1
        
    print('total_prob', total_prob)
    print('total_seqs', total_seqs/(2**args.sequence_length))
    
    
    corrects_dict = get_corrects(probs_dict)
    print('acc by mdl:', get_acc_by_mdl(corrects_dict, args.sequence_length))
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    filename = os.path.join(args.save_dir, 'probs_'+args.model+'_pretrained_length'+str(args.sequence_length)+'.pickle')
    with open(filename, 'wb') as handle:
        pickle.dump(probs_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved to:', filename)













