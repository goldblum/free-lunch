import torchtext
import os
import bz2
import argparse
import string
import random

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compress AmazonReviewFull')
    parser.add_argument('--data_root', default='~/data', type=str, help='data directory')
    parser.add_argument('--save_root', default='./', type=str, help='path for saving')
    parser.add_argument('--debug', action='store_true', help='debug mode, fast')
    args = parser.parse_args()
    print(args)


    if args.debug:
        num_lines = 5
    else:
        num_lines = -1

    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    train_iter = torchtext.datasets.AmazonReviewFull(args.data_root, split = 'train')
    lines = []
    counter = 0
    for label, line in train_iter:
        if counter == num_lines:
            break
        lines += line
        counter += 1
    lines=''.join(lines)
    print('dataset created')
    
    print('raw text:')
    text_path = os.path.join(args.save_root, 'temp'+suffix+'.txt')
    text_file = open(text_path, "w")
    text_file.write(lines)
    text_file.close()
    print('txt_bits', os.path.getsize(text_path)*8)

    bz2_path = os.path.join(args.save_root, 'temp'+suffix+'.bz2')
    tarbz2contents = bz2.compress(open(text_path, 'rb').read())
    fh = open(bz2_path, "wb")
    fh.write(tarbz2contents)
    fh.close()
    bz2_bits = os.path.getsize(bz2_path)*8
    print('bz2_bits', bz2_bits)

    os.remove(text_path)
    os.remove(bz2_path)


    print('shuffled text:')
    shuffled_lines=''.join(random.sample(lines,len(lines)))

    text_path = os.path.join(args.save_root, 'temp'+suffix+'.txt')
    text_file = open(text_path, "w")
    text_file.write(shuffled_lines)
    text_file.close()
    print('txt_bits', os.path.getsize(text_path)*8)

    bz2_path = os.path.join(args.save_root, 'temp'+suffix+'.bz2')
    tarbz2contents = bz2.compress(open(text_path, 'rb').read())
    fh = open(bz2_path, "wb")
    fh.write(tarbz2contents)
    fh.close()
    bz2_bits = os.path.getsize(bz2_path)*8
    print('bz2_bits', bz2_bits)

    os.remove(text_path)
    os.remove(bz2_path)


    print('random text:')
    chars = list(set(lines))
    lines = ''.join(random.choice(chars) for _ in range(len(lines)))

    text_path = os.path.join(args.save_root, 'temp'+suffix+'.txt')
    text_file = open(text_path, "w")
    text_file.write(lines)
    text_file.close()
    print('txt_bits', os.path.getsize(text_path)*8)

    bz2_path = os.path.join(args.save_root, 'temp'+suffix+'.bz2')
    tarbz2contents = bz2.compress(open(text_path, 'rb').read())
    fh = open(bz2_path, "wb")
    fh.write(tarbz2contents)
    fh.close()
    bz2_bits = os.path.getsize(bz2_path)*8
    print('bz2_bits', bz2_bits)

    os.remove(text_path)
    os.remove(bz2_path)
