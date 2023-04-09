try:
   import cPickle as pickle
except:
   import pickle
import os
import zipfile
import bz2
import numpy as np
from PIL import Image
import shutil
import csv
import string
import random

def get_directory_size(directory):
    """Returns the `directory` size in bits."""
    total = 0
    try:
        # print("[+] Getting the size of", directory)
        for entry in os.scandir(directory):
            if entry.is_file():
                # if it's a file, use stat() function
                total += entry.stat().st_size
            elif entry.is_dir():
                # if it's a directory, recursively call this function
                try:
                    total += get_directory_size(entry.path)
                except FileNotFoundError:
                    pass
    except NotADirectoryError:
        # if `directory` isn't a directory, get the file size then
        return os.path.getsize(directory)
    except PermissionError:
        # if for whatever reason we can't open the folder, return 0
        return 0
    return total*8

def numpy_dataset_size_asarray(dataset_list, save_root = './', bits_per_entry = 8):
    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    #size of raw dataset
    if isinstance(dataset_list, list):
        raw_bits = np.prod([dataset.shape for dataset in dataset_list])*bits_per_entry
    else:
        raw_bits = dataset_list.size*bits_per_entry
    
    #serialize with pickle
    serialize_path = os.path.join(save_root, 'temp'+suffix+'.pickle')
    pickled = pickle.dump(dataset_list, open(serialize_path, 'wb'))
    pickle_bits = os.path.getsize(serialize_path)*8
    
    #compress with zip
    #zip_path = os.path.join(save_root, 'temp.zip')
    #f = zipfile.ZipFile(zip_path, mode='w')
    #f.write(serialize_path)
    #f.close()
    #zip_bits = os.path.getsize(zip_path)*8  
    
    bz2_path = os.path.join(save_root, 'temp'+suffix+'.bz2')
    tarbz2contents = bz2.compress(open(serialize_path, 'rb').read())
    fh = open(bz2_path, "wb")
    fh.write(tarbz2contents)
    fh.close()
    bz2_bits = os.path.getsize(bz2_path)*8
    
    #delete temporary files
    os.remove(serialize_path)
    #os.remove(zip_path)
    os.remove(bz2_path)    
    
    #return raw_bits, pickle_bits, zip_bits, bz2_bits
    return raw_bits, pickle_bits, bz2_bits
    
    
def numpy_dataset_size_asimagefolder(dataset_list, save_root='./', file_type = 'WebP'):
    suffix = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    temp_path = os.path.join(save_root,'temp'+suffix+'/')
    if os.path.isdir(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(temp_path, )
    
    #make folder of images
    if isinstance(dataset_list, list):
        counter = 0
        for dataset in dataset_list:
            for idx in range(len(dataset)):
                im = Image.fromarray(dataset[idx])
                if file_type == 'PNG':
                    im.save(os.path.join(temp_path,"temp"+str(counter)+".png"), 'PNG')   
                elif file_type =='JPEG2000':
                    im.save(os.path.join(temp_path,"temp"+str(counter)+".jp2"), 'JPEG2000')  
                elif file_type == 'WebP': 
                    im.save(os.path.join(temp_path,"temp"+str(counter)+".webp"), 'WebP')  
                counter+=1 
    else:
        for idx in range(len(dataset_list)):
            im = Image.fromarray(dataset_list[idx])
            if file_type == 'PNG':
                im.save(os.path.join(temp_path,"temp"+str(idx)+".png"), 'PNG')   
            elif file_type =='JPEG2000':
                im.save(os.path.join(temp_path,"temp"+str(idx)+".jp2"), 'JPEG2000')   
            elif file_type == 'WebP': 
                im.save(os.path.join(temp_path,"temp"+str(idx)+".webp"), 'WebP', lossless=True)  
    folder_bits = get_directory_size(temp_path)
    
    #compress with tar
    #shutil.make_archive('temp', 'tar', temp_path)
    #tar_path = os.path.join(save_root, 'temp.tar')
    #tar_bits = os.path.getsize(tar_path)*8
    
    #compress with tar and bz2
    shutil.make_archive('temp'+suffix, 'bztar', temp_path)
    bztar_path = os.path.join(save_root, 'temp'+suffix+'.tar.bz2')
    bztar_bits = os.path.getsize(bztar_path)*8
    
    #delete temporary files
    shutil.rmtree(temp_path)
    #os.remove(tar_path)
    os.remove(bztar_path)
    #return folder_bits, tar_bits, bztar_bits
    return folder_bits, bztar_bits

















