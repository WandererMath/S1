import os
import random
import bisect
import numpy as np



def get_folder_names(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]


def get_first_fna_file(folder_path, suffix=".fna"):
    for file in os.listdir(folder_path):
        if file.lower().endswith(suffix):
            #return os.path.join(folder_path, file)
            return file
    return None  # If no .fna file is found

def refine(ecoli_data, ecoli_labels):
    seqs_refined=[]
    labels_refined=[]
    for i in range(len(ecoli_data)):
        if ecoli_labels[i] == 1:
            seqs_refined.append(ecoli_data[i])
            labels_refined.append(ecoli_labels[i])
        elif ecoli_labels[i] == 0 and random.random() < 1/100:
            seqs_refined.append(ecoli_data[i])
            labels_refined.append(ecoli_labels[i])

    return seqs_refined, labels_refined





def get_paths_ends_with_something(folder_path, suffix):
    return [os.path.abspath(os.path.join(folder_path, f)) for f in os.listdir(folder_path) if f.endswith(suffix)]


def closest_binary_search(nums_sorted, x):
    pos = bisect.bisect_left(nums_sorted, x)
    if pos == 0:
        return nums_sorted[0]
    if pos == len(nums_sorted):
        return nums_sorted[-1]
    before = nums_sorted[pos - 1]
    after = nums_sorted[pos]
    if after - x < x - before:
        return after
    else:
        return before


def rna_encoding(seq):


    # Define one-hot mapping
    mapping = {'A': [1, 0, 0, 0],
            'T': [0, 1, 0, 0],
            'G': [0, 0, 1, 0],
            'C': [0, 0, 0, 1]}

    # One-hot encode each nucleotide
    one_hot = np.array([mapping[nt] for nt in seq])  # shape (30, 4)

    # Transpose to get shape (4, 30)
    return one_hot.T


def rna_decoding(one_hot_T):
    import numpy as np
    
    # Reverse mapping
    reverse_mapping = {
        (1, 0, 0, 0): 'A',
        (0, 1, 0, 0): 'T',
        (0, 0, 1, 0): 'G',
        (0, 0, 0, 1): 'C'
    }
    
    # Transpose back to get shape (seq_len, 4)
    one_hot = one_hot_T.T
    
    # Decode each nucleotide
    seq = ''.join(reverse_mapping[tuple(row)] for row in one_hot)
    return seq


def structure_encoding(seq):

    # Define one-hot mapping
    mapping = {'(': [1, 0, 0],
            ')': [0, 1, 0],
            '.': [0, 0, 1]}

    # One-hot encode each nucleotide
    one_hot = np.array([mapping[nt] for nt in seq])  # shape (30, 4)

    # Transpose to get shape (4, 30)
    return one_hot.T

def match2start(tree, x):
    matches = tree[x]
    if matches:
        for m in matches:
            return m.begin
    else:
        return None
