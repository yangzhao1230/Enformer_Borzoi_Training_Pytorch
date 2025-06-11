import torch
from torch.utils.data import Dataset
import h5py
import numpy as np
import pandas as pd

class GEPH5Dataset(Dataset):
    """
    GEP means Gene Expression Prediction
    """
    def __init__(self, file_path, bed_path, seqlen, genome_dict, shift_aug, rc_aug):
        self.file_path = file_path
        self.bed_path = bed_path
        
        self.h5_file = h5py.File(self.file_path, 'r')
        self.targets = self.h5_file['targets']
        self.bed_file = pd.read_csv(bed_path, sep='\t')
        assert len(self.bed_file) == len(self.targets)

        self.seqlen = seqlen
        self.genome_dict = genome_dict
        self.chrom_length = {chrom: len(genome_dict[chrom]) for chrom in genome_dict}

        self.shift_aug = shift_aug
        self.rc_aug = rc_aug

    def resize_interval(self, chrom, start, end):
        mid_point = (start + end) // 2
        extend_start = mid_point - self.seqlen // 2
        extend_end = mid_point + self.seqlen // 2
        trimmed_start = max(0, extend_start)
        left_pad = trimmed_start - extend_start
        trimmed_end = min(self.chrom_length[chrom], extend_end)
        right_pad = extend_end - trimmed_end
        return trimmed_start, trimmed_end, left_pad, right_pad

    def get_sequence(self, chrom, start, end):
        trimmed_start, trimmed_end, left_pad, right_pad = self.resize_interval(chrom, start, end)
        sequence = str(self.genome_dict[chrom].seq[trimmed_start:trimmed_end]).upper()
        left_pad_seq = 'N' * left_pad
        right_pad_seq = 'N' * right_pad
        sequence = left_pad_seq + sequence + right_pad_seq
        return sequence

    def sequence_to_onehot(self, sequence):
        mapping = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1],
                   'N': [0, 0, 0, 0]}
        onehot = np.array([mapping[base] for base in sequence], dtype=np.float32)
        return onehot
    
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # sequence = self.sequences[idx]
        # # bool -> float32
        # sequence = sequence.astype(np.float32)
        target = self.targets[idx]
        row = self.bed_file.iloc[idx]
        chrom, start, end = row['chrom'], row['start'], row['end']

        if self.shift_aug:
            shift = np.random.randint(-3, 4)
            start += shift
            end += shift

        sequence = self.get_sequence(chrom, start, end)

        onehot = self.sequence_to_onehot(sequence)

        if self.rc_aug:
            if np.random.rand() < 0.5:
                onehot = onehot[::-1, ::-1] # reverse the sequence and the onehot
                target = target[::-1]

        return {
            'x': onehot,
            'labels': target,
            # "head": "human",
            # "target_length": 896,
        }
    
    def close(self):
        self.h5_file.close()
