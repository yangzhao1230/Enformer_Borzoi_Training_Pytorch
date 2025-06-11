import os

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GEPNpyDataset(Dataset):

    def __init__(self, npy_dir, bed_path, seqlen, genome_dict, 
                 shift_aug=False, rc_aug=False):
        self.npy_dir = npy_dir
        self.bed_path = bed_path
        self.seqlen = seqlen
        self.genome_dict = genome_dict
        self.chrom_length = {chrom: len(genome_dict[chrom]) for chrom in genome_dict}
        self.shift_aug = shift_aug
        self.rc_aug = rc_aug
        self.bed_file = pd.read_csv(bed_path, sep='\t')

        # assert number of npy files in npy_dir is equal to number of rows in bed_file
        assert len(os.listdir(npy_dir)) == len(self.bed_file)

    def resize_interval(self, chrom, start, end):
        """Resize genomic interval to fixed length, centered on midpoint"""
        mid_point = (start + end) // 2
        extend_start = mid_point - self.seqlen // 2
        extend_end = mid_point + self.seqlen // 2
        trimmed_start = max(0, extend_start)
        left_pad = trimmed_start - extend_start
        trimmed_end = min(self.chrom_length[chrom], extend_end)
        right_pad = extend_end - trimmed_end
        return trimmed_start, trimmed_end, left_pad, right_pad

    def get_sequence(self, chrom, start, end):
        """Extract genomic sequence with padding if necessary"""
        trimmed_start, trimmed_end, left_pad, right_pad = self.resize_interval(chrom, start, end)
        sequence = str(self.genome_dict[chrom].seq[trimmed_start:trimmed_end]).upper()
        left_pad_seq = 'N' * left_pad
        right_pad_seq = 'N' * right_pad
        sequence = left_pad_seq + sequence + right_pad_seq
        return sequence

    def sequence_to_onehot(self, sequence):
        """Convert DNA sequence to one-hot encoding"""
        mapping = {'A': [1, 0, 0, 0],
                   'C': [0, 1, 0, 0],
                   'G': [0, 0, 1, 0],
                   'T': [0, 0, 0, 1],
                   'N': [0, 0, 0, 0]}
        onehot = np.array([mapping[base] for base in sequence], dtype=np.float32)
        return onehot

    def __len__(self):
        return len(self.bed_file)

    def __getitem__(self, idx):

        npy_file_path = os.path.join(self.npy_dir, f"{idx}.npy")
        target = np.load(npy_file_path) # (896, 5313)

        # get genomic coordinates from bed_file
        row = self.bed_file.iloc[idx]
        chrom, start, end = row['chrom'], row['start'], row['end']

        sequence = self.get_sequence(chrom, start, end)
        if self.shift_aug:
            shift = np.random.randint(-3, 4)
            start += shift
            end += shift

        sequence = self.get_sequence(chrom, start, end)

        onehot = self.sequence_to_onehot(sequence)

        if self.rc_aug:
            if np.random.rand() < 0.5:
                onehot = onehot[::-1, ::-1].copy() # reverse the sequence and the onehot
                target = target[::-1].copy()

        return {
            'x': onehot,
            'labels': target,
        }
