import sys
import argparse
import tempfile
import time
import gzip
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset, Dataset

from ..common import constants, utils

### 08/23/2025 ###
import torch
import torch.nn.functional as F

def tensor_to_sequence(tensor, alphabet=constants.STANDARD_NT):
    """
    Convert a one-hot encoded tensor to a DNA sequence string.
    """
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    
    indices = tensor.argmax(dim=0)
    sequence = ''.join([alphabet[i] for i in indices])
    return sequence

class OriginalOneHotSlicer(torch.nn.Module):
    """
    Exact replica of the original OneHotSlicer for sequence extraction.
    """
    
    def __init__(self, in_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.register_buffer('weight', self.set_weight(in_channels, kernel_size))
        
    def set_weight(self, in_channels, kernel_size):
        outter_cat = []
        for i in range(in_channels):
            inner_stack = [torch.zeros((kernel_size, kernel_size)) for x in range(in_channels)]
            inner_stack[i] = torch.eye(kernel_size)
            outter_cat.append(torch.stack(inner_stack, dim=1))
        return torch.cat(outter_cat, dim=0)
    
    def forward(self, input):
        hook = F.conv1d(input, self.weight)
        # Original OneHotSlicer logic - matches exactly
        hook = hook.permute(0,2,1).flatten(0,1).unflatten(1,(self.in_channels, self.kernel_size))
        return hook

def extract_sequences_original_method(vcf_data, sample_indices=None, alphabet=constants.STANDARD_NT):
    """
    Extract sequences using the exact original OneHotSlicer method.
    Works entirely on GPU to avoid CPU/GPU precision differences.
    """
    if sample_indices is None:
        sample_indices = range(len(vcf_data))
    
    # Create the original OneHotSlicer and keep it on the same device as the main pipeline
    window_slicer = OriginalOneHotSlicer(len(alphabet), vcf_data.window_size)
    if torch.cuda.is_available():
        window_slicer = window_slicer.cuda()
    
    ref_sequences = []
    alt_sequences = []
    metadata = []
    
    print(f"Extracting sequences using original OneHotSlicer method (GPU if available)...")
    print(f"Processing {len(sample_indices)} samples...")
    
    for idx in sample_indices:
        print(f"\n--- Processing sample {idx} ---")
        try:
            # Get the VCF record
            record = vcf_data.vcf.iloc[idx]
            print(f"Record: {record['chrom']}:{record['pos']} {record['ref']}->{record['alt']}")
            
            # Extract genomic sequences (before windowing)
            ref_encoded = vcf_data.encode(record['ref'])
            alt_encoded = vcf_data.encode(record['alt'])
            print(f"Encoded alleles - ref: {ref_encoded.shape}, alt: {alt_encoded.shape}")
            
            var_loc = record['pos'] - 1
            start = var_loc - vcf_data.relative_end + 1
            trail_start = var_loc + ref_encoded.shape[1]
            trail_end = start + vcf_data.grab_size
            len_dif = alt_encoded.shape[1] - ref_encoded.shape[1]
            
            print(f"Genomic coordinates - start: {start}, var_loc: {var_loc}, trail_start: {trail_start}, trail_end: {trail_end}")
            
            # Build genomic sequences exactly like original VcfDataset
            contig = vcf_data.fasta[record['chrom']]
            print(f"Contig length: {contig.shape[1]}")
            
            # Check if coordinates are valid
            if start < 0:
                print(f"ERROR: start coordinate {start} is negative!")
                continue
            if trail_end > contig.shape[1]:
                print(f"ERROR: trail_end {trail_end} exceeds contig length {contig.shape[1]}!")
                continue
            
            # Reference sequence
            leader = contig[:, start:var_loc]
            trailer = contig[:, trail_start:trail_end]
            ref_segments = [leader, ref_encoded, trailer]
            ref_genomic = np.concatenate(ref_segments, axis=-1)
            print(f"Reference genomic sequence: {ref_genomic.shape}")
            
            # Alternative sequence (using original genomic shifting)
            start_adjust = len_dif // 2
            end_adjust = len_dif - start_adjust
            
            leader_alt = contig[:, start+start_adjust:var_loc]
            trailer_alt = contig[:, trail_start:trail_end-end_adjust]
            alt_segments = [leader_alt, alt_encoded, trailer_alt]
            alt_genomic = np.concatenate(alt_segments, axis=-1)
            print(f"Alternative genomic sequence: {alt_genomic.shape}")
            
            # Convert to tensors and keep on same device as main pipeline
            ref_tensor = torch.tensor(ref_genomic[np.newaxis].astype(np.float32))
            alt_tensor = torch.tensor(alt_genomic[np.newaxis].astype(np.float32))
            
            # Move to GPU if available (same as main pipeline)
            if torch.cuda.is_available():
                ref_tensor = ref_tensor.cuda()
                alt_tensor = alt_tensor.cuda()
            
            print(f"Tensor shapes - ref: {ref_tensor.shape}, alt: {alt_tensor.shape}")
            
            # Apply original OneHotSlicer
            ref_windows = window_slicer(ref_tensor)
            alt_windows = window_slicer(alt_tensor)
            print(f"OneHotSlicer output - ref: {ref_windows.shape}, alt: {alt_windows.shape}")
            
            # Calculate the number of windows
            num_windows = ref_windows.shape[0]
            print(f"Total windows: {num_windows}")
            
            # Apply step_size subsampling (original method)
            selected_indices = torch.arange(0, num_windows, vcf_data.step_size)
            # Keep indices on same device
            if torch.cuda.is_available():
                selected_indices = selected_indices.cuda()
            
            print(f"Selected indices: {selected_indices}")
            
            if len(selected_indices) == 0:
                print("ERROR: No windows selected after step_size subsampling!")
                continue
                
            ref_subsampled = ref_windows[selected_indices]
            alt_subsampled = alt_windows[selected_indices]
            print(f"After subsampling - ref: {ref_subsampled.shape}, alt: {alt_subsampled.shape}")
            
            # Apply reverse complements if needed
            if vcf_data.reverse_complements:
                # Keep complement_matrix on same device
                complement_matrix = vcf_data.complement_matrix
                if torch.cuda.is_available():
                    complement_matrix = complement_matrix.cuda()
                
                ref_rc = torch.flip(complement_matrix @ ref_subsampled, dims=[-1])
                ref_final = torch.cat([ref_subsampled, ref_rc], dim=0)
                
                alt_rc = torch.flip(complement_matrix @ alt_subsampled, dims=[-1])
                alt_final = torch.cat([alt_subsampled, alt_rc], dim=0)
            else:
                ref_final = ref_subsampled
                alt_final = alt_subsampled
            
            print(f"Final tensor shapes - ref: {ref_final.shape}, alt: {alt_final.shape}")
            
            # Convert tensors back to sequences
            # ONLY move to CPU at the very end for sequence conversion
            ref_final_cpu = ref_final.cpu()
            alt_final_cpu = alt_final.cpu()
            
            ref_sample_seqs = []
            alt_sample_seqs = []
            
            # Determine number of strands and windows
            if vcf_data.reverse_complements:
                num_strands = 2
                windows_per_strand = ref_final_cpu.shape[0] // 2
            else:
                num_strands = 1
                windows_per_strand = ref_final_cpu.shape[0]
            
            print(f"Processing {num_strands} strands with {windows_per_strand} windows each")
            
            # Extract sequences from each strand and window
            for strand_idx in range(num_strands):
                strand_name = "forward" if strand_idx == 0 else "reverse"
                start_idx = strand_idx * windows_per_strand
                
                for window_idx in range(windows_per_strand):
                    global_idx = start_idx + window_idx
                    ref_seq = tensor_to_sequence(ref_final_cpu[global_idx], alphabet)
                    alt_seq = tensor_to_sequence(alt_final_cpu[global_idx], alphabet)
                    
                    ref_sample_seqs.append(f"strand_{strand_name}_window_{window_idx:02d}:{ref_seq}")
                    alt_sample_seqs.append(f"strand_{strand_name}_window_{window_idx:02d}:{alt_seq}")
            
            ref_sequences.append(ref_sample_seqs)
            alt_sequences.append(alt_sample_seqs)
            
            # Store metadata
            metadata.append({
                'variant_idx': idx,
                'chrom': record['chrom'],
                'pos': record['pos'],
                'ref_allele': record['ref'],
                'alt_allele': record['alt'],
                'genomic_ref_length': ref_genomic.shape[1],
                'genomic_alt_length': alt_genomic.shape[1],
                'num_windows': windows_per_strand,
                'num_strands': num_strands
            })
            
            print(f"Successfully processed sample {idx}")
            
        except Exception as e:
            print(f"ERROR processing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return ref_sequences, alt_sequences, metadata

def save_original_sequences(vcf_data, output_file, sample_indices=None):
    """
    Extract and save sequences using original OneHotSlicer method.
    """
    ref_sequences, alt_sequences, metadata = extract_sequences_original_method(
        vcf_data, sample_indices
    )
    
    # Create output DataFrame
    sequence_data = []
    
    for ref_seq_list, alt_seq_list, meta in zip(ref_sequences, alt_sequences, metadata):
        for ref_seq_entry, alt_seq_entry in zip(ref_seq_list, alt_seq_list):
            row = meta.copy()
            row['ref_sequence'] = ref_seq_entry
            row['alt_sequence'] = alt_seq_entry
            sequence_data.append(row)
    
    seq_df = pd.DataFrame(sequence_data)
    seq_df.to_csv(output_file, sep='\t', index=False)
    print(f"Original method sequences saved to: {output_file}")
    print(f"Total sequence entries: {len(sequence_data)}")
    
    return seq_df
# Add these functions to the end of your fasta_datamodule.py file
def extract_flank_sequences(vcf_data, sample_indices=None):
    """
    Extracts the 200bp left and right flank sequences used by the VcfDataset_SimplePadding method.
    Since the MPRA flanks are constant, this function primarily formats them for comparison
    with the center sequence files.
    """
    if sample_indices is None:
        sample_indices = range(len(vcf_data))

    all_left_flanks = []
    all_right_flanks = []
    all_metadata = []

    print(f"Extracting 200bp flank sequences from the new method for {len(sample_indices)} samples...")

    # Since flanks are constant, convert them to strings once for efficiency
    fwd_left_flank_str = array_to_sequence(vcf_data.left_flank_encoded)
    fwd_right_flank_str = array_to_sequence(vcf_data.right_flank_encoded)

    # The reverse complement of the left flank is the new right flank, and vice-versa
    rev_left_flank_str = dna_reverse_complement(fwd_right_flank_str)
    rev_right_flank_str = dna_reverse_complement(fwd_left_flank_str)

    for idx in tqdm.tqdm(sample_indices):
        record = vcf_data.vcf.iloc[idx]
        
        variant_left_flanks = []
        variant_right_flanks = []

        try:
            # Generate forward strand flank entries for each window
            for window_start_pos in range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size):
                variant_left_flanks.append(f"forward_window_{window_start_pos:03d}:{fwd_left_flank_str}")
                variant_right_flanks.append(f"forward_window_{window_start_pos:03d}:{fwd_right_flank_str}")

            # Generate reverse strand flank entries for each window
            if vcf_data.reverse_complements:
                for window_start_pos in range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size):
                    variant_left_flanks.append(f"reverse_window_{window_start_pos:03d}:{rev_left_flank_str}")
                    variant_right_flanks.append(f"reverse_window_{window_start_pos:03d}:{rev_right_flank_str}")
            
            all_left_flanks.append(variant_left_flanks)
            all_right_flanks.append(variant_right_flanks)
            all_metadata.append({'variant_idx': idx, 'chrom': record['chrom'], 'pos': record['pos'], 'ref': record['ref'], 'alt': record['alt']})

        except Exception as e:
            print(f"Skipping flank extraction for index {idx} due to error: {e}")
            continue
            
    return all_left_flanks, all_right_flanks, all_metadata


def save_flank_sequences(vcf_data, output_file, sample_indices=None):
    """
    Extracts and saves the 200bp flank sequences from the new method to a TSV file.
    """
    left_flanks, right_flanks, metadata = extract_flank_sequences(
        vcf_data, sample_indices
    )
    
    output_data = []
    for i, meta in enumerate(metadata):
        for j in range(len(left_flanks[i])):
            row = meta.copy()
            # The left_flanks list already contains the formatted window string
            row['left_flank_sequence'] = left_flanks[i][j]
            row['right_flank_sequence'] = right_flanks[i][j]
            output_data.append(row)
            
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nNew method's 200bp flank sequences saved to: {output_file}")

def array_to_sequence(array, alphabet=constants.STANDARD_NT):
    """
    Convert a one-hot encoded numpy array to a DNA sequence string.
    """
    if array.ndim > 2:
        array = array.squeeze()
    
    indices = array.argmax(axis=0)
    sequence = ''.join([alphabet[i] for i in indices])
    return sequence

def dna_reverse_complement(sequence):
    """
    Returns the reverse complement of a DNA sequence.
    """
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join(complement_dict.get(base, base) for base in reversed(sequence))

def extract_center_sequences_new_method(vcf_data, sample_indices=None):
    """
    Extracts the 200bp "center" sequences generated by the VcfDataset_SimplePadding method.
    This function replicates the logic from __getitem__ to isolate the exact genomic
    and padded sequences before the 200bp MPRA flanks are added.
    """
    if sample_indices is None:
        sample_indices = range(len(vcf_data))

    all_ref_sequences = []
    all_alt_sequences = []
    all_metadata = []

    print(f"Extracting 200bp center sequences from the new method for {len(sample_indices)} samples...")

    for idx in tqdm.tqdm(sample_indices):
        record = vcf_data.vcf.iloc[idx]
        
        variant_ref_windows = []
        variant_alt_windows = []

        try:
            var_loc = record['pos'] - 1
            contig = vcf_data.fasta[record['chrom']]
            
            ref_allele_encoded = vcf_data.encode(record['ref'])
            alt_allele_encoded = vcf_data.encode(record['alt'])
            len_diff = alt_allele_encoded.shape[1] - ref_allele_encoded.shape[1]

            # Replicate the logic from __getitem__ to get the exact same center sequences
            for window_start_pos in range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size):
                leader_len = window_start_pos
                trailer_len = vcf_data.window_size - leader_len
                
                ref_leader_genomic = contig[:, var_loc - leader_len : var_loc]
                ref_trailer_genomic = contig[:, var_loc + ref_allele_encoded.shape[1] : var_loc + ref_allele_encoded.shape[1] + trailer_len - ref_allele_encoded.shape[1]]
                ref_center = np.concatenate([ref_leader_genomic, ref_allele_encoded, ref_trailer_genomic], axis=-1)

                alt_leader_genomic = contig[:, var_loc - leader_len : var_loc]
                alt_trailer_genomic = contig[:, var_loc + ref_allele_encoded.shape[1] : var_loc + ref_allele_encoded.shape[1] + trailer_len - alt_allele_encoded.shape[1]]
                
                if len_diff > 0: # Insertion
                    pad_seq = constants.MPRA_UPSTREAM[-len_diff:]
                    ref_center = np.concatenate([ref_center, vcf_data.encode(pad_seq)], axis=-1)

                alt_center = np.concatenate([alt_leader_genomic, alt_allele_encoded, alt_trailer_genomic], axis=-1)
                
                if len_diff < 0: # Deletion
                    pad_seq = constants.MPRA_UPSTREAM[-abs(len_diff):]
                    alt_center = np.concatenate([alt_center, vcf_data.encode(pad_seq)], axis=-1)

                ref_center = ref_center[:, :vcf_data.window_size]
                alt_center = alt_center[:, :vcf_data.window_size]
                
                ref_seq_str = array_to_sequence(ref_center)
                alt_seq_str = array_to_sequence(alt_center)

                variant_ref_windows.append(f"forward_window_{window_start_pos:03d}:{ref_seq_str}")
                variant_alt_windows.append(f"forward_window_{window_start_pos:03d}:{alt_seq_str}")
            
            # Manually create reverse complements
            if vcf_data.reverse_complements:
                forward_ref_seqs = [s.split(':')[1] for s in variant_ref_windows]
                forward_alt_seqs = [s.split(':')[1] for s in variant_alt_windows]
                
                for i, window_start_pos in enumerate(range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size)):
                    rc_ref = dna_reverse_complement(forward_ref_seqs[i])
                    rc_alt = dna_reverse_complement(forward_alt_seqs[i])
                    variant_ref_windows.append(f"reverse_window_{window_start_pos:03d}:{rc_ref}")
                    variant_alt_windows.append(f"reverse_window_{window_start_pos:03d}:{rc_alt}")

            all_ref_sequences.append(variant_ref_windows)
            all_alt_sequences.append(variant_alt_windows)
            all_metadata.append({'variant_idx': idx, 'chrom': record['chrom'], 'pos': record['pos'], 'ref': record['ref'], 'alt': record['alt']})
        
        except Exception as e:
            print(f"Skipping sequence extraction for index {idx} due to error: {e}")
            continue
            
    return all_ref_sequences, all_alt_sequences, all_metadata

def save_center_sequences(vcf_data, output_file, sample_indices=None):
    """
    Extracts and saves the 200bp center sequences from the new method to a TSV file.
    """
    ref_sequences, alt_sequences, metadata = extract_center_sequences_new_method(
        vcf_data, sample_indices
    )
    
    output_data = []
    for i, meta in enumerate(metadata):
        for j in range(len(ref_sequences[i])):
            row = meta.copy()
            row['ref_center_sequence'] = ref_sequences[i][j]
            row['alt_center_sequence'] = alt_sequences[i][j]
            output_data.append(row)
            
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nNew method's 200bp center sequences saved to: {output_file}")
def extract_full_sequences_new_method(vcf_data, sample_indices=None):
    """
    Extracts the complete 600bp sequences (flanks + center) generated by the
    VcfDataset_SimplePadding method for each window.
    """
    if sample_indices is None:
        sample_indices = range(len(vcf_data))

    all_ref_sequences = []
    all_alt_sequences = []
    all_metadata = []

    print(f"Extracting full 600bp sequences from the new method for {len(sample_indices)} samples...")

    for idx in tqdm.tqdm(sample_indices):
        record = vcf_data.vcf.iloc[idx]
        
        variant_ref_windows = []
        variant_alt_windows = []

        try:
            var_loc = record['pos'] - 1
            contig = vcf_data.fasta[record['chrom']]
            
            ref_allele_encoded = vcf_data.encode(record['ref'])
            alt_allele_encoded = vcf_data.encode(record['alt'])
            len_diff = alt_allele_encoded.shape[1] - ref_allele_encoded.shape[1]

            # Replicate the logic from __getitem__ to get the exact same center sequences
            for window_start_pos in range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size):
                leader_len = window_start_pos
                trailer_len = vcf_data.window_size - leader_len
                
                ref_leader_genomic = contig[:, var_loc - leader_len : var_loc]
                ref_trailer_genomic = contig[:, var_loc + ref_allele_encoded.shape[1] : var_loc + ref_allele_encoded.shape[1] + trailer_len - ref_allele_encoded.shape[1]]
                ref_center = np.concatenate([ref_leader_genomic, ref_allele_encoded, ref_trailer_genomic], axis=-1)

                alt_leader_genomic = contig[:, var_loc - leader_len : var_loc]
                alt_trailer_genomic = contig[:, var_loc + ref_allele_encoded.shape[1] : var_loc + ref_allele_encoded.shape[1] + trailer_len - alt_allele_encoded.shape[1]]
                
                if len_diff > 0: # Insertion
                    pad_seq = constants.MPRA_UPSTREAM[-len_diff:]
                    ref_center = np.concatenate([ref_center, vcf_data.encode(pad_seq)], axis=-1)

                alt_center = np.concatenate([alt_leader_genomic, alt_allele_encoded, alt_trailer_genomic], axis=-1)
                
                if len_diff < 0: # Deletion
                    pad_seq = constants.MPRA_UPSTREAM[-abs(len_diff):]
                    alt_center = np.concatenate([alt_center, vcf_data.encode(pad_seq)], axis=-1)

                ref_center = ref_center[:, :vcf_data.window_size]
                alt_center = alt_center[:, :vcf_data.window_size]
                
                # Assemble the final 600bp sequence by adding the flanks
                ref_full_seq_encoded = np.concatenate([vcf_data.left_flank_encoded, ref_center, vcf_data.right_flank_encoded], axis=-1)
                alt_full_seq_encoded = np.concatenate([vcf_data.left_flank_encoded, alt_center, vcf_data.right_flank_encoded], axis=-1)

                # Convert the full 600bp encoded sequence to a string
                ref_seq_str = array_to_sequence(ref_full_seq_encoded)
                alt_seq_str = array_to_sequence(alt_full_seq_encoded)

                variant_ref_windows.append(f"forward_window_{window_start_pos:03d}:{ref_seq_str}")
                variant_alt_windows.append(f"forward_window_{window_start_pos:03d}:{alt_seq_str}")

            # Manually create reverse complements
            if vcf_data.reverse_complements:
                forward_ref_seqs = [s.split(':')[1] for s in variant_ref_windows]
                forward_alt_seqs = [s.split(':')[1] for s in variant_alt_windows]
                
                for i, window_start_pos in enumerate(range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size)):
                    rc_ref = dna_reverse_complement(forward_ref_seqs[i])
                    rc_alt = dna_reverse_complement(forward_alt_seqs[i])
                    variant_ref_windows.append(f"reverse_window_{window_start_pos:03d}:{rc_ref}")
                    variant_alt_windows.append(f"reverse_window_{window_start_pos:03d}:{rc_alt}")

            all_ref_sequences.append(variant_ref_windows)
            all_alt_sequences.append(variant_alt_windows)
            all_metadata.append({'variant_idx': idx, 'chrom': record['chrom'], 'pos': record['pos'], 'ref': record['ref'], 'alt': record['alt']})
        
        except Exception as e:
            print(f"Skipping full sequence extraction for index {idx} due to error: {e}")
            continue
            
    return all_ref_sequences, all_alt_sequences, all_metadata


def save_full_sequences(vcf_data, output_file, sample_indices=None):
    """
    Extracts and saves the full 600bp sequences from the new method to a TSV file.
    """
    ref_sequences, alt_sequences, metadata = extract_full_sequences_new_method(
        vcf_data, sample_indices
    )
    
    output_data = []
    for i, meta in enumerate(metadata):
        for j in range(len(ref_sequences[i])):
            row = meta.copy()
            row['ref_full_sequence'] = ref_sequences[i][j]
            row['alt_full_sequence'] = alt_sequences[i][j]
            output_data.append(row)
            
    df = pd.DataFrame(output_data)
    df.to_csv(output_file, sep='\t', index=False)
    print(f"\nNew method's full 600bp sequences saved to: {output_file}")
###
def alphabet_onehotizer(seq, alphabet):
    """
    Convert a sequence of characters into a one-hot encoded array based on the provided alphabet.

    Args:
        seq (str): The input sequence to be one-hot encoded.
        alphabet (list): The alphabet of characters used for encoding.

    Returns:
        np.ndarray: A one-hot encoded array where each row corresponds to a character in 'seq'
                    and each column corresponds to a character in the 'alphabet'. The value at
                    each position is True if the character matches the alphabet element, False otherwise.
    """
    char_array = np.expand_dims( np.array([*seq]), 0 )
    alph_array = np.expand_dims( np.array(alphabet), 1 )
    
    return char_array == alph_array

class OneHotSlicer(nn.Module):
    """
    A PyTorch module that slices the one-hot encoded input along specified dimensions.

    Args:
        in_channels (int): Number of input channels (alphabet size) for the one-hot encoding.
        kernel_size (int): Size of the kernel used for slicing.

    Attributes:
        in_channels (int): Number of input channels (alphabet size) for the one-hot encoding.
        kernel_size (int): Size of the kernel used for slicing.

    Methods:
        set_weight(in_channels, kernel_size): Helper method to generate the weight tensor for slicing.
        forward(input): Forward pass through the slicing operation.

    Note:
        This module assumes that the input tensor is in the shape (batch_size, sequence_length, in_channels),
        representing one-hot encoded sequences.

    Returns:
        torch.Tensor: Sliced tensor of shape (batch_size, sequence_length, in_channels, kernel_size).
    """
    
    def __init__(self, in_channels, kernel_size):
        """
        Initializes the OneHotSlicer module with the given input channels and kernel size.
        """
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.register_buffer('weight', self.set_weight(in_channels, kernel_size))
        
    def set_weight(self, in_channels, kernel_size):
        """
        Generates a weight tensor for the slicing operation.

        Args:
            in_channels (int): Number of input channels (alphabet size) for the one-hot encoding.
            kernel_size (int): Size of the kernel used for slicing.

        Returns:
            torch.Tensor: Weight tensor for the slicing operation.
        """
        outter_cat = []
        for i in range(in_channels):
            inner_stack = [ torch.zeros((kernel_size,kernel_size)) for x in range(in_channels) ]
            inner_stack[i] = torch.eye(kernel_size)
            outter_cat.append( torch.stack(inner_stack, dim=1) )
        return torch.cat(outter_cat, dim=0)
    
    def forward(self, input):
        """
        Performs the forward pass through the slicing operation.

        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, sequence_length, in_channels).

        Returns:
            torch.Tensor: Sliced tensor of shape (batch_size, sequence_length, in_channels, kernel_size).
        """
        hook = F.conv1d(input, self.weight)
        hook = hook.permute(0,2,1).flatten(0,1) \
                 .unflatten(1,(self.in_channels, self.kernel_size))
        return hook

class Fasta:
    """
    A class for reading and processing sequences from a FASTA file.

    Args:
        fasta_path (str): Path to the FASTA file containing sequences.
        all_upper (bool, optional): Whether to convert sequences to uppercase. Default is True.
        alphabet (str, optional): The alphabet of characters used for encoding sequences. Default is constants.STANDARD_NT.

    Attributes:
        fasta_path (str): Path to the FASTA file containing sequences.
        all_upper (bool): Whether sequences should be converted to uppercase.
        alphabet (str): The alphabet of characters used for encoding sequences.
        fasta (dict): Dictionary mapping contig keys to one-hot encoded sequences.
        contig_lengths (dict): Dictionary mapping contig keys to their respective sequence lengths.
        contig_index2key (dict): Dictionary mapping contig indices to contig keys.
        contig_key2index (dict): Dictionary mapping contig keys to their respective indices.
        contig_descriptions (list): List of contig descriptions parsed from the FASTA file.

    Methods:
        read_fasta(): Reads and processes sequences from the FASTA file.
    """
    
    def __init__(self, fasta_path, all_upper=True, 
                 alphabet=constants.STANDARD_NT):
        """
        Initializes the Fasta object with the specified parameters and reads the FASTA file.
        """
        self.fasta_path = fasta_path
        self.all_upper = all_upper
        self.alphabet = alphabet
        self.read_fasta()
        
    def read_fasta(self):
        """
        Reads and processes sequences from the FASTA file, populating relevant attributes.
        """
        self.fasta = {}
        self.contig_lengths   = {}
        self.contig_index2key = {}
        self.contig_key2index = {}
        self.contig_descriptions = {}
        
        print('pre-reading fasta into memory', file=sys.stderr)
        with open(self.fasta_path, 'r') as f:
            fa = np.array(
                [ x.rstrip() for x in tqdm.tqdm(f.readlines()) ]
            )
            print('finding keys', file=sys.stderr)
            fa_idx = np.where( np.char.startswith(fa, '>') )[0]
            print('parsing', file=sys.stderr)
            
            for idx, contig_loc in tqdm.tqdm(list(enumerate(fa_idx))):
                contig_info = fa[contig_loc][1:]
                contig_key, *contig_des = contig_info.split()
                
                start_block = fa_idx[idx] + 1
                try:
                    end_block = fa_idx[idx+1]
                except IndexError:
                    end_block = None
                    
                get_blocks = fa[start_block:end_block]
                if self.all_upper:
                    contig_seq = ''.join( np.char.upper(get_blocks) )
                else:
                    contig_seq = ''.join( get_blocks )

                self.fasta[contig_key] = alphabet_onehotizer(
                    contig_seq, self.alphabet
                )
                self.contig_lengths[contig_key] = len(contig_seq)
                self.contig_index2key[idx] = contig_key
                self.contig_key2index[contig_key] = idx
                self.contig_descriptions = contig_des
                    
        print('done',file=sys.stderr)


class FastaDataset(Dataset):
    """
    A PyTorch Dataset class for generating sequence windows from a Fasta object.

    Args:
        fasta_obj (Fasta): An instance of the Fasta class containing sequence data.
        window_size (int): Size of the sliding window used to extract sequences.
        step_size (int): Step size for sliding the window.
        reverse_complements (bool, optional): Whether to include reverse complements of the sequences. Default is True.
        alphabet (str, optional): The alphabet of characters used for encoding sequences. Default is constants.STANDARD_NT.
        complement_dict (dict, optional): A dictionary mapping characters to their complements. Default is constants.DNA_COMPLEMENTS.
        pad_final (bool, optional): Whether to pad the final window if it doesn't fit perfectly within the sequence. Default is False.

    Attributes:
        fasta (Fasta): An instance of the Fasta class containing sequence data.
        window_size (int): Size of the sliding window used to extract sequences.
        step_size (int): Step size for sliding the window.
        reverse_complements (bool): Whether reverse complements of sequences are included.
        alphabet (str): The alphabet of characters used for encoding sequences.
        complement_dict (dict): A dictionary mapping characters to their complements.
        complement_matrix (numpy.ndarray): A matrix representing character complement relationships.
        pad_final (bool): Whether the final window is padded.
        n_keys (int): Number of keys (contigs) in the Fasta object.
        key_lens (dict): Dictionary mapping contig keys to their respective sequence lengths.
        key_n_windows (dict): Dictionary mapping contig keys to the number of windows.
        key_rolling_n (numpy.ndarray): Array of cumulative sums of windows for each key.
        key2idx (dict): Dictionary mapping contig keys to their indices.
        idx2key (list): List of contig keys corresponding to indices.
        n_unstranded_windows (int): Total number of unstranded windows.

    Methods:
        count_windows(): Count the number of windows for each contig.
        get_fasta_coords(idx): Get the start and end coordinates of a window for a given index.
        parse_complements(): Parse the complement matrix based on the provided alphabet and complement dictionary.
    """
    
    def __init__(self, 
                 fasta_obj, window_size, step_size, 
                 reverse_complements=True,
                 alphabet=constants.STANDARD_NT,
                 complement_dict=constants.DNA_COMPLEMENTS,
                 pad_final=False):
        """
        Initializes the FastaDataset object with the specified parameters and precomputes necessary attributes.
        """
        super().__init__()
        
        assert step_size <= window_size, "Gaps will form if step_size > window_size"
        
        self.fasta = fasta_obj
        self.window_size = window_size
        self.step_size = step_size
        
        self.reverse_complements = reverse_complements
        
        self.alphabet = alphabet
        self.complement_dict = complement_dict
        self.complement_matrix = self.parse_complements()
        
        self.pad_final  = pad_final
        
        self.n_keys = len(self.fasta.keys())
        self.key_lens =  { k: self.fasta[k].shape[-1] for k in self.fasta.keys() }
        self.key_n_windows = self.count_windows()
        self.key_rolling_n = np.cumsum([ self.key_n_windows[k] for k in self.fasta.keys() ])
        
        self.key2idx  = { k:i for i,k in enumerate(self.fasta.keys()) }
        self.idx2key  = list(self.fasta.keys())
        
        self.n_unstranded_windows = sum( self.key_n_windows.values() )
                    
    def count_windows(self):
        """
        Count the number of windows for each contig based on the window size and step size.
        """
        key_n_windows = {}
        
        for k, v in self.key_lens.items():
            
            if v >= self.window_size:
                n = 1
                n += (v - self.window_size) // self.step_size
                if self.pad_final:
                    n += 1 if (v - self.window_size) % self.step_size > 0 else 0
                
            else:
                n = 0
                
            key_n_windows[k] = n
        
        return key_n_windows
        
    def get_fasta_coords(self, idx):
        """
        Get the start and end coordinates of a window for a given index.

        Args:
            idx (int): Index of the desired window.

        Returns:
            dict: A dictionary containing the contig key, start, and end coordinates of the window.
        """
        k_id = self.n_keys - sum(self.key_rolling_n > idx)
        n_past = 0 if k_id == 0 else self.key_rolling_n[k_id-1]
        window_idx = idx - n_past
        
        k = self.idx2key[k_id]
        start = window_idx * self.step_size
        end   = min(start + self.window_size, self.key_lens[k])
        start = end - self.window_size
        
        return {'key': k, 'start': start, 'end': end}

    def parse_complements(self):
        """
        Parse the complement matrix based on the provided alphabet and complement dictionary.

        Returns:
            numpy.ndarray: A matrix representing character complement relationships.
        """
        comp_mat = np.zeros( (len(self.alphabet),len(self.alphabet)) )
        
        for i in range(len(self.alphabet)):
            target_index = self.alphabet.index( self.complement_dict[ self.alphabet[i] ] )
            comp_mat[target_index,i] = 1
        return comp_mat
    
    def __len__(self):
        """
        Get the total number of windows in the dataset.

        Returns:
            int: Total number of windows.
        """
        strands = 2 if self.reverse_complements else 1
        
        return self.n_unstranded_windows * strands
    
    def __getitem__(self, idx):
        """
        Get the data for a specific window at the given index.

        Args:
            idx (int): Index of the desired window.

        Returns:
            tuple: A tuple containing the location tensor and the one-hot encoded sequence tensor.
        """

        if self.reverse_complements:
            strand = 1 if idx % 2 == 0 else -1
            u_idx = idx // 2
        else:
            u_idx = idx
            strand = 1
        
        fasta_loc = self.get_fasta_coords(u_idx)
        k, start, end = [fasta_loc[x] for x in ['key', 'start', 'end']]
        
        fasta_seq = self.fasta[k][:,start:end].astype(np.float32)
        fasta_seq = fasta_seq if strand == 1 else np.flip( self.complement_matrix @ fasta_seq, axis=-1)
        fasta_seq = torch.tensor(fasta_seq.copy())
        
        loc_tensor= torch.tensor([self.key2idx[k], start, end, strand])
        
        return loc_tensor, fasta_seq

class VCF:
    """
    A class for reading and handling Variant Call Format (VCF) files.

    Args:
        vcf_path (str): Path to the VCF file.
        max_allele_size (int, optional): Maximum allowed allele size. Default is 10000.
        max_indel_size (int, optional): Maximum allowed indel size. Default is 10000.
        alphabet (list[str], optional): List of allowed characters for alleles. Default is constants.STANDARD_NT.
        strict (bool, optional): Whether to raise an error if unknown tokens are found in alleles. Default is False.
        all_upper (bool, optional): Whether to convert alleles to uppercase. Default is True.
        chr_prefix (str, optional): Prefix to add to chromosome names. Default is an empty string.
        verbose (bool, optional): Whether to print verbose messages during processing. Default is False.

    Attributes:
        vcf_path (str): Path to the VCF file.
        max_allele_size (int): Maximum allowed allele size.
        max_indel_size (int): Maximum allowed indel size.
        alphabet (list[str]): List of allowed characters for alleles.
        strict (bool): Whether to raise an error if unknown tokens are found in alleles.
        all_upper (bool): Whether alleles are converted to uppercase.
        chr_prefix (str): Prefix to add to chromosome names.
        verbose (bool): Whether verbose messages are printed.
        vcf (pd.DataFrame): DataFrame containing the VCF data.

    Methods:
        _open_vcf(): Open and preprocess the VCF file, returning a DataFrame.
        __call__(loc_idx=None, iloc_idx=None): Get a VCF record by location or index.

    """
    
    def __init__(self, 
                 vcf_path, 
                 max_allele_size=10000,
                 max_indel_size=10000,
                 alphabet=constants.STANDARD_NT, 
                 strict=False, 
                 all_upper=True, chr_prefix='', 
                 verbose=False
                ):
        """
        Initialize the VCF object and read the VCF file.

        Args:
            vcf_path (str): Path to the VCF file.
            max_allele_size (int, optional): Maximum allowed allele size. Default is 10000.
            max_indel_size (int, optional): Maximum allowed indel size. Default is 10000.
            alphabet (list[str], optional): List of allowed characters for alleles. Default is constants.STANDARD_NT.
            strict (bool, optional): Whether to raise an error if unknown tokens are found in alleles. Default is False.
            all_upper (bool, optional): Whether to convert alleles to uppercase. Default is True.
            chr_prefix (str, optional): Prefix to add to chromosome names. Default is an empty string.
            verbose (bool, optional): Whether to print verbose messages during processing. Default is False.
        """
        self.vcf_path = vcf_path
        self.max_allele_size = max_allele_size
        self.max_indel_size = max_indel_size
        self.alphabet = [ x.upper() for x in alphabet ] if all_upper else alphabet
        self.strict   = strict
        self.all_upper= all_upper
        self.chr_prefix = chr_prefix
        self.verbose = verbose
        
        self.vcf = self._open_vcf()
        #self.read_vcf()
        
    def _open_vcf(self):
        """
        Open and preprocess the VCF file, returning a DataFrame.

        Returns:
            pd.DataFrame: DataFrame containing the VCF data.
        """
        vcf_colnames = ['chrom','pos','id','ref','alt','qual','filter','info']
        re_pat = matcher = f'[^{"".join(self.alphabet)}]'
        
        # Loading to DataFrame
        print('loading DataFrame', file=sys.stderr)
        if self.vcf_path.endswith('gz'):
            data = pd.read_csv(self.vcf_path, sep='\t', comment='#', header=None, compression='gzip', usecols=[0,1,2,3,4])
        else:
            data = pd.read_csv(self.vcf_path, sep='\t', comment='#', header=None, usecols=[0,1,2,3,4])
        
        print(f'loaded shape: {data.shape}', file=sys.stderr)
        data.columns = vcf_colnames[:data.shape[1]]
        data['chrom']= self.chr_prefix + data['chrom'].astype(str)
        
        # Checking and filtering tokens
        print('Checking and filtering tokens', file=sys.stderr)
        if self.all_upper:
            data['ref'] = data['ref'].str.upper()
            data['alt'] = data['alt'].str.upper()
        
        ref_filter = data['ref'].str.contains(re_pat,regex=True)
        alt_filter = data['alt'].str.contains(re_pat,regex=True)
        
        if self.strict:
            assert ref_filter.sum() > 0, "Found unknown token in ref. Abort."
            assert alt_filter.sum() > 0, "Found unknown token in alt. Abort."
        else:
            total_filter = ~(ref_filter | alt_filter)
            data = data.loc[ total_filter ]
        
        print(f'passed shape: {data.shape}', file=sys.stderr)
        # Length checks
        print('Allele length checks', file=sys.stderr)
        ref_lens = data['ref'].str.len()
        alt_lens = data['alt'].str.len()
        
        max_sizes   = np.maximum(ref_lens, alt_lens)
        indel_sizes = np.abs(ref_lens - alt_lens)
        
        size_filter = (max_sizes < self.max_allele_size) & (indel_sizes < self.max_indel_size)
        data = data.loc[size_filter]
        
        print(f'final shape: {data.shape}', file=sys.stderr)
        print('Done', file=sys.stderr)
        return data.reset_index(drop=True)
        
    def __call__(self, loc_idx=None, iloc_idx=None):
        """
        Get a VCF record by location or index.

        Args:
            loc_idx (int, optional): Location-based index of the desired record.
            iloc_idx (int, optional): Integer-based index of the desired record.

        Returns:
            pd.Series: A pandas Series representing the selected VCF record.
        """
        assert (loc_idx is None) ^ (iloc_idx is None), "Use loc XOR iloc"
        
        if loc_idx is not None:
            record = self.vcf.loc[loc_idx]
        else:
            record = self.vcf.iloc[iloc_idx]
            
        return record
        
    
class VcfDataset(Dataset):
    """
    A PyTorch dataset class for processing variant data from a VCF file and corresponding genomic sequences from a FASTA file.

    Args:
        vcf_obj (VCF): VCF object containing variant call data.
        fasta_obj (Fasta): Fasta object containing genomic sequences.
        window_size (int): Size of the data windows.
        relative_start (int): Relative start position within the window.
        relative_end (int): Relative end position within the window.
        step_size (int, optional): Step size for window sliding. Default is 1.
        reverse_complements (bool, optional): Whether to include reverse complements. Default is True.
        left_flank (str, optional): Left flank sequence to add to each window. Default is an empty string.
        right_flank (str, optional): Right flank sequence to add to each window. Default is an empty string.
        all_upper (bool, optional): Whether to convert sequences to uppercase. Default is True.
        use_contigs (list[str], optional): List of contig names to include. Default is an empty list.
        alphabet (list[str], optional): List of allowed characters for sequences. Default is constants.STANDARD_NT.
        complement_dict (dict[str, str], optional): Dictionary of nucleotide complements. Default is constants.DNA_COMPLEMENTS.

    Attributes:
        vcf (VCF): VCF object containing variant call data.
        fasta (Fasta): Fasta object containing genomic sequences.
        window_size (int): Size of the data windows.
        relative_start (int): Relative start position within the window.
        relative_end (int): Relative end position within the window.
        grab_size (int): Size of the genomic region to grab.
        step_size (int): Step size for window sliding.
        reverse_complements (bool): Whether reverse complements are included.
        left_flank (str): Left flank sequence added to each window.
        right_flank (str): Right flank sequence added to each window.
        all_upper (bool): Whether sequences are converted to uppercase.
        use_contigs (list[str]): List of contig names to include.
        alphabet (list[str]): List of allowed characters for sequences.
        complement_dict (dict[str, str]): Dictionary of nucleotide complements.
        complement_matrix (torch.Tensor): Matrix for nucleotide complement transformation.
        window_slicer (OneHotSlicer): Slicer for encoding sequences.

    Methods:
        parse_complements(): Parse the complement matrix for nucleotide transformation.
        encode(allele): Encode an allele sequence.
        filter_vcf(): Filter VCF records based on contigs and other criteria.
        __len__(): Get the number of samples in the dataset.
        __getitem__(idx): Get a sample from the dataset.
    """
    
    def __init__(self, 
                 vcf_obj, fasta_obj, window_size, 
                 relative_start, relative_end,  
                 step_size=1,
                 reverse_complements=True,
                 left_flank='', right_flank='', 
                 all_upper=True, use_contigs=[],
                 alphabet=constants.STANDARD_NT,
                 complement_dict=constants.DNA_COMPLEMENTS):
        """
        Initialize the VcfDataset object and preprocess the data.

        Args:
            vcf_obj (VCF): VCF object containing variant call data.
            fasta_obj (Fasta): Fasta object containing genomic sequences.
            window_size (int): Size of the data windows.
            relative_start (int): Relative start position within the window.
            relative_end (int): Relative end position within the window.
            step_size (int, optional): Step size for window sliding. Default is 1.
            reverse_complements (bool, optional): Whether to include reverse complements. Default is True.
            left_flank (str, optional): Left flank sequence to add to each window. Default is an empty string.
            right_flank (str, optional): Right flank sequence to add to each window. Default is an empty string.
            all_upper (bool, optional): Whether to convert sequences to uppercase. Default is True.
            use_contigs (list[str], optional): List of contig names to include. Default is an empty list.
            alphabet (list[str], optional): List of allowed characters for sequences. Default is constants.STANDARD_NT.
            complement_dict (dict[str, str], optional): Dictionary of nucleotide complements. Default is constants.DNA_COMPLEMENTS.
        """
        super().__init__()
        
        self.vcf   = vcf_obj
        self.fasta = fasta_obj
        self.window_size = window_size
        self.relative_start = relative_start
        self.relative_end   = relative_end
        self.grab_size = self.window_size-self.relative_start+self.relative_end-1
        
        self.step_size = step_size
        self.reverse_complements = reverse_complements
        
        self.left_flank = left_flank
        self.right_flank= right_flank
        self.all_upper = all_upper
        self.use_contigs = use_contigs
        self.alphabet = alphabet
        self.complement_dict = complement_dict
        self.complement_matrix = torch.tensor( self.parse_complements() ).float()
        
        self.window_slicer = OriginalOneHotSlicer(len(alphabet), window_size)
        
        self.filter_vcf()

    def parse_complements(self):
        """
        Parse the complement matrix for nucleotide transformation.

        Returns:
            torch.Tensor: Complement matrix for nucleotide transformation.
        """
        comp_mat = np.zeros( (len(self.alphabet),len(self.alphabet)) )
        
        for i in range(len(self.alphabet)):
            target_index = self.alphabet.index( self.complement_dict[ self.alphabet[i] ] )
            comp_mat[target_index,i] = 1
        return comp_mat
    
    def encode(self, allele):
        """
        Encode an allele sequence.

        Args:
            allele (str): Allele sequence to be encoded.

        Returns:
            torch.Tensor: One-hot encoded allele sequence.
        """
        my_allele = allele.upper() if self.all_upper else allele
        return alphabet_onehotizer(my_allele, self.alphabet)
        
    def filter_vcf(self):
        """
        Filter VCF records based on contigs and other criteria.
        """
        pre_len = self.vcf.shape[0]
        
        contig_filter = self.vcf['chrom'].isin(self.fasta.keys())
        print(f"{contig_filter.sum()}/{pre_len} records have matching contig in FASTA", file=sys.stderr)
        if len(self.use_contigs) > 0:
            contig_filter = contig_filter & self.vcf['chrom'].isin(self.use_contigs)
            print(f"removing {np.sum(~self.vcf['chrom'].isin(self.use_contigs))}/{pre_len} records based on contig blacklist", file=sys.stderr)
            
        if contig_filter.sum() < 1:
            print('No contigs passed. Check filters.', file=sys.stderr)
        
        self.vcf = self.vcf.loc[ contig_filter ]
        print(f"returned {self.vcf.shape[0]}/{pre_len} records", file=sys.stderr)
        return None
    
    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
            int: Number of samples.
        """
        return self.vcf.shape[0]
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.

        Args:
            idx (int): Index of the sample.

        Returns:
            dict: Dictionary containing 'ref' and 'alt' sequences.
        """
        record = self.vcf.iloc[idx]
        
        ref = self.encode(record['ref'])
        alt = self.encode(record['alt'])
        
        var_loc = record['pos'] - 1
        start   = var_loc - self.relative_end + 1

        trail_start = var_loc + ref.shape[1]
        trail_end   = start + self.grab_size

        len_dif = alt.shape[1] - ref.shape[1]
        start_adjust = len_dif // 2
        end_adjust   = len_dif - start_adjust

        try:
            # Collect reference
            contig = self.fasta[ record['chrom'] ]
            assert var_loc < contig.shape[1], "Variant position outside of chromosome bounds. Check VCF/FASTA build version."
            leader = contig[:, start:var_loc]
            trailer= contig[:, trail_start:trail_end]
            
            ref_segments = [leader, ref, trailer]
            
            # Collect alternate
            leader = contig[:, start+start_adjust:var_loc]
            trailer= contig[:, trail_start:trail_end-end_adjust]
            
            alt_segments = [leader, alt, trailer]
            
            # Combine segments
            ref = np.concatenate(ref_segments, axis=-1)
            alt = np.concatenate(alt_segments, axis=-1)
            
            ref = torch.tensor(ref[np.newaxis].astype(np.float32))
            alt = torch.tensor(alt[np.newaxis].astype(np.float32))

            try:
                ref_slices = self.window_slicer(ref)[::self.step_size]
                alt_slices = self.window_slicer(alt)[::self.step_size]
            except RuntimeError:
                print(ref)
                print(ref.shape)
                print(alt)
                print(alt.shape)
                
                raise RuntimeError


            if self.reverse_complements:
                ref_rc = torch.flip(self.complement_matrix @ ref_slices, dims=[-1])
                ref_slices = torch.cat([ref_slices,ref_rc], dim=0)

                alt_rc = torch.flip(self.complement_matrix @ alt_slices, dims=[-1])
                alt_slices = torch.cat([alt_slices,alt_rc], dim=0)

            return {'ref': ref_slices, 'alt': alt_slices}

        except KeyError:
            print(f"No contig: {record['chrom']} in FASTA, skipping", file=sys.stderr)
            return {'ref': None, 'alt': None}
### 082425 UPDATED LOGIC ###
# Gemini reversion - 082725 - 1:02 AM
# --- START: Replace from this line down in fasta_datamodule.py ---
class VcfDataset_SimplePadding(VcfDataset):
    """
    Final definitive VcfDataset. This class uses a single, unified, and arithmetically
    correct logic that adapts to each variant type to ensure perfect alignment and
    prediction consistency.
    """
    def __init__(self, *args, **kwargs):
        # The __init__ from the parent class is sufficient. We only override __getitem__.
        parent_kwargs = {k: v for k, v in kwargs.items() if k in VcfDataset.__init__.__code__.co_varnames}
        super().__init__(*args, **parent_kwargs)
        print("Using VcfDataset_SimplePadding (Final Unified Formula) to generate 600bp windows.", file=sys.stderr)


    def __getitem__(self, idx):
        record = self.vcf.iloc[idx]
        ref_windows, alt_windows = [], []
        try:
            var_loc = record['pos'] - 1
            contig = self.fasta[record['chrom']]
            ref_allele_encoded = self.encode(record['ref'])
            alt_allele_encoded = self.encode(record['alt'])
            ref_len, alt_len = ref_allele_encoded.shape[1], alt_allele_encoded.shape[1]

            for window_start_pos in range(self.relative_start, self.relative_end, self.step_size):
                
                # --- Final Unified Formula Logic ---
                insertion_size = alt_len - ref_len
                
                # 1. Calculate the alignment shift based on insertion size.
                shift = 1 - (insertion_size // 2) if insertion_size > 0 else 1
                
                # 2. Determine the anchor allele for context sizing.
                anchor_len = alt_len if insertion_size > 0 else ref_len
                
                # 3. Apply the calculated rules.
                leader_len = window_start_pos + shift
                trailer_len = self.window_size - leader_len - anchor_len
                
                leader_genomic = contig[:, var_loc - leader_len : var_loc]
                trailer_start_pos = var_loc + ref_len
                trailer_genomic = contig[:, trailer_start_pos : trailer_start_pos + trailer_len]
                
                ref_insert = np.concatenate([leader_genomic, ref_allele_encoded, trailer_genomic], axis=-1)
                alt_insert = np.concatenate([leader_genomic, alt_allele_encoded, trailer_genomic], axis=-1)

                ref_flank_total_len = 600 - ref_insert.shape[1]
                ref_left_len, ref_right_len = ref_flank_total_len // 2, ref_flank_total_len - (ref_flank_total_len // 2)
                alt_flank_total_len = 600 - alt_insert.shape[1]
                alt_left_len, alt_right_len = alt_flank_total_len // 2, alt_flank_total_len - (alt_flank_total_len // 2)

                ref_left_flank = self.encode(constants.MPRA_UPSTREAM[-ref_left_len:])
                ref_right_flank = self.encode(constants.MPRA_DOWNSTREAM[:ref_right_len])
                alt_left_flank = self.encode(constants.MPRA_UPSTREAM[-alt_left_len:])
                alt_right_flank = self.encode(constants.MPRA_DOWNSTREAM[:alt_right_len])
                
                ref_full_seq = np.concatenate([ref_left_flank, ref_insert, ref_right_flank], axis=-1)
                alt_full_seq = np.concatenate([alt_left_flank, alt_insert, alt_right_flank], axis=-1)
                
                if ref_full_seq.shape[1] < 600:
                    padding = np.zeros((len(self.alphabet), 600 - ref_full_seq.shape[1])); ref_full_seq = np.concatenate([ref_full_seq, padding], axis=-1)
                if alt_full_seq.shape[1] < 600:
                    padding = np.zeros((len(self.alphabet), 600 - alt_full_seq.shape[1])); alt_full_seq = np.concatenate([alt_full_seq, padding], axis=-1)
                
                ref_windows.append(ref_full_seq[:, :600]); alt_windows.append(alt_full_seq[:, :600])

            ref_slices_fwd = torch.from_numpy(np.stack(ref_windows, axis=0).astype(np.float32))
            alt_slices_fwd = torch.from_numpy(np.stack(alt_windows, axis=0).astype(np.float32))

            if self.reverse_complements:
                ref_centers_rc = torch.flip(self.complement_matrix @ ref_slices_fwd[:, :, 200:400], dims=[-1])
                alt_centers_rc = torch.flip(self.complement_matrix @ alt_slices_fwd[:, :, 200:400], dims=[-1])
                
                fwd_flanks_L_ref = ref_slices_fwd[:, :, :200]; fwd_flanks_R_ref = ref_slices_fwd[:, :, 400:]
                ref_rc_full = torch.cat([fwd_flanks_L_ref, ref_centers_rc, fwd_flanks_R_ref], dim=2)

                fwd_flanks_L_alt = alt_slices_fwd[:, :, :200]; fwd_flanks_R_alt = alt_slices_fwd[:, :, 400:]
                alt_rc_full = torch.cat([fwd_flanks_L_alt, alt_centers_rc, fwd_flanks_R_alt], dim=2)
                
                ref_slices = torch.cat([ref_slices_fwd, ref_rc_full], dim=0)
                alt_slices = torch.cat([alt_slices_fwd, alt_rc_full], dim=0)
            else:
                ref_slices, alt_slices = ref_slices_fwd, alt_slices_fwd
            
            return {'ref': ref_slices, 'alt': alt_slices}
        except Exception as e:
            print(f"Error processing {record['chrom']}:{record['pos']}: {e}", file=sys.stderr)
            return {'ref': torch.empty(0), 'alt': torch.empty(0)}
            
# --- VALIDATION AND EXTRACTION FUNCTIONS (Updated with final logic) ---
def array_to_sequence(array, alphabet=constants.STANDARD_NT):
    if array.ndim > 2: array = array.squeeze()
    indices = array.argmax(axis=0)
    return ''.join([alphabet[i] for i in indices])

def dna_reverse_complement(sequence):
    complement_dict = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return "".join(complement_dict.get(base, base) for base in reversed(sequence))

def _extract_sequences_core_logic(vcf_data, idx):
    """Helper function with the final, formulaic, and aligned sequence generation logic."""
    record = vcf_data.vcf.iloc[idx]
    results = []
    var_loc = record['pos'] - 1
    contig = vcf_data.fasta[record['chrom']]
    ref_allele_encoded = vcf_data.encode(record['ref'])
    alt_allele_encoded = vcf_data.encode(record['alt'])
    ref_len, alt_len = ref_allele_encoded.shape[1], alt_allele_encoded.shape[1]
    
    for window_start_pos in range(vcf_data.relative_start, vcf_data.relative_end, vcf_data.step_size):
        insertion_size = alt_len - ref_len
        
        shift = 1 - (insertion_size // 2) if insertion_size > 0 else 1
        anchor_len = alt_len if insertion_size > 0 else ref_len
        
        leader_len = window_start_pos + shift
        trailer_len = vcf_data.window_size - leader_len - anchor_len
        
        leader_genomic = contig[:, var_loc - leader_len: var_loc]
        trailer_start_pos = var_loc + ref_len
        trailer_genomic = contig[:, trailer_start_pos: trailer_start_pos + trailer_len]
        
        ref_insert = np.concatenate([leader_genomic, ref_allele_encoded, trailer_genomic], axis=-1)
        alt_insert = np.concatenate([leader_genomic, alt_allele_encoded, trailer_genomic], axis=-1)

        ref_flank_total_len = 600 - ref_insert.shape[1]
        ref_left_len, ref_right_len = ref_flank_total_len // 2, ref_flank_total_len - (ref_flank_total_len // 2)
        alt_flank_total_len = 600 - alt_insert.shape[1]
        alt_left_len, alt_right_len = alt_flank_total_len // 2, alt_flank_total_len - (alt_flank_total_len // 2)

        ref_left_flank_encoded = vcf_data.encode(constants.MPRA_UPSTREAM[-ref_left_len:])
        ref_right_flank_encoded = vcf_data.encode(constants.MPRA_DOWNSTREAM[:ref_right_len])
        alt_left_flank_encoded = vcf_data.encode(constants.MPRA_UPSTREAM[-alt_left_len:])
        alt_right_flank_encoded = vcf_data.encode(constants.MPRA_DOWNSTREAM[:alt_right_len])
        results.append({
            'window_start_pos': window_start_pos, 'ref_center': ref_insert, 'alt_center': alt_insert,
            'ref_left_flank': ref_left_flank_encoded, 'ref_right_flank': ref_right_flank_encoded,
            'alt_left_flank': alt_left_flank_encoded, 'alt_right_flank': alt_right_flank_encoded
        })
    return results, record

def _format_extraction_output(vcf_data, sample_indices, extractor_func):
    if sample_indices is None: sample_indices = range(len(vcf_data))
    all_primary_seqs, all_secondary_seqs, all_metadata = [], [], []
    for idx in tqdm.tqdm(sample_indices):
        try:
            window_results, record = _extract_sequences_core_logic(vcf_data, idx)
            primary_fwd, secondary_fwd = extractor_func(window_results, vcf_data)
            if vcf_data.reverse_complements:
                primary_rev, secondary_rev = extractor_func(window_results, vcf_data, reverse_complement=True)
                primary_fwd.extend(primary_rev); secondary_fwd.extend(secondary_rev)
            all_primary_seqs.append(primary_fwd); all_secondary_seqs.append(secondary_fwd)
            all_metadata.append({'variant_idx': idx, 'chrom': record['chrom'], 'pos': record['pos'], 'ref': record['ref'], 'alt': record['alt']})
        except Exception as e: print(f"Skipping extraction for index {idx} due to error: {e}"); continue
    return all_primary_seqs, all_secondary_seqs, all_metadata

def save_center_sequences(vcf_data, output_file, sample_indices=None):
    print(f"Extracting 200bp center sequences from the new method for {len(sample_indices or vcf_data)} samples...")
    def extractor(results, vcf_data, reverse_complement=False):
        refs, alts = [], []
        for r in results:
            ref_str, alt_str, pos = array_to_sequence(r['ref_center']), array_to_sequence(r['alt_center']), r['window_start_pos']
            if reverse_complement:
                refs.append(f"reverse_window_{pos:03d}:{dna_reverse_complement(ref_str)}"); alts.append(f"reverse_window_{pos:03d}:{dna_reverse_complement(alt_str)}")
            else:
                refs.append(f"forward_window_{pos:03d}:{ref_str}"); alts.append(f"forward_window_{pos:03d}:{alt_str}")
        return refs, alts
    refs, alts, meta = _format_extraction_output(vcf_data, sample_indices, extractor)
    output_data = [{'ref_center_sequence': r, 'alt_center_sequence': a, **m} for i, m in enumerate(meta) for r, a in zip(refs[i], alts[i])]
    pd.DataFrame(output_data).to_csv(output_file, sep='\t', index=False)
    print(f"\nNew method's 200bp center sequences saved to: {output_file}")

def save_flank_sequences(vcf_data, output_file, sample_indices=None):
    print(f"Extracting flank sequences from the new method for {len(sample_indices or vcf_data)} samples...")
    def extractor(results, vcf_data, reverse_complement=False):
        lefts, rights = [], []
        for r in results:
            # --- CORRECTED LOGIC: Use the pre-encoded flank arrays ---
            ref_left_seq = array_to_sequence(r['ref_left_flank'])
            ref_right_seq = array_to_sequence(r['ref_right_flank'])
            ref_left_len = len(ref_left_seq)
            ref_right_len = len(ref_right_seq)
            pos = r['window_start_pos']

            if reverse_complement:
                # RC center uses the same FWD flanks
                lefts.append(f"reverse_window_{pos:03d}:L({ref_left_len}bp):{ref_left_seq}")
                rights.append(f"reverse_window_{pos:03d}:R({ref_right_len}bp):{ref_right_seq}")
            else:
                lefts.append(f"forward_window_{pos:03d}:L({ref_left_len}bp):{ref_left_seq}")
                rights.append(f"forward_window_{pos:03d}:R({ref_right_len}bp):{ref_right_seq}")
        return lefts, rights
    
    # This function now needs to be rewritten to output both ref and alt flanks correctly
    if sample_indices is None: sample_indices = range(len(vcf_data))
    output_data = []
    for idx in tqdm.tqdm(sample_indices):
        try:
            window_results, record = _extract_sequences_core_logic(vcf_data, idx)
            meta_info = {
                'variant_idx': idx, 'chrom': record['chrom'], 'pos': record['pos'],
                'ref': record['ref'], 'alt': record['alt']
            }
            # This loop structure correctly creates one row per window
            for r in window_results:
                pos = r['window_start_pos']
                # Fwd
                row_fwd = meta_info.copy()
                row_fwd['ref_flank_sequences'] = f"forward_window_{pos:03d}:L({r['ref_left_flank'].shape[1]}bp):{array_to_sequence(r['ref_left_flank'])};R({r['ref_right_flank'].shape[1]}bp):{array_to_sequence(r['ref_right_flank'])}"
                row_fwd['alt_flank_sequences'] = f"forward_window_{pos:03d}:L({r['alt_left_flank'].shape[1]}bp):{array_to_sequence(r['alt_left_flank'])};R({r['alt_right_flank'].shape[1]}bp):{array_to_sequence(r['alt_right_flank'])}"
                output_data.append(row_fwd)
                # RC
                if vcf_data.reverse_complements:
                    row_rc = meta_info.copy()
                    row_rc['ref_flank_sequences'] = f"reverse_window_{pos:03d}:L({r['ref_left_flank'].shape[1]}bp):{array_to_sequence(r['ref_left_flank'])};R({r['ref_right_flank'].shape[1]}bp):{array_to_sequence(r['ref_right_flank'])}"
                    row_rc['alt_flank_sequences'] = f"reverse_window_{pos:03d}:L({r['alt_left_flank'].shape[1]}bp):{array_to_sequence(r['alt_left_flank'])};R({r['alt_right_flank'].shape[1]}bp):{array_to_sequence(r['alt_right_flank'])}"
                    output_data.append(row_rc)
        except Exception as e:
            print(f"Skipping flank extraction for index {idx} due to error: {e}")
            continue

    pd.DataFrame(output_data).to_csv(output_file, sep='\t', index=False)
    print(f"\nNew method's flank sequences saved to: {output_file}")


def save_full_sequences(vcf_data, output_file, sample_indices=None):
    print(f"Extracting full 600bp sequences from the new method for {len(sample_indices or vcf_data)} samples...")
    def extractor(results, vcf_data, reverse_complement=False):
        refs, alts = [], []
        for r in results:
            # --- CORRECTED LOGIC: Use the pre-encoded flank and center arrays ---
            ref_center_seq = array_to_sequence(r['ref_center'])
            alt_center_seq = array_to_sequence(r['alt_center'])
            
            ref_left_flank_seq = array_to_sequence(r['ref_left_flank'])
            ref_right_flank_seq = array_to_sequence(r['ref_right_flank'])
            alt_left_flank_seq = array_to_sequence(r['alt_left_flank'])
            alt_right_flank_seq = array_to_sequence(r['alt_right_flank'])
            
            pos = r['window_start_pos']
            if reverse_complement:
                ref_center_rc = dna_reverse_complement(ref_center_seq)
                alt_center_rc = dna_reverse_complement(alt_center_seq)
                # RC center uses FWD flanks
                ref_full_str = ref_left_flank_seq + ref_center_rc + ref_right_flank_seq
                alt_full_str = alt_left_flank_seq + alt_center_rc + alt_right_flank_seq
                refs.append(f"reverse_window_{pos:03d}:{ref_full_str}"); alts.append(f"reverse_window_{pos:03d}:{alt_full_str}")
            else:
                ref_full_str = ref_left_flank_seq + ref_center_seq + ref_right_flank_seq
                alt_full_str = alt_left_flank_seq + alt_center_seq + alt_right_flank_seq
                refs.append(f"forward_window_{pos:03d}:{ref_full_str}"); alts.append(f"forward_window_{pos:03d}:{alt_full_str}")
        return refs, alts
    refs, alts, meta = _format_extraction_output(vcf_data, sample_indices, extractor)
    output_data = [{'ref_full_sequence': r, 'alt_full_sequence': a, **m} for i, m in enumerate(meta) for r, a in zip(refs[i], alts[i])]
    pd.DataFrame(output_data).to_csv(output_file, sep='\t', index=False)
    print(f"\nNew method's full 600bp sequences saved to: {output_file}")

# Note: The original save_original_sequences function and its helpers should be kept above this block if you still want to use it.