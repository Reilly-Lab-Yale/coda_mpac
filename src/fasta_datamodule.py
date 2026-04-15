import sys
import argparse
import tempfile
import time
import gzip
from functools import partial
from collections import defaultdict

import numpy as np
import pandas as pd
import math

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from torch.utils.data import random_split, DataLoader, TensorDataset, ConcatDataset, Dataset

from ..common import constants, utils

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
        window_alignmnet (str, optional): {left, right, middle} allele position on which to center sliding windows. Default is left.
        window_symmetric (bool): {True, False} Whether sliding windows are symmetric. Overrides relative_start/end if True. Default is False.
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
        window_alignment (str): Alignmnet position for sliding windows (left, right, middle)
        window_symmetric (bool): Whether sliding windows are symmetric around alignment point.
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
                 window_alignment='left',
                 window_symmetric=False,
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
        self.window_alignment = window_alignment
        self.window_symmetric = window_symmetric
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
        
        self.window_slicer = OneHotSlicer(len(alphabet), window_size)
        
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
        self.setwindows()
        print("Using VcfDataset_SimplePadding (Final Unified Formula) to generate 600bp windows.", file=sys.stderr)

    def setwindows(self):
        """
        Set start and end windows for each variant. Filter VCF records that cannot be windowed VCF records based on contigs and other criteria.
        """
        pre_len = self.vcf.shape[0]
        self.vcf['start'] = self.relative_start
        self.vcf['end'] = self.relative_end
        
        for idx in range(0, pre_len):
            record = self.vcf.iloc[idx]
            ref_len = self.encode(record['ref']).shape[1]
            alt_len = self.encode(record['alt']).shape[1]
            anchor_len = alt_len if alt_len > ref_len else ref_len

            if self.window_symmetric: #Ensure windows are symmetric around the anchor point
                if (self.window_alignment == 'left') or (self.window_alignment == 'right'):
                    true_flank = (self.window_size - 2*anchor_len) // 2
                elif self.window_alignment == 'middle':
                    true_flank = (self.window_size - anchor_len) // 2
                half_window = self.window_size // 2
                user_flank = max(half_window - self.relative_start, self.relative_end - half_window)
                min_flank = min(user_flank, true_flank)
                num_windows = (2*min_flank) // self.step_size
                window_range = num_windows*self.step_size
                start = half_window - window_range // 2
                end = half_window + window_range // 2
            else: # Ensure allele always fits into the window
                true_flank = self.window_size #dummy
                if (self.window_alignment == 'left') or (self.window_alignment == 'right'):
                    start = self.relative_start
                    end = min(self.relative_end, self.window_size - anchor_len)
                elif self.window_alignment == 'middle':
                    start = max(self.relative_start, anchor_len // 2)
                    end = min(self.relative_end, self.window_size - anchor_len // 2)
            if (true_flank <= 0):
                print(f"Allele length for index={idx} does not fit into window/orientation: {anchor_len}, {self.window_size}, {self.window_alignment}")
                self.vcf.iloc[idx, self.vcf.columns.get_loc('start')] = -1
                self.vcf.iloc[idx, self.vcf.columns.get_loc('end')] = -1
            elif (end < start):
                print(f"Invalid window bounds start={start}, end={end} for index={idx}")
                self.vcf.iloc[idx, self.vcf.columns.get_loc('start')] = -1
                self.vcf.iloc[idx, self.vcf.columns.get_loc('end')] = -1
            else:
                self.vcf.iloc[idx, self.vcf.columns.get_loc('start')] = start
                self.vcf.iloc[idx, self.vcf.columns.get_loc('end')] = end

        self.vcf = self.vcf.loc[self.vcf['start'] >=0] 

        print(f"{self.vcf.shape[0]}/{pre_len} records passed window filter", file=sys.stderr)
        return None
 
    def __getitem__(self, idx):
        record = self.vcf.iloc[idx]
        ref_windows, alt_windows = [], []

        try:
            # Variant setup
            var_loc = record['pos'] - 1
            contig = self.fasta[record['chrom']]

            ref_allele_encoded = self.encode(record['ref'])
            alt_allele_encoded = self.encode(record['alt'])

            ref_len = ref_allele_encoded.shape[1]
            alt_len = alt_allele_encoded.shape[1]

            insertion_size = alt_len - ref_len
            #shift = - (insertion_size // 2) if insertion_size > 0 else 0

            anchor_len = alt_len if insertion_size > 0 else ref_len

            start = record['start']
            end = record['end']
            print(f"Using window bounds start={start}, end={end} for {record['chrom']}:{record['pos']}", file=sys.stderr)
 
            # Sliding windows
            for window_start_pos in range(start, end + 1, self.step_size):
                # LEFT aligned
                # allele is placed immediately after the leader region
                if self.window_alignment == 'left':
                    leader_len = window_start_pos #+ shift
                    trailer_len = self.window_size - leader_len - anchor_len

                # RIGHT aligned
                # allele is placed immediately before the trailer region
                elif self.window_alignment == 'right':
                    trailer_len = window_start_pos #+ shift
                    leader_len = self.window_size - trailer_len - anchor_len
            
                # MIDDLE aligned
                # half of allele is included in leader
                elif self.window_alignment == 'middle':
                    leader_len =  window_start_pos - (anchor_len // 2)
                    trailer_len = self.window_size - leader_len - anchor_len

                # Edge case protection: clamp lengths
                if leader_len < 0: leader_len = 0
                if trailer_len < 0: trailer_len = 0

                # Extract genomic leader (left flank)
                leader_start = var_loc - leader_len
                leader_end   = var_loc
                # clamp boundaries
                leader_start = max(0, leader_start)
                leader_end = min(leader_end, contig.shape[1])
                leader_end = max(leader_start, leader_end)

                leader_genomic = contig[:, leader_start:leader_end]

                # Extract genomic trailer (right flank)
                trailer_start = var_loc + ref_len
                trailer_end   = trailer_start + trailer_len
                # clamp boundaries
                trailer_end = min(trailer_end, contig.shape[1])
                trailer_start = max(0, trailer_start)
                trailer_start = min(trailer_start, trailer_end)

                trailer_genomic = contig[:, trailer_start:trailer_end]
            
                # Build ref/alt inserts
                ref_insert = np.concatenate([leader_genomic, ref_allele_encoded, trailer_genomic], axis=-1)
                alt_insert = np.concatenate([leader_genomic, alt_allele_encoded, trailer_genomic], axis=-1)

                # MPRA flanking to ensure final length = 600
                def add_flanks(insert):
                    flank_total = 600 - insert.shape[1]
                    left_len = flank_total // 2
                    right_len = flank_total - left_len

                    if flank_total < 0:
                        # window too long; trim equally from both sides
                        trim_left = flank_total // -2
                        trim_right = -flank_total - trim_left
                        insert = insert[:, trim_left:-trim_right]
                        left_len = 0
                        right_len = 0

                    left_flank = self.encode(constants.MPRA_UPSTREAM[-left_len:])
                    right_flank = self.encode(constants.MPRA_DOWNSTREAM[:right_len])

                    full = np.concatenate([left_flank, insert, right_flank], axis=-1)

                    # pad if shorter
                    if full.shape[1] < 600:
                        pad = np.zeros((len(self.alphabet), 600 - full.shape[1]))
                        full = np.concatenate([full, pad], axis=-1)

                    return full[:, :600]

                ref_windows.append(add_flanks(ref_insert))
                alt_windows.append(add_flanks(alt_insert))
        
            # Tensor stacking and reverse complement handling
            ref_slices_fwd = torch.from_numpy(np.stack(ref_windows).astype(np.float32))
            alt_slices_fwd = torch.from_numpy(np.stack(alt_windows).astype(np.float32))

            if self.reverse_complements:
                # reverse-complement only center 200bp
                ref_centers_rc = torch.flip(self.complement_matrix @ ref_slices_fwd[:, :, 200:400], dims=[-1])
                alt_centers_rc = torch.flip(self.complement_matrix @ alt_slices_fwd[:, :, 200:400], dims=[-1])

                ref_rc = torch.cat([ref_slices_fwd[:, :, :200], ref_centers_rc, ref_slices_fwd[:, :, 400:]], dim=2)
                alt_rc = torch.cat([alt_slices_fwd[:, :, :200], alt_centers_rc, alt_slices_fwd[:, :, 400:]], dim=2)

                ref_slices = torch.cat([ref_slices_fwd, ref_rc], dim=0)
                alt_slices = torch.cat([alt_slices_fwd, alt_rc], dim=0)
            else:
                ref_slices = ref_slices_fwd
                alt_slices = alt_slices_fwd


            return {"ref": ref_slices, "alt": alt_slices}

        except Exception as e:
            print(f"Error processing {record['chrom']}:{record['pos']}: {e}", file=sys.stderr)
            return {"ref": torch.empty(0), "alt": torch.empty(0)}

# --------------------------------------------------------------------------------------------------------------------------
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
    insertion_size = alt_len - ref_len
    #shift = - (insertion_size // 2) if insertion_size > 0 else 0
    anchor_len = alt_len if insertion_size > 0 else ref_len
    #print(f"Anchor_len: {anchor_len}")

    if vcf_data.window_symmetric: #Ensure windows are symmetric around the anchor point
        if (vcf_data.window_alignment == 'left') or (vcf_data.window_alignment == 'right'):
            true_flank = (vcf_data.window_size - 2*anchor_len) // 2
        elif vcf_data.window_alignment == 'middle':
            true_flank = (vcf_data.window_size - anchor_len) // 2
        if true_flank < 0:
            raise ValueError(f"Allele length does not fit into window/orientation: {anchor_len}, {vcf_data.window_size}, {vcf_data.window_alignment}")
        half_window = vcf_data.window_size // 2
        user_flank = max(half_window - vcf_data.relative_start, vcf_data.relative_end - half_window)
        min_flank = min(user_flank, true_flank)
        num_windows = (2*min_flank) // vcf_data.step_size
        window_range = num_windows*vcf_data.step_size
        start = half_window - window_range // 2
        end = half_window + window_range // 2
    else: #Ensure allele fits into window
        if (vcf_data.window_alignment == 'left') or (vcf_data.window_alignment == 'right'):
            start = vcf_data.relative_start
            end = min(vcf_data.relative_end, vcf_data.window_size - anchor_len)
        elif vcf_data.window_alignment == 'middle':
            start = max(vcf_data.relative_start, anchor_len // 2)
            end = min(vcf_data.relative_end, vcf_data.window_size - anchor_len // 2)

    if end < start:
        raise ValueError(f"Invalid window bounds start={start}, end={end}", file=sys.stderr)
    print(f"Using window bounds start={start}, end={end} for index {idx}", file=sys.stderr) 

    for window_start_pos in range(start, end+1, vcf_data.step_size):
        if vcf_data.window_alignment == 'left':
            leader_len = window_start_pos #+ shift
            trailer_len = vcf_data.window_size - leader_len - anchor_len
        elif vcf_data.window_alignment == 'right':
            trailer_len = window_start_pos #+ shift
            leader_len = vcf_data.window_size - trailer_len - anchor_len
        elif vcf_data.window_alignment == 'middle':
            leader_len =  window_start_pos - (anchor_len // 2)
            trailer_len = vcf_data.window_size - leader_len - anchor_len

        if leader_len < 0: leader_len = 0
        if trailer_len < 0: trailer_len = 0
        leader_start = var_loc - leader_len
        leader_end   = var_loc
        leader_start = max(0, leader_start)
        leader_end = min(leader_end, contig.shape[1])
        leader_end = max(leader_start, leader_end)
        leader_genomic = contig[:, leader_start:leader_end]
        trailer_start = var_loc + ref_len
        trailer_end   = trailer_start + trailer_len
        trailer_end = min(trailer_end, contig.shape[1])
        trailer_start = max(0, trailer_start)
        trailer_start = min(trailer_start, trailer_end)
        trailer_genomic = contig[:, trailer_start:trailer_end]

        ref_insert = np.concatenate([leader_genomic, ref_allele_encoded, trailer_genomic], axis=-1)
        alt_insert = np.concatenate([leader_genomic, alt_allele_encoded, trailer_genomic], axis=-1)

        def add_flanks(insert):
            flank_total = 600 - insert.shape[1]
            left_len = flank_total // 2
            right_len = flank_total - left_len

            if flank_total < 0:
                # window too long; trim equally from both sides
                trim_left = flank_total // -2
                trim_right = -flank_total - trim_left
                insert = insert[:, trim_left:-trim_right]
                left_len = 0
                right_len = 0

            left_flank = vcf_data.encode(constants.MPRA_UPSTREAM[-left_len:])
            right_flank = vcf_data.encode(constants.MPRA_DOWNSTREAM[:right_len])
            # SKIP PADDING
            #full = np.concatenate([left_flank, insert, right_flank], axis=-1)
            #if full.shape[1] < 600:
            #    pad = np.zeros((len(self.alphabet), 600 - full.shape[1]))
            #    full = np.concatenate([full, pad], axis=-1)
            return left_flank, insert, right_flank
        
        ref_left, ref_center, ref_right = add_flanks(ref_insert)
        alt_left, alt_center, alt_right = add_flanks(alt_insert)

        results.append({
            'window_start_pos': window_start_pos, 'ref_center': ref_center, 'alt_center': alt_center,
            'ref_left_flank': ref_left, 'ref_right_flank': ref_right,
            'alt_left_flank': alt_left, 'alt_right_flank': alt_right
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
        except Exception as e: print(f"Skipping extraction for index {idx} due to error: {e}", file=sys.stderr); continue
    return all_primary_seqs, all_secondary_seqs, all_metadata

def save_center_sequences(vcf_data, output_file, sample_indices=None):
    print(f"Extracting 200bp center sequences from the new method for {len(sample_indices or vcf_data)} samples...", file=sys.stderr)
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
        refs, alts = [], []
        for r in results:
            ref_left, ref_right, alt_left, alt_right, pos = array_to_sequence(r['ref_left_flank']), array_to_sequence(r['ref_right_flank']), array_to_sequence(r['alt_left_flank']), array_to_sequence(r['alt_right_flank']), r['window_start_pos']
            if reverse_complement:
                # RC uses same FWD flanks
                refs.append(f"reverse_window_{pos:03d}:L({r['ref_left_flank'].shape[1]}bp):{array_to_sequence(r['ref_left_flank'])};R({r['ref_right_flank'].shape[1]}bp):{array_to_sequence(r['ref_right_flank'])}")
                alts.append(f"reverse_window_{pos:03d}:L({r['alt_left_flank'].shape[1]}bp):{array_to_sequence(r['alt_left_flank'])};R({r['alt_right_flank'].shape[1]}bp):{array_to_sequence(r['alt_right_flank'])}")
            else:
                refs.append(f"forward_window_{pos:03d}:L({r['ref_left_flank'].shape[1]}bp):{array_to_sequence(r['ref_left_flank'])};R({r['ref_right_flank'].shape[1]}bp):{array_to_sequence(r['ref_right_flank'])}")
                alts.append(f"forward_window_{pos:03d}:L({r['alt_left_flank'].shape[1]}bp):{array_to_sequence(r['alt_left_flank'])};R({r['alt_right_flank'].shape[1]}bp):{array_to_sequence(r['alt_right_flank'])}")
        return refs, alts

    refs, alts, meta = _format_extraction_output(vcf_data, sample_indices, extractor)
    output_data = [{'ref_flank_sequence': r, 'alt_flank_sequence': a, **m} for i, m in enumerate(meta) for r, a in zip(refs[i], alts[i])]
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

