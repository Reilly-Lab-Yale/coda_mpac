import subprocess
import sys
import os
import shutil
import gzip
import csv
import argparse
import multiprocessing

import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from functorch import combine_state_for_ensemble, vmap

import numpy as np
import pandas as pd
import boda
from boda.common import constants, utils
from boda.common.utils import unpack_artifact, model_fn


def load_model(artifact_path):
    """
    Load a trained model from the specified artifact path.

    Args:
        artifact_path (str): Path to the model artifact.

    Returns:
        nn.Module: The loaded trained model.
    """
    USE_CUDA = torch.cuda.device_count() >= 1
    if os.path.isdir('./artifacts'):
        shutil.rmtree('./artifacts')

    unpack_artifact(artifact_path)

    model_dir = './artifacts'

    my_model = model_fn(model_dir)
    my_model.eval()
    if USE_CUDA:
        my_model.cuda()
    
    return my_model

def combine_ref_alt_skew_tensors(ref, alt, skew, ids=None):
    """
    Combine reference, alternative, and skew tensors into a single DataFrame.

    Args:
        ref (torch.Tensor): Reference tensor.
        alt (torch.Tensor): Alternative tensor.
        skew (torch.Tensor): Skew tensor.
        ids (list): List of identifiers for each tensor element.

    Returns:
        pd.DataFrame: A DataFrame containing combined information from the input tensors.
    """
    result = []
    
    for tag, data in zip(['ref', 'alt', 'skew'], [ref, alt, skew]):
        hold = pd.DataFrame( data.numpy(), columns=ids )
        for col in hold.columns:
            hold[col] = f'{col}__{tag}=' + hold[col].astype(str)
            
        result.append( hold.agg(';'.join, axis=1) )
        
    result = pd.concat(result, axis=1)
    
    return result.agg(';'.join, axis=1)

class ConsistentModelPool(nn.Module):
    """
    Ensemble of consistent models.

    This class creates an ensemble of consistent models from a list of model paths.
    
    Args:
        path_list (list): List of paths to model artifacts.

    Attributes:
        fmodel (nn.Module): The ensemble forward model.
        params (dict): Parameters shared across models.
        buffers (dict): Buffers shared across models.
    """
    
    def __init__(self,
                 path_list
                ):
        """
        Initialize the ConsistentModelPool with a list of model paths.

        Args:
            path_list (list): List of paths to model artifacts.
        """
        super().__init__()
        
        models = [ load_model(model_path) for model_path in path_list ]
        self.fmodel, self.params, self.buffers = combine_state_for_ensemble(models)
            
    def forward(self, batch):
        """
        Forward pass through the ensemble.

        Args:
            batch (torch.Tensor): Input data batch.

        Returns:
            torch.Tensor: Predictions from the ensemble.
        """
        preds = vmap(self.fmodel, in_dims=(0, 0, None))(self.params, self.buffers, batch)
        return preds.mean(dim=0)
            
class VariableModelPool(nn.Module):
    """
    Ensemble of variable models.

    This class creates an ensemble of variable models from a list of model paths.
    
    Args:
        path_list (list): List of paths to model artifacts.

    Attributes:
        models (list): List of loaded models.
    """
    
    def __init__(self,
                 path_list
                ):
        """
        Initialize the VariableModelPool with a list of model paths.

        Args:
            path_list (list): List of paths to model artifacts.
        """
        super().__init__()
        
        self.models = [ load_model(model_path) for model_path in path_list ]
            
    def forward(self, batch):
        """
        Forward pass through the ensemble.

        Args:
            batch (torch.Tensor): Input data batch.

        Returns:
            torch.Tensor: Predictions from the ensemble.
        """
        return torch.stack([model(batch) for model in self.models]).mean(dim=0)
            
class VepTester(nn.Module):
    """
    Variant Effect Predictor Tester module.

    This class tests the variant effect predictor model on reference and alternate batches of data.

    Args:
        model (nn.Module): A PyTorch model for variant effect prediction.

    Attributes:
        use_cuda (bool): Flag indicating whether CUDA is available.
        model (nn.Module): The model to be tested.
    """
    
    def __init__(self,
                 model
                ):
        """
        Initialize the VepTester with the variant effect predictor model.

        Args:
            model (nn.Module): A PyTorch model for variant effect prediction.
        """
        super().__init__()
        self.use_cuda = torch.cuda.device_count() >= 1
        self.model = torch.nn.DataParallel(model) if torch.cuda.device_count() > 1 else model
        
    def forward(self, ref_batch, alt_batch, average_full_revcomp=False):
        """
        Perform a forward pass through the model with reference and alternate batches.

        Args:
            ref_batch (torch.Tensor): Reference data batch.
            alt_batch (torch.Tensor): Alternate data batch.

        Returns:
            dict: A dictionary containing predictions for reference, alternate, and skew.
        """
        ref_shape, alt_shape = ref_batch.shape, alt_batch.shape
        #(batch_size, n_windows, n_tokens, length)
        #n_windows = (2 if reverse_complments else 1) * (relative_end - relative_start) // step_size
        assert ref_shape == alt_shape
        
        ref_batch = ref_batch.flatten(0,1)
        alt_batch = alt_batch.flatten(0,1)
        
        with torch.cuda.amp.autocast():
            ref_preds = self.model(ref_batch.contiguous())
            alt_preds = self.model(alt_batch.contiguous())
            
            if average_full_revcomp:
                ref_preds = torch.stack([
                    ref_preds,
                    self.model(ref_batch.flip(dims=[1,2]).contiguous())
                ], dim=0).mean(dim=0, keepdim=False)
                alt_preds = torch.stack([
                    alt_preds,
                    self.model(alt_batch.flip(dims=[1,2]).contiguous())
                ], dim=0).mean(dim=0, keepdim=False)

        ref_preds = ref_preds.unflatten(0, ref_shape[0:2])
        ref_preds = ref_preds.unflatten(1, (2, ref_shape[1]//2))
        
        alt_preds = alt_preds.unflatten(0, alt_shape[0:2])
        alt_preds = alt_preds.unflatten(1, (2, alt_shape[1]//2))
            
        skew_preds = alt_preds - ref_preds

        return {'ref': ref_preds, 
                'alt': alt_preds, 
                'skew': skew_preds}
    
class reductions(object):
    """
    A collection of static methods for various tensor reduction operations.

    These methods provide reduction operations on tensors such as mean, sum, max, min,
    absolute maximum, absolute minimum, and gather.

    Attributes:
        None
    """
    
    @staticmethod
    def mean(tensor, dim):
        """
        Compute the mean along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to compute the mean.

        Returns:
            torch.Tensor: The mean values along the specified dimension.
        """
        return tensor.mean(dim=dim)
    
    @staticmethod
    def sum(tensor, dim):
        """
        Compute the sum along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to compute the sum.

        Returns:
            torch.Tensor: The sum values along the specified dimension.
        """
        return tensor.sum(dim=dim)
    
    @staticmethod
    def max(tensor, dim):
        """
        Compute the maximum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to compute the maximum values.

        Returns:
            torch.Tensor: The maximum values along the specified dimension.
        """
        return tensor.amax(dim=dim)
    
    @staticmethod
    def min(tensor, dim):
        """
        Compute the minimum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to compute the minimum values.

        Returns:
            torch.Tensor: The minimum values along the specified dimension.
        """
        return tensor.amin(dim=dim)
    
    @staticmethod
    def abs_max(tensor, dim):
        """
        Compute the absolute maximum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to compute the absolute maximum values.

        Returns:
            torch.Tensor: The absolute maximum values along the specified dimension.
        """
        n_dims = len(tensor.shape)
        get_idx= tensor.abs().argmax(dim=dim)
        ##### Following can be replaced by torch.gather #####
        slicer = []
        for i in range(n_dims):
            if i != dim:
                viewer = [1] * n_dims
                dim_size = tensor.shape[i]
                viewer[i] = dim_size
                viewer.pop(dim)
                slicer.append( torch.arange(dim_size).view(*viewer).expand(*get_idx.shape) )
            else:
                slicer.append( get_idx )
        ##### Above can be replaced by torch.gather #####
            
        return tensor[slicer]
    
    @staticmethod
    def abs_min(tensor, dim):
        """
        Compute the absolute minimum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to compute the absolute minimum values.

        Returns:
            torch.Tensor: The absolute minimum values along the specified dimension.
        """
        n_dims = len(tensor.shape)
        get_idx= tensor.abs().argmin(dim=dim)
        ##### Following can be replaced by torch.gather #####
        slicer = []
        for i in range(n_dims):
            if i != dim:
                viewer = [1] * n_dims
                dim_size = tensor.shape[i]
                viewer[i] = dim_size
                viewer.pop(dim)
                slicer.append( torch.arange(dim_size).view(*viewer).expand(*get_idx.shape) )
            else:
                slicer.append( get_idx )
        ##### Above can be replaced by torch.gather #####
            
        return tensor[slicer]
    
    @staticmethod
    def gather(tensor, dim, index):
        """
        Gather elements along a specified dimension of the input tensor using given indices.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to gather elements.
            index (torch.Tensor): The indices along the specified dimension.

        Returns:
            torch.Tensor: Gathered elements from the input tensor.
        """
        return torch.gather(tensor, dim, index).squeeze(dim=dim)
    
    @staticmethod
    def pick_forward(tensor, dim):
        """
        Use with strand reduction to pick forward strand predictions.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to pick 0th indexed values.

        Returns:
            torch.Tensor: The 0th values along the specified dimension.
        """
        return tensor.index_select(dim=dim, index=torch.tensor(0)).squeeze(dim)
        
    @staticmethod
    def pick_reverse(tensor, dim):
        """
        Use with strand reduction to pick reverse strand predictions.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to pick 1st indexed values.

        Returns:
            torch.Tensor: The 1st values along the specified dimension.
        """
        return tensor.index_select(dim=dim, index=torch.tensor(1)).squeeze(dim)
        
    
class gatherings(object):
    """
    A collection of static methods for gathering indices corresponding to specific tensor operations.

    These methods provide operations to gather indices of maximum, minimum, absolute maximum,
    and absolute minimum values along specified dimensions of a tensor.

    Attributes:
        None
    """
    
    @staticmethod
    def max(tensor, dim):
        """
        Gather indices of maximum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to gather indices of maximum values.

        Returns:
            torch.Tensor: Indices of maximum values along the specified dimension.
        """
        return tensor.max(dim=dim, keepdim=True)[1]
    
    @staticmethod
    def min(tensor, dim):
        """
        Gather indices of minimum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to gather indices of minimum values.

        Returns:
            torch.Tensor: Indices of minimum values along the specified dimension.
        """
        return tensor.min(dim=dim, keepdim=True)[1]
    
    @staticmethod
    def abs_max(tensor, dim):
        """
        Gather indices of absolute maximum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to gather indices of absolute maximum values.

        Returns:
            torch.Tensor: Indices of absolute maximum values along the specified dimension.
        """
        return tensor.abs().max(dim=dim, keepdim=True)[1]
    
    @staticmethod
    def abs_min(tensor, dim):
        """
        Gather indices of absolute minimum values along the specified dimension of the input tensor.

        Args:
            tensor (torch.Tensor): Input tensor.
            dim (int): The dimension along which to gather indices of absolute minimum values.

        Returns:
            torch.Tensor: Indices of absolute minimum values along the specified dimension.
        """
        return tensor.abs().min(dim=dim, keepdim=True)[1]
    
class activity_filtering(object):
    """
    A collection of static methods for filtering and adjusting tensor activity based on thresholds.

    These methods provide operations to filter tensor activity based on maximum, minimum, absolute maximum,
    and absolute minimum values, while applying adjustments using specified thresholds and epsilon values.

    Attributes:
        None
    """
    
    @staticmethod
    def max(preds, indexer, threshold, epsilon):
        """
        Apply maximum value-based filtering to tensor activity and adjust values below the threshold.

        Args:
            preds (dict): Dictionary containing tensor activity data.
            indexer (str): Key to select the tensor activity to be indexed.
            threshold (float): Threshold for filtering values.
            epsilon (float): Value added to the indexing tensor to adjust values below the threshold.

        Returns:
            torch.Tensor: Indexed tensor activity after applying filtering and adjustments.
        """
        ref_filter = preds['ref'].abs() > threshold
        alt_filter = preds['alt'].abs() > threshold
        _filter = ref_filter | alt_filter
        _mask   = 1. - _filter.type(torch.float32)
        indexing_tensor = preds[indexer] + _mask.mul(-1/epsilon)
        return indexing_tensor
    
    @staticmethod
    def min(preds, indexer, threshold, epsilon):
        """
        Apply minimum value-based filtering to tensor activity and adjust values below the threshold.

        Args:
            preds (dict): Dictionary containing tensor activity data.
            indexer (str): Key to select the tensor activity to be indexed.
            threshold (float): Threshold for filtering values.
            epsilon (float): Value added to the indexing tensor to adjust values below the threshold.

        Returns:
            torch.Tensor: Indexed tensor activity after applying filtering and adjustments.
        """
        ref_filter = preds['ref'].abs() > threshold
        alt_filter = preds['alt'].abs() > threshold
        _filter = ref_filter | alt_filter
        _mask   = 1. - _filter.type(torch.float32)
        indexing_tensor = preds[indexer] + _mask.mul( 1/epsilon)
        return indexing_tensor
    
    @staticmethod
    def abs_max(preds, indexer, threshold, epsilon):
        """
        Apply absolute maximum value-based filtering to tensor activity and adjust values above the threshold.

        Args:
            preds (dict): Dictionary containing tensor activity data.
            indexer (str): Key to select the tensor activity to be indexed.
            threshold (float): Threshold for filtering values.
            epsilon (float): Value added to the indexing tensor to adjust values above the threshold.

        Returns:
            torch.Tensor: Indexed tensor activity after applying filtering and adjustments.
        """
        ref_filter = preds['ref'].abs() > threshold
        alt_filter = preds['alt'].abs() > threshold
        _filter = ref_filter | alt_filter
        _mask   = 1. - _filter.type(torch.float32)
        indexing_tensor = preds[indexer] * _filter.add(epsilon)
        return indexing_tensor
    
    @staticmethod
    def abs_min(preds, indexer, threshold, epsilon):
        """
        Apply absolute minimum value-based filtering to tensor activity and adjust values below the threshold.

        Args:
            preds (dict): Dictionary containing tensor activity data.
            indexer (str): Key to select the tensor activity to be indexed.
            threshold (float): Threshold for filtering values.
            epsilon (float): Value added to the indexing tensor to adjust values below the threshold.

        Returns:
            torch.Tensor: Indexed tensor activity after applying filtering and adjustments.
        """
        ref_filter = preds['ref'].abs() > threshold
        alt_filter = preds['alt'].abs() > threshold
        _filter = ref_filter | alt_filter
        _mask   = 1. - _filter.type(torch.float32)
        indexing_tensor = preds[indexer] * _filter.add(_mask.mul(1/epsilon))
        return indexing_tensor

# Add this function to your script (before main function)
def tensor_to_sequence(tensor, alphabet=constants.STANDARD_NT):
    """
    Convert a one-hot encoded tensor to a DNA sequence string.
    
    Args:
        tensor (torch.Tensor): One-hot encoded tensor of shape (channels, length)
        alphabet (list): List of nucleotide characters
    
    Returns:
        str: DNA sequence string
    """
    if tensor.dim() > 2:
        tensor = tensor.squeeze()
    
    # Get the argmax along the channel dimension
    indices = tensor.argmax(dim=0)
    
    # Convert indices to characters
    sequence = ''.join([alphabet[i] for i in indices])
    return sequence

def extract_sequences_from_batch_universal(ref_batch, alt_batch, alphabet=constants.STANDARD_NT):
    """
    Universal sequence extraction that works with both VcfDataset and VcfDataset_Indel formats.
    
    Args:
        ref_batch (torch.Tensor): Reference tensor batch
        alt_batch (torch.Tensor): Alternative tensor batch  
        alphabet (list): Nucleotide alphabet
        
    Returns:
        tuple: (ref_sequences, alt_sequences) - lists of sequences for each batch item
    """
    ref_sequences = []
    alt_sequences = []
    
    # Handle 6D tensors: [batch, 1, strands, windows, channels, length] (VcfDataset_Indel)
    if len(ref_batch.shape) == 6:
        batch_size = ref_batch.shape[0]
        
        for batch_idx in range(batch_size):
            ref_batch_seqs = []
            alt_batch_seqs = []
            
            # Remove singleton dimension: [strands, windows, channels, length]
            ref_item = ref_batch[batch_idx].squeeze(0)  
            alt_item = alt_batch[batch_idx].squeeze(0)
            
            strands, windows = ref_item.shape[0], ref_item.shape[1]
            
            for strand_idx in range(strands):
                strand_name = "forward" if strand_idx == 0 else "reverse"
                
                for window_idx in range(windows):
                    ref_seq = tensor_to_sequence(ref_item[strand_idx, window_idx], alphabet)
                    alt_seq = tensor_to_sequence(alt_item[strand_idx, window_idx], alphabet)
                    
                    ref_batch_seqs.append(f"strand_{strand_name}_window_{window_idx:02d}:{ref_seq}")
                    alt_batch_seqs.append(f"strand_{strand_name}_window_{window_idx:02d}:{alt_seq}")
            
            ref_sequences.append(ref_batch_seqs)
            alt_sequences.append(alt_batch_seqs)
    
    # Handle 5D tensors: [batch, strands, windows, channels, length] (original VcfDataset)
    elif len(ref_batch.shape) == 5:
        batch_size = ref_batch.shape[0]
        
        for batch_idx in range(batch_size):
            ref_batch_seqs = []
            alt_batch_seqs = []
            
            strands, windows = ref_batch.shape[1], ref_batch.shape[2]
            
            for strand_idx in range(strands):
                strand_name = "forward" if strand_idx == 0 else "reverse"
                
                for window_idx in range(windows):
                    ref_seq = tensor_to_sequence(ref_batch[batch_idx, strand_idx, window_idx], alphabet)
                    alt_seq = tensor_to_sequence(alt_batch[batch_idx, strand_idx, window_idx], alphabet)
                    
                    ref_batch_seqs.append(f"strand_{strand_name}_window_{window_idx:02d}:{ref_seq}")
                    alt_batch_seqs.append(f"strand_{strand_name}_window_{window_idx:02d}:{alt_seq}")
            
            ref_sequences.append(ref_batch_seqs)
            alt_sequences.append(alt_batch_seqs)
    
    # Handle 4D tensors: [batch, windows, channels, length] (no strands)
    elif len(ref_batch.shape) == 4:
        batch_size, n_windows = ref_batch.shape[:2]
        
        for batch_idx in range(batch_size):
            ref_batch_seqs = []
            alt_batch_seqs = []
            
            for window_idx in range(n_windows):
                ref_seq = tensor_to_sequence(ref_batch[batch_idx, window_idx], alphabet)
                alt_seq = tensor_to_sequence(alt_batch[batch_idx, window_idx], alphabet)
                
                ref_batch_seqs.append(f"window_{window_idx:02d}:{ref_seq}")
                alt_batch_seqs.append(f"window_{window_idx:02d}:{alt_seq}")
            
            ref_sequences.append(ref_batch_seqs)
            alt_sequences.append(alt_batch_seqs)
    
    else:
        print(f"Warning: Unsupported tensor shape {ref_batch.shape} for sequence extraction")
        return [], []
    
    return ref_sequences, alt_sequences

def main(args):
    """
    Run the main processing pipeline for the given command-line arguments.

    This function executes the main processing pipeline, including loading models, processing input data from FASTA and VCF files,
    and generating predictions. The resulting predictions are then post-processed based on specified reduction and filtering methods.

    Args:
        args (argparse.Namespace): Parsed command-line arguments.

    Returns:
        None
    """
    USE_CUDA = torch.cuda.device_count() >= 1
    print(sys.argv)
    ##################
    ## Import Model ##
    ##################
    if len(args.artifact_path) == 1:
        my_model = load_model(args.artifact_path[0])
    elif len(args.artifact_path) > 1 and args.use_vmap:
        my_model = ConsistentModelPool(args.artifact_path)
    elif len(args.artifact_path) > 1:
        my_model = VariableModelPool(args.artifact_path)
    
    #########################
    ## Setup FASTA and VCF ##
    #########################
    fasta_data = boda.data.Fasta(args.fasta_file)
    
    vcf = boda.data.VCF(
        args.vcf_file, chr_prefix=args.vcf_contig_prefix, 
        max_allele_size=20, max_indel_size=20,
    )
    
    WINDOW_SIZE = args.window_size
    RELATIVE_START = args.relative_start
    RELATIVE_END = args.relative_end
    
    vcf_data = boda.data.VcfDataset(
        vcf.vcf, fasta_data.fasta, WINDOW_SIZE, 
        RELATIVE_START, RELATIVE_END, step_size=args.step_size, 
        left_flank='', right_flank='', use_contigs=args.use_contigs,
    )

    ########################
    ## determine chunking ##
    ########################
    if args.n_jobs > 1:
        extra_tasks = len(vcf_data) % args.n_jobs
        if extra_tasks > 0:
            subset_size = ((len(vcf_data) // args.n_jobs) + 1)
        else:
            subset_size = len(vcf_data) // args.n_jobs
        start_idx = subset_size*args.job_id
        stop_idx  = min(len(vcf_data), subset_size*(args.job_id+1))
        vcf_subset = torch.utils.data.Subset(vcf_data, np.arange(start_idx, stop_idx))
        vcf_table  = vcf_data.vcf.iloc[start_idx:stop_idx]
    else:
        vcf_subset = vcf_data
        vcf_table  = vcf_data.vcf

    print(f"Dataset length: {len(vcf_subset)}, VCF length: {vcf_table.shape}")
    assert len(vcf_subset) == vcf_table.shape[0], "size mismatch"
    ###########################
    ## prepare data pipeline ##
    ###########################
    vcf_loader = torch.utils.data.DataLoader( vcf_subset, batch_size=args.batch_size*max(1,torch.cuda.device_count()) )
    
    left_flank = boda.common.utils.dna2tensor( 
        args.left_flank 
    ).unsqueeze(0).unsqueeze(0)

    right_flank= boda.common.utils.dna2tensor( 
        args.right_flank
    ).unsqueeze(0).unsqueeze(0)
    
    flank_builder = utils.FlankBuilder(
        left_flank=left_flank,
        right_flank=right_flank,
    )
    if USE_CUDA:
        flank_builder.cuda()
    
    vep_tester = VepTester(my_model)
    
    ref_preds = []
    alt_preds = []
    skew_preds= []

    ######################
    ## run through data ##
    ######################
    with torch.no_grad():
        for i, batch in enumerate(tqdm.tqdm(vcf_loader)):
            ref_allele, alt_allele = batch['ref'], batch['alt']
            
            if USE_CUDA:
                ref_allele = ref_allele.cuda()
                alt_allele = alt_allele.cuda()

            ref_allele = flank_builder(ref_allele).contiguous()
            alt_allele = flank_builder(alt_allele).contiguous()
            if args.output_sequences:
                # Store sequences before predictions
                ref_seqs, alt_seqs = extract_sequences_from_batch_universal(
                    ref_allele.cpu(), alt_allele.cpu(), constants.STANDARD_NT
                )
                
                # Add to cumulative storage
                if 'ref_sequences_all' not in locals():
                    ref_sequences_all = []
                    alt_sequences_all = []
                
                ref_sequences_all.extend(ref_seqs)
                alt_sequences_all.extend(alt_seqs)

            all_preds = vep_tester(
                ref_allele, alt_allele, 
                average_full_revcomp=args.average_full_revcomp
            )
            
            if not args.raw_predictions:
                
                if args.strand_reduction == 'gather':
                    strand_index = getattr(gatherings, args.strand_gathering) \
                                   (all_preds[args.gather_source], dim=1)
                    strand_kwargs= {'index': strand_index}
                else:
                    strand_kwargs= {}
                
                proc_preds = {}
                proc_preds['ref'] = getattr(reductions, args.strand_reduction) \
                                    (all_preds['ref'], dim=1, **strand_kwargs)
                proc_preds['alt'] = getattr(reductions, args.strand_reduction) \
                                    (all_preds['alt'], dim=1, **strand_kwargs)
                proc_preds['skew']= getattr(reductions, args.strand_reduction) \
                                    (all_preds['skew'], dim=1, **strand_kwargs)
                
                if args.window_reduction == 'gather':
                    if args.activity_filter is not None:
                        try:
                            indexing_tensor = getattr(activity_filtering, args.window_gathering) \
                                              (proc_preds, args.gather_source, args.activity_filter, args.epsilon)
                        except AttributeError as e:
                            errmsg = "activity_filter not implmented for selected "
                            errmsg+= f"window_gathering: {args.window_gathering}"
                            raise Exception(errmsg) from e
                    else:
                        indexing_tensor = proc_preds[args.gather_source]
                        
                    window_index = getattr(gatherings, args.window_gathering) \
                                   (indexing_tensor, dim=1)
                    window_kwargs = {'index': window_index}
                else:
                    window_kwargs = {}
                    
                proc_preds['ref'] = getattr(reductions, args.window_reduction) \
                                    (proc_preds['ref'], dim=1, **window_kwargs)
                proc_preds['alt'] = getattr(reductions, args.window_reduction) \
                                    (proc_preds['alt'], dim=1, **window_kwargs)
                proc_preds['skew']= getattr(reductions, args.window_reduction) \
                                    (proc_preds['skew'], dim=1, **window_kwargs)
                
                ref_preds.append(proc_preds['ref'].cpu())
                alt_preds.append(proc_preds['alt'].cpu())
                skew_preds.append(proc_preds['skew'].cpu())
            
            else:
                ref_preds.append(all_preds['ref'].cpu())
                alt_preds.append(all_preds['alt'].cpu())

    ##################
    ## dump outputs ##
    ##################
    if not args.raw_predictions:
        ref_preds = torch.cat(ref_preds, dim=0)
        alt_preds = torch.cat(alt_preds, dim=0)
        skew_preds= torch.cat(skew_preds, dim=0)
        
        print(f"ref dims: {ref_preds.shape}, alt dims: {alt_preds.shape}, skew dims: {skew_preds.shape}")
        
        full_results = combine_ref_alt_skew_tensors(ref_preds, alt_preds, skew_preds, args.feature_ids)
        
        print(f"results table shape: {full_results.shape}")
        vcf_table = pd.concat([vcf_table.reset_index(drop=True), pd.DataFrame(full_results, columns=['INFO'])], axis=1)
        vcf_table.to_csv(args.output, sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    else:
        ref_preds = torch.cat(ref_preds, dim=0)
        alt_preds = torch.cat(alt_preds, dim=0)
        torch.save({'ref': ref_preds, 'alt': alt_preds, 'vcf': vcf_table}, args.output)
    ##################
    ## output seqs ###
    ##################
    if args.output_sequences:
        # Determine output file path
        if args.sequence_output_file:
            seq_output_path = args.sequence_output_file
        else:
            base_path = args.output.rsplit('.', 1)[0]  # Remove extension
            seq_output_path = f"{base_path}_sequences.tsv"
        
        # Create sequence output DataFrame
        sequence_data = []
        
        for idx, (vcf_row, ref_seq_list, alt_seq_list) in enumerate(zip(vcf_table.iterrows(), ref_sequences_all, alt_sequences_all)):
            _, vcf_record = vcf_row
            
            for ref_seq_entry, alt_seq_entry in zip(ref_seq_list, alt_seq_list):
                sequence_data.append({
                    'variant_idx': idx,
                    'chrom': vcf_record['chrom'],
                    'pos': vcf_record['pos'],
                    'ref_allele': vcf_record['ref'],
                    'alt_allele': vcf_record['alt'],
                    'ref_sequence': ref_seq_entry,
                    'alt_sequence': alt_seq_entry
                })
        
        seq_df = pd.DataFrame(sequence_data)
        seq_df.to_csv(seq_output_path, sep='\t', index=False)
        print(f"Sequences saved to: {seq_output_path}")
        print(f"Total sequence entries: {len(sequence_data)}")

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="Contribution scoring tool.")
    # Input info
    parser.add_argument('--artifact_path', type=str, nargs='*', required=True, help='Pre-trained model artifacts. Supply multiple to ensemble.')
    parser.add_argument('--use_vmap', type=utils.str2bool, default=False, help='If ensemble members have consistent architecture can speed up with functorch.vmap.')
    parser.add_argument('--vcf_file', type=str, required=True, help='Variants to test in VCF format.')
    parser.add_argument('--fasta_file', type=str, required=True, help='FASTA reference file.')
    # Output info
    parser.add_argument('--output', type=str, required=True, help='Output path. Simple VCF if not RAW_PREDICTIONS else PT pickle.')
    parser.add_argument('--feature_ids', type=str, nargs='*', help='Custom feature IDs for outputs in INFO column.')
    parser.add_argument('--raw_predictions', type=utils.str2bool, default=False, help='Dump raw ref/alt predictions as tensors. Output will be a PT pickle.')
    # Data preprocessing
    parser.add_argument('--window_size', type=int, default=200, help='Window size to be extracted from the genome.')
    parser.add_argument('--left_flank', type=str, default=boda.common.constants.MPRA_UPSTREAM[-200:], help='Upstream padding.')
    parser.add_argument('--right_flank', type=str, default=boda.common.constants.MPRA_DOWNSTREAM[:200], help='Downstream padding.')
    parser.add_argument('--vcf_contig_prefix', type=str, default='', help='Prefix to append VCF contig IDs to match FASTA contig IDs.')
    # VEP testing conditions
    parser.add_argument('--relative_start', type=int, default=0, help='Leftmost position where variant is tested, 0-based inclusive.')
    parser.add_argument('--relative_end', type=int, default=200, help='Rightmost position where variant is tested, 1-based exclusive.')
    parser.add_argument('--step_size', type=int, default=1, help='Step size between positions where variants are tested. Validate tested variant indices using list(range(start, end))[::-step].')
    parser.add_argument('--strand_reduction', type=str, choices=('mean', 'sum', 'max', 'min', 'abs_max', 'abs_min', 'gather', 'pick_forward', 'pick_reverse'), default='mean', help='Specify reduction over strands. Options: mean, sum, max, min, abs_max, abs_min, gather, pick_forward, pick_reverse')
    parser.add_argument('--window_reduction', type=str, choices=('mean', 'sum', 'max', 'min', 'abs_max', 'abs_min', 'gather'), default='mean', help='Specify reduction over testing windows. Options: mean, sum, max, min, abs_max, abs_min, gather.')
    # Conditional VEP testing args
    parser.add_argument('--strand_gathering', type=str, choices=('max', 'min', 'abs_max', 'abs_min'), help='If using a gather reduction over strands, specify index sorting function.')
    parser.add_argument('--window_gathering', type=str, choices=('max', 'min', 'abs_max', 'abs_min'), help='If using a gather reduction of testing windows, specify index sorting function.')
    parser.add_argument('--activity_filter', type=float, help='Minimum activity theshold to consider variant (checks both ref and alt).')
    parser.add_argument('--epsilon', type=float, default=1e-4, help='Small factor restore default behavior for variants where no orientation passes the activity_filter.')
    parser.add_argument('--gather_source', type=str, choices=('ref', 'alt', 'skew'), help='Variant prediction type to use for gathering. Choose from (ref, alt, skew)')
    parser.add_argument('--average_full_revcomp', type=utils.str2bool, default=False, help='Compute and average over full reverse complmenents before applying strand reduction.')
    # Throughput management
    parser.add_argument('--use_contigs', type=str, nargs='*', default=[], help='Optional list of contigs (space seperated) to restrict testing to.')    
    parser.add_argument('--batch_size', type=int, default=10, help='Batch size during sequence extraction from FASTA.')
    parser.add_argument('--job_id', type=int, default=0, help='Job partition index for distributed computing.')
    parser.add_argument('--n_jobs', type=int, default=1, help='Total number of job partitions.')
    # add sequence output
    parser.add_argument('--output_sequences', type=utils.str2bool, default=False, 
                   help='Output all generated sequences along with predictions.')
    parser.add_argument('--sequence_output_file', type=str, 
                    help='File path for sequence output. If not specified, uses output path with _sequences.tsv suffix.')
    
    args = parser.parse_args()
    main(args)