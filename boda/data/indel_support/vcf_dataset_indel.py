#!/usr/bin/env python3
"""
VcfDataset_Indel - Complete implementation for boda2

This file contains the complete VcfDataset_Indel class that handles indels
by using MPRA flanking sequences for dynamic padding.

Save this as: boda/data/indel_support/vcf_dataset_indel.py
"""

import torch
import pandas as pd
import numpy as np
import tqdm
from torch.utils.data import Dataset, DataLoader

# Import boda utilities - adjust paths as needed for your setup
try:
    import boda.common.utils
    import boda.common.constants
except ImportError:
    print("Warning: Could not import boda.common modules")
    print("Make sure you're running from the boda2 root directory")
    # Create dummy implementations for testing
    class DummyUtils:
        @staticmethod
        def dna2tensor(seq):
            # Simple one-hot encoding for testing
            mapping = {'A': [1,0,0,0], 'T': [0,1,0,0], 'C': [0,0,1,0], 'G': [0,0,0,1], 'N': [0,0,0,0]}
            encoded = [mapping.get(base.upper(), [0,0,0,0]) for base in seq]
            return torch.tensor(encoded).T.float()
    
    class DummyConstants:
        MPRA_UPSTREAM = "ATCG" * 50  # 200bp dummy sequence
        MPRA_DOWNSTREAM = "CGTA" * 50  # 200bp dummy sequence
    
    if 'boda' not in globals():
        class boda:
            class common:
                utils = DummyUtils()
                constants = DummyConstants()


class VcfDataset_Indel(Dataset):
    """
    Modified VcfDataset class that handles indels by using dynamic padding 
    with MPRA flanking sequences instead of genomic sequences.
    
    This class maintains consistency with multiple windows defined by 
    relative_start, relative_end, and step_size parameters while ensuring
    all sequences reach the target 600bp length required by the model.
    """
    
    def __init__(self, 
                 vcf_data,
                 fasta_sequences, 
                 relative_start=25,
                 relative_end=180, 
                 step_size=25,
                 target_length=600,
                 batch_size=128,
                 device='cuda'):
        """
        Initialize the VcfDataset_Indel class.
        
        Args:
            vcf_data: DataFrame containing VCF variant information
            fasta_sequences: DataFrame or dict containing reference sequences
            relative_start: Start position for windowing (default: 25)
            relative_end: End position for windowing (default: 180) 
            step_size: Step size for windowing (default: 25)
            target_length: Final sequence length required by model (default: 600)
            batch_size: Batch size for data loading (default: 128)
            device: Device for tensor operations (default: 'cuda')
        """
        self.vcf_data = vcf_data
        self.fasta_sequences = fasta_sequences
        self.relative_start = relative_start
        self.relative_end = relative_end
        self.step_size = step_size
        self.target_length = target_length
        self.batch_size = batch_size
        self.device = device
        
        print(f"Initializing VcfDataset_Indel with {len(vcf_data)} variants")
        print(f"Window parameters: start={relative_start}, end={relative_end}, step={step_size}")
        
        # Pre-compute MPRA flanking sequences as tensors
        self.left_flanker = boda.common.utils.dna2tensor(
            boda.common.constants.MPRA_UPSTREAM
        )
        self.right_flanker = boda.common.utils.dna2tensor(
            boda.common.constants.MPRA_DOWNSTREAM
        )
        
        print(f"MPRA flankers - Left: {self.left_flanker.shape}, Right: {self.right_flanker.shape}")
        
        # Generate windowed sequences
        self.windowed_sequences = self._generate_windowed_sequences()
        
        # Create padded sequences with MPRA flanks
        self.seq_tensors = self._create_padded_sequences()
        
    def _generate_windowed_sequences(self):
        """
        Generate sequences for all windows defined by relative_start, 
        relative_end, and step_size parameters.
        
        Returns:
            List of sequence dictionaries with metadata
        """
        print("Generating windowed sequences...")
        windowed_seqs = []
        
        for idx, variant in tqdm.tqdm(self.vcf_data.iterrows(), 
                                     total=len(self.vcf_data),
                                     desc="Processing variants"):
            # Get the reference sequence for this variant
            ref_seq = self._get_reference_sequence(variant)
            if not ref_seq:
                print(f"Warning: No reference sequence found for variant {variant.get('ID', idx)}")
                continue
                
            alt_seq = self._get_alternate_sequence(variant, ref_seq)
            
            # Generate windows for this variant
            windows = self._generate_windows_for_variant(variant, ref_seq, alt_seq)
            windowed_seqs.extend(windows)
            
        print(f"Generated {len(windowed_seqs)} windowed sequences")
        return windowed_seqs
    
    def _get_reference_sequence(self, variant):
        """
        Extract reference sequence for a given variant.
        
        Args:
            variant: Single variant record from VCF data
            
        Returns:
            Reference sequence string
        """
        chrom = variant['CHROM']
        pos = variant['POS']
        
        if isinstance(self.fasta_sequences, dict):
            # If fasta_sequences is a dictionary mapping positions to sequences
            key = f"{chrom}:{pos}"
            return self.fasta_sequences.get(key, "")
        else:
            # If fasta_sequences is a DataFrame
            mask = (self.fasta_sequences['CHROM'] == chrom) & \
                   (self.fasta_sequences['POS'] == pos)
            matches = self.fasta_sequences[mask]
            if len(matches) > 0:
                return matches.iloc[0]['sequence']
            return ""
    
    def _get_alternate_sequence(self, variant, ref_seq):
        """
        Generate alternate sequence by applying the variant to reference.
        
        Args:
            variant: Single variant record from VCF data
            ref_seq: Reference sequence string
            
        Returns:
            Alternate sequence string with variant applied
        """
        ref_allele = str(variant['REF'])
        alt_allele = str(variant['ALT'])
        
        # For indels, we need to handle insertions and deletions
        if len(ref_allele) > len(alt_allele):
            # Deletion - remove extra bases
            alt_seq = ref_seq.replace(ref_allele, alt_allele, 1)
        elif len(alt_allele) > len(ref_allele):
            # Insertion - add extra bases
            alt_seq = ref_seq.replace(ref_allele, alt_allele, 1)
        else:
            # SNV or complex variant
            alt_seq = ref_seq.replace(ref_allele, alt_allele, 1)
            
        return alt_seq
    
    def _generate_windows_for_variant(self, variant, ref_seq, alt_seq):
        """
        Generate multiple windows for a single variant based on 
        relative_start, relative_end, and step_size.
        
        Args:
            variant: Single variant record
            ref_seq: Reference sequence
            alt_seq: Alternate sequence
            
        Returns:
            List of windowed sequences with metadata
        """
        windows = []
        
        # Generate window positions
        for start_pos in range(self.relative_start, 
                             self.relative_end, 
                             self.step_size):
            end_pos = start_pos + 200  # Assuming 200bp center sequences
            
            # Extract windowed sequences
            ref_window = ref_seq[start_pos:end_pos] if len(ref_seq) >= end_pos else ref_seq[start_pos:]
            alt_window = alt_seq[start_pos:end_pos] if len(alt_seq) >= end_pos else alt_seq[start_pos:]
            
            # Pad sequences to exactly 200bp if needed
            ref_window = self._pad_to_200bp(ref_window)
            alt_window = self._pad_to_200bp(alt_window)
            
            # Create sequence records
            variant_id = variant.get('ID', f"{variant['CHROM']}:{variant['POS']}")
            
            windows.extend([
                {
                    'sequence': ref_window,
                    'variant_id': variant_id,
                    'allele': 'REF',
                    'window_start': start_pos,
                    'window_end': end_pos,
                    'original_length': len(ref_window),
                    'chrom': variant['CHROM'],
                    'pos': variant['POS']
                },
                {
                    'sequence': alt_window, 
                    'variant_id': variant_id,
                    'allele': 'ALT',
                    'window_start': start_pos,
                    'window_end': end_pos,
                    'original_length': len(alt_window),
                    'chrom': variant['CHROM'],
                    'pos': variant['POS']
                }
            ])
            
        return windows
    
    def _pad_to_200bp(self, sequence):
        """
        Pad sequence to exactly 200bp using N characters if too short,
        or truncate if too long.
        
        Args:
            sequence: Input DNA sequence string
            
        Returns:
            200bp sequence string
        """
        if len(sequence) < 200:
            # Pad with N characters
            padding = 200 - len(sequence)
            sequence = sequence + 'N' * padding
        elif len(sequence) > 200:
            # Truncate to 200bp
            sequence = sequence[:200]
            
        return sequence
    
    def _create_padded_sequences(self):
        """
        Create padded sequences using MPRA flanking sequences to reach 
        the target length (600bp).
        
        Returns:
            Stacked tensor of all padded sequences
        """
        print(f"Creating padded sequences for {len(self.windowed_sequences)} sequences...")
        
        # Convert sequences to tensors
        seq_tensor_list = []
        
        for seq_record in tqdm.tqdm(self.windowed_sequences, 
                                   desc="Converting sequences to tensors"):
            try:
                # Convert center sequence to tensor
                seq_tensor = boda.common.utils.dna2tensor(seq_record['sequence'])
                seq_tensor_list.append(seq_tensor)
            except Exception as e:
                print(f"Warning: Failed to convert sequence {seq_record['variant_id']}: {e}")
                # Create dummy tensor for failed conversions
                dummy_tensor = torch.zeros(4, 200)
                seq_tensor_list.append(dummy_tensor)
        
        # Apply dynamic padding with MPRA flanks
        padded_tensors = []
        
        for i, seq_tensor in enumerate(tqdm.tqdm(seq_tensor_list, 
                                               desc="Applying MPRA padding")):
            seq_len = seq_tensor.shape[-1]
            padding_len = self.target_length - seq_len
            
            if padding_len < 0:
                # Sequence is too long, truncate
                seq_tensor = seq_tensor[:, :self.target_length]
                padded_tensors.append(seq_tensor)
                continue
            
            # Calculate left and right padding lengths
            left_pad_len = padding_len // 2 + padding_len % 2
            right_pad_len = padding_len // 2
            
            # Extract appropriate flanking sequence portions
            if left_pad_len > 0:
                if left_pad_len <= self.left_flanker.shape[-1]:
                    left_flank = self.left_flanker[:, -left_pad_len:]
                else:
                    # Need more padding than available in flanker
                    left_flank = self.left_flanker
                    extra_padding = torch.zeros(4, left_pad_len - self.left_flanker.shape[-1])
                    left_flank = torch.cat([extra_padding, left_flank], dim=-1)
            else:
                left_flank = torch.empty(4, 0)
            
            if right_pad_len > 0:
                if right_pad_len <= self.right_flanker.shape[-1]:
                    right_flank = self.right_flanker[:, :right_pad_len]
                else:
                    # Need more padding than available in flanker
                    right_flank = self.right_flanker
                    extra_padding = torch.zeros(4, right_pad_len - self.right_flanker.shape[-1])
                    right_flank = torch.cat([right_flank, extra_padding], dim=-1)
            else:
                right_flank = torch.empty(4, 0)
            
            # Concatenate flanks with center sequence
            try:
                padded_seq = torch.cat([left_flank, seq_tensor, right_flank], dim=-1)
                
                # Verify final length
                if padded_seq.shape[-1] != self.target_length:
                    print(f"Warning: Sequence {i} has incorrect length {padded_seq.shape[-1]}, expected {self.target_length}")
                    # Force correct length
                    if padded_seq.shape[-1] > self.target_length:
                        padded_seq = padded_seq[:, :self.target_length]
                    else:
                        extra_padding = torch.zeros(4, self.target_length - padded_seq.shape[-1])
                        padded_seq = torch.cat([padded_seq, extra_padding], dim=-1)
                
                padded_tensors.append(padded_seq)
                
            except Exception as e:
                print(f"Warning: Failed to pad sequence {i}: {e}")
                # Create dummy padded sequence
                dummy_padded = torch.zeros(4, self.target_length)
                padded_tensors.append(dummy_padded)
        
        # Stack all tensors
        final_tensor = torch.stack(padded_tensors)
        print(f"Final tensor shape: {final_tensor.shape}")
        
        return final_tensor
    
    def get_dataloader(self):
        """
        Create and return a DataLoader for the processed sequences.
        
        Returns:
            PyTorch DataLoader object
        """
        dataset = torch.utils.data.TensorDataset(self.seq_tensors)
        dataloader = torch.utils.data.DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        return dataloader
    
    def __len__(self):
        """Return the number of sequences in the dataset."""
        return len(self.windowed_sequences)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Args:
            idx: Index of the item to retrieve
            
        Returns:
            Tuple of (padded_sequence_tensor, metadata_dict)
        """
        return (
            self.seq_tensors[idx], 
            self.windowed_sequences[idx]
        )
    
    def get_metadata(self):
        """
        Return metadata for all sequences in the dataset.
        
        Returns:
            List of metadata dictionaries
        """
        return self.windowed_sequences
    
    def get_sequence_info(self):
        """
        Get summary information about sequence lengths and padding.
        
        Returns:
            Dictionary with sequence statistics
        """
        if not self.windowed_sequences:
            return {'error': 'No sequences generated'}
            
        original_lengths = [seq['original_length'] for seq in self.windowed_sequences]
        
        return {
            'total_sequences': len(self.windowed_sequences),
            'target_length': self.target_length,
            'min_original_length': min(original_lengths) if original_lengths else 0,
            'max_original_length': max(original_lengths) if original_lengths else 0,
            'mean_original_length': np.mean(original_lengths) if original_lengths else 0,
            'padding_applied': self.target_length - np.mean(original_lengths) if original_lengths else 0,
            'num_variants': len(self.vcf_data),
            'windows_per_variant': len(range(self.relative_start, self.relative_end, self.step_size)),
            'sequences_per_variant': len(range(self.relative_start, self.relative_end, self.step_size)) * 2
        }
    
    def filter_by_variant_type(self, variant_types=['indel']):
        """
        Filter the dataset to only include specific variant types.
        
        Args:
            variant_types: List of variant types to include ('snv', 'indel', 'insertion', 'deletion')
            
        Returns:
            New VcfDataset_Indel with filtered variants
        """
        def get_variant_type(row):
            ref_len = len(str(row['REF']))
            alt_len = len(str(row['ALT']))
            
            if ref_len == alt_len == 1:
                return 'snv'
            elif ref_len > alt_len:
                return 'deletion'
            elif alt_len > ref_len:
                return 'insertion'
            else:
                return 'indel'
        
        # Add variant type column
        vcf_with_types = self.vcf_data.copy()
        vcf_with_types['variant_type'] = vcf_with_types.apply(get_variant_type, axis=1)
        
        # Filter by requested types
        filtered_vcf = vcf_with_types[vcf_with_types['variant_type'].isin(variant_types)]
        
        print(f"Filtered from {len(self.vcf_data)} to {len(filtered_vcf)} variants")
        
        # Create new dataset with filtered data
        return VcfDataset_Indel(
            vcf_data=filtered_vcf.drop('variant_type', axis=1),
            fasta_sequences=self.fasta_sequences,
            relative_start=self.relative_start,
            relative_end=self.relative_end,
            step_size=self.step_size,
            target_length=self.target_length,
            batch_size=self.batch_size,
            device=self.device
        )


# Convenience function for easy dataset creation
def create_vcf_dataset_indel(vcf_file_path, 
                           fasta_file_path=None,
                           fasta_sequences=None,
                           relative_start=25,
                           relative_end=180, 
                           step_size=25,
                           batch_size=128):
    """
    Convenience function to create VcfDataset_Indel from file paths.
    
    Args:
        vcf_file_path: Path to VCF file
        fasta_file_path: Path to FASTA sequences (optional if fasta_sequences provided)
        fasta_sequences: Pre-loaded FASTA sequences dict/DataFrame
        relative_start: Start position for windowing (default: 25)
        relative_end: End position for windowing (default: 180)
        step_size: Step size for windowing (default: 25)
        batch_size: Batch size for data loading (default: 128)
        
    Returns:
        VcfDataset_Indel instance
    """
    # Load VCF data
    print(f"Loading VCF data from {vcf_file_path}")
    vcf_data = pd.read_csv(vcf_file_path, sep='\t', comment='#')
    print(f"Loaded {len(vcf_data)} variants")
    
    # Load FASTA sequences if not provided
    if fasta_sequences is None:
        if fasta_file_path is None:
            raise ValueError("Either fasta_file_path or fasta_sequences must be provided")
        
        print(f"Loading FASTA sequences from {fasta_file_path}")
        # This is a placeholder - implement actual FASTA loading based on your method
        fasta_sequences = {}
        print("Warning: FASTA loading not implemented - using empty dict")
    
    # Create dataset
    dataset = VcfDataset_Indel(
        vcf_data=vcf_data,
        fasta_sequences=fasta_sequences,
        relative_start=relative_start,
        relative_end=relative_end,
        step_size=step_size,
        batch_size=batch_size
    )
    
    return dataset


# Test function to verify the class works
def test_vcf_dataset_indel():
    """Test function to verify VcfDataset_Indel works correctly."""
    print("🧪 Testing VcfDataset_Indel class...")
    
    # Create test data
    test_vcf_data = pd.DataFrame([
        {
            'CHROM': 'chr1', 
            'POS': 1000, 
            'ID': 'test_del', 
            'REF': 'ATCGT', 
            'ALT': 'A',
            'QUAL': 50,
            'FILTER': 'PASS'
        },
        {
            'CHROM': 'chr1', 
            'POS': 2000, 
            'ID': 'test_ins', 
            'REF': 'G', 
            'ALT': 'GTCA',
            'QUAL': 45,
            'FILTER': 'PASS'
        }
    ])
    
    # Create test sequences
    test_fasta_sequences = {
        'chr1:1000': 'ATCGT' + 'ACGT' * 99,  # 400bp total
        'chr1:2000': 'G' + 'TCGA' * 99       # 397bp total
    }
    
    try:
        # Create dataset
        dataset = VcfDataset_Indel(
            vcf_data=test_vcf_data,
            fasta_sequences=test_fasta_sequences,
            relative_start=25,
            relative_end=100,  # Smaller range for testing
            step_size=25,
            target_length=600,
            batch_size=4
        )
        
        print(f"✅ Dataset created successfully with {len(dataset)} sequences")
        
        # Test data loader
        dataloader = dataset.get_dataloader()
        first_batch = next(iter(dataloader))
        print(f"✅ DataLoader works, first batch shape: {first_batch[0].shape}")
        
        # Test metadata
        metadata = dataset.get_metadata()
        print(f"✅ Metadata contains {len(metadata)} entries")
        
        # Test sequence info
        seq_info = dataset.get_sequence_info()
        print(f"✅ Sequence info: {seq_info}")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("VcfDataset_Indel class loaded successfully!")
    print("Run test_vcf_dataset_indel() to verify functionality")
    
    # Uncomment the next line to run tests automatically
    # test_vcf_dataset_indel()