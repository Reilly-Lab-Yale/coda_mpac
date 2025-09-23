"""
Indel support module for boda2 VCF processing
"""

try:
    from .vcf_dataset_indel import VcfDataset_Indel
    __all__ = ['VcfDataset_Indel']
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import VcfDataset_Indel: {e}")
    __all__ = []
