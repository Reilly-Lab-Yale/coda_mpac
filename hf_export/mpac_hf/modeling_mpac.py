"""
Standalone definition of the MPAC model architecture (`BassetBranched`).

This module is deliberately self-contained: it depends only on `torch` (plus
`huggingface_hub` for the `from_pretrained` mixin). It does not import
`boda`, `lightning`, or any of the training-time machinery. Layer classes and
the forward pass are transcribed from `boda/model/basset.py` and
`boda/model/custom_layers.py` so that state dicts load with identical keys and
produce bitwise-identical outputs.

MIT License

Copyright (c) 2025 Sagar Gosai, Rodrigo Castro

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import math
import os
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.func import functional_call, stack_module_state, vmap

try:
    from huggingface_hub import PyTorchModelHubMixin
except ImportError:  # keeps the file usable as a plain torch module offline
    class PyTorchModelHubMixin:
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__()


__all__ = [
    'STANDARD_NT', 'MPRA_UPSTREAM', 'MPRA_DOWNSTREAM', 'CELL_TYPES',
    'dna2tensor', 'MPACModel', 'MPACEnsemble', 'fold_for_chromosome',
]

# -----------------------------------------------------------------------------
# Assay constants
# -----------------------------------------------------------------------------

STANDARD_NT = ['A', 'C', 'G', 'T']

# Vector context flanking the 200 bp variable region in the MPRA library. The
# model is trained on the full 600 bp construct, so predictions on a bare 200mer
# are only meaningful once these are attached (see `MPACModel.add_flanks`).
MPRA_UPSTREAM = 'ACGAAAATGTTGGATGCTCATACTCGTCCTTTTTCAATATTATTGAAGCATTTATCAGGGTTACTAGTACGTCTCTCAAGGATAAGTAAGTAATATTAAGGTACGGGAGGTATTGGACAGGCCGCAATAAAATATCTTTATTTTCATTACATCTGTGTGTTGGTTTTTTGTGTGAATCGATAGTACTAACATACGCTCTCCATCAAAACAAAACGAAACAAAACAAACTAGCAAAATAGGCTGTCCCCAGTGCAAGTGCAGGTGCCAGAACATTTCTCTGGCCTAACTGGCCGCTTGACG'
MPRA_DOWNSTREAM = 'CACTGCGGCTCCTGCGATCTAACTGGCCGGTACCTGAGCTCGCTAGCCTCGAGGATATCAAGATCTGGCCTCGGCGGCCAAGCTTAGACACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGTTGGTAAAGCCACCATGGTGAGCAAGGGCGAGGAGCTGTTCACCGGGGTGGTGCCCATCCTGGTCGAGCTGGACGGCGACGTAAACGGCCACAAGTTCAGCGTGTCCGGCGAGGGCGAGGGCGATGCCACCTACGGCAAGCTGACCCTGAAGTTCATCT'

CELL_TYPES = ['K562', 'HepG2', 'SKNSH']


def dna2tensor(sequence_str, vocab_list=STANDARD_NT):
    """One-hot encode a DNA string as a (4, len) float tensor."""
    seq_tensor = torch.zeros((len(vocab_list), len(sequence_str)))
    for i, letter in enumerate(sequence_str):
        seq_tensor[vocab_list.index(letter), i] = 1.
    return seq_tensor


def get_padding(kernel_size):
    left = (kernel_size - 1) // 2
    right = kernel_size - 1 - left
    return [max(0, x) for x in [left, right]]


# -----------------------------------------------------------------------------
# Layers
# -----------------------------------------------------------------------------

class Conv1dNorm(nn.Module):
    """Conv1d with optional weight norm and batch norm."""

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1,
                 bias=True, batch_norm=True, weight_norm=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              stride, padding, dilation, groups, bias)
        if weight_norm:
            self.conv = nn.utils.weight_norm(self.conv)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_channels, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=True)

    def forward(self, input):
        try:
            return self.bn_layer(self.conv(input))
        except AttributeError:
            return self.conv(input)


class LinearNorm(nn.Module):
    """Linear with optional weight norm and batch norm."""

    def __init__(self, in_features, out_features, bias=True,
                 batch_norm=True, weight_norm=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)
        if weight_norm:
            self.linear = nn.utils.weight_norm(self.linear)
        if batch_norm:
            self.bn_layer = nn.BatchNorm1d(out_features, eps=1e-05, momentum=0.1,
                                           affine=True, track_running_stats=True)

    def forward(self, input):
        try:
            return self.bn_layer(self.linear(input))
        except AttributeError:
            return self.linear(input)


class GroupedLinear(nn.Module):
    """Independent linear map per group, applied to a (batch, groups*in) tensor."""

    def __init__(self, in_group_size, out_group_size, groups):
        super().__init__()

        self.in_group_size = in_group_size
        self.out_group_size = out_group_size
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(groups, in_group_size, out_group_size))
        self.bias = nn.Parameter(torch.zeros(groups, 1, out_group_size))

        self.reset_parameters(self.weight, self.bias)

    def reset_parameters(self, weights, bias):
        nn.init.kaiming_uniform_(weights, a=math.sqrt(3))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(weights)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(bias, -bound, bound)

    def forward(self, x):
        reorg = x.permute(1, 0).reshape(self.groups, self.in_group_size, -1).permute(0, 2, 1)
        hook = torch.bmm(reorg, self.weight) + self.bias
        reorg = hook.permute(0, 2, 1).reshape(self.out_group_size * self.groups, -1).permute(1, 0)
        return reorg


class RepeatLayer(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x):
        return x.repeat(*self.args)


class BranchedLinear(nn.Module):
    """Per-output-branch MLP tower built from GroupedLinear layers."""

    def __init__(self, in_features, hidden_group_size, out_group_size,
                 n_branches=1, n_layers=1, activation='ReLU', dropout_p=0.5):
        super().__init__()

        self.in_features = in_features
        self.hidden_group_size = hidden_group_size
        self.out_group_size = out_group_size
        self.n_branches = n_branches
        self.n_layers = n_layers

        self.branches = OrderedDict()

        self.nonlin = getattr(nn, activation)()
        self.dropout = nn.Dropout(p=dropout_p)

        self.intake = RepeatLayer(1, n_branches)
        cur_size = in_features

        for i in range(n_layers):
            if i + 1 == n_layers:
                setattr(self, f'branched_layer_{i+1}', GroupedLinear(cur_size, out_group_size, n_branches))
            else:
                setattr(self, f'branched_layer_{i+1}', GroupedLinear(cur_size, hidden_group_size, n_branches))
            cur_size = hidden_group_size

    def forward(self, x):
        hook = self.intake(x)

        i = -1
        for i in range(self.n_layers - 1):
            hook = getattr(self, f'branched_layer_{i+1}')(hook)
            hook = self.dropout(self.nonlin(hook))
        hook = getattr(self, f'branched_layer_{i+2}')(hook)

        return hook


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------

class MPACModel(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='mpac',
    tags=['biology', 'genomics', 'dna', 'mpra', 'cis-regulatory'],
    license='mit',
):
    """The `BassetBranched` architecture used by every MPAC checkpoint.

    Consumes one-hot DNA of shape (batch, 4, input_len) and returns one activity
    value per output branch, shape (batch, n_outputs). For the released weights
    the branches are `CELL_TYPES` and `input_len` is 600.
    """

    def __init__(self, input_len=600,
                 conv1_channels=300, conv1_kernel_size=19,
                 conv2_channels=200, conv2_kernel_size=11,
                 conv3_channels=200, conv3_kernel_size=7,
                 n_linear_layers=2, linear_channels=1000,
                 linear_activation='ReLU', linear_dropout_p=0.3,
                 n_branched_layers=1, branched_channels=250,
                 branched_activation='ReLU6', branched_dropout_p=0.,
                 n_outputs=280,
                 use_batch_norm=True, use_weight_norm=False,
                 variable_region_len=200, output_names=None):
        super().__init__()

        self.input_len = input_len

        self.conv1_channels = conv1_channels
        self.conv1_kernel_size = conv1_kernel_size
        self.conv1_pad = get_padding(conv1_kernel_size)

        self.conv2_channels = conv2_channels
        self.conv2_kernel_size = conv2_kernel_size
        self.conv2_pad = get_padding(conv2_kernel_size)

        self.conv3_channels = conv3_channels
        self.conv3_kernel_size = conv3_kernel_size
        self.conv3_pad = get_padding(conv3_kernel_size)

        self.n_linear_layers = n_linear_layers
        self.linear_channels = linear_channels
        self.linear_activation = linear_activation
        self.linear_dropout_p = linear_dropout_p

        self.n_branched_layers = n_branched_layers
        self.branched_channels = branched_channels
        self.branched_activation = branched_activation
        self.branched_dropout_p = branched_dropout_p

        self.n_outputs = n_outputs

        self.use_batch_norm = use_batch_norm
        self.use_weight_norm = use_weight_norm

        self.variable_region_len = variable_region_len
        self.output_names = list(output_names) if output_names is not None else None
        assert self.output_names is None or len(self.output_names) == n_outputs, \
            f"output_names has {len(self.output_names)} entries but n_outputs is {n_outputs}"

        self.pad1 = nn.ConstantPad1d(self.conv1_pad, 0.)
        self.conv1 = Conv1dNorm(4, self.conv1_channels, self.conv1_kernel_size,
                                stride=1, padding=0, dilation=1, groups=1, bias=True,
                                batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm)
        self.pad2 = nn.ConstantPad1d(self.conv2_pad, 0.)
        self.conv2 = Conv1dNorm(self.conv1_channels, self.conv2_channels, self.conv2_kernel_size,
                                stride=1, padding=0, dilation=1, groups=1, bias=True,
                                batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm)
        self.pad3 = nn.ConstantPad1d(self.conv3_pad, 0.)
        self.conv3 = Conv1dNorm(self.conv2_channels, self.conv3_channels, self.conv3_kernel_size,
                                stride=1, padding=0, dilation=1, groups=1, bias=True,
                                batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm)

        self.pad4 = nn.ConstantPad1d((1, 1), 0.)

        self.maxpool_3 = nn.MaxPool1d(3, padding=0)
        self.maxpool_4 = nn.MaxPool1d(4, padding=0)

        next_in_channels = self.conv3_channels * self.get_flatten_factor(self.input_len)

        for i in range(self.n_linear_layers):
            setattr(self, f'linear{i+1}',
                    LinearNorm(next_in_channels, self.linear_channels, bias=True,
                               batch_norm=self.use_batch_norm, weight_norm=self.use_weight_norm))
            next_in_channels = self.linear_channels

        self.branched = BranchedLinear(next_in_channels, self.branched_channels,
                                       self.branched_channels, self.n_outputs,
                                       self.n_branched_layers, self.branched_activation,
                                       self.branched_dropout_p)

        self.output = GroupedLinear(self.branched_channels, 1, self.n_outputs)

        self.nonlin = getattr(nn, self.linear_activation)()

        self.dropout = nn.Dropout(p=self.linear_dropout_p)

        self._register_flanks()

    def get_flatten_factor(self, input_len):
        hook = input_len
        assert hook % 3 == 0
        hook = hook // 3
        assert hook % 4 == 0
        hook = hook // 4
        assert (hook + 2) % 4 == 0
        return (hook + 2) // 4

    # -- MPRA vector context ---------------------------------------------------

    def _register_flanks(self):
        """Precompute the one-hot flanks that pad a variable region up to input_len.

        Registered non-persistently so they stay out of the state dict, which
        keeps key parity with the original `boda` checkpoints.
        """
        pad_total = self.input_len - self.variable_region_len
        if pad_total <= 0:
            self.register_buffer('left_flank', None, persistent=False)
            self.register_buffer('right_flank', None, persistent=False)
            return

        left_len = pad_total // 2
        right_len = pad_total - left_len
        assert left_len <= len(MPRA_UPSTREAM) and right_len <= len(MPRA_DOWNSTREAM), \
            f"need {left_len}/{right_len} bp of flank, have {len(MPRA_UPSTREAM)}/{len(MPRA_DOWNSTREAM)}"

        self.register_buffer('left_flank', dna2tensor(MPRA_UPSTREAM[-left_len:]).unsqueeze(0),
                             persistent=False)
        self.register_buffer('right_flank', dna2tensor(MPRA_DOWNSTREAM[:right_len]).unsqueeze(0),
                             persistent=False)

    def add_flanks(self, x):
        """Concatenate MPRA vector context onto a (batch, 4, variable_region_len) tensor."""
        assert x.shape[-1] == self.variable_region_len, \
            f"expected variable region of {self.variable_region_len} bp, got {x.shape[-1]}"
        *batch_dims, _, _ = x.shape
        pieces = []
        if self.left_flank is not None:
            pieces.append(self.left_flank.expand(*batch_dims, -1, -1))
        pieces.append(x)
        if self.right_flank is not None:
            pieces.append(self.right_flank.expand(*batch_dims, -1, -1))
        return torch.cat(pieces, axis=-1)

    # -- computation -----------------------------------------------------------

    def encode(self, x):
        hook = self.nonlin(self.conv1(self.pad1(x)))
        hook = self.maxpool_3(hook)
        hook = self.nonlin(self.conv2(self.pad2(hook)))
        hook = self.maxpool_4(hook)
        hook = self.nonlin(self.conv3(self.pad3(hook)))
        hook = self.maxpool_4(self.pad4(hook))
        hook = torch.flatten(hook, start_dim=1)
        return hook

    def decode(self, x):
        hook = x
        for i in range(self.n_linear_layers):
            hook = self.dropout(self.nonlin(getattr(self, f'linear{i+1}')(hook)))
        hook = self.branched(hook)
        return hook

    def classify(self, x):
        return self.output(x)

    def forward(self, x):
        """Predict activity from a fully assembled (batch, 4, input_len) one-hot tensor."""
        return self.classify(self.decode(self.encode(x)))

    # -- convenience -----------------------------------------------------------

    @torch.no_grad()
    def predict(self, sequences, batch_size=128, rc_average=True, device=None):
        """Predict activity for a list of bare variable-region DNA strings.

        Handles the two steps that are easy to get wrong: attaching the MPRA
        vector context, and averaging the forward and reverse-complement passes
        (the convention used throughout the CODA papers).

        Returns a (len(sequences), n_outputs) float tensor on the CPU, with
        columns ordered as `self.output_names`.
        """
        if isinstance(sequences, str):
            raise TypeError("pass a list of sequences, not a single string")

        device = device if device is not None else next(self.parameters()).device
        was_training = self.training
        self.eval()

        results = []
        try:
            for start in range(0, len(sequences), batch_size):
                chunk = sequences[start:start + batch_size]
                batch = torch.stack([dna2tensor(s.upper()) for s in chunk]).to(device)
                preds = self(self.add_flanks(batch))
                if rc_average:
                    # The reverse strand is the reverse complement of the INSERT ONLY,
                    # re-flanked in the forward orientation -- not a flip of the
                    # assembled 600 bp tensor. This looks like a bug and is not: it
                    # matches `src/vcf_predict.py` in sjgosai/boda2, which produced the
                    # published MPAC predictions, and it models the real experiment
                    # (a fixed plasmid with the insert cloned backwards).
                    #
                    # Flipping the flanked tensor instead scores ~0.035 higher against
                    # Table S2, so the temptation to "fix" this is real. Don't: it would
                    # silently desynchronise this model from every published MPAC number.
                    rc = self.add_flanks(batch.flip(dims=[1, 2]))
                    preds = (preds + self(rc)).div(2.)
                results.append(preds.cpu())
        finally:
            self.train(was_training)

        return torch.cat(results, dim=0)


class MPACEnsemble(nn.Module):
    """Mean prediction over a set of architecturally identical `MPACModel`s.

    Uses `torch.func.vmap` over stacked parameters, matching `ConsistentModelPool`
    in the CODA inference scripts.
    """

    def __init__(self, models):
        super().__init__()

        models = list(models)
        assert len(models) > 0, "need at least one model"
        for m in models:
            m.eval()

        self._template = models[0]
        self.n_models = len(models)
        self.output_names = self._template.output_names
        self.variable_region_len = self._template.variable_region_len
        self.input_len = self._template.input_len

        params, buffers = stack_module_state(models)
        # Keep the stacked tensors visible to .to()/.cuda() by registering them.
        self.params = nn.ParameterDict(
            {k.replace('.', '/'): nn.Parameter(v, requires_grad=False) for k, v in params.items()}
        )
        self._buffer_keys = list(buffers.keys())
        for k, v in buffers.items():
            self.register_buffer(k.replace('.', '/'), v)

    def _unpack(self):
        params = {k.replace('/', '.'): v for k, v in self.params.items()}
        buffers = {k: getattr(self, k.replace('.', '/')) for k in self._buffer_keys}
        return params, buffers

    def forward(self, x):
        params, buffers = self._unpack()

        def fmodel(p, b, data):
            return functional_call(self._template, (p, b), (data,))

        preds = vmap(fmodel, in_dims=(0, 0, None))(params, buffers, x)
        return preds.mean(dim=0)

    def add_flanks(self, x):
        return self._template.add_flanks(x)

    predict = MPACModel.predict

    @classmethod
    def from_pretrained(cls, repo_id, chromosome, device='cpu', **kwargs):
        """Load the ten MPAC models that held `chromosome` out as their test fold.

        This is the intended entry point. Picking a fold by hand is easy to get
        wrong, and getting it wrong silently leaks training data into your
        predictions rather than raising an error.

        `chromosome` accepts '7', 7, or 'chr7'.
        """
        import json

        from huggingface_hub import hf_hub_download, snapshot_download
        from safetensors.torch import load_file

        chrom = str(chromosome).lower().replace('chr', '')

        provenance = json.load(open(hf_hub_download(repo_id, 'provenance.json', **kwargs)))
        fold = fold_for_chromosome(provenance, chrom)

        config = json.load(open(hf_hub_download(repo_id, 'config.json', **kwargs)))
        local = snapshot_download(repo_id, allow_patterns=[f'{fold}/*'], **kwargs)

        models = []
        for record in sorted(r['file'] for r in provenance
                             if os.path.dirname(r['file']) == fold):
            model = MPACModel(**config)
            model.load_state_dict(load_file(os.path.join(local, record)))
            models.append(model.eval().to(device))

        assert len(models) == 10, \
            f"expected 10 replicates for {fold}, found {len(models)}"
        return cls(models).to(device)


def fold_for_chromosome(provenance, chromosome):
    """Return the directory of the fold that held `chromosome` out as test data."""
    chrom = str(chromosome).lower().replace('chr', '')
    folds = {os.path.dirname(r['file']) for r in provenance
             if chrom in [str(c) for c in (r.get('test_chrs') or [])]}
    assert len(folds) == 1, (
        f"chromosome {chrom} maps to {len(folds)} folds ({sorted(folds)}); "
        f"MPAC covers autosomes 1-22 only, so chrX, chrY and non-human sequence "
        f"have no held-out ensemble"
    )
    return folds.pop()
