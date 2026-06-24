# +
import math
import torch
import netCDF4
import numpy as np
from itertools import cycle
from textwrap import dedent
from collections import defaultdict, deque
from graphlib import TopologicalSorter
from contextlib import nullcontext

from torch import nn
from tsnecuda import TSNE as TSNECuda
from torch._subclasses import FakeTensorMode

import fnmatch
import resource
import psutil
import socket
import pandas as pd
from partnn.io_utils import get_git_commit

import typing
import hashlib
import tarfile, io, json, pathlib, time
from partnn.io_cfg import cache_dir
from os.path import exists
import scipy

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE as TSNESK
from matplotlib.collections import EllipseCollection
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import SymLogNorm, LogNorm, PowerNorm
from matplotlib.ticker import EngFormatter, ScalarFormatter, FuncFormatter

from partnn.io_utils import resio
from partnn.io_utils import eval_formula, filter_rowdicts
from partnn.io_utils import hie2deep, deep2hie, get_subdict
from partnn.tch_utils import BatchRNG, ShapeChecker, parse_config, FunctionalLayer
from partnn.tch_utils import torch_quantile, torch_cdf, torch_astype, torch_unique, make_nn, get_srtalgnidxs, odict
from partnn.aerometrics import get_ccnerr, get_inperr, get_opticalerr, interp_refr, get_histrelerrs

import faiss
import faiss.contrib.torch_utils
import math
import warnings

warnings.filterwarnings('ignore', message='TypedStorage is deprecated')

# +
def get_snrtspltidxs(cfg_dict: dict, n_snr: int, n_t: int, n_seeds: int,
                     rng: BatchRNG, device: torch.device, pop_opts: bool=True):
    """
    Generates split indices, one for the training portion, and the other
    for testing portion.

    Args:
        cfg_dict (dict): The specifications of the splitting process. This
            defines whether each of the senario and time dimensions should
            be splitted (1) separately, or (2) jointly combined and then
            splitted.

            Example 1:

                `cfg_dict = {'vars': ['snr', 't'],
                             'snr/shuffle': True,
                             'snr/low': 0,
                             'snr/high': 0.8,
                             't/shuffle': False,
                             't/low': 0,
                             't/high': 1}`

            Example 2:

                `cfg_dict = {'vars': [snr+t],
                             'snr+t/shuffle': True,
                             'snr+t/low': 0,
                             'snr+t/high': 1}`
            `

        n_snr (int): The number of scenarios in the data.

        n_t (int): The number of time-steps in the data.

        n_seeds (int): The number of independent runs with
            different RNG seeds.

        rng (BatchRNG): The batch random number generator. This should
            have a shape of `(n_seeds,)`.

        device (torch.device | str): The pytorch device of the output.

        pop_opts (bool, optional): Whether to pop out the options out of the
            input `cfg_dict` dictionary. Defaults to False.


    Returns:
        dict: A dictionary with the `split_idxs` and `negsplit_idxs` keys:

            * `split_idxs` is a torch tensor of shape `(n_seeds, n_split)`,
                where `n_split` is the (train) split size.

            * `negsplit_idxs` is a torch tensor of shape `(n_seeds,
                n_negsplit)`, where `n_negsplit` is the (test) split size.


        Both tensors contain integers between 0 and `n_snr * n_t`, and are
            indices of training points. These values can be easily converted
            to a pair of scenario and time indices for locating the data.
    """

    # The number of seeds (i.e., independent runs)
    assert rng.shape == (n_seeds,)

    # The `split_vars` is a list of variables that need to be independently splitted:
    #
    #   * `split_vars` should either be `['snr', 't']` or `['snr+t']`.
    #
    #   * When `split_vars == ['snr+t']`, all the scenario and time pairs
    #     are compiled, possibly shuffled, and then splitted. This gives
    #     the most typical, and the easiest, kind of train/test split.
    split_vars = cfg_dict.pop('vars')

    #######################################
    #  Shuffling and Subsetting Indices   #
    #  along Scnario and Time (or both)   #
    #            Dimensions               #
    #######################################
    # The variable to size translation
    var2n = {'snr': n_snr, 't': n_t, 'snr+t': n_snr * n_t}
    var2spltinfo = dict()
    for var in split_vars:
        assert var in ('snr', 't', 'snr+t')

        # The variable size
        n_var = var2n[var]
        # Whether to shuffle the indecis before splitting
        shuffle = cfg_dict.pop(f'{var}/shuffle')
        # The starting point of the split. This should be in the [0, 1] range.
        q_low = cfg_dict.pop(f'{var}/low')
        assert 1 >= q_low >= 0
        # The ending part of the split. This should be in the [0, 1] range.
        q_high = cfg_dict.pop(f'{var}/high')
        assert 1 >= q_high >= 0

        # In the following step, we will create `n_seeds` independent randomizations
        # if the `shuffle` option was set.
        if shuffle:
            # In this case, `perm_idx1d` contains randomized permutations of `range(n_var)`.
            perm_idxs = rng.rand(n_seeds, n_var).argsort(dim=-1)
            assert perm_idxs.shape == (n_seeds, n_var)
        else:
            # In this case, `perm_idxs` is basically the same as `range(n_var)`.
            perm_idxs = torch.arange(n_var, device=device).reshape(1, n_var).expand(n_seeds, n_var)
            assert perm_idxs.shape == (n_seeds, n_var)

        # The start and end indecis along the `n_var` dimension.
        i_low, i_high = int(n_var * q_low), int(n_var * q_high)
        n_spltvar = i_high - i_low

        # The indecis of data points used in the training split
        split_idxvar = perm_idxs[:, i_low: i_high]
        assert split_idxvar.shape == (n_seeds, n_spltvar)

        # Next, we convert the array of indecis `split_idxvar` into a boolean tensor indicating
        # whether each data point is part of the split (True) or not part of the split (False).
        is_inspltvar = torch.full((n_seeds, n_var), False, device=device, dtype=torch.bool)
        assert is_inspltvar.shape == (n_seeds, n_var)
        is_inspltvar.scatter_(dim=-1, index=split_idxvar, src=(split_idxvar >= 0))

        # Storing the `is_inspltvar` indicator array.
        var2spltinfo[var] = (is_inspltvar, n_spltvar)

    if split_vars == ['snr', 't']:
        snr_insplt, n_snrsplt = var2spltinfo.pop('snr')
        assert snr_insplt.shape == (n_seeds, n_snr)
        assert (snr_insplt.sum(dim=-1) == n_snrsplt).all()

        t_insplt, n_tsplt = var2spltinfo.pop('t')
        assert t_insplt.shape == (n_seeds, n_t)
        assert (t_insplt.sum(dim=-1) == n_tsplt).all()

        # An indicator variable of whether each scenario/time combination
        # is present in a particular seed's training split.
        is_insplit_ = snr_insplt.reshape(n_seeds, n_snr, 1) *  t_insplt.reshape(n_seeds, 1, n_t)
        assert is_insplit_.shape == (n_seeds, n_snr, n_t)
        is_insplit = is_insplit_.reshape(n_seeds, n_snr * n_t)
        assert is_insplit.shape == (n_seeds, n_snr * n_t)

        n_split = n_snrsplt * n_tsplt
    elif split_vars == ['snr+t']:
        is_insplit, n_split = var2spltinfo.pop('snr+t')
        assert is_insplit.shape == (n_seeds, n_snr * n_t)
    else:
        raise ValueError(f'split_vars={split_vars} undefined')

    #######################################
    #       Creating Split Indices        #
    #######################################
    # At this point, two key variables are produced:
    #   1. `is_insplit` is a boolean array indicating whether each (snr, t)
    #       pair is in the (training) split.
    #   2. `n_split` is an integer showing the (training) split size in each
    #       of the independent runs/seeds.

    # Creating the split indices (e.g., the train split indecis).
    #   * The `split_idxs` is the output of this step.
    #   * The `split_idxs` values should be between 0 and `n_snr * n_t`.
    split_vals, split_idxs = is_insplit.to(torch.uint8).topk(n_split,
                                                             dim=-1, largest=True, sorted=True)
    assert split_idxs.shape == (n_seeds, n_split)
    assert split_vals.all()
    assert (split_idxs >=           0).all()
    assert (split_idxs <  n_snr * n_t).all()

    # Creating the negative split idxs. The negative split indices are basically
    # the indices of the complement split (e.g., test vs train).
    n_negsplit = n_snr * n_t - n_split
    is_innegsplit = is_insplit.logical_not()
    negsplit_vals, negsplit_idxs = is_innegsplit.to(torch.uint8).topk(n_negsplit,
                                                                      dim=-1, largest=True, sorted=True)
    assert negsplit_idxs.shape == (n_seeds, n_negsplit)
    assert negsplit_vals.all()
    assert (negsplit_vals >=           0).all()
    assert (negsplit_vals <  n_snr * n_t).all()

    outdict = dict(split_idxs=split_idxs, negsplit_idxs=negsplit_idxs)

    return outdict

def load_histdata(data_path: str, data_chems: None | list = None, 
    data_binsi1 : int | None = 0, data_binsi2 : int | None = None):
    """
    Loads histogram data from a netcdf file and returns the data pointers.

    Parameters
    ----------
    data_path: (str) The path to the netCDF data file.

    data_chems: (list[str]) The list of chemical species to keep.

    data_binsi1:  (int) The starting diameter bin index to load.

    data_binsi2:  (int) The ending diameter bin index to load.

    Returns
    -------
    data_hist: (dict) The loaded data with the following keys:

        * `m_chmprthst`: The speciated mass distribution of particles.
        
        * `m_prthst`: The total mass of particles within each bin.
        
        * `n_prthst`: The number concentration of the particles.
        
        * `rho_chm`: The density of chemical species.
        
        * `kappa_chm`: The Kappa constant for chemical species.
        
        * `len_wvref`: The reference wave-lengths.
        
        * `refr_wvchmref`: The refraction indices of chemicals 
            within each reference wavelength.
        
        * `n_snr`: The number of scenarios.
        
        * `n_t`: The number of time-steps within each scenario.
        
        * `n_chem`: The number of chemical species.
        
        * `n_bins`: The number of diameter bins.
        
        * `n_waveref`: The number of reference wave lengths.
        
        * `chem_species`: The list of chemical speicies names.
        
        * `d_histbins`: The diameter bin thresholds.
    
    Example
    -------
    ```
        # Loading the original data
        data_path = f'{data_dir}/02_masshist/03_bwchisamp.nc'
        data_hist = load_histdata(data_path)

        n_snr = data_hist['n_snr']
        n_t = data_hist['n_t']
        n_chem = data_hist['n_chem']
        n_bins = data_hist['n_bins']
        n_waveref = data_hist['n_waveref']

        m_chmprthst = data_hist['m_chmprthst']
        assert m_chmprthst.shape == (n_snr, n_t, n_chem, n_bins)
        m_prthst = data_hist['m_prthst']
        assert m_prthst.shape == (n_snr, n_t, n_bins)
        n_prthst = data_hist['n_prthst']
        assert n_prthst.shape == (n_snr, n_t, n_bins)
        rho_chm = data_hist['rho_chm']
        assert rho_chm.shape == (n_chem,)
        kappa_chm = data_hist['kappa_chm']
        assert kappa_chm.shape == (n_chem,)
        len_wvref = data_hist['len_wvref']
        assert len_wvref.shape == (n_waveref,)
        refr_wvchmref = data_hist['refr_wvchmref']
        assert refr_wvchmref.shape == (n_waveref, n_chem)
        chem_species = data_hist['chem_species']
        assert len(chem_species) == n_chem
        d_histbins = data_hist['d_histbins']
        assert d_histbins.shape == (n_bins + 1,)
    ```
    """
    #########################################################
    #################### Loading the Data ###################
    #########################################################
    grid_ssfrqs = None
    with netCDF4.Dataset(data_path, "r") as fp:
        ################# The Array Dimensions #################
        # The number of scenarios
        n_snrall = fp.dimensions['n_snr'].size
        # The number of time steps in each scenario
        n_t = fp.dimensions['n_t'].size
        # The number of diameter histogram bins
        n_bins = fp.dimensions['n_bins'].size
        # The number of chemical specices
        n_chem = fp.dimensions['n_chem'].size
        # The number of reference wave lengths for optical property calculations
        n_waveref = fp.dimensions['n_wave'].size

        # Applying the sub-sampling
        if grid_ssfrqs is not None:
            x_ssfrq, y_ssfrq, z_ssfrq, t_ssfrq = grid_ssfrqs
            assert n_snrall == (n_grid_x * n_grid_y * n_grid_z)
            i_snrs1 = np.arange(n_snrall * n_t).reshape(n_grid_x, n_grid_y, n_grid_z, n_t)
            assert i_snrs1.shape == (n_grid_x, n_grid_y, n_grid_z, n_t)
            i_snrs2 = i_snrs1[::x_ssfrq, ::y_ssfrq, ::z_ssfrq, ::t_ssfrq]
            n_samp = i_snrs2.size
            n_snr, n_t = math.prod(i_snrs2.shape[:-1]), i_snrs2.shape[-1]
            i_snrs = i_snrs2.ravel()
            assert i_snrs.shape == (n_samp,)
        else:
            n_snr, n_samp = n_snrall, n_snrall * n_t
            i_snrs = slice(None, None, None)

        #################### The Data Arrays ####################
        # The number of particles within each diameter bin
        n_prthst_ = fp.variables['n_prthst'][i_snrs]
        assert n_prthst_.shape == (n_samp, n_bins)
        n_prthst = n_prthst_.reshape(n_snr, n_t, n_bins)
        assert n_prthst.shape == (n_snr, n_t, n_bins)

        # The mass within each diameter bin (the mass histogram)
        m_prthst_ = fp.variables['m_prthst'][i_snrs]
        assert m_prthst_.shape == (n_samp, n_bins)
        m_prthst = m_prthst_.reshape(n_snr, n_t, n_bins)
        assert m_prthst.shape == (n_snr, n_t, n_bins)

        # The mass of each chemical species within the diameter bins
        m_chmprthst_ = fp.variables['m_chmprthst'][i_snrs]
        assert m_chmprthst_.shape == (n_samp, n_chem, n_bins)
        m_chmprthst = m_chmprthst_.reshape(n_snr, n_t, n_chem, n_bins)
        assert m_chmprthst.shape == (n_snr, n_t, n_chem, n_bins)

        ####################### Meta-Data #######################
        # The Chemical Species
        chem_species_str = fp.variables['chem_species'].names
        # The list of chemcial species
        chem_species = chem_species_str.split(',')

        # The chemical density for each species in (kg/m^3)
        rho_chm = fp.variables['chem_rho'][:]
        assert rho_chm.shape == (n_chem,)
        # The kappa of the chemical species.
        kappa_chm = fp.variables['chem_kappa'][:]
        assert kappa_chm.shape == (n_chem,)
        # The reference wave lengths for which the chemical refraction indices are provided.
        len_wvref = fp.variables['wave_len'][:]
        assert len_wvref.shape == (n_waveref,)
        # The real refraction indices of the chemical species at various wave lengths.
        refr_wvchmref_real = fp.variables['chem_refr_real'][:]
        assert refr_wvchmref_real.shape == (n_waveref, n_chem)
        # The imaginary refraction indices of the chemical species at various wave lengths.
        refr_wvchmref_imag = fp.variables['chem_refr_imag'][:]
        assert refr_wvchmref_imag.shape == (n_waveref, n_chem)
        # The complex refraction indices of the chemical species at various wave lengths.
        refr_wvchmref = refr_wvchmref_real + 1j * refr_wvchmref_imag
        assert refr_wvchmref.shape == (n_waveref, n_chem)

        # The histogram bin lower range of diameters
        diam_low = fp.variables['diam_low'][:].item()
        # The histogram bin higher range of diameters
        diam_high = fp.variables['diam_high'][:].item()
        # The diameter log base
        diam_logbase = fp.variables['diam_logbase'][:].item()

        logdiam_low, logdiam_high = np.log(diam_low) / diam_logbase, np.log(diam_high) / diam_logbase
        logdiams = np.linspace(logdiam_low, logdiam_high, n_bins + 1, endpoint=True)
        d_histbins = np.exp(logdiams * diam_logbase)
        assert d_histbins.shape == (n_bins + 1,)

    # Revising the chemical species if necessary
    if data_chems not in (None, 'all'):
        i_keepchems = [chem_species.index(chem) for chem in data_chems]
        chem_species = [chem_species[i_chm] for i_chm in i_keepchems]
        chem_species_str = ','.join(chem_species)

        n_chem = len(chem_species)
        m_chmprthst = m_chmprthst[:, :, i_keepchems, :]
        assert m_chmprthst.shape == (n_snr, n_t, n_chem, n_bins)

        m_prthst = m_chmprthst.sum(axis=-2)
        assert m_prthst.shape == (n_snr, n_t, n_bins)

        rho_chm = rho_chm[i_keepchems]
        assert rho_chm.shape == (n_chem,)
        kappa_chm = kappa_chm[i_keepchems]
        assert kappa_chm.shape == (n_chem,)
        refr_wvchmref = refr_wvchmref[:, i_keepchems]
        assert refr_wvchmref.shape == (n_waveref, n_chem)

    # Revising the histogram bins if necessary
    data_binsi2 = n_bins if data_binsi2 is None else data_binsi2
    if (data_binsi1, data_binsi2) != (0, n_bins):
        i_keepbins = list(range(data_binsi1, data_binsi2))
        n_bins = len(i_keepbins)

        # The number of particles within each diameter bin
        n_prthst = n_prthst[:, :, i_keepbins]
        assert n_prthst.shape == (n_snr, n_t, n_bins)

        # The mass within each diameter bin (the mass histogram)
        m_prthst = m_prthst[:, :, i_keepbins]
        assert m_prthst.shape == (n_snr, n_t, n_bins)

        # The mass of each chemical species within the diameter bins
        m_chmprthst = m_chmprthst[:, :, :, i_keepbins]
        assert m_chmprthst.shape == (n_snr, n_t, n_chem, n_bins)

        # The histogram bin ranges of value
        logdiams = logdiams[i_keepbins + [i_keepbins[-1] + 1]]
        assert logdiams.shape == (n_bins + 1,)
        logdiam_low, logdiam_high = logdiams[0], logdiams[-1]
        diam_low, diam_high, diam_logbase = np.exp(logdiam_low), np.exp(logdiam_high), diam_logbase
        d_histbins = np.exp(logdiams * diam_logbase)
        assert d_histbins.shape == (n_bins + 1,)

    data_hist = dict(
        m_chmprthst=m_chmprthst, m_prthst=m_prthst, n_prthst=n_prthst,
        rho_chm=rho_chm, kappa_chm=kappa_chm, len_wvref=len_wvref, refr_wvchmref=refr_wvchmref,
        n_snr=n_snr, n_t=n_t, n_chem=n_chem, n_bins=n_bins, n_waveref=n_waveref,
        chem_species=chem_species, d_histbins=d_histbins)

    return data_hist



class IsPNorm:
    def __init__(self, pp_layers, lyrnrmlz_dimset):
        """
        Utility class for facilitating the internal inverse
        operations of the `PreProcessor` class.
        """
        self.pp_layers = pp_layers
        self.lyrnrmlz_dimset = lyrnrmlz_dimset

    def __call__(self, layer_name):
        layer_opts = self.pp_layers[layer_name]
        layer_type = layer_opts['type']
        if layer_type != 'pnorm':
            return False
        layer_pnrmdim = layer_opts['dim']
        return set(layer_pnrmdim).issubset(self.lyrnrmlz_dimset)

class PreProcessor(nn.Module):
    def __init__(self, pp_config: dict | None, device: torch.device, 
        dtype: torch.dtype, cdtype: torch.dtype):
        """
        Generates the transformation parameters as a dictionary of pytorch tensors.

        Args:
            pp_config (dict): Transformation specification dictionary.

                Example:
                    ```
                    pp_hieopts = {n_prthst/type: input
                        n_prthst/shape: [n_seeds, n_mb, 1, k_bins]

                        01_add/type: add
                        01_add/value: 1000.0
                        01_add/input: n_prthst
                        01_add/shape: [n_seeds, n_mb, 1, k_bins]

                        02_log/type: log
                        02_log/base: e
                        02_log/input: 01_add
                        02_log/shape: [n_seeds, n_mb, 1, k_bins]

                        03_cntr/type: shift
                        03_cntr/input: 02_log
                        03_cntr/shape: [n_seeds, n_mb, 1, k_bins]

                        04_scale/type: scale
                        04_scale/input: 03_cntr
                        04_scale/shape: [n_seeds, n_mb, 1, k_bins]}
                    ```

            device (torch.device | str): The pytorch device of the tensors.

            dtype (torch.dtype | str): The pytorch dtype of the tensors.
        """
        super().__init__()

        # Obtaining general information about layer types and their inversion priorities.
        invfams, tchfuncs, invltypes = self.get_layertypeinfo()

        if pp_config is not None:
            # Converting the hierarchical options dictionary into a deep one.
            pp_confighie = deep2hie(pp_config, dictcls=odict, sep="/")
            pp_configdeep = hie2deep(pp_confighie, dictcls=odict, sep="/")
            pp_name = 'Pre-Processing'

            # Parsing the input config
            parsed_config = parse_config(pp_name, pp_configdeep, pop_opts=True)
            pp_input = parsed_config['input']
            pp_ishape = parsed_config['shape']

            pp_hparams = parsed_config['hparams']
            pp_hpdefs = parsed_config['hpdefs']

            # These two need to be further processed
            pp_output_ = parsed_config['output']
            pp_layers1 = parsed_config['layers']

            # Translating the options with `eval_formula` and `pp_hpdefs`
            self.tnslt_hpdefs(pp_layers1, pp_hpdefs)

            # Inserting the virtual input/output layers into the graph
            pp_layers2, pp_output = self.insert_iolayers(pp_input, pp_ishape, pp_output_, pp_layers1)

            # Removing the layers with `enable=False`
            pp_layers3 = self.enable_layers(pp_layers2)

            # Sanitizing and double-checking the input layers
            self.sanitize_inputs(pp_layers3, tchfuncs)

            # Sorting the keys of the `pp_layers3` dictionary topologically.
            pp_layers4, invpp_tsfull = self.sort_topological(pp_layers3, direction='forward')

            # Populates the 'round' attribute for each layer of the DAG.
            self.calc_rounds(pp_layers4, tchfuncs)

            # Extracting the parameter portion of the options
            pp_layers, pp_params = self.xtrct_params(pp_layers4, tchfuncs, device, dtype)

            # Getting the output layer renamings (both forward and inverse).
            rnm_outfwd, rnm_outinv = self.get_outputrnmngs(pp_layers)

            # Getting the inversion topological orders, only including layers that create the inputs.
            pp_layers, invpp_ts = self.get_invppts(pp_layers, invfams)
        else:
            pp_layers = odict()
            pp_params = odict()
            pp_hpdefs = odict()
            invpp_ts = []
            rnm_outfwd, rnm_outinv = odict(), odict()

        # The key defining variabls
        self.pp_layers = pp_layers
        self.pp_params = nn.ParameterDict(pp_params)
        self.pp_hpdefs = pp_hpdefs
        # Some attributes that can be derived from the ones above
        self.rnm_outfwd = rnm_outfwd
        self.rnm_outinv = rnm_outinv
        self.invpp_ts = invpp_ts
        # Provided to the constructor every time
        self.device = device
        self.dtype = dtype
        self.cdtype = cdtype
        # These are static variables created depending on no arguments 
        self.invfams = invfams
        self.tchfuncs = tchfuncs
        self.invltypes = invltypes

    @staticmethod
    def get_graph(pp_layers: dict, direction: str = 'forward'):
        """
        Creating the DAG representation of the transformation process.

        The output will be two dictionaries:
          1. `pp_graph`:    a layer name to the list of parents mapping, and
          2. `pp_invgraph`: a layer name to the list of children mapping.

        Args:
            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}

        Returns:
            A pair of dictionaries:
                1. `pp_graph`: a layer name to the list of parents mapping.

                    Example:
                        pp_graph    -> {child:  list[parents]}
                        pp_graph = {01_nrmlzd: [m_chmprthst],
                                    02_norm: [03_lognorm],
                                    03_lognorm: [04_lognorm],
                                    01_nrmlzd: [05_nrmlzd],
                                    04_lognorm: [06_cat],
                                    05_nrmlzd: [06_cat],
                                    06_cat: [07_out]}

                2. `pp_invgraph`: a layer name to the list of children mapping.

                    Example:
                        pp_invgraph -> {parent: list[children]}
                        pp_invgraph = {01_nrmlzd: [m_chmprthst],
                                          02_norm: [m_chmprthst],
                                          03_lognorm: [02_norm],
                                          04_lognorm: [03_lognorm],
                                          05_nrmlzd: [01_nrmlzd],
                                          06_cat: [04_lognorm, 05_nrmlzd],
                                          07_out: [06_cat]}
        """
        pp_graph, pp_invgraph = odict(), odict()
        for layer_name, layer_opts in pp_layers.items():
            layer_type = layer_opts['type']
            if direction == 'forward':
                if layer_type == 'input':
                    continue
                assert 'inplst' in layer_opts, f'{layer_name} has no inplst specified'
                chlds_lst = layer_opts['inplst']
                assert isinstance(chlds_lst, list)
            elif direction == 'inverse':
                if layer_type == 'output':
                    continue
                assert 'invinputs' in layer_opts, f'{layer_name} has no invinputs specified'
                chlds_lst = layer_opts['invinputs']
                assert isinstance(chlds_lst, list)
            else:
                raise ValueError(f'undefined direction={direction}')
            pp_graph[layer_name] = chlds_lst[:]

            for chld in chlds_lst:
                pp_invgraph.setdefault(chld, [])
                pp_invgraph[chld].append(layer_name)

        return pp_graph, pp_invgraph

    @staticmethod
    def sort_topological(pp_layers: dict, direction: str = 'forward'):
        """
        Performs a topological sort on the graph defined by the `pp_layers`
        dictionary, and returns the same dictionary but ordered topologically.

        Why do we need a topological sorter? Why can't we just implement
        a simple DFS?

        The reason is that we're dealing with a DAG, not a tree!
        a pre-order traversal yields a topological sort. This should be easy
        to see - by definition, a pre-order traversal visits each layer before
        visiting the layer's children, which is precisely the property you want
        to have for a topological sort.

        On general DAGs, though, this reasoning doesn't work. You can have
        portions where two layers have an edge to the same layer (think of a
        diamond shape) - pre-order might visit one of the two parents first,
        and then visit the child before visiting the other parent.

        Research "pre-order traversal vs topological sort" to learn more:
        https://www.quora.com/Is-topological-sort-on-a-tree-equivalent-to-a-pre-order-traversal

        Args:

            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}

        Returns:
            pp_layerssrtd (dict): The same as the `pp_layers` dictionary, except
                that the keys (i.e., DAG layers) are sorted topologically.

                In other words, the input layers are ordered first, and
                the output layers are last.

                Example:
                pp_layerssrtd = {'m_chmprthst': {'type': ...},
                    '01_nrmlzd': {'type': ...},
                    '02_norm': {'type': ...},
                    '05_nrmlzd': {'type': ...},
                    '03_lognorm': {'type': ...},
                    '04_lognorm': {'type': ...},
                    '07_out': {'type': ...}}

            invpp_ts (list): The list of layers in the inverse topological
                order. This is useful for performing the inverse operations.

                In other words, the output layers are ordered first, and
                the input layers are last.
        """
        pp_graph, invpp_graph = PreProcessor.get_graph(pp_layers, direction=direction)

        pp_tsorter = TopologicalSorter(graph=pp_graph)

        pp_ts = list(pp_tsorter.static_order())
        # Example:
        #   pp_ts = ['m_chmprthst', '01_nrmlzd', '02_norm', '05_nrmlzd',
        #               '03_lognorm', '04_lognorm', '07_out']

        # Doing the same process for the inverted DAG
        invpp_tsorter = TopologicalSorter(graph=invpp_graph)
        invpp_ts = list(invpp_tsorter.static_order())

        # Sorting the forward and inverse transformation dictionary
        pp_layerssrtd = odict([(layer_name, pp_layers[layer_name]) for layer_name in pp_ts])

        # Inserting the output attribute into the options
        for layer_name in invpp_ts:
            layer_opts = pp_layerssrtd[layer_name]
            layer_output = invpp_graph.get(layer_name, [])
            if direction == 'forward':
                if len(layer_output) == 0:
                    assert layer_opts['type'] == 'output', dedent(f'''
                        The "{layer_name}" has no outputs and is not an output layer.
                        Dangling layers are not permitted in the transformation graph.
                        Either fix this or remove such dangling layers.''')
                layer_opts['output'] = layer_output
            elif direction == 'inverse':
                pass
            else:
                raise ValueError(f'undefined direction={direction}')

        return pp_layerssrtd, invpp_ts

    @staticmethod
    def calc_rounds(pp_layers: dict, tchfuncs: dict):
        """
        Finding the appropriate parameter inference rounds.

        This method does not return anything. Instead, it populates
        a 'round' key within the options of each layer in the
        `pp_layers` dictionary.

        Why do we need to define rounds?
          1. The shift/scale parameter inferences require a full
             sweep over the data.
          2. We cannot perform a forward transformation over the
             entire data in one mini-batch; the data may not fit
             the memory.
          3. A shift/scale layer might come after another shift/scale
             layer. For this, the first parameters must be inferred
             first, so that the evaluated mini-batches are correct
             for the second layer!

        Args:
            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}

            tchfuncs (dict): A set of pytorch non-parametric function names 
                mapped to the torch functions.

                Example:

                    tchfuncs = {
                        'sum': torch.sum,
                        'torch.sum': torch.sum,
                        'mean': torch.mean,
                        'torch.mean': torch.mean,
                        ...}
        """
        for layer_name, layer_opts in pp_layers.items():
            # Example:
            #   layer_opts = {'type': 'input',
            #              'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
            #   layer_type = 'input'
            layer_type = layer_opts['type']

            #  Examples:
            #    1. layer_type = 'input'
            #       layer_input = None
            #    2. layer_type = 'log'
            #       layer_input = '02_norm'
            #    3. layer_type = 'cat'
            #       layer_input = ['04_lognorm', '05_nrmlzd']
            layer_input = layer_opts.get('input', None)
            assert isinstance(layer_input, (type(None), str, list))

            if layer_type == 'input':
                assert layer_input is None
                layer_round = 0
            elif layer_type in ('add', 'mul', 'log', 'exp', 'pwr', 'pow', 'sgnpow', 'diff',
                'normalize', 'pnorm', 'output', 'pipe', 'slice', 'reshape'):
                assert isinstance(layer_input, str), dedent(f'''
                    layer_name = {layer_name}
                    layer_type = {layer_type}
                    layer_input = {layer_input}''')
                layer_round = pp_layers[layer_input]['round']
            elif layer_type in ('cat', 'stack'):
                assert isinstance(layer_input, list)
                layer_round = max(pp_layers[layer_name_]['round'] for layer_name_ in layer_input)
            elif layer_type in ('layeradd', 'layermul', 'layersub', 'layerdiv'):
                assert isinstance(layer_input, str), dedent(f'''
                    layer_name = {layer_name}
                    layer_type = {layer_type}
                    layer_input = {layer_input}''')
                assert 'other' in layer_opts, f'Layer {layer_name} has no "other" argument'
                layer_other = layer_opts['other']
                assert layer_other in pp_layers, dedent(f"""
                    Layer {layer_name}'s "other" argument value is not a 
                    valid layer name: 
                        layer_name = {layer_name}
                        other = {layer_other}""")
                layer_input2 = [layer_input, layer_other]
                layer_round = max(pp_layers[layer_name_]['round'] for layer_name_ in layer_input2)
            elif layer_type in tchfuncs:
                assert isinstance(layer_input, (str, list))
                layer_input2 = [layer_input] if isinstance(layer_input, str) else layer_input
                for layer_opt, layer_optval in layer_opts.items():
                    if isinstance(layer_optval, str) and (layer_optval in pp_layers):
                        layer_input2.append(layer_optval)
                layer_round = max(pp_layers[layer_name_]['round'] for layer_name_ in layer_input2)
            elif layer_type in ('shift', 'scale', 'cdf'):
                assert isinstance(layer_input, str)
                layer_round = pp_layers[layer_input]['round'] + 1
            else:
                raise ValueError(f'layer type {layer_type} undefined')
            layer_opts['round'] = layer_round

    @staticmethod
    def tnslt_hpdefs(pp_layers: dict, pp_hpdefs: dict):
        """
        Translates the options in `pp_layers` in-place using the
        variables in `pp_hpdefs` and the `eval_formula`.

        Args:

            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}

            pp_hpdefs (dict): The collection of hyper-parameters
                and defined variables. This will be used to
                translate the options in `pp_layers`.
        """
        for layer_name, layer_opts in pp_layers.items():
            for hpname, hpval in list(layer_opts.items()):
                # The 'input', 'output', 'enable' can still be parametric. This 
                # was done to accomodate parametric architecture shapes.
                if hpname in ('type', 'shape', 'round'):
                    continue
                layer_opts[hpname] = eval_formula(hpval, pp_hpdefs, catch=True)

    @staticmethod
    def insert_iolayers(pp_input: list, pp_ishape: dict,
        pp_out: str | list | tuple | dict, pp_lyrs: dict):
        """
        Adds the input and output layers into the layers. The `pp_out`
        is also parsed into a dictionary and returned.

        Args:
            pp_input (list): a list containing the input layer names
                that don't yet exist in the graph.

            pp_ishape (dict): a dictionary mapping the input layer
                names to their expected shapes.

            pp_out (str | list | tuple | dict): the non-resolved
                output layer names. This function will turn this into
                a dictionary no matter the type and return it.

            pp_lyrs (dict): the layers' config.

        Returns:
            pp_layers (dict): the amended layers config with the
                input/output layers inserted.

            pp_output (dict): a dictionary mapping the output layer
                names in the graph to their respective inputs.
        """
        pp_layers = dict()
        for layer_name in pp_input:
            assert layer_name not in pp_lyrs, dedent(f'''
                The input layer name "{layer_name}" is already defined in 
                the layers. Make sure the layer names are different 
                than the input names:
                    tsnfm_input = {pp_input}
                    pp_lyrs.keys() = {pp_lyrs.keys()}''')
            layer_shape = pp_ishape.get(layer_name, None)
            layer_opts = dict(type='input')
            if layer_shape is not None:
                layer_opts['shape'] = layer_shape
            pp_layers[layer_name] = layer_opts

        for layer_name, layer_opts in pp_lyrs.items():
            pp_layers[layer_name] = layer_opts

        if isinstance(pp_out, str):
            pp_output = {'__output__': pp_out}
        elif isinstance(pp_out, (list, tuple)):
            pp_output = {f'__output__{i+1:02d}': layer_name_
                for i, layer_name_ in enumerate(pp_out)}
        elif isinstance(pp_out, dict):
            pp_output = pp_out
        else:
            raise ValueError(dedent(f'''
                undefined "{type(pp_out)}" type for pp_output: 
                    pp_output = {pp_out}'''))

        for layer_name, layer_input in pp_output.items():
            assert isinstance(layer_input, str), dedent(f'''
                The output layer name "{layer_name}" has a non-string 
                input specified with the "{type(layer_input)}" type:
                    layer_input = {layer_input}''')
            
            assert layer_name not in pp_lyrs, dedent(f'''
                The output layer name "{layer_name}" is already defined in 
                the layers. Make sure the layer names are different 
                than the output names:
                    pp_out = {pp_out}
                    pp_lyrs.keys() = {pp_lyrs.keys()}''')

            assert layer_input in pp_lyrs, dedent(f'''
                The output layer "{layer_input}" is not defined in 
                the layers:
                    pp_out = {pp_out}
                    pp_lyrs.keys() = {pp_lyrs.keys()}''')

            layer_shape = pp_lyrs[layer_input].get('shape', None)
            layer_opts = dict(type='output', input=layer_input)
            if layer_shape is not None:
                layer_opts['shape'] = layer_shape
            pp_layers[layer_name] = layer_opts

        return pp_layers, pp_output

    @staticmethod
    def enable_layers(pp_lyrs: dict):
        """
        Keeps the layers with the `enable` option being `True` or unspecified. 
        If the layer specifies `enable` being `False`, the layer will be 
        removed from the layers' config dictionary. The output is an independent 
        copy of the input layer configs.

        Args:
            pp_lyrs (dict): the layers' config.

        Returns:
            pp_layers (dict): the amended layers config with the disabled layers 
                being stripped out of the config dictionary.
        """
        pp_layers = dict()
        for layer_name, layer_opts_ in pp_lyrs.items():
            layer_opts = dict(layer_opts_)
            layer_en = layer_opts.pop('enable', True)
            assert isinstance(layer_en, bool), dedent(f'''
                The enable option for layer "{layer_name}" must be a bool type 
                (i.e., either `True` or `False`):
                    enable: {layer_en}
                ''')
            if layer_en:
                pp_layers[layer_name] = layer_opts
        
        return pp_layers

    @staticmethod
    def sanitize_inputs(pp_layers: dict, tchfuncs: dict):
        """
        Sanitizes the layer inputs and makes sure they're okay. It also 
        converts singleton input lists to strings for unary input layers.

        Args:
            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}
            
            tchfuncs (dict): A set of pytorch non-parametric function names 
                mapped to the torch functions.

                Example:

                    tchfuncs = {
                        'sum': torch.sum,
                        'torch.sum': torch.sum,
                        'mean': torch.mean,
                        'torch.mean': torch.mean,
                        ...}
        """
        all_layernames = list(pp_layers.keys())
        for layer_name, layer_opts in pp_layers.items():
            # Example:
            #   layer_opts = {'type': 'input',
            #              'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
            #   layer_type = 'input'
            assert 'type' in layer_opts, dedent(f'''
                Layer "{layer_name}" has no type specified.
            ''')
            layer_type = layer_opts['type']

            #  Examples:
            #    1. layer_type = 'input'
            #       layer_input = None
            #    2. layer_type = 'log'
            #       layer_input = '02_norm'
            #    3. layer_type = 'cat'
            #       layer_input = ['04_lognorm', '05_nrmlzd']
            layer_input = layer_opts.get('input', None)
            layer_input2 = None
            layer_inplst = None

            if layer_type == 'input':
                assert layer_input is None
                layer_input2 = None
                layer_inplst = []
            elif layer_type in ('add', 'mul', 'log', 'exp', 'pwr', 'pow', 'sgnpow', 'diff',
                'normalize', 'pnorm', 'output', 'pipe', 'reshape', 'slice', 
                'shift', 'scale', 'cdf'):
                layer_input2 = layer_input
                if isinstance(layer_input, (list, tuple)):
                    assert len(layer_input) == 1, dedent(f'''
                        The layer "{layer_name}" with the "{layer_type}" type should 
                        have a single input specified. However, it has {len(layer_input)}
                        elements:
                            layer_input = {layer_input}''')
                    layer_input2 = layer_input[0]  
                assert isinstance(layer_input2, str), dedent(f'''
                        The layer "{layer_name}" with the "{layer_type}" type should 
                        have a single input layer name specified. However, it specified 
                        a non-string input value of the "{type(layer_input2)}" type:
                            layer_input = {layer_input2}''')
                assert layer_input2 in all_layernames, dedent(f'''
                    The layer "{layer_name}" with the "{layer_type}" type should 
                    have a single input layer name specified. However, it specified 
                    "{layer_input2}" which is not one of the layers:
                        layer_name = {layer_name}
                        layer_type = {layer_type}
                        layer_input = {layer_input2}
                        all_layernames = {all_layernames}''')
                layer_inplst = [layer_input2]
            elif layer_type in ('cat', 'stack'):
                assert isinstance(layer_input, (list, tuple)), dedent(f'''
                    The layer "{layer_name}" with the "{layer_type}" type should 
                    have a list/tuple of inputs specifed:
                        layer_input = {layer_input}''')
                
                for lyr_input in layer_input:
                    assert isinstance(lyr_input, str), dedent(f'''
                        The layer "{layer_name}" with the "{layer_type}" type should 
                        have a list of input layer names specified. However, it specified
                        a non-string input value "{lyr_input}" with the type "{type(lyr_input)}":
                            layer_input = {layer_input}''')

                    assert lyr_input in all_layernames, dedent(f'''
                        The layer "{layer_name}" with the "{layer_type}" type should 
                        have a list of layer names specified. However, it specified 
                        "{lyr_input}" as an input, which is not one of the layers:
                            layer_name = {layer_name}
                            layer_type = {layer_type}
                            offending input = {lyr_input}
                            layer_input = {layer_input}
                            all_layernames = {all_layernames}''')
                
                layer_input2 = layer_input
                layer_inplst = layer_input
            elif layer_type in ('layeradd', 'layermul', 'layersub', 'layerdiv'):
                layer_input2 = layer_input
                assert 'other' in layer_opts, dedent(f'''
                    Could not find the "other" argument to the layer "{layer_name}" 
                    with the "{layer_type}" type:
                        layer_opts: {layer_opts}''')
                layer_other = layer_opts['other']
                assert layer_other in all_layernames, dedent(f'''
                    The "{layer_other}" layer name input to the layer "{layer_name}" 
                    with the "{layer_type}" type does not exist:
                        other: {layer_other}
                        layer_opts: {layer_opts}''')
                layer_inplst = [layer_input2, layer_other]
            elif layer_type in tchfuncs:
                layer_input2 = None
                layer_inplst = []
                for layer_opt, layer_optval in layer_opts.items():
                    if isinstance(layer_optval, str) and (layer_optval in all_layernames):
                        layer_inplst.append(layer_optval)
                    
                assert len(layer_inplst) > 0, dedent(f'''
                    Could not find any inputs to the layer "{layer_name}" with 
                    the "{layer_type}" type:
                        layer_opts: {layer_opts}''')
            else:
                raise ValueError(f'layer type {layer_type} undefined')
            
            assert 'inplst' not in layer_opts
            assert layer_inplst is not None
            layer_opts['inplst'] = layer_inplst
            if layer_input2 is not None:
                layer_opts['input'] = layer_input2

    @staticmethod
    def xtrct_params(pp_layers: dict, tchfuncs: dict, device: any, dtype: any):
        """
        Extracts the torch parameter options from `pp_layers` and
        stores them in a separate `pp_params` dictionary. This `pp_params`
        dictionary will later be passed to `nn.ParameterDict`.

        Args:

            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}
            
            tchfuncs (dict): A set of pytorch non-parametric function names 
                mapped to the torch functions.

                Example:

                    tchfuncs = {
                        'sum': torch.sum,
                        'torch.sum': torch.sum,
                        'mean': torch.mean,
                        'torch.mean': torch.mean,
                        ...}

            device (torch.device | str): The pytorch device of the tensors.

            dtype (torch.dtype | str): The pytorch dtype of the tensors.

        Returns:
            pp_layers (dict): The same thing as the `pp_layers` input,
                only with the parameter options popped out.

            pp_params (dict): A dictionary with the layer `nn.Parameter`
                values.

        """
        pp_params = dict()
        for layer_name, layer_opts in pp_layers.items():
            # Example:
            #   layer_opts = {
            #       'type': 'input',
            #       'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
            #   layer_type = 'input'
            layer_type = layer_opts['type']

            #### Evaluating the mini-batch array ####
            if layer_type in ('input', 'normalize', 'pnorm', 'cat', 'diff',
                'layeradd', 'layermul', 'layersub', 'layerdiv', 
                'pipe', 'stack', 'reshape', 'slice', 'output'):
                pass
            elif layer_type == 'log':
                layer_base = layer_opts.pop('base', 'e')
                layer_logbase = 1.0 if layer_base == 'e' else np.log(layer_base)
                layer_logbaseth = torch.from_numpy(np.array(layer_logbase)).to(device=device, dtype=dtype)
                pp_params[f'{layer_name}/logbase'] = nn.Parameter(layer_logbaseth)
            elif layer_type == 'exp':
                layer_base = layer_opts.pop('base', 'e')
                layer_expbase = 1.0 if layer_base == 'e' else np.log(layer_base)
                layer_expbaseth = torch.from_numpy(np.array(layer_expbase)).to(device=device, dtype=dtype)
                pp_params[f'{layer_name}/expbase'] = nn.Parameter(layer_expbaseth)
            elif layer_type in ('pwr', 'pow'):
                layer_pow = layer_opts.pop('exponent')
                layer_powth = torch.from_numpy(np.array(layer_pow)).to(device=device, dtype=dtype)
                pp_params[f'{layer_name}/exponent'] = nn.Parameter(layer_powth)
            elif layer_type == 'sgnpow':
                layer_pow = layer_opts.pop('exponent')
                layer_powth = torch.from_numpy(np.array(layer_pow)).to(device=device, dtype=dtype)
                pp_params[f'{layer_name}/exponent'] = nn.Parameter(layer_powth)
            elif layer_type in ('add', 'mul'):
                # `layer_addval` is a native python object and needs to
                # be converted to a pytorch tensor.
                assert 'value' in layer_opts, dedent(f'''
                    During pre-processor construction, the "{layer_name}" layer did not 
                    have a `value` property. This is a required option.''')

                layer_value = layer_opts.pop('value')
                assert (isinstance(layer_value, (int, float, list, tuple, np.ndarray)) 
                    or torch.is_tensor(layer_value)), dedent(f'''
                        The value for layer "{layer_name}" has a type of 
                        "{type(layer_value)}", which is not defined:
                            layer_name = {layer_name}
                            layer_type = {layer_type}
                            layer_value = {layer_value}
                    ''')

                if torch.is_tensor(layer_value):
                    layer_valth = layer_value
                elif isinstance(layer_value, (int, float)):
                    layer_valth = torch.tensor(layer_value, device=device, dtype=dtype)
                else:
                    layer_valnp = np.array(layer_value)
                    layer_valth = torch.from_numpy(layer_valnp).to(device=device, dtype=dtype)

                pp_params[f'{layer_name}/value'] = nn.Parameter(layer_valth)
            elif layer_type in ('shift', 'scale', 'cdf'):
                pkey = {'shift': 'value', 'scale': 'value', 'cdf': 'quantiles'}[layer_type]
                if pkey not in layer_opts:
                    continue

                layer_value = layer_opts.pop(pkey)
                assert isinstance(layer_value, (int, float, list, tuple, np.ndarray)) or torch.is_tensor(layer_value)

                if torch.is_tensor(layer_value):
                    layer_valth = layer_value
                elif isinstance(layer_value, (int, float)):
                    layer_valth = torch.tensor(layer_value).to(device=device, dtype=dtype)
                else:
                    layer_valnp = np.array(layer_value)
                    layer_valth = torch.from_numpy(layer_valnp).to(device=device, dtype=dtype)
                    
                pp_params[f'{layer_name}/{pkey}'] = nn.Parameter(layer_valth)
            elif layer_type in tchfuncs:
                pass
            else:
                raise ValueError(f'layer type {layer_type} undefined')

        return pp_layers, pp_params
    
    @staticmethod
    def get_outputrnmngs(pp_layers: dict):
        """
        Returns a forward and inverse output layer renaming dictionary. 
        These output renamings will be defined based on the `rename` 
        field in the layer specifications.

        Args:
            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}

        Returns:
            rnm_outfwd (dict): The forward layer renaming dictionary.

            rnm_outinv (dict): The inverse layer renaming dictionary.
        """

        rnm_outfwd, rnm_outinv = dict(), dict()
        for layer_name, layer_opts in pp_layers.items():
            # Example:
            #   layer_opts = {
            #       'type': 'input',
            #       'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
            #   layer_type = 'input'
            layer_type = layer_opts['type']
            if (layer_type == 'output') and ('rename' in layer_opts):
                layer_rename = layer_opts['rename']
                rnm_outfwd[layer_name] = layer_rename
                assert layer_rename not in rnm_outinv, dedent(f'''
                    Multiple output layers defined the same renaming:
                        offending layer name: {layer_name}
                        offending layer rename: {layer_rename}
                        other layer with the same rename: {rnm_outinv[layer_rename]}
                    ''')
                rnm_outinv[layer_rename] = layer_name
        
        return rnm_outfwd, rnm_outinv

    @staticmethod
    def get_layertypeinfo():
        """
        Generates general information about layer types.

        Returns:

            tchfuncs (dict): A set of pytorch non-parametric function names 
                mapped to the torch functions.

                Example:

                    tchfuncs = {
                        'sum': torch.sum,
                        'torch.sum': torch.sum,
                        'mean': torch.mean,
                        'torch.mean': torch.mean,
                        ...}

            
            invfams (dict): A mapping of layer types to their inversion 
                priority values.

                Example:

                    invfams = {
                        ('output',): 17, ('input',): 16, ('pipe',): 15, 
                        ('reshape',): 14, ('cat',): 13, ('stack',): 12, 
                        ('add',): 11, ('shift',): 10, 
                        ('mul',): 9, ('scale',): 8, ('sgnpow',): 7, 
                        ('pwr',): 6, ('pow',): 5, 
                        ('exp',): 4, ('log',): 3, 
                        ('normalize', 'pnorm'): 2, ('normalize',): 1, 
                        ('cdf',): 0,
                        ('sum',): -1, ('torch.sum',): -2, 
                        ('mean',): -3, ('torch.mean',): -4, 
                        ...}

            invltypes (set): The set of all invertible layer types.

                Example:

                    invltypes = {'output', 'input', 'add', 'mul', 'log', 'exp', 
                        ...}
        """
        # The invertible layer priorities
        invfams = {
            ('output',): 23, ('input',): 22, ('pipe',): 21, 
            ('reshape',): 20, ('cat',): 19, ('stack',): 18, 
            ('add',): 17, ('shift',): 16, ('diff',): 15,
            ('layeradd',): 14, ('layersub',): 13, 
            ('mul',): 12, ('scale',): 11, ('sgnpow',): 10, 
            ('layermul',): 9, ('layerdiv',): 8, 
            ('pwr',): 7, ('pow',): 6, 
            ('exp',): 5, ('log',): 4, 
            ('normalize', 'pnorm'): 3, ('normalize',): 2, 
            ('cdf',): 1, ('slice',): 0}

        # The invertible layer types
        invltypes = set()
        for ltypes in invfams:
            for ltype in ltypes:
                invltypes.add(ltype)

        # 'pnorm' itself is not invertible, so it has to be added later
        invfams[('pnorm',)] = -1

        # The non-invertible torch functions
        tchfuncs = dict()
        for layer_type in FunctionalLayer.torch_functions:
            layer_func =  getattr(torch, layer_type)
            if layer_type not in invltypes:
                tchfuncs[layer_type] = layer_func
            tchfuncs[f'torch.{layer_type}'] = layer_func
        for layer_type in FunctionalLayer.torch_nn_functionals:
            layer_func = getattr(torch.nn.functional, layer_type)
            if (layer_type not in invltypes) and (layer_type not in tchfuncs):
                tchfuncs[layer_type] = layer_func
            tchfuncs[f'torch.nn.functional.{layer_type}'] = layer_func
        tchfuncs['torch.astype'] = torch_astype
        tchfuncs['astype'] = torch_astype
        
        # The non-invertible torch function priorities are the lowest
        min_pri = min(invfams.values()) - 1
        for layer_type in tchfuncs:
            invfams[(layer_type,)], min_pri = min_pri, min_pri - 1
        
        return invfams, tchfuncs, invltypes

    @staticmethod
    def get_invppts(pp_layers, invfams):
        """
        This function generates an inverse topological sort of the layers.
        The important thing about this function is that it will restrict the 
        graph to only the edges that are necessary for recovering the input 
        layers.

        It also inserts the `'invinputs'` key in each of the layer options.

        Args:
            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}
            
            invfams (dict): A mapping of layer types to their inversion 
                priority values.

                Example:

                    invfams = {
                        ('output',): 17, ('input',): 16, ('pipe',): 15, 
                        ('reshape',): 14, ('cat',): 13, ('stack',): 12, 
                        ('add',): 11, ('shift',): 10, 
                        ('mul',): 9, ('scale',): 8, ('sgnpow',): 7, 
                        ('pwr',): 6, ('pow',): 5, 
                        ('exp',): 4, ('log',): 3, 
                        ('normalize', 'pnorm'): 2, ('normalize',): 1, 
                        ('cdf',): 0
                        ('sum',): -1, ('torch.sum',): -2, 
                        ('mean',): -3, ('torch.mean',): -4, 
                        ...}
        
        Returns:
            inv_ppts (list): The inverse topological order of traversing 
                the graph from outputs to inputs. This is a list of layer 
                names. Only the layers used for reconstructing the input 
                layers will be included.

                Example:

                    inv_ppts = ['m_chmprtnrm', 'l12_something', ..., 
                        'l01_something', 'm_chmprthst']
        """
        inv_pptsfull = list(pp_layers.keys())[::-1]
        for layer_name in inv_pptsfull:
            layer_opts = pp_layers[layer_name]
            layer_type = layer_opts['type']
            layer_outslst = layer_opts['output']
            assert isinstance(layer_outslst, list)
            if len(layer_outslst) == 0:
                assert layer_type == 'output', dedent(f'''
                    The "{layer_name}" has no outputs and is not an output layer.
                    Dangling layers are not permitted in the transformation graph.
                    Either fix this or remove such dangling layers.''')
                layer_invinputs = []
            else:
                # The mapping of each layer type to all the output layer names of that type
                layer_typ2nms = defaultdict(list)
                for lname in layer_outslst:
                    lopts = pp_layers[lname]
                    ltype = lopts['type']
                    layer_typ2nms[ltype].append(lname)
                
                # `invfam_nms2pri` will be a mapping of all possible layer families to their priorities.
                invfam_nms2pri = dict()
                for invfam_types, invfam_pri in invfams.items():
                    if (len(invfam_types) == 1) and (invfam_types[0] in ('layeradd', 'layersub', 'layermul', 'layerdiv')):
                        linv_type, = invfam_types
                        for linv_name in layer_typ2nms[linv_type]:
                            linv_other = pp_layers[linv_name]['other']
                            invfam_nms2pri[(linv_name, linv_other)] = invfam_pri
                    elif len(invfam_types) == 1:
                        linv_type, = invfam_types
                        for linv_name in layer_typ2nms[linv_type]:
                            invfam_nms2pri[(linv_name,)] = invfam_pri
                    elif invfam_types == ('normalize', 'pnorm'):
                        for lyrnrmlz in layer_typ2nms['normalize']:
                            lyrnrmlz_opts = pp_layers[lyrnrmlz]
                            lyrnrmlz_dim = lyrnrmlz_opts['dim']
                            lyrnrmlz_dimset = set(lyrnrmlz_dim)
                            is_pnorm = IsPNorm(pp_layers, lyrnrmlz_dimset)
                            for lrypnrm in layer_typ2nms['pnorm']:
                                if is_pnorm(lrypnrm):
                                    invfam_nms2pri[(lyrnrmlz, lrypnrm)] = invfam_pri
                    else:
                        raise ValueError(f'undefined case invfam_types={invfam_types}')

                # Picking the best family for inversion
                layer_invinputstup, linv_pri = max(invfam_nms2pri.items(), key=lambda kv: kv[1])
                layer_invinputs = list(layer_invinputstup)

                assert len(layer_invinputs) > 0, dedent(f'''
                    Something odd happened: 
                        layer_name: {layer_name}
                        layer_typ2nms: {layer_typ2nms}
                        invfam_nms2pri: {invfam_nms2pri}
                        layer_invinputs: {layer_invinputs}''')
            
            # Here, we will insert the 'invinputs' key into the layer options.
            if 'invinputs' in layer_opts:
                assert layer_opts['invinputs'] == layer_invinputs, dedent(f'''
                    Somehow the `invinputs` entry for a layer was provided 
                    before, and it is not the same as the one I figured.
                    Maybe a bad loading is happening:
                        layer_name: {layer_name}
                        old invinputs: {layer_opts['invinputs']}
                        new invinputs: {layer_invinputs}''')
            else:
                layer_opts['invinputs'] = layer_invinputs
        
        # Some layers are complicated for inversion: 
        #   c = layermul(a, b) 
        #   a = layermulinv(c, b)
        # Note how `b` is one of the inputs in both forward and inverse!
        # For this reason, we need to re-run the graph.
        pp_layersinv, _ = PreProcessor.sort_topological(pp_layers, direction='inverse')

        # Applying a simple BFS to see which layers are explicitly needed 
        # for inversion to recover the input layers.
        queue = deque(layer_name for layer_name, layer_opts in pp_layers.items() 
            if layer_opts['type'] == 'input')
        layers_invrqrd = set()
        while len(queue) > 0:
            layer_name = queue.popleft()
            layer_opts = pp_layers[layer_name]
            layer_invinputs = layer_opts['invinputs']
            layers_invrqrd.add(layer_name)
            for lname in layer_invinputs:
                queue.append(lname)

        # Now `layers_invrqrd` includes the layers that are strictly 
        # necessary for making it back to the input layers. All we 
        # have to do is to restrict the inverse order to those layers.
        inv_ppts = [layer_name for layer_name in pp_layersinv 
            if layer_name in layers_invrqrd]

        return pp_layers, inv_ppts

    @staticmethod
    def vrfy_inversions(pp_layers: dict, invpp_ts: list, invltypes: set):
        """
        This function goes through the inversion order, and makes sure all inversion steps 
        involve truly invertible and non torch-function layers.

        Args:

            pp_layers (dict): Transformation specification dictionary.

                Example:

                    pp_layers = {
                        'n_prthst': {'type': 'input',
                                     'shape': [n_seeds, n_mb, 1, k_bins]},

                        '01_add': {'type': 'add',
                                   'value': 1000.0,
                                   'input': 'n_prthst',
                                   'shape': ['n_seeds', 'n_mb', 1, 'k_bins']},
                        ...}
            
            inv_ppts (list): The inverse topological order of traversing 
                the graph from outputs to inputs. This is a list of layer 
                names. Only the layers used for reconstructing the input 
                layers will be included.

                Example:

                    inv_ppts = ['m_chmprtnrm', 'l12_something', ..., 
                        'l01_something', 'm_chmprthst']

            invltypes (set): The set of all invertible layer types.

                Example:

                    invltypes = {'output', 'input', 'add', 'mul', 'log', 'exp', 
                        ...}
        """
        for layer_name in invpp_ts:
            layer_opts = pp_layers[layer_name]
            layer_type = layer_opts['type']
            layer_outslst = layer_opts['output']
            layer_invinputs = layer_opts['invinputs']
            assert isinstance(layer_outslst, list)
            
            for linv_name in layer_invinputs:
                linv_opts = pp_layers[linv_name]
                linv_type = linv_opts['type']
                assert linv_type in invltypes, dedent(f'''
                    Layer "{linv_name}" is a non-invertible pre-processing layer 
                    and has no inversion procedures defined.

                    For instance, there are no inverse operations defined for torch 
                    function layers. 
                    
                    Layer "{linv_name}" has a type of "{linv_type}" and cannot 
                    be inverted to obtain the "{layer_name}" layer values.''')

    def forward(self, inputs_mb: dict, full: bool = False, skip_noninps: bool = True):
        """
        Performs a forward transformation on the pre-processing tree.

        Args:
            inputs_mb (dict): The mapping of the input layers to mini-batch
                arrays. The arrays can be either `np.ndarray` or
                `torch.tensor` types.

                Example:
                    ```
                    inputs_mb = {
                        'm_chmprthst': np.array(n_seed, n_mb, k_bins, n_chem),
                        'n_prthst': np.array(n_seed, n_mb, k_bins),
                        'm_prthst': np.array(n_seed, n_mb, k_bins)
                    }
                    ```

            full (bool): Whether to return the computed mini-batch arrays for all
                the layers (true), or only returning the mini-batch arrays for
                the output layers (false).

            skip_noninps (bool): Whether to ignore the layer name and values in 
                `inputs_mb` that are not "input" layer types in the graph. 
                
                When set to False, the mini-batch values for such layers will 
                not be re-calculated from previous layers, and will be taken from 
                the `inputs_mb` dictionary instead.

        Returns:

            layers_mb: A dictionary of layer names (str) to values (torch.tensor).
        """
        pp_layers = self.pp_layers
        pp_params = self.pp_params
        device, dtype = self.device, self.dtype
        pp_hpdefs = self.pp_hpdefs
        rnm_outfwd = self.rnm_outfwd
        tchfuncs = self.tchfuncs
        invltypes = self.invltypes
        check_shapestepwise = True

        shaper = ShapeChecker()
        shaper.update(pp_hpdefs)

        layers_mb = odict()
        for layer_name, layer_opts_ in pp_layers.items():
            # Taking a shallow copy
            layer_opts = dict(layer_opts_)

            # Example:
            #   layer_opts = {'type': 'input',
            #              'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
            #   layer_type = 'input'
            layer_type = layer_opts.pop('type')

            #### Evaluating the expected shape ####
            # Example:
            #   layer_expctdshp = ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']
            #   layer_shape = (100, 256, 51, 20)
            layer_expctdshp = layer_opts.pop('shape')

            #  Examples:
            #    1. layer_type = 'input'
            #       layer_input = None
            #    2. layer_type = 'log'
            #       layer_input = '02_norm'
            #    3. layer_type = 'cat'
            #       layer_input = ['04_lognorm', '05_nrmlzd']
            layer_input = layer_opts.pop('input', None) if (layer_type in invltypes) else None
            assert isinstance(layer_input, (type(None), str, list))

            layer_inplst = layer_opts.pop('inplst')
            assert isinstance(layer_inplst, list)

            layer_output = layer_opts.pop('output', None)
            assert isinstance(layer_output, list)

            layer_invinputs = layer_opts.pop('invinputs')
            assert isinstance(layer_invinputs, list)

            layer_round = layer_opts.pop('round', None)
            assert isinstance(layer_round, (type(None), int))

            #### Evaluating the mini-batch array ####
            if layer_type == 'input':
                assert layer_name in inputs_mb, dedent(f'''
                    Could not locate "{layer_name}" in vars: 
                        {inputs_mb.keys()}
                    ''')
                layer_mbnp = inputs_mb[layer_name]
                layer_mb = layer_mbnp if torch.is_tensor(layer_mbnp) else (
                    torch.from_numpy(layer_mbnp).to(device=device, dtype=dtype))
            elif not(skip_noninps) and (layer_name in inputs_mb):
                layer_mb = inputs_mb[layer_name]
                layer_opts.clear()
            elif layer_type in ('normalize', 'pnorm'):
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                layer_pnorm = layer_opts.pop('pnorm')
                layer_dim = layer_opts.pop('dim')
                # Example:
                #   layer_pnorm = 2
                #   layer_dim = [-1, -2]

                # `torch.linalg.norm`` is super slow because of some pytorch bug!
                # layer_inpmbnorm = torch.linalg.norm(layer_inpmb, ord=layer_pnorm,
                #     dim=layer_dim, keepdim=True)
                layer_inpmbnorm = layer_inpmb.abs().pow(layer_pnorm).sum(dim=layer_dim,
                    keepdim=True).pow(1./layer_pnorm)

                if layer_type == 'pnorm':
                    layer_mb = layer_inpmbnorm
                elif layer_type == 'normalize':
                    layer_mb = layer_inpmb / layer_inpmbnorm
                    if full:
                        layers_mb[f'{layer_name}__NRMLZD'] = layer_mb
                else:
                    raise ValueError(f'layer type {layer_type} not defined')
            elif layer_type == 'log':
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                layer_logbase = pp_params[f'{layer_name}/logbase'].detach()
                layer_mb = layer_inpmb.log() / layer_logbase
            elif layer_type == 'exp':
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                layer_expbase = pp_params[f'{layer_name}/expbase'].detach()
                layer_mb =  (layer_expbase * layer_inpmb).exp()
            elif layer_type in ('pwr', 'pow'):
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                layer_power = pp_params[f'{layer_name}/exponent'].detach()
                layer_mb =  layer_inpmb.pow(layer_power)
            elif layer_type == 'sgnpow':
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                layer_power = pp_params[f'{layer_name}/exponent'].detach()
                layer_mb =  layer_inpmb.abs().pow(layer_power) * layer_inpmb.sign()
            elif layer_type in ('add', 'mul', 'shift', 'scale'):
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                assert torch.is_tensor(layer_inpmb)
                layer_dim = layer_opts.pop('dim') if layer_type in ('shift', 'scale') else None
                layer_eps = layer_opts.pop('eps', None) if (layer_type == 'scale') else None
                layer_pnorm = layer_opts.pop('pnorm', None) if (layer_type == 'scale') else None

                assert f'{layer_name}/value' in pp_params, dedent(f'''
                    "{layer_name}" does not have a value property. Are you sure the proper 
                    shifts/scales were inferred from the training data by calling 
                    the `infer` method?''')
                layer_value = pp_params[f'{layer_name}/value'].detach()

                if layer_type == 'add':
                    layer_mb = layer_inpmb + layer_value
                elif layer_type == 'mul':
                    layer_mb = layer_inpmb * layer_value
                elif layer_type == 'shift':
                    layer_mb = layer_inpmb - layer_value
                elif layer_type == 'scale':
                    layer_mb = layer_inpmb / layer_value
                else:
                    raise ValueError(f'layer type {layer_type} not defined')
            elif layer_type in ('layeradd', 'layermul', 'layersub', 'layerdiv'):
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                assert torch.is_tensor(layer_inpmb)
                layer_other = layer_opts.pop('other')
                layer_othmb = layers_mb[layer_other]
                assert torch.is_tensor(layer_othmb)

                if layer_type == 'layeradd':
                    layer_mb = layer_inpmb + layer_othmb
                elif layer_type == 'layermul':
                    layer_mb = layer_inpmb * layer_othmb
                elif layer_type == 'layersub':
                    layer_mb = layer_inpmb - layer_othmb
                elif layer_type == 'layerdiv':
                    layer_mb = layer_inpmb / layer_othmb
                else:
                    raise ValueError(f'layer type {layer_type} not defined')
            elif layer_type == 'diff':
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]

                layer_dim = layer_opts.pop('dim')
                assert isinstance(layer_dim, int)
                
                layer_dir = layer_opts.pop('dir')
                assert layer_dir in ('fwd', 'inv')

                if layer_dir == 'fwd':
                    layer_inpmb2 = layer_inpmb
                elif layer_dir == 'inv':
                    layer_inpmb2 = layer_inpmb.flip(dims=(layer_dim,))
                else:
                    raise ValueError(f'undefined layer_dir={layer_dir}')
                
                zeros_shape = list(layer_inpmb2.shape)
                zeros_shape[layer_dim] = 1
                left_padding = torch.zeros(*zeros_shape, device=layer_inpmb2.device, dtype=layer_inpmb2.dtype)
                
                layer_mb2 = torch.diff(layer_inpmb2, n=1, dim=layer_dim, prepend=left_padding)
                assert layer_mb2.shape == layer_inpmb2.shape
                
                if layer_dir == 'fwd':
                    layer_mb = layer_mb2
                elif layer_dir == 'inv':
                    layer_mb = layer_mb2.flip(dims=(layer_dim,))
                else:
                    raise ValueError(f'undefined layer_dir={layer_dir}')
                assert layer_mb.shape == layer_inpmb.shape
            elif layer_type == 'cdf':
                assert isinstance(layer_input, str)
                layer_inpmb = layers_mb[layer_input]
                assert torch.is_tensor(layer_inpmb)

                layer_dim = layer_opts.pop('dim')
                n_qnts = layer_opts.pop('bins') + 1
                cdf_frame = layer_opts.pop('frame')
                mb_ordcdf = layer_opts.pop('mb_order', None)
                domain_min, domain_max = layer_opts.pop('domain')

                assert f'{layer_name}/quantiles' in pp_params, dedent(f'''
                    "{layer_name}" does not have a `quantiles` property. Are you sure the proper 
                    empirical quantiles were inferred from the training data by calling 
                    the `infer` method?''')
                layer_qntls = pp_params[f'{layer_name}/quantiles'].detach()

                mb_ordcdf = layer_opts.pop('mb_order', None)
                assert mb_ordcdf is not None, dedent(f'''
                    "{layer_name}" does not have a `mb_order` property. Are you sure the proper 
                    empirical mb_order were inferred from the training data by calling 
                    the `infer` method?''')

                # Note: we will avoid providing `torch_cdf` with the actual `domain` to 
                #       make sure it would not stack the domain min and max to the quantiles 
                #       since we already did that at the `layer_qntls` inference time.
                layer_cdfs = torch_cdf(input=layer_inpmb, qntls=layer_qntls, dim=mb_ordcdf, 
                    frame=frame, domain=None)
                assert layer_cdfs.shape == layer_inpmb.shape

                layer_mb = layer_cdfs
                assert layer_mb.shape == layer_inpmb.shape
            elif layer_type in ('cat', 'stack'):
                assert isinstance(layer_input, list)
                layer_inpmblst = [layers_mb[layer_name_] for layer_name_ in layer_input]
                layer_dim = layer_opts.pop('dim')
                assert isinstance(layer_dim, int)

                if layer_type == 'cat':
                    layer_mb = torch.cat(layer_inpmblst, dim=layer_dim)
                elif layer_type == 'stack':
                    layer_mb = torch.stack(layer_inpmblst, dim=layer_dim)
                else:
                    raise ValueError(f'layer type {layer_type} not defined')
            elif layer_type == 'reshape':
                layer_inpmb = layers_mb[layer_input]
                layer_shape = shaper.eval(layer_expctdshp)
                layer_mb = layer_inpmb.reshape(*layer_shape)
            elif layer_type == 'pipe':
                layer_mb = layers_mb[layer_input]
            elif layer_type == 'slice':
                layer_dim = layer_opts.pop('dim')
                slc_start = layer_opts.pop('start', None)
                slc_end = layer_opts.pop('end', None)
                slc_step = layer_opts.pop('step', None)
                layer_slc = slice(slc_start, slc_end, slc_step)
                layer_mb = torch_slice(layers_mb[layer_input], layer_dim, layer_slc)
            elif layer_type == 'output':
                layer_rename = layer_opts.pop('rename', None)
                layer_mb = layers_mb[layer_input]
            elif layer_type in tchfuncs:
                layer_kwargs, layer_inputs2 = dict(), set()
                for optn in tuple(layer_opts):
                    optval = layer_opts.pop(optn)
                    if isinstance(optval, str) and (optval in layers_mb):
                        layer_kwargs[optn] = layers_mb[optval]
                        layer_inputs2.add(optval)
                    else:
                        layer_kwargs[optn] = optval

                assert layer_inputs2 == set(layer_inplst), dedent(f'''
                    There was a mismatch in input collection for the "{layer_name}" 
                    layer with the "{layer_type}" torch function type:
                        Missing inputs: {set(layer_inplst) - layer_inputs2}
                        Extra inputs: {layer_inputs2 - set(layer_inplst)}
                        layer_kwargs: {layer_kwargs}''')

                # `torch.index_select` should have natively supported lists of 
                # integers as index. In any case, we have to make that exception.
                if layer_type in ('index_select', 'torch.index_select'):
                    idxselindx = layer_kwargs.get('index', None)
                    if isinstance(idxselindx, (list, tuple)):
                        layer_kwargs['index'] = torch.tensor(idxselindx, 
                            device=device, dtype=torch.long)
                    
                layer_func = tchfuncs[layer_type]
                
                try:
                    layer_mb = layer_func(**layer_kwargs)
                except Exception as exc:
                    layer_kwargs2 = {lkw: 
                        f'torch.tensor(shape={lkv.shape}, device={lkv.device}, dtype={lkv.dtype})'
                        if torch.is_tensor(lkv) else lkv 
                        for lkw, lkv in layer_kwargs.items()}
                    exc.add_note(dedent(f'''
                        There was an error applying a torch function:
                            layer name: {layer_name}
                            layer type: {layer_type}
                            arguments: {layer_kwargs2}'''))
                    raise exc
            else:
                raise ValueError(f'layer type {layer_type} undefined')

            layers_mb[layer_name] = layer_mb

            shaper.append({layer_name: layer_expctdshp}, layers_mb)
            if check_shapestepwise:
                shaper.check()

            assert len(layer_opts) == 0, f'unused options for layer {layer_name}: {layer_opts}'

        if not check_shapestepwise:
            shaper.check()

        # Preparing the output dictionary
        layers_mbout_ = dict()
        for layer_name, layer_mb in layers_mb.items():
            # Excluding the unofficial layers (e.g., the `*_nrmlzd` layers)
            if layer_name not in pp_layers:
                continue
            layer_opts = pp_layers[layer_name]
            layer_type = layer_opts['type']
            if layer_type != 'output':
                continue
            layer_input = layer_opts.get('input', None)
            # Making sure the output layer name is legit
            layer_outname = layer_input if layer_name.startswith('__output__') else layer_name
            layers_mbout_[layer_outname] = layer_mb

        layers_mbout = layers_mb if full else layers_mbout_

        # Applying the output layer renamings. This feature was mainly implemented for 
        # identity-like pp configs, where the user needed identical input and output names.
        for layer_name, layer_rename in rnm_outfwd.items():
            if layer_rename in layers_mbout:
                layer_mb1 = layers_mbout[layer_name]
                layer_mb2 = layers_mbout[layer_rename]
                assert id(layer_mb1) == id(layer_mb2), dedent(f'''
                    The output layer renamings is re-writing a layer's output.
                    This can happen if the output layer renaming is identical 
                    to an existing layer name and the full argument was set 
                    to True. The values for these two layers is not identical 
                    as well:
                        source layer name: {layer_name}
                        conflicting target name: {layer_rename}''')
            
            layers_mbout[layer_rename] = layers_mbout.pop(layer_name)

        return layers_mbout, shaper

    def inverse(self, outputs_mb: dict, shaper: any = None, strict: bool = True, full: bool = False):
        """
        Performs an inverse transformation on the pre-processing tree.

        Args:
            outputs_mb (dict): The mapping of the output layers to mini-batch
                arrays. The arrays can be either `np.ndarray` or
                `torch.tensor` types.

                Example:
                    ```
                    inputs_mb = {
                        'm_chmprthst': np.array(n_seed, n_mb, k_bins, n_chem),
                        'n_prthst': np.array(n_seed, n_mb, k_bins),
                        'm_prthst': np.array(n_seed, n_mb, k_bins)}
                    ```

            shaper (ShapeChecker): The shape checker object holding all the layers'
                shape information such as the shape variables and layer shapes. This
                method will only use it to evaluate the expected shape formulas, and
                will not rely on the forward layer shapes memory of the `shaper` object.

            strict (bool): Whether to enforce the one-to-one properties of the inversion
                map. If the pre-processing transformation is not one-to-one, having this
                option equal to `True` will lead to `ValueError` exceptions.

            full (bool): Whether to return the computed mini-batch arrays for all
                the layers (true), or only returning the mini-batch arrays for
                the output layers (false).

        Returns:
            layers_mb (dict): A mapping of layer names (str) to values (torch.tensor).
        """
        pp_layers = self.pp_layers
        pp_params = self.pp_params
        device, dtype = self.device, self.dtype
        pp_hpdefs = self.pp_hpdefs
        rnm_outinv = self.rnm_outinv
        tchfuncs = self.tchfuncs
        invltypes = self.invltypes
        check_shapestepwise = False
        invpp_ts = self.invpp_ts

        # Making sure there are no torch function inversions necessary.
        self.vrfy_inversions(pp_layers, invpp_ts, invltypes)

        # Applying the output layer renamings. This feature was mainly implemented for 
        # identity-like pp configs, where the user needed identical input and output names.
        assert isinstance(outputs_mb, dict)
        if len(rnm_outinv) > 0:
            outputs_mb = outputs_mb.copy()
            for layer_rename, layer_name in rnm_outinv.items():          
                outputs_mb[layer_name] = outputs_mb.pop(layer_rename)

        for layer_name, layer_opts in pp_layers.items():
            layer_type = layer_opts['type']
            if layer_type in ('cat', 'stack', 'reshape'):
                assert shaper is not None, dedent(f'''
                    In the pre-processing graph, the layer "{layer_name}" 
                    has a "{layer_type}" type and the `shaper` argument
                    is null. If any `cat`, `reshape`, or `stack` layers 
                    exist in the graph, the `shaper` argument should be 
                    provided, since inverting those layers requires the 
                    knowledge of their input layers shape variables. This 
                    is a physical requirement, and is not a mere sanity 
                    assertion tool.
                ''')

        if shaper is None:
            shaper = ShapeChecker()
            shaper.update(pp_hpdefs)

        # Translating the external output layers to
        # the actual output-type layers in the graph
        invpp_output = {layer_opts['input']: layer_name
            for layer_name, layer_opts in pp_layers.items()
            if layer_opts['type'] == 'output'}

        outputs_mb, outputs_mb_inp = dict(), outputs_mb
        for layer_name, layer_tnsr in outputs_mb_inp.items():
            if layer_name.endswith('__NRMLZD'):
                continue
            layer_opts = pp_layers[layer_name]
            layer_type = layer_opts['type']
            if layer_type == 'output':
                out_layername = layer_name
            elif layer_name in invpp_output:
                out_layername = invpp_output[layer_name]
            else:
                continue
            assert pp_layers[out_layername]['type'] == 'output'
            outputs_mb[out_layername] = layer_tnsr
        
        for layer_name, layer_opts in pp_layers.items():
            if layer_opts['type'] == 'output':
                assert layer_name in outputs_mb, dedent(f'''
                    Neither the "{layer_name}" or "{layer_opts["input"]}" layer 
                    mini-batch values were provided to the pre-proecessing inverse 
                    function. Since these represent an output layer, they must be 
                    provided as an input to the inverse transformation.''')
        
        layers_mb = odict()
        for layer_name in invpp_ts:
            layer_opts = pp_layers[layer_name]
            # Example:
            #   layer_opts = {'type': 'output',
            #              'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
            #   layer_type = 'output'
            layer_type = layer_opts['type']

            #### Evaluating the expected shape ####
            # Example:
            #   layer_expctdshp = ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']
            #   layer_shape = (100, 256, 51, 20)
            layer_expctdshp = layer_opts['shape']

            # `layer_invinputs` contains the layers that will be used for 
            # reconstructing this layer. These are the highest priority 
            # layers we could find.
            # Example:
            #       layer_invinputs = ['05_nrmlzd']
            layer_invinputs = layer_opts['invinputs']

            if len(layer_invinputs) == 0:
                layer_output = None
                layer_outopts = None
                layer_outtype = None
            else:
                layer_output = layer_invinputs[0]
                layer_outopts = pp_layers[layer_output]
                layer_outtype = layer_outopts['type']

            
            # Preventing any leakage from the previous layer
            layer_mb = None

            #### Evaluating the mini-batch array ####
            if layer_outtype is None:
                assert layer_output is None
                assert layer_type == 'output', dedent(f'''
                    Layer "{layer_name}" has no outtype, yet its type is 
                    "{layer_type}". 

                    This maybe because Layer "{layer_name}" is a dangling layer (i.e., has 
                    no outputs). Dangling layers are not permitted in the transformation 
                    graph. Either fix this or remove such dangling layers.

                        outputs_mb.keys() = {outputs_mb.keys()}
                ''')
                assert layer_name in outputs_mb, dedent(f'''
                    could not locate "{layer_name}" in vars: 
                        outputs_mb.keys() = {outputs_mb.keys()}
                ''')

                layer_mbnp = outputs_mb[layer_name]
                layer_mb = layer_mbnp if torch.is_tensor(layer_mbnp) else (
                    torch.from_numpy(layer_mbnp).to(device=device, dtype=dtype))
            elif layer_outtype == 'normalize':
                assert len(layer_invinputs) in (1, 2)
                # The normalized (i.e., $x/||x||$) layer, parameters and values
                lyrnrmlz = layer_output
                lyrnrmlz_mb = layers_mb[lyrnrmlz]

                if len(layer_invinputs) == 2:
                    lyrpnrm = layer_invinputs[1]
                    # The norm (i.e., $||x||$) layer, parameters and values
                    lyrpnrm_opts = pp_layers[lyrpnrm]
                    lyrpnrm_dim = lyrpnrm_opts['dim']
                    lyrpnrm_pnorm = lyrpnrm_opts['pnorm']
                    lyrpnrm_mb = layers_mb[lyrpnrm]

                    lyrnrmlz_mbpnrm = lyrnrmlz_mb.abs().pow(lyrpnrm_pnorm).sum(dim=lyrpnrm_dim,
                        keepdim=True).pow(1./lyrpnrm_pnorm)

                    layer_mbnrmlzd = lyrnrmlz_mb / lyrnrmlz_mbpnrm
                    layer_mb = lyrpnrm_mb * layer_mbnrmlzd
                    if full:
                        layers_mb[f'{lyrnrmlz}__NRMLZD'] = layer_mbnrmlzd
                elif strict:
                    raise ValueError(dedent(f'''
                        Cannot invert the normalized layer "{layer_output}" into the 
                        original "{layer_name}" without having the p-norm values. Either 
                        sibling pnorm layers did not exist or had too many 
                        reductions in the dimensions.
                    '''))
                elif len(layer_invinputs) == 1:
                    layer_mb = lyrnrmlz_mb
                else:
                    raise ValueError(f'undefined case for {layer_invinputs}')
            elif layer_outtype == 'log':
                assert isinstance(layer_output, str)
                layer_outmb = layers_mb[layer_output]
                if strict:
                    assert not layer_outmb.isnan().any(), dedent(f'''
                        The mini-batch values for the "{layer_output}" layer 
                        contain NaN values. This is possibly due to taking 
                        the log of some negative numbers. Therfore, the "{layer_name}" 
                        layer values cannot be strictly inverted.
                    ''')
                layer_outlogbase = pp_params[f'{layer_output}/logbase'].detach()
                layer_mb = (layer_outmb * layer_outlogbase).exp()
            elif layer_outtype == 'exp':
                assert isinstance(layer_output, str)
                layer_outmb = layers_mb[layer_output]
                if strict:
                    assert not layer_outmb.isnan().any(), dedent(f'''
                        The mini-batch values for the "{layer_output}" layer 
                        contain NaN values. This is possibly due to taking 
                        the log of some negative numbers. Therfore, the "{layer_name}" 
                        layer values cannot be strictly inverted.
                    ''')
                layer_outexpbase = pp_params[f'{layer_output}/expbase'].detach()
                layer_mb = layer_outmb.log() / layer_outexpbase
            elif layer_outtype in ('pwr', 'pow', 'sgnpow'):
                assert isinstance(layer_output, str)
                layer_outmb = layers_mb[layer_output]
                layer_outpower = pp_params[f'{layer_output}/exponent'].detach()
                if strict:
                    assert not layer_outmb.isnan().any(), dedent(f'''
                        The mini-batch values for the "{layer_output}" layer 
                        contain NaN values. This is possibly due to taking 
                        the log of some negative numbers. Therfore, the "{layer_name}" 
                        layer values cannot be strictly inverted.
                    ''')
                    assert not (layer_outpower == 0).any(), dedent(f'''
                        The power exponents for the "{layer_name}" layer contain zero. 
                        Therfore, the "{layer_name}" layer values cannot be 
                        strictly inverted.
                    ''')
                if layer_outtype in ('pwr', 'pow'):
                    layer_mb = layer_outmb.pow(1.0 / layer_outpower)
                elif layer_outtype == 'sgnpow':
                    layer_mb = layer_outmb.abs().pow(1.0 / layer_outpower) * layer_outmb.sign()
                else:
                    raise ValueError(f'undefined layer_outtype = {layer_outtype}')
            elif layer_outtype in ('add', 'mul', 'shift', 'scale'):
                assert isinstance(layer_output, str)
                layer_outmb = layers_mb[layer_output]
                assert torch.is_tensor(layer_outmb)

                # `layer_addval` is a native python object and needs to
                # be converted to a pytorch tensor.
                assert f'{layer_output}/value' in pp_params, dedent(f'''
                    The "{layer_output}" layer does not have a `value` parameter.
                    Are you sure the proper shifts/scales were inferred from
                    data by calling the `infer` method?''')
                layer_outvalue = pp_params[f'{layer_output}/value'].detach()
                assert torch.is_tensor(layer_outvalue)

                if strict and (layer_outtype == 'mul'):
                    assert layer_outvalue.all(), dedent(f'''
                        The mini-batch values for the "{layer_name}" cannot be strictly inferred from 
                        the "{layer_output}". The reason is that the multiplication values within 
                        "{layer_name}" contain zero values.
                    ''')
                elif strict  and (layer_outtype == 'scale'):
                    assert not layer_outmb.isnan().any(), dedent(f'''
                        The mini-batch values for the "{layer_output}" layer 
                        contain NaN values. This is possibly due to dividing 
                        by zero values in the forward call. Therfore, the "{layer_name}" 
                        layer values cannot be strictly inverted.
                    ''')

                if layer_outtype == 'add':
                    layer_mb = layer_outmb - layer_outvalue
                elif layer_outtype == 'mul':
                    layer_mb = layer_outmb / layer_outvalue
                elif layer_outtype == 'shift':
                    layer_mb = layer_outmb + layer_outvalue
                elif layer_outtype == 'scale':
                    layer_mb = layer_outmb * layer_outvalue
                else:
                    raise ValueError(f'layer type {layer_outtype} not defined')
            elif layer_outtype in ('layeradd', 'layermul', 'layersub', 'layerdiv'):
                layer_other = layer_outopts['other']
                layer_outmb = layers_mb[layer_output]
                layer_othmb = layers_mb[layer_other]

                if strict and (layer_outtype == 'layermul'):
                    assert layer_othmb.all(), dedent(f'''
                        The mini-batch values for the "{layer_name}" cannot be strictly inferred from 
                        the "{layer_output}". The reason is that the multiplication values within 
                        "{layer_name}" contain zero values.''')
                elif strict and (layer_outtype == 'layerdiv'):
                    assert not layer_outmb.isnan().any(), dedent(f'''
                        The mini-batch values for the "{layer_output}" layer 
                        contain NaN values. This is possibly due to dividing 
                        by zero values in the forward call. Therfore, the "{layer_name}" 
                        layer values cannot be strictly inverted.''')

                if layer_outtype == 'layeradd':
                    layer_mb = layer_outmb - layer_othmb
                elif layer_outtype == 'layermul':
                    layer_mb = layer_outmb / layer_othmb
                elif layer_outtype == 'layersub':
                    layer_mb = layer_outmb + layer_othmb
                elif layer_outtype == 'layerdiv':
                    layer_mb = layer_outmb * layer_othmb
                else:
                    raise ValueError(f'layer type {layer_outtype} not defined')
            elif layer_outtype == 'diff':
                assert isinstance(layer_output, str)
                layer_outmb = layers_mb[layer_output]

                layer_outdim = layer_outopts['dim']
                assert isinstance(layer_outdim, int)
                
                layer_outdir = layer_outopts['dir']
                assert layer_outdir in ('fwd', 'inv')

                if layer_outdir == 'fwd':
                    layer_outmb2 = layer_outmb
                elif layer_outdir == 'inv':
                    layer_outmb2 = layer_outmb.flip(dims=(layer_outdim,))
                else:
                    raise ValueError(f'undefined layer_outdir={layer_outdir}')
                
                layer_mb2 = torch.cumsum(layer_outmb2, dim=layer_outdim)
                assert layer_mb2.shape == layer_outmb2.shape
                
                if layer_outdir == 'fwd':
                    layer_mb = layer_mb2
                elif layer_outdir == 'inv':
                    layer_mb = layer_mb2.flip(dims=(layer_outdim,))
                else:
                    raise ValueError(f'undefined layer_outdir={layer_outdir}')
            elif layer_outtype == 'cdf':
                layer_dim = layer_outopts['dim']
                n_qnts = layer_outopts['bins'] + 1
                cdf_frame = layer_outopts['frame']
                mb_ordcdf = layer_outopts.get('mb_order', None)
                domain_min, domain_max = layer_outopts['domain']
                cdf_maxerr = 1e-5

                n_qnts2 = n_qnts + 2
                layer_qntls = pp_params[f'{layer_output}/quantiles'].detach()
                assert layer_qntls.shape[mb_ordcdf] == n_qnts2
                
                assert isinstance(layer_output, str)
                layer_cdfs = layers_mb[layer_output]

                if cdf_frame == 'domain':
                    layer_cdfs2 = layer_cdfs
                    assert layer_cdfs2.shape == layer_cdfs.shape
                elif cdf_frame == 'data':
                    # Shifting the domain min and max out of the [0, 1] quantile region
                    q0, q1 = 1 / (n_qnts2 - 1), (n_qnts2 - 2) / (n_qnts2 - 1)
                    layer_cdfs2 = q0 + (q1 - q0) * layer_cdfs
                    assert layer_cdfs2.shape == layer_cdfs.shape
                else:
                    raise ValueError(f'undefined cdf_frame={cdf_frame}')

                if strict:
                    min_cndtn = (layer_cdfs2.min() >= (0 - cdf_maxerr))
                    max_cndtn = (layer_cdfs2.max() <= (1 + cdf_maxerr))
                    assert min_cndtn and max_cndtn, dedent(f'''
                        Some out of domain data was passed for cdf inversion.
                        Make sure the cdf domain is extended enough.
                            min effective cdf: {layer_cdfs2.min().item()}
                            max effective cdf: {layer_cdfs2.max().item()}
                            domain min: {domain_min}
                            domain_max: {domain_max}''') 

                layer_cdfs3 = (layer_cdfs2 * (n_qnts2 - 1))
                assert layer_cdfs3.shape == layer_cdfs.shape
                
                alpha3 = layer_cdfs3 - layer_cdfs3.floor()
                assert alpha3.shape == layer_cdfs.shape

                srch_idxslo2 = (layer_cdfs3 - alpha3).round().long()
                assert srch_idxslo2.shape == layer_cdfs.shape
                assert (srch_idxslo2 < n_qnts2).all()
                assert (srch_idxslo2 >= 0).all()

                srch_valslo2 = torch.take_along_dim(layer_qntls, srch_idxslo2, dim=mb_ordcdf)
                assert srch_valslo2.shape == layer_cdfs.shape

                srch_idxshi2 = (srch_idxslo2 + 1).clamp(max=n_qnts2-1)
                assert srch_idxshi2.shape == layer_cdfs.shape
                assert (srch_idxshi2 < n_qnts2).all()
                assert (srch_idxshi2 >= 0).all()

                srch_valshi2 = torch.take_along_dim(layer_qntls, srch_idxshi2, dim=mb_ordcdf)
                assert srch_valshi2.shape == layer_cdfs.shape

                layer_mb = alpha3 * srch_valshi2 + (1 - alpha3) * srch_valslo2
                assert layer_mb.shape == layer_cdfs.shape
            elif layer_outtype == 'cat':
                # Example:
                #   '04_lognorm' --> shape == [n_seeds, n_mb, 25, 35]
                #   '07_nrmlzd'  --> shape == [n_seeds, n_mb, 25, 15]
                #   '09_scaled'  --> shape == [n_seeds, n_mb, 25, 50]
                #   '10_concat'  --> torch.cat(['04_lognorm', '07_nrmlzd',
                #                               '09_scaled'], dim=-1)
                #
                # Let's say we're trying to reconstruct the `'07_nrmlzd'`
                # layer from the `'10_concat'` layer. In this example, we have
                #    layer = '07_nrmlzd'
                #    layer_outtype = 'cat'
                #    layer_output = '10_concat'
                #    layer_outcatdim = -1
                #
                #    `layer_outcatdim` is the concatenation dimension.

                # The `torch.cat` dimension in the output layer.
                layer_outcatdim = layer_outopts['dim']
                assert isinstance(layer_outcatdim, int)

                # The list of sibling layers; i.e., the layers passed to the
                # `torch.cat` function.
                # Example:
                #   sblnglayers_lst == ['04_lognorm', '07_nrmlzd', '09_scaled']
                #   sbnd2catdim = {'04_lognorm': 35, '07_nrmlzd': 15, '09_scaled': 50}
                sblnglayers_lst = layer_outopts['input']
                sblyr2catdim = odict()
                for lyrsb in sblnglayers_lst:
                    # Example:
                    #    lyrsb = '04_lognorm'
                    #    lyrsb_expctdshp = ['n_seeds', 'n_mb', 25, 35]
                    #    lyrsb_shape = [100, 32, 25, 35]
                    #    lyrsb_shape[layer_outcatdim] = 35

                    # Calculating the sibling layer's shape
                    lyrsb_opts = pp_layers[lyrsb]
                    lyrsb_expctdshp = lyrsb_opts['shape']
                    lyrsb_shape = shaper.eval(lyrsb_expctdshp)
                    # Finding the sibling layer's share in the `torch.cat` dimension
                    sblyr2catdim[lyrsb] = lyrsb_shape[layer_outcatdim]

                # Finding the start and end indices within the `torch.cat`
                # dimension for each sibling layer.
                # Example:
                #    i1_lst = [0,  35,  50]
                #    i2_lst = [35, 50, 100]
                i2_lst = np.cumsum(list(sblyr2catdim.values())).tolist()
                i1_lst = [0] + i2_lst[:-1]

                # Example:
                #   lyrsb2catdimidxs = {'04_lognorm': (0, 35), '07_nrmlzd': (35, 50),
                #                      '09_scaled': (50, 100)}
                lyrsb2catdimidxs = {lyrsb: (i1, i2)
                                    for lyrsb, i1, i2 in zip(sblnglayers_lst, i1_lst, i2_lst)}

                # Example:
                #   layer = '07_nrmlzd'
                #   layer_i1, layer_i2 = (35, 50)
                layer_i1, layer_i2 = lyrsb2catdimidxs[layer_name]
                assert isinstance(layer_i1, int)
                assert isinstance(layer_i2, int)

                # Example:
                #   layer_outmb = torch.randn(n_seeds, n_mb, 25, 100)
                #   layer_mb = torch.randn(n_seeds, n_mb, 25, 15)
                layer_outmb = layers_mb[layer_output]
                i1i2index = torch.arange(layer_i1, layer_i2, device=device, dtype=torch.long)
                layer_mb = layer_outmb.index_select(dim=layer_outcatdim, index=i1i2index)
            elif layer_outtype == 'stack':
                # Example:
                #   '04_lognorm' --> shape == [n_seeds, n_mb, 25]
                #   '07_nrmlzd'  --> shape == [n_seeds, n_mb, 25]
                #   '09_scaled'  --> shape == [n_seeds, n_mb, 25]
                #   '10_concat'  --> torch.stack(['04_lognorm', '07_nrmlzd',
                #                                 '09_scaled'], dim=-1)
                #   '10_concat'  --> shape == [n_seeds, n_mb, 25, 3]
                #
                # Let's say we're trying to reconstruct the `'07_nrmlzd'`
                # layer from the `'10_concat'` layer. In this example, we have
                #    layer = '07_nrmlzd'
                #    layer_outtype = 'stack'
                #    layer_output = '10_concat'
                #    layer_outstckdim = -1
                #
                #    `layer_outstckdim` is the concatenation dimension.

                # The `torch.cat` dimension in the output layer.
                layer_outstckdim = layer_outopts['dim']
                assert isinstance(layer_outstckdim, int)

                # The list of sibling layers; i.e., the layers passed to the
                # `torch.cat` function.
                # Example:
                #   sblnglayers_lst == ['04_lognorm', '07_nrmlzd', '09_scaled']
                #   layer_i = 1
                sblnglayers_lst = layer_outopts['input']
                layer_i = sblnglayers_lst.index(layer_name)

                # Example:
                #   layer_outmb = torch.randn(n_seeds, n_mb, 25, 3)
                #   layer_mb =    torch.randn(n_seeds, n_mb, 25)
                layer_outmb = layers_mb[layer_output]
                layer_mb = torch.select(layer_outmb, layer_outstckdim, layer_i)
            elif layer_outtype == 'reshape':
                assert isinstance(layer_output, str)
                layer_shape = shaper.eval(layer_expctdshp)
                layer_outmb = layers_mb[layer_output]
                layer_mb = layer_outmb.reshape(*layer_shape)
            elif layer_outtype in ('output', 'pipe'):
                layer_mb = layers_mb[layer_output]
            elif (layer_outtype in tchfuncs) or (layer_outtype == 'slice'):
                raise ValueError(dedent(f'''
                    There are no inverse operations defined for torch function 
                    layers. Specifically, layer "{layer_output}" has a type of
                    "{layer_outtype}" and cannot be inverted to obtain the 
                    "{layer_name}" layer values.'''))
            else:
                raise ValueError(f'layer type {layer_outtype} undefined')

            layers_mb[layer_name] = layer_mb
            shaper.append({layer_name: layer_expctdshp}, layers_mb)
            if check_shapestepwise:
                shaper.check()
 
        if not check_shapestepwise:
            shaper.check()

        if full:
            # If there were non-native PP layers (e.g., torch functions) with a null mini-batch 
            # value, then we have to apply a forward call to populate them.
            layers_mb, _ = self.forward(layers_mb, full=True, skip_noninps=False)
            layers_mbout = layers_mb
        else:
            layers_mbout = {layer_name: layer_mb for layer_name, layer_mb in layers_mb.items()
                if (layer_name in pp_layers) and (pp_layers[layer_name]['type'] == 'input')}

        if strict:
            nan_layers = [layer_name for layer_name, layer_mb in layers_mb.items() 
                if layer_mb.isnan().any()]
            assert len(nan_layers) == 0, dedent(f'''
                The following layers had a NaN value in inversion:
                    nan_layers = {nan_layers}
            ''')

        return layers_mbout

    def get_shaper(self, inputs_shp: dict):
        """
        Runs a simulated forward loop without any calculations and tries to infer the
        layer tensor shapes using first principles about the operations. It will return
        a `ShapeChcker` object that

        Args:
            inputs_shp (dict): The mapping of the input names (str) to mini-batch
                shapes (tuple/list).

                Example:
                    ```
                    inputs_mb = {
                        'm_chmprthst': (n_seed, n_mb, k_bins, n_chem),
                        'n_prthst': (n_seed, n_mb, k_bins),
                        'm_prthst': (n_seed, n_mb, k_bins)
                    }
                    ```
        Returns:
            shaper (ShapeChecker): The shape checker object holding all the layers'
                shape information such as the shape variables and layer output shapes.

        """
        pp_layers = self.pp_layers
        pp_params = self.pp_params
        pp_hpdefs = self.pp_hpdefs
        tchfuncs = self.tchfuncs
        invltypes = self.invltypes
        shaper = ShapeChecker()
        shaper.update(pp_hpdefs)
        for layer_name, layer_opts in pp_layers.items():
            # Taking a shallow copy
            layer_type = layer_opts['type']
            layer_expctdshp = layer_opts['shape']

            layer_input = layer_opts.get('input', None)
            assert isinstance(layer_input, (type(None), str, list))

            if layer_type in ('log', 'exp', 'pwr', 'pow', 'diff', 'sgnpow', 'normalize', 'output', 'pipe'):
                assert isinstance(layer_input, str)
                layer_shape = shaper.shapes[layer_input]
            elif layer_type in ('add', 'mul', 'shift', 'scale', 'cdf', 
                'layeradd', 'layermul', 'layersub', 'layerdiv'):

                pkey, scnd_type = {'add': ('value', 'param'), 
                    'mul': ('value', 'param'), 'shift': ('value', 'param'), 
                    'scale': ('value', 'param'), 'cdf': ('quantiles', 'param'), 
                    'layeradd': ('other', 'layer'), 'layermul': ('other', 'layer'), 
                    'layersub': ('other', 'layer'), 'layerdiv': ('other', 'layer')}[layer_type]
                assert isinstance(layer_input, str)
                layer_inpshp = shaper.shapes[layer_input]

                if scnd_type == 'param':
                    assert f'{layer_name}/{pkey}' in pp_params, dedent(f'''
                        "{layer_name}" does not have a {pkey} property. Are you sure the proper 
                        shifts/scales were inferred from the training data by calling 
                        the `infere` method?''')
                    layer_scndshp = pp_params[f'{layer_name}/{pkey}'].shape
                elif scnd_type == 'layer':
                    assert pkey in layer_opts, dedent(f'''
                        "{layer_name}" does not have a {pkey} property. Are you sure the proper 
                        shifts/scales were inferred from the training data by calling 
                        the `infere` method?''')
                    layer_scnd = layer_opts[pkey]
                    layer_scndshp = shaper.shapes[layer_scnd]
                else:
                    raise ValueError(f'undefined scnd_type={scnd_type}')

                if layer_type == 'cdf':
                    assert 'mb_order' in layer_opts, dedent(f'''
                        "{layer_name}" does not have a value property. Are you sure the proper 
                        shifts/scales were inferred from the training data by calling 
                        the `infere` method?''')
                    mb_ordcdf = layer_opts['mb_order']
                    layer_scndshp = list(layer_scndshp)
                    layer_scndshp[mb_ordcdf] = 1
                    layer_scndshp = tuple(layer_scndshp)
                
                try:
                    layer_shape = torch.broadcast_shapes(layer_inpshp, layer_scndshp)
                except RuntimeError as exc:
                    exc.add_note(dedent(f'''
                        This error happend because the the input 
                        shapes for the "{layer_name}" layer were not 
                        broadcastable:
                            layer_inpshp = {layer_inpshp}
                            layer_scndshp = {layer_scndshp}'''))
                    raise exc
            elif layer_type == 'input':
                assert layer_name in inputs_shp, dedent(f'''
                    Could not locate "{layer_name}" in vars: 
                        {inputs_shp.keys()}
                    ''')
                layer_inpmb = inputs_shp[layer_name]
                if hasattr(layer_inpmb, 'shape'):
                    layer_shape = layer_inpmb.shape
                elif isinstance(layer_inpmb, (list, tuple)):
                    layer_shape = layer_inpmb
                else:
                    raise ValueError(dedent(f'''
                        Unusual type for the "{layer_name}" input's value: 
                            type(layer_inpmb) = {type(layer_inpmb)}
                        '''))
            elif layer_type == 'pnorm':
                assert isinstance(layer_input, str)
                layer_inpshp = shaper.shapes[layer_input]

                layer_dim = layer_opts['dim']
                layer_dimlst = [layer_dim] if isinstance(layer_dim, int) else layer_dim
                assert isinstance(layer_dimlst, (list, tuple))
                assert all(isinstance(dim_, int) for dim_ in layer_dimlst)

                layer_shape = list(layer_inpshp)[:]
                for i_dim in layer_dimlst:
                    layer_shape[i_dim] = 1
                layer_shape = tuple(layer_shape)
            elif layer_type in ('cat', 'stack'):
                assert isinstance(layer_input, list)
                layer_inpshplst = [shaper.shapes[layer_name_] for layer_name_ in layer_input]
                layer_dim = layer_opts['dim']
                assert isinstance(layer_dim, int)
                assert all(len(shp) == len(layer_inpshplst[0]) for shp in layer_inpshplst)
                shps_arr = np.array(layer_inpshplst)

                if layer_type == 'cat':
                    shps_arr[:, layer_dim] = shps_arr[:, layer_dim].sum()
                    assert np.all(shps_arr == shps_arr[:1, :])
                    layer_shape = tuple(shps_arr[0].astype(int).tolist())
                elif layer_type == 'stack':
                    assert np.all(shps_arr == shps_arr[:1, :])
                    layer_shape = shps_arr[0].astype(int).tolist()
                    layer_shape.insert(layer_dim, len(layer_inpshplst))
                else:
                    raise ValueError(f'layer type {layer_type} not defined')
            elif layer_type == 'reshape':
                layer_shape = shaper.eval(layer_expctdshp)
            elif layer_type == 'slice':
                assert isinstance(layer_input, str)
                layer_inpshp = shaper.shapes[layer_input]
                layer_dim = layer_opts['dim']
                assert isinstance(layer_dim, int)
                layer_start = layer_opts.get('start', None)
                assert isinstance(layer_start, (int, type(None)))
                layer_end = layer_opts.get('end', None)
                assert isinstance(layer_end, (int, type(None)))
                layer_step = layer_opts.get('step', None)
                assert isinstance(layer_step, (int, type(None)))

                i_inpdims = list(range(layer_inpshp[layer_dim]))
                i_outdims = i_inpdims[slice(layer_start, layer_end, layer_step)]

                layer_shape = list(layer_inpshp)
                layer_shape[layer_dim] = len(i_outdims)
            elif layer_type in tchfuncs:
                layer_shape = shaper.eval(layer_expctdshp)
                layer_inplst = layer_opts['inplst']
                layer_func = tchfuncs[layer_type]
                with FakeTensorMode():
                    layer_kwargs, layer_inputs2 = dict(), set()
                    for optn, optval in layer_opts.items():
                        if optn in ('type', 'inplst', 'output', 'round', 'enable', 'shape', 'invinputs'):
                            pass
                        elif isinstance(optval, str) and (optval in shaper.shapes):
                            layer_kwargs[optn] = torch.empty(*shaper.shapes[optval])
                            layer_inputs2.add(optval)
                        else:
                            layer_kwargs[optn] = optval
                    
                    # TODO: Fix the dtype checking issue. Maybe run a whole fake tensor sim instead?
                    if layer_type in ('one_hot', 'torch.nn.functional.one_hot'):
                        if 'input' in layer_kwargs:
                            layer_kwargs['input'] = layer_kwargs['input'].to(dtype=torch.long)
                    if layer_type in ('index_select', 'torch.index_select'):
                        if ('index' in layer_kwargs) and isinstance(layer_kwargs['index'], (list, tuple)):
                            layer_kwargs['index'] = torch.tensor(layer_kwargs['index'], dtype=torch.long)

                    assert layer_inputs2 == set(layer_inplst), dedent(f'''
                        There was a mismatch in input collection for the "{layer_name}" 
                        layer with the "{layer_type}" torch function type:
                            Missing inputs: {set(layer_inplst) - layer_inputs2}
                            Extra inputs: {layer_inputs2 - set(layer_inplst)}''')

                    try:
                        layer_mb = layer_func(**layer_kwargs)
                    except Exception as exc:
                        layer_kwargs2 = {lkw: 
                            f'torch.tensor(shape={lkv.shape}, device={lkv.device}, dtype={lkv.dtype})'
                            if torch.is_tensor(lkv) else lkv for lkw, lkv in layer_kwargs.items()}
                        exc.add_note(dedent(f'''
                            There was an error applying a torch function to a fake input 
                            for the purposes of output shape prediction:
                                layer name: {layer_name}
                                layer type: {layer_type}
                                arguments: {layer_kwargs2}'''))
                        raise exc
                    
                    assert torch.is_tensor(layer_mb), dedent(f'''
                        The output of the the "{layer_name}" layer with a torch 
                        function type "{layer_type} is not a tensor":
                            layer output type: {type(layer_mb)}
                            layer output: {layer_mb}''')
                    
                    layer_shape2 = tuple(layer_mb.shape)
                    assert tuple(layer_shape) == layer_shape2, dedent(f'''
                        There was a shape mismatch in the "{layer_name}" layer 
                        with a torch function type "{layer_type}":
                            Expected layer shape: {layer_shape}
                            Evaluated layer shape: {layer_shape2}''')
            else:
                raise ValueError(f'layer type {layer_type} undefined')

            shaper.append({layer_name: layer_expctdshp}, {layer_name: layer_shape})
            shaper.check()

        return shaper

    @torch.no_grad()
    def infer(self, data: dict, trn_spltidxs: torch.tensor, n_seeds: int, 
        n_mb: int, mb_order: int = 1, hash_data: any = None, pre_pp: list | tuple | None = None, 
        cache_path: str | None = None):
        """
        This method infers the shift and scale parameters from the
        training portion of the data.

        The `shift` layers are supposed to perform a baseline shift by subtracting
        the mean value over the training data. Similarly, the `scale` layers are
        supposed to divide the values by the standard deviation over the
        training data.

        This function returns nothing. The inferred parameters will be stored in
        the internal `pp_layers` dictionary.

        Args:
            data (dict): A mapping of the possible input layers to their entire
                data arrays. These arrays can be instances of `torch.tensor` or
                `np.ndarray` or similar data structures. The key point is that
                these data structures must support array indexing (see the
                `layer_mbvals_ = layer_arr[i_mb.ravel()]` line)

            trn_spltidxs (torch.tensor | np.ndarray): The training split
                indices. This must have a shape of `(n_seeds, n_trn)`.

            n_seeds (int): The number of independent runs with
                different RNG seeds.

            n_mb (int): The mini-batch size.

            mb_order (int): The mini-batch dimension index. should be either 0 or 1.
                This determines where `n_mb` is located in the input's shape:

                ```
                if mb_order == 0:
                    assert x.shape == (n_mb, n_seeds, n_channel, *x_spatial_dims)
                elif mb_order == 1:
                    assert x.shape == (n_seeds, n_mb, n_channel, *x_spatial_dims)
                else:
                    raise NotImplementedError
                ```
        """
        pp_layers = self.pp_layers
        pp_params = self.pp_params
        device, dtype = self.device, self.dtype

        # Checking to see if there is anything to infer
        need_infer = any((layer_opts['type'] in ('shift', 'scale', 'cdf'))
            for layer_name, layer_opts in pp_layers.items())
        
        if not need_infer:
            # There is no need to do anything else
            return

        # Checking the cache to see if we have already inferred this data
        cache_status, cache_key = self.load_cache(trn_spltidxs, n_seeds, hash_data, cache_path)
        assert cache_status in ('hit', 'miss')

        if cache_status == 'hit': 
            # Looks like we had a hit. There is nothing else needed to 
            # do since the parameters are already loaded into self.
            return

        assert mb_order in (0, 1)

        n_trn = trn_spltidxs.shape[mb_order]
        if mb_order == 1:
            assert trn_spltidxs.shape == (n_seeds, n_trn)
        elif mb_order == 0:
            assert trn_spltidxs.shape == (n_trn, n_seeds)
        else:
            raise ValueError(f'undefined mb_order = {mb_order}')

        # Example:
        #    data = {'m_chmprthst': np.randn(n_snr * n_t, n_chem, k_bins),
        #            'n_prthst':    np.randn(n_snr * n_t,      1, k_bins),
        #            'm_prthst':    np.randn(n_snr * n_t,      1, k_bins)}

        # Making sure the shift/scale layer parameters are cleared
        for layer_name, layer_opts in pp_layers.items():
            layer_type = layer_opts['type']
            if layer_type == 'shift':
                assert f'{layer_name}/value' not in pp_params, dedent(f'''
                    looks like the `value` for the "{layer_name}" layer is 
                    already specified.''')
                valth = torch.zeros(1, device=device, dtype=dtype)
                pp_params[f'{layer_name}/value'] = nn.Parameter(valth)
            elif layer_type == 'scale':
                assert f'{layer_name}/value' not in pp_params, dedent(f'''
                    looks like the `value` for the "{layer_name}" layer is 
                    already specified.''')
                valth = torch.ones(1, device=device, dtype=dtype)
                pp_params[f'{layer_name}/value'] = nn.Parameter(valth)
            elif layer_type == 'cdf':
                assert f'{layer_name}/quantiles' not in pp_params, dedent(f'''
                    looks like the `quantiles` for the "{layer_name}" layer is 
                    already specified.''')
                assert 'mb_order' not in layer_opts, dedent(f'''
                    looks like the `mb_order` for the "{layer_name}" layer is 
                    already specified."''')

                n_dimscdf = len(layer_opts['shape'])
                n_qnts = layer_opts['bins'] + 1
                domain_min, domain_max = torch.finfo(dtype).min / 2, torch.finfo(dtype).max / 2
                
                # Making a fake quantiles vector so that the forward call would not break
                qntlsth1 = torch.linspace(domain_min, domain_max, n_qnts + 2, device=device, dtype=dtype)
                qshp1 = [1 for _ in range(n_dimscdf)]
                qshp1[mb_order] = n_qnts + 2
                qntlsth2 = qntlsth1.reshape(*qshp1)

                pp_params[f'{layer_name}/quantiles'] = qntlsth2
                layer_opts['mb_order'] = mb_order
                
        # Performing sanity checks over the prior PP functions provided by the user.
        if (pre_pp is None):
            pre_pplst = None
        elif isinstance(pre_pp, (list, tuple)) and (len(pre_pp) == 0):
            pre_pplst = None
        elif isinstance(pre_pp, (list, tuple)) and (len(pre_pp) > 0):
            pre_pplst = list(pre_pp)
            for i_ppp, pp_pre in enumerate(pre_pplst):
                assert callable(pp_pre), dedent(f'''
                    The "{i_ppp}"-th pre-PP is not callable''')
        elif callable(pre_pp):
            pre_pplst = [pre_pp]
            assert callable(pre_pp), 'The provided pre-PP is not callable.'
        else:
            raise ValueError(f'undefined type for pre_pp={pre_pp}')

        #################################################
        #    Iterating Mini-batch-wise Over the Data    #
        #    to infer the shift or scale parameters     #
        #################################################
        n_ioiters = int(np.ceil(n_trn / n_mb))

        # Comments:
        #   1. The data is being forwarded in min-batches. That's why we need
        #      many rounds of data sweeps to infer all the shift/scale parameters.
        #   2. `n_totrnds` denotes the total number of rounds we have to sweep
        #      the entire data to infer all the parameter layers.
        n_totrnds = max(pp_layers[layer_name]['round'] for layer_name, layer_opts in pp_layers.items())

        for i_rnd in range(n_totrnds):
            layer2params = odict()
            # Note: `layer2params` will only contain the parametrics layers whose round is current.
            for i_ioiter in range(n_ioiters):
                i1 = i_ioiter * n_mb
                i2 = min(i1 + n_mb, n_trn)
                n_iomb = i2 - i1

                if mb_order == 1:
                    i_mb = trn_spltidxs[:, i1: i2]
                    assert i_mb.shape == (n_seeds, n_iomb)
                elif mb_order == 0:
                    i_mb = trn_spltidxs[i1: i2, :]
                    assert i_mb.shape == (n_iomb, n_seeds)
                else:
                    raise ValueError(f'undefined mb_order = {mb_order}')

                inputs_mb1 = dict()
                for layer_name, layer_data in data.items():
                    if layer_name not in pp_layers:
                        continue
                    layer_opts = pp_layers[layer_name]
                    layer_type = layer_opts['type']
                    if layer_type != 'input':
                        continue

                    if torch.is_tensor(layer_data):
                        layer_mbvals_ = layer_data[i_mb.ravel()]
                        assert layer_mbvals_.shape[0] == n_seeds * n_iomb
                    else:
                        i_mbrvld = i_mb.ravel()
                        assert i_mbrvld.shape == (n_seeds * n_iomb,)

                        i_mbunqs, i_mbinvrs, i_mbcnts, i_mbindx = torch_unique(i_mbrvld)
                        assert i_mbinvrs.shape == (n_seeds * n_iomb,)
                        i_mbunqsnp = i_mbunqs.detach().cpu().numpy()
                        i_mbinvrsnp = i_mbinvrs.detach().cpu().numpy()
                        assert i_mbinvrsnp.shape == (n_seeds * n_iomb,)
                        layer_mbvalsnp_ = layer_data[i_mbunqsnp][i_mbinvrsnp]
                        assert layer_mbvalsnp_.shape[0] == n_seeds * n_iomb
                        layer_mbvals_ = torch.tensor(layer_mbvalsnp_).to(device=device, dtype=dtype)
                        assert layer_mbvals_.shape[0] == n_seeds * n_iomb

                    layer_mbvals = layer_mbvals_.reshape(*i_mb.shape, *layer_mbvals_.shape[1:])
                    assert layer_mbvals.shape[:2] == i_mb.shape
                    inputs_mb1[layer_name] = layer_mbvals

                ################## Applying any prior pre-processors ##################
                inputs_mb2 = inputs_mb1
                if pre_pplst is not None:
                    i_ppp = 0
                    try:
                        for i_ppp, pp_pre in enumerate(pre_pplst):
                            inputs_mb2, lyr_shprpre = pp_pre.forward(inputs_mb2, full=False)
                    except Exception as exc:
                        exc.add_note(dedent('There was an error applying the ' +
                            f'"{i_ppp}"-th prior pre-processor.'))
                        raise exc

                ########## Performing the mini-batch forward transformation ###########
                with torch.no_grad():
                    layers_mb, layers_shaper = self.forward(inputs_mb2, full=True)
                
                ####### Inferring the minibatch specific part of the parameters ########
                for layer_name, layer_opts in pp_layers.items():
                    # Example:
                    #   layer_opts = {'type': 'input',
                    #              'shape': ['n_seeds', 'n_mb', 'k_bins+1', 'n_chem']}
                    #   layer_type = 'input'
                    #   layer_rnd = 0
                    layer_type = layer_opts['type']
                    layer_round = layer_opts['round']
                    need_infer = ((i_rnd == layer_round - 1) and
                        (layer_type in ('shift', 'scale', 'cdf')))
                    if not need_infer: continue

                    layer_mbvals = layers_mb[layer_name]
                    assert layer_mbvals.shape[:2] == i_mb.shape

                    # Obtaining the input data to this layer
                    layer_inpname = layer_opts['input']
                    layer_inpmbvals = layers_mb[layer_inpname]
                    assert layer_inpmbvals.shape[:2] == i_mb.shape

                    layer_params = layer2params.get(layer_name, None)
                    if layer_type == 'shift':
                        assert (layer_inpmbvals == layer_mbvals).all()
                        mb_valsmean = layer_inpmbvals.sum(dim=mb_order, keepdim=True) / n_trn

                        if mb_order == 1:
                            assert mb_valsmean.shape[:2] == (n_seeds, 1)
                        elif mb_order == 0:
                            assert mb_valsmean.shape[:2] == (1, n_seeds)
                        else:
                            raise ValueError(f'undefined mb_order = {mb_order}')

                        if layer_params is None:
                            layer2params[layer_name] = mb_valsmean
                        else:
                            layer_params += mb_valsmean
                    elif layer_type == 'scale':
                        assert (layer_inpmbvals == layer_mbvals).all()
                        layer_pnorm = layer_opts.get('pnorm', 2)
                        mb_valssqmean = layer_inpmbvals.abs().pow(layer_pnorm).sum(dim=mb_order, keepdim=True) / n_trn

                        if mb_order == 1:
                            assert mb_valssqmean.shape[:2] == (n_seeds, 1)
                        elif mb_order == 0:
                            assert mb_valssqmean.shape[:2] == (1, n_seeds)
                        else:
                            raise ValueError(f'undefined mb_order = {mb_order}')

                        if layer_params is None:
                            layer2params[layer_name] = mb_valssqmean
                        else:
                            layer_params += mb_valssqmean
                    elif layer_type == 'cdf':
                        if layer_params is None:
                            layer2params[layer_name] = [layer_inpmbvals]
                        else:
                            layer_params.append(layer_inpmbvals)
                    else:
                        raise ValueError(f'undefined layer type {layer_type}')

            ### Finishing the inference procedure and storing the parameters ###
            ###            in the main `pp_layers` dictionary                ###
            for layer_name, layer_params in layer2params.items():
                layer_opts = pp_layers[layer_name]
                layer_type = layer_opts['type']
                layer_dim = layer_opts['dim']
                with torch.no_grad():
                    if (layer_type in ('shift', 'scale')) and ((layer_dim is None) or (len(layer_dim) == 0)):
                        layer_parammean = layer_params
                    elif (layer_type in ('shift', 'scale')):
                        layer_parammean = layer_params.mean(dim=layer_dim, keepdim=True)
                    elif (layer_type == 'cdf'):
                        layer_alldata = torch.cat(layer_params, dim=mb_order)
                        n_qnts = layer_opts['bins'] + 1
                        domain_min, domain_max = layer_opts['domain']

                        q_cdfs = torch.linspace(0, 1, n_qnts, device=device, dtype=dtype)
                        assert q_cdfs.shape == (n_qnts,)

                        # Making sure all dimension indices are positive
                        cdf_dims1 = [layer_dim] if isinstance(layer_dim, int) else layer_dim
                        cdf_dims2 = [] if (cdf_dims1 is None) else cdf_dims1
                        cdf_dims3 = [i_dim % layer_alldata.ndim for i_dim in cdf_dims2]
                        cdf_dims4 = cdf_dims3 + ([] if (mb_order in cdf_dims3) else [mb_order])
                        cdf_dims5 = sorted(cdf_dims4)

                        # Since quantile cannot take a list of dim as input, we have to call our version of the func.
                        d_qtmp = mb_order
                        layer_qntls1 = torch_quantile(layer_alldata, q_cdfs, dim=cdf_dims5, keepdim=True, qdim=d_qtmp)
                        assert layer_qntls1.shape[d_qtmp] == n_qnts
                        assert layer_qntls1.shape[mb_order + 1] == 1

                        shape_qntls1 = list(layer_alldata.shape)
                        for i_dim in cdf_dims5:
                            shape_qntls1[i_dim] = 1
                        shape_qntls1.insert(d_qtmp, n_qnts)
                        assert layer_qntls1.shape == tuple(shape_qntls1), dedent(f'''
                            Something unexpected happened:
                                actual shape: {layer_qntls1.shape}
                                expected shape: {shape_qntls1}
                                cdf_dims5: {cdf_dims5}
                                data shape: {layer_alldata.shape}''')

                        assert (layer_qntls1.select(dim=d_qtmp, index=0) > domain_min).all(), dedent(f'''
                            The provided domain_min is higher than that of the data. 
                            Please make sure the specified domain_min is the lowest 
                            possible in the data domain:
                                data min: {layer_qntls1.select(dim=d_qtmp, index=0).min().item()}
                                domain_min: {domain_min}''')
                        assert (layer_qntls1.select(dim=d_qtmp, index=-1) < domain_max).all(), dedent(f'''
                            The provided domain_max is lower than that of the data. 
                            Please make sure the specified domain_max is the lowest 
                            possible in the data domain:
                                data max: {layer_qntls1.select(dim=d_qtmp, index=-1).max().item()}
                                domain_max: {domain_max}''')

                        # Inserting the domain min and max as extreme quantiles
                        flshp = list(layer_qntls1.shape)
                        flshp[d_qtmp] = 1

                        layer_qntls2 = torch.cat([
                            torch.full(flshp, domain_min, device=device, dtype=dtype),
                            layer_qntls1,
                            torch.full(flshp, domain_max, device=device, dtype=dtype)], dim=d_qtmp)
                        assert layer_qntls2.shape[d_qtmp] == n_qnts + 2

                        # Moving the `n_qnts` dimension to mb_order
                        layer_qntls = layer_qntls2.transpose(d_qtmp, mb_order + 1).squeeze(dim=d_qtmp)
                        assert layer_qntls.shape[mb_order] == n_qnts + 2
                    else:
                        raise ValueError(f'undefined layer type {layer_type}')
                
                if layer_type == 'shift':
                    pp_params[f'{layer_name}/value'] = nn.Parameter(layer_parammean.detach())
                elif layer_type == 'scale':
                    layer_eps = layer_opts.get('eps', 0.0)
                    layer_pnorm = layer_opts.get('pnorm', 2)
                    assert isinstance(layer_eps, (int, float))
                    layer_parammean2 = (layer_eps ** layer_pnorm) + layer_parammean
                    pp_params[f'{layer_name}/value'] = nn.Parameter(layer_parammean2.pow(1.0 / layer_pnorm).detach())
                elif layer_type == 'cdf':
                    pp_params[f'{layer_name}/quantiles'] = nn.Parameter(layer_qntls.detach())
                    layer_opts['mb_order'] = mb_order
                else:
                    raise ValueError(f'undefined layer type {layer_type}')
        
        # Let's save the object's state into cache for future use.
        if cache_status == 'miss':
            self.save_cache(cache_key, cache_path)

    def get_dims(self, data_dims: dict, types: list, n_dims: int = None, n_seeds: int = 0,
                 n_mb: int = 0, mb_order: int = 1):
        """
        Returns a dictionary of layer names to numerical dimension tuples. This
        method omits the preceding `(n_seeds, n_mb)` part of the shape tuple.
        This function also can check that the number of dimensions is fixed to
        be `n_dims`.

        Args:
            data_dims (dict): A dictionary mapping the input variable names
                (str) to non-batch (i.e., not the `n_seeds` and `n_mb`) shapes.

                Example:
                    ```
                    data_dims = {
                        'n_prthst': (1, k_bins),
                        'm_prthst': (1, k_bins),
                        'm_chmprthst': (n_chem, k_bins)
                    }
                    ```
            types (str | list): Either a string or a list/tuple of strings.

                Example:

                    types = 'input'

                    types = ['input', 'output', 'log']

            n_dims (int): The number of dimensions. Only used for assertion checks.

            n_seeds (int): The number of independent runs with different RNG seeds.
                For the purposes of this function, it can be a meaningless value
                such as zero, since it will be dropped from the resulting shapes
                anyway.

            n_mb (int): The mini-batch size. For the purposes of this function,
                it can be a meaningless value such as zero, since it will be
                dropped from the resulting shapes anyway.

            mb_order (int): The mini-batch dimension index. should be either 0 or 1.
                This determines where `n_mb` is located in the input's shape:

                ```
                if mb_order == 0:
                    assert x.shape == (n_mb, n_seeds, n_channel, *x_spatial_dims)
                elif mb_order == 1:
                    assert x.shape == (n_seeds, n_mb, n_channel, *x_spatial_dims)
                else:
                    raise NotImplementedError
                ```

        Returns:

            dims (dict): a dictionary mapping the layer names to the
                evaluated numerical dimension tuples.

                Example:

                    dims = {'09_out': (10, 32)}
        """
        types = [types] if isinstance(types, str) else types
        assert isinstance(types, (list, tuple))
        assert all(isinstance(layer_type, str) for layer_type in types)
        assert mb_order in (0, 1)

        if mb_order == 1:
            lshape = (n_seeds, n_mb)
        elif mb_order == 0:
            lshape = (n_mb, n_seeds)
        else:
            raise ValueError(f'undefined mb_order = {mb_order}')

        inputs_shp = {key: [*lshape, *spdims] for key, spdims in data_dims.items()}
        shaper = self.get_shaper(inputs_shp)

        dims = dict()
        for layer_name, layer_opts in self.pp_layers.items():
            layer_type = layer_opts['type']
            if layer_type not in types:
                continue
            layer_shape = shaper.shapes[layer_name]
            assert layer_shape[:2] == (n_seeds, n_mb), dedent(f'''
                The shape of the "{layer_name}" layer does not start with 
                a (`n_seeds`, `n_mb`) preamble.
                    layer_shape = {layer_shape}
                    n_seeds, n_mb = {(n_seeds, n_mb)}''')
            layer_dims = layer_shape[2:]
            if n_dims is not None:
                assert len(layer_dims) == n_dims, dedent(f'''
                    Too few or too many dimensions for the "{layer_name}" layer with
                    the following dimensions: 
                        layer_dims = "{layer_dims}"
                ''')

            if layer_name.startswith('__output__'):
                layer_input = layer_opts['input']
                dims[layer_input] = layer_dims
            elif (layer_type == 'output') and ('rename' in layer_opts):
                layer_rename = layer_opts['rename']
                if layer_rename in dims:
                    layer_dims2 = dims[layer_rename]
                    assert layer_dims == layer_dims2, dedent(f'''
                        There is a name and rename conflict due to an output 
                        layer renaming. The "{layer_rename}" layer defined  
                        conflicting dimensions:
                            internal output layer name: {layer_name}
                            the output layer rename: {layer_rename}
                            existing dim: {layer_dims2}
                            offending dim: {layer_dims}''')
                dims[layer_rename] = layer_dims
            else:
                dims[layer_name] = layer_dims

        return dims

    def save(self):
        """
        Returns the full state of the instance such that it would be
        serializable and completely able to re-create the object on
        its own.

        Returns:
            state (dict): A hierarchical dictionary with all the
                object's information and parameters including the
                layers graph, the parameters, and the hyper-parameter
                definitions.
        """
        layers_hie = deep2hie(self.pp_layers, maxdepth=1, sep='/')
        state_hie = {'layers': layers_hie,
                     'params': self.pp_params.state_dict(),
                     'hpdefs': self.pp_hpdefs}
        state = deep2hie(state_hie, maxdepth=1, sep='/')
        return state

    def load_params(self, params: dict, inplace: bool = False):
        """
        Loads the input params dict into self.pp_params.

        Parameters
        ----------
            params: (dict) The parameters dictionary.

            inplace: (bool) Whether to update the data in the 
                existing tensors, or replace them entirely by 
                the new dictionary elements.
        """
        pp_paramsdict = dict()
        for name, param in params.items():
            if torch.is_tensor(param):
                param_tch1 = param 
            elif isinstance(param, (np.ndarray, float, int)):
                param_tch1 = torch.tensor(param)
            else:
                raise ValueError(f'undefined param {param}')
            param_tch2 = param_tch1.to(device=self.device, dtype=self.dtype)
            pp_paramsdict[name] = param_tch2
        
        if inplace:
            for name in self.pp_params:
                assert name in pp_paramsdict, dedent(f'''
                    Some params are missing in the input
                        missing param name: {name}''')
            for name, param_new in pp_paramsdict.items():
                if name in self.pp_params:
                    param = self.pp_params[name]
                    assert param.shape == param_new.shape, dedent(f'''
                        Cannot load the new parameters in-place, 
                        since the shapes are not the same for some:
                            param name: {name}
                            param old shape: {param.shape}
                            param new shape: {param_new.shape}''')
                    param.data.copy_(param_new.data)
                else:
                    param_inp = params.get(name, None)
                    if torch.is_tensor(param_inp) and (param_inp.data_ptr() == param_new.data_ptr()):
                        param_newcp = torch.empty_like(param_new).copy_(param_new)
                    else:
                        param_newcp = param_new
                    self.pp_params[name] = nn.Parameter(param_newcp)
        else:
            pp_params = nn.ParameterDict(pp_paramsdict)
            self.pp_params.clear()
            self.pp_params.update(pp_params)
        
    def load(self, state: dict):
        """
        Loads the state dictionary into self

        Args:
            state (dict): The hierarchical dictionary with all the
                object's information and parameters including the
                layers graph, the parameters, and the hyper-parameter
                definitions.

        Returns:
            pp (PreProcessor): A new instance of the PreProcessor object
                with all the state variables embedded in it.
        """
        state_hie = hie2deep(state, maxdepth=1, sep='/')
        layers_hie = hie2deep(state_hie['layers'], maxdepth=1, sep='/')
        self.pp_layers.clear()
        self.pp_layers.update(layers_hie)
        self.load_params(state_hie['params'], inplace=False)
        self.pp_hpdefs.clear()
        self.pp_hpdefs.update(state_hie['hpdefs'])
        # Getting the output layer renamings (both forward and inverse).
        self.rnm_outfwd, self.rnm_outinv = self.get_outputrnmngs(self.pp_layers)
        # Getting the inversion topological orders, only including layers that create the inputs.
        self.pp_layers, self.invpp_ts = self.get_invppts(self.pp_layers, self.invfams)

    @classmethod
    def make(cls, state: dict, device: torch.device, dtype: torch.dtype, cdtype: torch.dtype):
        """
        Creates a new instance of the PreProcessor Object using a state
        dictionary.

        Args:
            state (dict): The hierarchical dictionary with all the
                object's information and parameters including the
                layers graph, the parameters, and the hyper-parameter
                definitions.

            device (torch.device | str): The pytorch device of the tensors.

            dtype (torch.dtype | str): The pytorch dtype of the tensors.

        Returns:
            pp (PreProcessor): A new instance of the PreProcessor object
                with all the state variables embedded in it.
        """
        pp = cls(None, device, dtype, cdtype)
        pp.load(state)
        return pp

    def load_cache(self, trn_spltidxs, n_seeds, hash_data, cache_path):
        """
        Checks whether the parameters were 
        """
        pp_state = self.save()

        # Getting the full state of the inputs and self
        input_hash = {'pp': pp_state, 
            'split': trn_spltidxs,
            'n_seeds': n_seeds,
            'data': hash_data}

        # Applying the hash function
        cache_key = hash_dict(input_hash)

        # Loading the cached state if it exist
        if cache_path is None:
            cache_path = f'{cache_dir}/pp.tar'
        status = 'miss'
        if exists(cache_path):
            with tarfile.open(cache_path, 'r') as archive:
                if cache_key in archive.getnames():
                    fileobj = archive.extractfile(cache_key)
                    state_cached = torch.load(fileobj, weights_only=True,
                        map_location=self.device)
                    self.load(state_cached)
                    status = 'hit'

        return status, cache_key

    def save_cache(self, cache_key: str, cache_path: str | None):
        assert isinstance(cache_key, str)
        pathlib.Path(cache_dir).mkdir(parents=True, exist_ok=True)
        if cache_path is None:
            cache_path = f'{cache_dir}/pp.tar'

        pp_state = self.save()
        stream = io.BytesIO()   
        torch.save(pp_state, stream)
        
        append_to_tar(cache_path, cache_key, stream)


def make_pp(name, hparams, config, device, dtype, cdtype):
    name2cfg = hie2deep(config, maxdepth=2)
    assert name in name2cfg, dedent(f'''
        The pre-processer module "{name}" is not defined. 
        Available modules are "{name2cfg.keys()}".
    ''')
    pp_cfg = hie2deep(name2cfg[name])
    if 'hparams' not in pp_cfg:
        pp_cfg['hparams'] = dict()
    pp_cfg['hparams'].update(hparams)
    pp_cnstrctr = pp_cfg.pop('class', PreProcessor)
    pp = pp_cnstrctr(pp_cfg, device=device, dtype=dtype, cdtype=cdtype)
    return pp

def get_lbldims(lblnn, ul_dims, n_seeds, n_mb, tch_device, tch_dtype, lblnnkws=None):
    # lblnn_paramreprs = {key:
    #     f'torch.tensor(shape={val.shape}, device={val.device}, dtype={val.dtype})'
    #     for key, val in lblnn.named_parameters().items()}

    # assert len(lblnn_paramreprs) == 0, dedent(f'''
    #     label nn should have no trainable parameters:
    #         label params: {lblnn_paramreprs}''')

    with FakeTensorMode(allow_non_fake_inputs=True):
        n_mbfk = n_mb
        ul_mbsfk = {u_node: torch.empty(n_seeds, n_mbfk, n_uchnls, n_ulen,
            device=tch_device, dtype=tch_dtype)
            for u_node, (n_uchnls, n_ulen) in ul_dims.items()}

        for u_node, (n_uchnls, n_ulen) in ul_dims.items():
            u_mb = ul_mbsfk[u_node]
            assert u_mb.shape == (n_seeds, n_mbfk, n_uchnls, n_ulen)

        if (lblnnkws is not None):
            assert len(lblnnkws.keys() & ul_mbsfk.keys()) == 0
            ul_mbsfk.update(lblnnkws)
        y_mbsfk = lblnn(**ul_mbsfk)
        assert isinstance(y_mbsfk, dict)

        y_dims = dict()
        for y_node, y_mb in y_mbsfk.items():
            assert torch.is_tensor(y_mb), dedent(f'''
                    One label is not a torch tensor:
                        y node: {y_node}
                        y type: {str(type(y_mb))}''')
            assert y_mb.ndim == 4, dedent(f'''
                One label node is not 2-dimensional:
                    y node: {y_node}
                    y shape: {y_mb.shape}''')
            n_ychnls, n_ylen = y_mb.shape[2:]
            assert y_mb.shape == (n_seeds, n_mbfk, n_ychnls, n_ylen)
            y_dims[y_node] = (n_ychnls, n_ylen)

    return y_dims

def hash_dict(hash_input):
    key2hash = dict()
    for key, val in hash_input.items():
        assert isinstance(key, (str, int, float, type(None)))
        
        if isinstance(val, dict):
            hashed_val = hash_dict(val)
        elif isinstance(val, (tuple, list)):
            hashed_val = hash_dict(dict(enumerate(val)))
        elif isinstance(val, (int, float, str)) or (val is None):
            hashed_val = val
        elif isinstance(val, complex):
            hashed_val = hash_dict({'real': val.real, 'imag': val.imag})
        elif isinstance(val, np.ndarray):
            hashed_val = hashlib.md5(val.data, 
                usedforsecurity=False).hexdigest()
        elif torch.is_tensor(val):
            val_np = val.detach().cpu().numpy()
            hashed_val = hashlib.md5(val_np.data, 
                usedforsecurity=False).hexdigest()
        else:
            raise ValueError(dedent(f'''
                Encontered a value with the "{type(val)}" type:
                    val = {val}
            '''))
    
        key2hash[key] = hashed_val

    dhash = hashlib.md5(usedforsecurity=False)
    encoded = json.dumps(key2hash, sort_keys=True).encode()
    dhash.update(encoded)

    return dhash.hexdigest()

def append_to_tar(tar_path, file_name, file_like_obj):
    with tarfile.open(tar_path, "a") as archive:
        assert file_name not in archive.getnames(), dedent(f'''
            File "{file_name}" already exists in "{tar_path}"
            ''')
        info = tarfile.TarInfo(name=file_name)
        file_like_obj.seek(0, io.SEEK_END)
        info.size = file_like_obj.tell()
        info.mtime = time.time()
        file_like_obj.seek(0, io.SEEK_SET)
        archive.addfile(info, file_like_obj)
    file_like_obj.close()

def get_spltcfgs(e_spltcfgsraw, eid, n_evlpnts, e_sigscale, is_condvae, lblnn_enctype):
    """
    A utility helper function to manage the various split definition 
    standards in the evaluation configs.

    The output is a standard dictionary config type where all keys and 
    values have been sanity checked.

    Parameters
    ----------
    e_spltcfgsraw: (dict | list) The raw splits config

    eid: (str) The evaluation id. mostly used for assertion message 
        information.

    n_evlpnts: (int | None) The raw number of evaluation points. Can 
        be null if the configs are specific enough.

    e_sigscale: (float) Default reconstruction noise injection
        scale factor.
    
    Returns
    -------
    e_spltcfgs: (dict) a deep split config dictionary with the 
        standard format.
    """
    
    assert isinstance(n_evlpnts, (int, type(None))), dedent(f'''
        The "{eid}/n" option must always be an integer. If you need to customize 
        split-specific number of evaluation points, please define "{eid}/splits"
        as a "split to number of points" mapping dictionary''')
    
    if isinstance(e_spltcfgsraw, (list, tuple)):
        assert isinstance(n_evlpnts, int), dedent(f'''
            Need "eval/n" to determine the number of samples 
            within the "{eid}" eval profile.''')
        e_spltcfgs1 = {e_split: n_evlpnts for e_split in e_spltcfgsraw}
    else:
        e_spltcfgs1 = e_spltcfgsraw
    assert isinstance(e_spltcfgs1, dict)
    e_spltcfgs2 = hie2deep(deep2hie(e_spltcfgs1))

    e_spltcfgs3 = dict()
    for e_split, e_spltcfg1 in e_spltcfgs2.items():
        # Extracting the user-provided config for the split
        if isinstance(e_spltcfg1, int):
            e_spltcfg2 = {'n': e_spltcfg1}
        else:
            e_spltcfg2 = e_spltcfg1
        
        # Processing the split config
        assert isinstance(e_spltcfg2, dict), dedent(f'''
            split config undefined -> {e_spltcfg2}
                eval id: {eid}
                eval split: {e_split}''')
        
        n_spltmb = e_spltcfg2.pop('n', None)
        assert isinstance(n_spltmb, int) and (n_spltmb > 0), dedent(f'''
            split size undefined -> {n_spltmb}
                eval id: {eid}
                eval split: {e_split}
                n: {n_spltmb}''')

        # Processing the split variant configs
        e_vrntcfgs1 = get_subdict(e_spltcfg2, 'vrnts', pop=True)
        e_vrntcfgs1 = dict() if (e_vrntcfgs1 is None) else e_vrntcfgs1
        assert isinstance(e_vrntcfgs1, dict), f'"{eid}/{e_split}/vrnts" must be dict'
        ver_spltcfg = 'old' if (len(e_vrntcfgs1) == 0) else 'new'

        # Getting the y sigma scale for vae label networks
        if is_condvae and (lblnn_enctype == 'vae') and (ver_spltcfg == 'old'):
            e_spltlblcfg = get_subdict(e_spltcfg2, 'lbl', pop=True)
            ey_sigscale = e_spltlblcfg.pop('sigscale', e_sigscale)
            assert (ey_sigscale is not None), dedent(f'''
                Since this is a conditional model with a vae label network
                the split label sigma scale must be specified:
                    Missing option: {eid}/splits/{e_split}/lbl/sigscale''')
            assert len(e_spltlblcfg) == 0, dedent(f'''
                extra options at {eid}/splits/{e_split}/lbl: 
                    {e_spltlblcfg}''')
        elif is_condvae and (lblnn_enctype == 'vae') and (ver_spltcfg == 'new'):
            e_spltlblcfg = get_subdict(e_spltcfg2, 'lbl', pop=True)
            ey_sigscale = e_spltlblcfg.pop('sigscale', None)
            assert (ey_sigscale is not None), dedent(f'''
                Since this is a conditional model with a vae label network
                the split label sigma scale must be specified:
                    Missing option: {eid}/splits/{e_split}/lbl/sigscale''')
            assert len(e_spltlblcfg) == 0, dedent(f'''
                extra options at {eid}/splits/{e_split}/lbl: 
                    {e_spltlblcfg}''')
        elif is_condvae and (lblnn_enctype == 'plain'):
            ey_sigscale = 0
        elif not is_condvae:
            ey_sigscale = None
        else:
            raise ValueError(f'undefined case')

        if ver_spltcfg == 'new':
            assert e_sigscale is None, dedent(f'''
                With complete variant specification, there should be no general 
                `sigscale` option provided for the entire evaluation profile, and 
                the latent and label `sigscale` values should be provided 
                specifically for each split and variant explicitly:
                    eval id: {eid}
                    offending extra option: eval/{eid}/sigscale''')

        emixup_cfg1 = get_subdict(e_spltcfg2, 'mixup', pop=True)
        emixup_cfg1 = dict() if (emixup_cfg1 is None) else emixup_cfg1
        assert isinstance(emixup_cfg1, dict), f'"{eid}/{e_split}/mixup" must be dict'
        if 'enable' not in emixup_cfg1:
            emixup_cfg1['enable'] = False

        emixup_cfg2 = dict(emixup_cfg1)
        ehas_mixup = emixup_cfg2.pop('enable')
        if ehas_mixup:
            emixup_alpha = emixup_cfg2.pop('alpha')
            assert emixup_alpha is not None, dedent(f'''
                With mixup enabled, the mixup alpha parameter must be specified:
                    eval id: {eid}
                    eval split: {e_split}''')
            assert isinstance(emixup_alpha, (float, int)) and (emixup_alpha >= 0), dedent(f'''
                With mixup enabled, the mixup alpha parameter must be non-negative:
                    eval id: {eid}
                    eval split: {e_split}''')
            emixup_nper = emixup_cfg2.pop('n', None)
            assert emixup_nper is not None, dedent(f'''
                With mixup enabled, the per-sample number of mixup interpolations
                must be specified:
                    eval id: {eid}
                    eval split: {e_split}''')
            assert isinstance(emixup_nper, int) and (emixup_nper > 0), dedent(f'''
                With mixup enabled, the per-sample number of mixup interpolations
                must be a positive integer:
                    eval id: {eid}
                    eval split: {e_split}''')
            emixup_flipmax = emixup_cfg2.pop('flipmax', None)
            assert emixup_flipmax is not None, dedent(f'''
                With mixup enabled, the flipmax option must be specified:
                    eval id: {eid}
                    eval split: {e_split}''')
            assert isinstance(emixup_flipmax, bool), dedent(f'''
                With mixup enabled, the flipmax option must be a boolean:
                    eval id: {eid}
                    eval split: {e_split}''')
            emixup_detspc = emixup_cfg2.pop('detspc', None)
            assert emixup_detspc is not None, dedent(f'''
                With mixup enabled, the detspc option must be specified:
                    eval id: {eid}
                    eval split: {e_split}''')
            assert isinstance(emixup_detspc, bool), dedent(f'''
                With mixup enabled, the detspc option must be a boolean:
                    eval id: {eid}
                    eval split: {e_split}''')
        else:
            assert len(emixup_cfg2) == 0, dedent(f'''
                With mixup disabled, no mixup options must be specified:
                    eval id: {eid}
                    eval split: {e_split}
                    unused options: {emixup_cfg2}''')
            emixup_cfg1.update({'alpha': None, 'n': None, 'flipmax': None, 'detspc': None})

        assert len(emixup_cfg2) == 0, dedent(f'''
            unused options in mixup config -> {emixup_cfg2}
                eval id: {eid}
                eval split: {e_split}''')

        assert len(e_spltcfg2) == 0, dedent(f'''
            unused options in split config -> {e_spltcfg2}
                eval id: {eid}
                eval split: {e_split}''')

        if len(e_vrntcfgs1) == 0:
            if e_split in ('train', 'test', 'snr'):
                e_vrntcfgs1.update({
                    'orig/type': 'orig', 'rcnst/type': 'rcnst', 
                    'rcnst/ltnt/type': 'orig', 'rcnst/ltnt/n': 1,
                    'rcnst/ltnt/sigscale': e_sigscale})
                if is_condvae:
                    e_vrntcfgs1['rcnst/lbl/type'] = 'orig'
            elif e_split in ('normal',):
                e_vrntcfgs1.update({
                    'genr/type': 'genr', 'genr/ltnt/type': 'normal',
                    'genr/ltnt/n': 1, 'genr/ltnt/sigscale': 1})
                if is_condvae:
                    e_vrntcfgs1['genr/lbl/type'] = 'iid'
                    e_vrntcfgs1['genr/lbl/split'] = 'train'
            else:
                raise ValueError(f'undefined split: {e_split}')

        # Sanity checking the quality of the variant configs
        e_vrntcfgs = hie2deep(deep2hie(e_vrntcfgs1), maxdepth=1)
        for e_vrnt, e_vrntcfg in e_vrntcfgs.items():
            e_vrntcfg2 = dict(e_vrntcfg)
            vrnt_type = e_vrntcfg2.pop('type', None)
            assert vrnt_type in ('orig', 'rcnst', 'genr'), dedent(f'''
                Bad variant type -> {vrnt_type}
                    eval id: {eid}
                    eval split: {e_split}
                    variant name: {e_vrnt}''')
            if vrnt_type in ('rcnst', 'genr'):
                z_typvrnt = e_vrntcfg2.pop('ltnt/type', None)
                assert z_typvrnt in ('orig', 'normal', 'knn'), dedent(f'''
                    Bad ltnt type -> {z_typvrnt}
                        eval id: {eid}
                        eval split: {e_split}
                        variant name: {e_vrnt}''')
                if vrnt_type == 'genr':
                    assert z_typvrnt in ('normal', 'knn'), dedent(f'''
                        Bad ltnt type for a genr variant -> {z_typvrnt}
                            eval id: {eid}
                            eval split: {e_split}
                            variant name: {e_vrnt}''')
                if e_split == 'normal':
                    assert z_typvrnt not in ('orig',), dedent(f'''
                        Bad ltnt type for a normal split -> {z_typvrnt}
                            eval id: {eid}
                            eval split: {e_split}
                            variant name: {e_vrnt}''')

                if z_typvrnt == 'normal':
                    mu_spltvrnt = e_vrntcfg2.pop('ltnt/mean/type', 'zero')
                    
                    assert mu_spltvrnt in (list(e_spltcfgs2) + ['zero']), dedent(f'''
                        The normal sampling mean should either be 'zero' or an 
                        evaluated split. "{mu_spltvrnt}" was not an evaluated split:
                            ltnt/mean: {mu_spltvrnt}
                            eval name: {eid}
                            split name: {e_split}
                            variant name: {e_vrnt}''')

                    sigma_spltvrnt = e_vrntcfg2.pop('ltnt/cov/type', 'identity')
                    
                    assert sigma_spltvrnt in (list(e_spltcfgs2) + ['identity']), dedent(f'''
                        The normal sampling cov should either be 'identity' or an 
                        evaluated split. "{sigma_spltvrnt}" was not an evaluated split:
                            ltnt/cov: {sigma_spltvrnt}
                            eval name: {eid}
                            split name: {e_split}
                            variant name: {e_vrnt}''')
                    
                    if sigma_spltvrnt != 'identity':
                        assert sigma_spltvrnt in e_spltcfgs2
                        # The subsampling size of the covariance matrix data just for 
                        # estimating the slicing directions. A null value will default to using the 
                        # entire covariance data for the related computations.
                        n_sampsig = e_vrntcfg2.pop('ltnt/cov/nsamp', None)
                        assert (n_sampsig is None) or (isinstance(n_sampsig, int) and (n_sampsig > 0)), dedent(f'''
                            invalid number of sub-samples for the cov matrix: {n_sampsig}
                                ltnt/cov/nsamp: {n_sampsig}
                                eval name: {eid}
                                split name: {e_split}
                                variant name: {e_vrnt}''')

                if z_typvrnt == 'knn':
                    assert is_condvae, dedent(f'''
                        cannot run label knn on a non-conditional vae:
                            eval name: {eid}
                            split name: {e_split}
                            variant name: {e_vrnt}''')
                    z_spltvrnt = e_vrntcfg2.pop('ltnt/split', None)
                    assert z_spltvrnt in e_spltcfgs2, dedent(f'''
                        label knn split will not exist -> {z_spltvrnt}
                            eval id: {eid}
                            eval split: {e_split}
                            variant name: {e_vrnt}''')
                    asgn_cfg = get_subdict(e_vrntcfg2, 'ltnt/asgn', pop=True)
                    assert len(asgn_cfg) > 0, dedent(f'''
                        assignment config is empty -> {asgn_cfg}
                            eval id: {eid}
                            eval split: {e_split}
                            variant name: {e_vrnt}''')

                e_vsigscl = e_vrntcfg2.pop('ltnt/sigscale', None)
                assert (isinstance(e_vsigscl, (int, float))
                    and (e_vsigscl >= 0)), dedent(f'''
                    Bad sigscale -> {e_vsigscl}
                        eval id: {eid}
                        eval split: {e_split}
                        variant name: {e_vrnt}''')
                n_rcns = e_vrntcfg2.pop('ltnt/n', None)
                assert isinstance(n_rcns, int), dedent(f'''
                    Bad number of latent z's per sample -> {n_rcns}
                        eval id: {eid}
                        eval split: {e_split}
                        variant name: {e_vrnt}''')
                if is_condvae:
                    y_typvrnt = e_vrntcfg2.pop('lbl/type', None)
                    assert y_typvrnt in ('orig', 'fixed', 'iid'), dedent(f'''
                        Bad label type -> {y_typvrnt}
                            eval id: {eid}
                            eval split: {e_split}
                            variant name: {e_vrnt}''')
                    if y_typvrnt == 'iid':
                        y_spltvrnt = e_vrntcfg2.pop('lbl/split', None)
                        assert y_spltvrnt in ('train', 'test'), dedent(f'''
                            Bad label split -> {y_spltvrnt}
                                eval id: {eid}
                                eval split: {e_split}
                                variant name: {e_vrnt}''')
                        assert y_spltvrnt in e_spltcfgs2, dedent(f'''
                            label split will not exist -> {y_spltvrnt}
                                eval id: {eid}
                                eval split: {e_split}
                                variant name: {e_vrnt}''')
                    if y_typvrnt == 'fixed':
                        y_nodevals = get_subdict(e_vrntcfg2, 'lbl', pop=True)
                else:
                    lblcfgs = get_subdict(e_vrntcfg2, 'lbl', pop=True)
                    assert len(lblcfgs) == 0, dedent(f'''
                        non-conditional models should not have lbl/* variant options:
                            eval id: {eid}
                            eval split: {e_split}
                            variant name: {e_vrnt}
                            lbl/* options: {lblcfgs}''')

            assert len(e_vrntcfg2) == 0, dedent(f'''
                Unused options:
                    eval id: {eid}
                    eval split: {e_split}
                    variant name: {e_vrnt}
                    extra options: {e_vrntcfg2}''')
        
        e_spltcfgs3[f'{e_split}/n'] = n_spltmb
        e_spltcfgs3[f'{e_split}/idxs'] = None
        e_spltcfgs3[f'{e_split}/iidxs'] = None
        e_spltcfgs3[f'{e_split}/vrnts'] = e_vrntcfgs
        e_spltcfgs3[f'{e_split}/lbl/sigscale'] = ey_sigscale
        e_spltcfgs3[f'{e_split}/mixup'] = emixup_cfg1
        

    e_spltcfgs = hie2deep(deep2hie(e_spltcfgs3), maxdepth=3)
    
    return e_spltcfgs

def torch_slice(tensor, dim, slc):
    return tensor[(dim if dim >= 0 else dim + tensor.dim()) * (slice(None), ) + (slc, )]

def get_nbcst(n1: int, n2: int):
    assert isinstance(n1, int), n1
    assert isinstance(n2, int), n2
    assert n1 >= 1, n1
    assert n2 >= 1, n2

    if n1 == n2:
        nbcst = n1
    elif n1 == 1:
        nbcst = n2
    elif n2 == 1:
        nbcst = n1
    else:
        nbcst = None

    return nbcst

def get_nn_sol(model, x, n_eval=None, mborder_in=1, mborder_out=None, model_kwargs=None):
    """
    Gets a model and evaluates it minibatch-wise on the tensor x.
    The minibatch size is capped at n_eval. The output will have the
    predicted potentials and the vector fields at them.

    Parameters
    ----------
    model: (nn.module) the batched neural network.

    x: (tuple | list | dict | torch.tensor) the evaluation points.

    n_eval: (int or None) the maximum mini-batch size. If None is
        given, the entire batch size will be used as `n_eval`.
    
    mborder_in: (int | tuple | list | dict) the dimension of the tensor 
        denoting the mini-batches for the input. If non-int values were 
        given, they should be compatible with the `x` inputs.

    mborder_out: (int | tuple | list | dict) the dimension of the tensor 
        denoting the mini-batches for the neural output. If non-int values 
        were given, they should be compatible with the neural outputs.

    Returns
    ----------
    xoutcats: (list | tuple | dict | torch.tensor) the evaluated 
        neural output.
    """
    
    if (mborder_out is None) and isinstance(mborder_in, int):
        mborder_out = mborder_in
    elif (mborder_out is None):
        raise ValueError(dedent('''
            mborder_out needs to be specified.
        '''))
    
    if isinstance(x, (tuple, list)):
        x_itrtr = tuple(enumerate(x))
        if isinstance(mborder_in, int):
            mborder_in = ((mborder_in,) * len(x))
    elif isinstance(x, dict):
        x_itrtr = tuple(x.items())
        if isinstance(mborder_in, int):
            mborder_in = {x_name: mborder_in for x_name in x}
    else:
        x_itrtr = ((0, x),)
        assert isinstance(mborder_in, int)
        mborder_in = (mborder_in,)
    
    n_xs = dict()
    for x_name, x_tnsr in x_itrtr:
        assert torch.is_tensor(x_tnsr), dedent(f'''
            The provided value for "{x_name}" is not a tensor.
        ''')
        x_mborder = mborder_in[x_name]
        assert isinstance(x_mborder, int)
        assert x_mborder < x_tnsr.ndim
        assert x_mborder >= -x_tnsr.ndim
        n_xs[x_name] = x_tnsr.shape[x_mborder]
    
    n_xunqs = list(set(n_xs.values()))
    assert len(n_xunqs) == 1, dedent(f'''
        The inferred mini-batch size for x members was not unique:
            n_xs = {n_xs}
        ''')
    n_x = n_xunqs[0]

    model_kwargs = dict() if model_kwargs is None else model_kwargs
    n_eval = n_x if n_eval is None else n_eval
    n_batches = int(np.ceil(n_x / n_eval))
    assert n_batches > 0

    nn_outslst = []
    empty_dict = dict()
    for i in range(n_batches):
        i1 = i * n_eval
        i2 = min(n_x, (i + 1) * n_eval)
        n_mb = i2 - i1
        
        if isinstance(x, (tuple, list)):
            x_mbargs = tuple(
                torch_slice(x_tnsr, mborder_in[x_name], slice(i1, i2)) 
                for x_name, x_tnsr in x_itrtr)
            x_mbkwargs = empty_dict
        elif isinstance(x, dict):
            x_mbargs = tuple()
            x_mbkwargs = {
                x_name: torch_slice(x_tnsr, mborder_in[x_name], slice(i1, i2)) 
                for x_name, x_tnsr in x.items()}
        else:
            x_mbargs = (torch_slice(x, mborder_in[0], slice(i1, i2)), )
            x_mbkwargs = empty_dict
        
        nnout_i = model(*x_mbargs, **x_mbkwargs, **model_kwargs)
        nn_outslst.append(nnout_i)
    
    nnout_0 = nn_outslst[0]
    if isinstance(nnout_0, (tuple, list)):
        n_xouts = len(nnout_0)

        if isinstance(mborder_out, int):
            mborder_out = ((mborder_out,) * n_xouts)
        assert isinstance(mborder_out, (list, tuple))
        assert len(mborder_out) == n_xouts

        for nnouts_i in nn_outslst:
            assert isinstance(nnouts_i, (tuple, list))
            assert len(nnouts_i) == n_xouts
            assert torch.is_tensor(nnouts_i)

        xoutcats = []
        for i_xout in range(n_xouts):
            catlst_i = [nnouts_i[i_xout] for nnouts_i in nn_outslst]
            xoutcat_i = torch.cat(catlst_i, dim=mborder_out[i_xout])
            xoutcats.append(xoutcat_i)
    elif isinstance(nnout_0, dict):
        xout_keys = set(nnout_0.keys())

        if isinstance(mborder_out, int):
            mborder_out = {x_name: mborder_out for x_name in nnout_0}
        
        assert isinstance(mborder_out, dict)
        assert set(mborder_out.keys()) == xout_keys

        for nnouts_i in nn_outslst:
            assert isinstance(nnouts_i, dict)
            assert set(nnouts_i.keys()) == xout_keys
            assert all(torch.is_tensor(x_tnsr) 
                for x_name, x_tnsr in nnouts_i.items())

        xoutcats = dict()
        for x_name in nnout_0:
            catlst_i = [nnouts_i[x_name] for nnouts_i in nn_outslst]
            xoutcats[x_name] = torch.cat(catlst_i, dim=mborder_out[x_name])
    elif torch.is_tensor(nnout_0):
        for nnouts_i in nn_outslst:
            assert torch.is_tensor(nnouts_i)
        assert isinstance(mborder_out, int)
        xoutcats = torch.cat(nn_outslst, dim=mborder_out)
    else:
        raise ValueError(dedent(f'''
            unknown output type for neural model: {type(nnout_0)}
        '''))
    
    return xoutcats

def unflatten_cfg(cfg_dict: dict, prefix: str, sep: str='/',
                  keydepth: int = 1, dictcls = None, pop_opts: bool = False):
    """
    Defines a simple hie2deep operation outlined below. See the input/output examples for
    quickly understanding what this function does.

        1. Narrows down the `cfg_dict` to keys starting with `prefix`.

            Example: ignore all keys not starting with `prefix = 'eval'`.

        2. Removes the prefix from the keys.

            Example: Turn 'eval/01_trn/lvl2a/n' into `'01_trn/lvl2a/n'
                     for the rest of the process.

        3. Splits the keys by `sep`.

            Example: Turn '01_trn/lvl2a/n' into ['01_trn', 'lvl2a', 'n'].

        4. Take the first `keydepth` elements to define the output keys.

            Example: The output key for '01_trn/lvl2a/n' when `keydepth=2`
                     would be `01_trn/lvl2a`.

        5. The rest of the splitted elements will go into the value dictionary.

            Example: All in all, {`eval/01_trn/lvl2a/n`: 100} with `keydepth=2` will
                     define {'01_trn/lvl2a': {'n': 100}}.

    Args:
        cfg_dict (dict): A hierarchical config dictionary.

            Example:

                `cfg_dict = {
                    'eval/01_trn/lvl2a/n': 100,
                    'eval/01_trn/lvl2a/frq': 200,
                    'eval/01_trn/lvl2b/n': 300,
                    'eval/02_tst/lvl2c/n': 400,
                    'eval/02_tst/lvl2c/frq': 500
                }`

        prefix (str): The string prefix of the relevant options.

            Example:
                `prefix = 'eval'`

        keydepth (int): The depth of the key level.

        pop_opts (bool, optional): Whether to pop out the options out of the
            input `cfg_dict` dictionary. Defaults to False.


    Returns:
        dict: The slightly deeper version of `cfg_dict` with the
            `prefix` being stripped.

        Example:

            `{'01_trn/lvl2a': {'n': 100, 'frq: 200},
              '01_trn/lvl2b': {'n': 300},
              '02_tst/lvl2c': {'n': 400, 'frq: 500}}`
    """

    if not pop_opts:
        cfg_dict = cfg_dict.copy()

    if dictcls is None:
        dictcls = odict if isinstance(cfg_dict, odict) else dict

    # The first `keydepth` elements define the keys.
    eid_list_dup = [sep.join(opt.split(sep)[1:keydepth+1]) for opt in cfg_dict
                    if opt.startswith(f'{prefix}{sep}')]
    eid_list = list(dictcls.fromkeys(eid_list_dup))
    evalcfgs = dictcls()
    for eid in eid_list:
        evalcfgs[eid] = dictcls()
        cfgopts = list(cfg_dict.keys())
        for opt in cfgopts:
            prfxopt = f'{prefix}{sep}{eid}{sep}'
            if opt.startswith(prfxopt):
                optn = opt[len(prfxopt):]
                optv = cfg_dict.pop(opt)
                evalcfgs[eid][optn] = optv

    return evalcfgs

def clear_figaxes(fig, axes):
    """
    Clears the figure and axes for re-plotting

    Parameters
    ----------
    fig: The matplotlib Figure.

    axes: (np.ndarray) The matplotlib axes as a numpy array.

    Returns
    ----------
    fig: (plt.Figure) The matplotlib Figure.

    axes: (np.ndarray) The matplotlib axes as a numpy array.
    """
    fig.legends, fig.stale = [], True
    for ax in axes.ravel():
        ax.set_xscale('linear')
        ax.set_yscale('linear')
        ax.clear()

    return fig, axes

def plot_scat(data: dict, fig=None, axes=None, caxes=None):
    """
    Generates a scatter plot

    Args:
        data (dict): The 2-d scattering data with a shape of `(n, 2)`.

        fig (plt.figure): The matplotlib figure.

        axes (plt.axes): The matplotlib axis.

        caxes (plt.axes): The matplotlib axis for colorbar.

    Returns:
        The matplotlib figure, axis, and color bar objects.
    """
    if fig is None:
        assert axes is None
        assert caxes is None
        fig, axes = plt.subplots(1, 1, figsize=(3.0, 2.5), dpi=72, squeeze=False)
        ax = axes[0, 0]
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        caxes = np.array([[cax]])
    else:
        assert axes is not None
        ax = axes[0, 0]
        cax = caxes[0, 0]

    colors = ['#001c7f', '#b1400d', '#12711c', '#8c0800', '#591e71',
              '#592f0d', '#a23582', '#3c3c3c', '#b8850a', '#006374']
    for i, (lbl, np_arr) in enumerate(data.items()):
        ax.scatter(np_arr[:, 0], np_arr[:, 1], label=lbl, c=colors[i], s=1, alpha=0.5)
    ax.set_xlabel(r'$MDS_1$')
    ax.set_ylabel(r'$MDS_2$')
    fig.legend()

    return fig, axes, caxes

def sciformatter(x, pos, precision=None, n_maxprez=None, force_exp=False):
    precision = 3 if precision is None else precision
    n_maxprez = 5 if n_maxprez is None else n_maxprez
    
    x_sign = '-' if x < 0 else ''
    x_abs = abs(x)
    assert n_maxprez <= 5
    if abs(x) == 0:
        x_str = '$0$'
    else:
        cross = r'\!\times\!\!'
        genfmtr = f'{{:.{precision}g}}'
        x_gstr = genfmtr.format(x_abs)
        if 'e' in x_gstr:
            x_mulstr, x_log10str = x_gstr.split('e')
            x_log10 = int(x_log10str)
        elif x_gstr.startswith('0.' + '0' * n_maxprez) or force_exp:
            x_log10 = math.floor(math.log10(x_abs))
            x_mul = x_abs / math.pow(10, x_log10)
            x_mulstr = genfmtr.format(x_mul)
        else:
            x_mulstr, x_log10 = x_gstr, 0
        
        x_mulflt = float(x_mulstr)
        x_mulstr3 = x_mulstr if (x_mulflt != 1) or ((x_mulflt == 1) and (x_log10 == 0)) else None
        x_log10str3 = None if (x_log10 == 0) else f'{x_log10}'
        x_str = f'${x_sign}'
        if (x_mulstr3 is not None):
            x_str += x_mulstr
        if (x_mulstr3 is not None) and (x_log10str3 is not None):
            x_str += cross
        if (x_log10str3 is not None):
            x_str += r'{\vphantom{\big|}10}'
            if x_log10 != 1:
                x_str += '^{' + f'{x_log10}' + '}'
            # x_str += r'{\vphantom{\Big|}10}'
            # if x_log10 != 1:
            #     x_str += '^{\hspace{-0.3em}' + f'{x_log10}' + '}'
        x_str += '$'
    return x_str

class SciFormatter:
    def __init__(self, **scikwa):
        self.scikwa = scikwa
    def __call__(self, x, pos):
        return sciformatter(x, pos, **self.scikwa)

class SciLabeler:
    def __init__(self, scikwa):
        self.scikwa = scikwa
    
    def __call__(self, ticks):
        scikwa = self.scikwa
        ticklabels1 = [sciformatter(tck, None, **scikwa) for tck in ticks]
        has_scinot = any('\\times' in tcklbl for tcklbl in ticklabels1)
        if has_scinot:
            scikwa2 = {**scikwa, 'force_exp': True}
            ticklabels2 = [sciformatter(tck, None, **scikwa2) for tck in ticks]
        else:
            ticklabels2 = ticklabels1
        
        return ticklabels2

def mark_rect(ax, x_boxdata, y_boxdata=None, w_boxfrac=0.1, h_boxfrac=None, 
    label=None, **rect_kwargs):
    """
    Adds a rectangle to the plot. No annotation is performed.

    Parameters
    ----------
    ax: (plt.axis) The matplotlib axis.

    x_boxdata: (float) the box center x value in data coordinates.

    y_boxdata: (float | None) the box center x value in data coordinates. 
        If not specified, a label for the corresponding line must be specified.

    w_boxfrac: (float) The width of box in axis fraction coordinates.

    h_boxfrac: (float | None) The height of the box in axis fraction coordinates. 
        If not specified, it defaults to `w_boxfrac`.

    label: (str | None) The label of the line from which the y value of the box must 
        be infered.

    rect_kwargs: (dict) The arguments passed to `plt.Rectangle`
    """

    h_boxfrac = h_boxfrac if (h_boxfrac is not None) else w_boxfrac

    rect_kwargs = dict(edgecolor='black', linewidth=2, facecolor='none')
    rect_kwargs.update(rect_kwargs)

    assert x_boxdata is not None

    # Computing the proper center y value if not specified.
    if y_boxdata is None:
        lbl2xdata = {line.get_label(): line.get_xdata() for line in ax.lines}
        lbl2ydata = {line.get_label(): line.get_ydata() for line in ax.lines}

        assert label in lbl2xdata, dedent(f'''
            When the box y is not specifed, the label must be specified''')

        x_ax, y_ax = lbl2xdata[label], lbl2ydata[label]
        y_boxdata = np.interp(x_boxdata, x_ax, y_ax)

    # Compiling the right axis transformation.
    #   1. First, the data coordinates must be transformed into the standard and global 
    #      display coordinates using `ax.transData`
    #   2. Then, the display coordinates must be translated to axis fractions using 
    #      `ax.transAxes.inverted()`.
    ax_trans1 = ax.transData + ax.transAxes.inverted()

    # The rectangle center xy in the axes fraction coords
    x_cntrax, y_cntrax = ax_trans1.transform((x_boxdata, y_boxdata))

    # The lower left corner of the rectangle in axis fraction coords.
    x1_rectax = max(0, x_cntrax - w_boxfrac / 2)
    y1_rectax = max(0, y_cntrax - h_boxfrac / 2)

    # The top right corner of the rectangle in axis fraction coords.
    x2_rectax = min(1, x1_rectax + w_boxfrac)
    y2_rectax = min(1, y1_rectax + h_boxfrac)

    # Translating the rectangle corner corrdinates back to data coords.
    ax_trans2 = ax_trans1.inverted()
    x1_rectdata, y1_rectdata = ax_trans2.transform((x1_rectax, y1_rectax))
    x2_rectdata, y2_rectdata = ax_trans2.transform((x2_rectax, y2_rectax))

    # Computing the width and height
    w_bbdata = x2_rectdata - x1_rectdata
    h_bbdata = y2_rectdata - y1_rectdata

    # Creating a rectangle patch
    rect = plt.Rectangle([x1_rectdata, y1_rectdata], width=w_bbdata, height=h_bbdata, 
        **rect_kwargs)

    # Adding the rectangle to the axis
    ax.add_patch(rect)

def print_axheader(ax, text, loc, pad=None, rotation=None, **kwargs):
    """
    Adds a top (column) or left (row) header to an axis.

    Parameters
    ----------
    ax: (plt.axis) a single matplotlib axis.

    text: (str) the header text.

    loc: (str) either `top` (for column headers) or `left` (for row headers).

    pad: (int | None) The amount of padding in points (point = 1/72 inch).

    rotation: (float | None) The text rotation in degrees.

    kwargs: (dict) all arguments passed to `ax.annotate`.
    """
    assert loc in ('top', 'left')
    pad = pad if (pad is not None) else {'top': 1, 'left': 5}[loc]
    rotation = rotation if (rotation is not None) else {'top': 0, 'left': 90}[loc]

    # Getting the ax coordinates in point units relative to the figure lower-left corner
    ax_bbpnts = ax.get_tightbbox().get_points().ravel()
    x_leftpnts, y_btmpnts, x_rightpnts, y_uppnts = ax_bbpnts
    assert y_uppnts >= y_btmpnts
    assert x_rightpnts >= x_leftpnts
    
    if loc == 'top':
        # Transforming the corrdinates to fraction units
        text_xypnts = ((x_leftpnts + x_rightpnts) / 2, y_uppnts + pad)
        x_midfrac, y_upfrac = ax.transAxes.inverted().transform(text_xypnts)
        
        ax.annotate(text, xy=(0.5, 1), xytext=(0.5, y_upfrac),
            xycoords="axes fraction", textcoords="axes fraction",
            ha="center", va="bottom", rotation=rotation, **kwargs)
    elif loc == 'left':
        # Transforming the corrdinates to fraction units
        text_xypnts = (x_leftpnts - pad, (y_btmpnts + y_uppnts) / 2)
        x_leftfrac, y_midfrac = ax.transAxes.inverted().transform(text_xypnts)

        ax.annotate(text, xy=(0, 0.5), xytext=(x_leftfrac, 0.5),
            xycoords="axes fraction", textcoords="axes fraction",
            ha="right", va="center", rotation=rotation, **kwargs)

        # ax.annotate(text, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
        #     xycoords=ax.yaxis.label, textcoords="offset points",
        #     ha="right", va="center", rotation=rotation, **kwargs)
    else:
        raise ValueError(f'undefined loc={loc}')

def tag_axis(ax: plt.axis, text: str, xy: tuple = (0, 1), fontsize: None | int = None, pad: float = 0.3):
    """
    Adds an april tag to a matplotlib axis.

    All x, y, width, and height values are fractions relative to the axis dimensions.

    Parameters
    ----------
    text: (str) The tag text.

    ax: (plt.axis) The matplotlib axis to insert the tag into.

    xy: (tuple) a pair of floats. the bottom left position of the box.

    pad: (float) The padding around the text relative to fontsize.
    """
    fontsize = ax.xaxis.label.get_fontsize() if (fontsize is None) else fontsize
    pad_x, pad_y = (pad, pad) if isinstance(pad, (int, float, type(None))) else pad

    # Finding the appropriate width and height
    txt_tmp = ax.text(xy[0], xy[1], text, transform=ax.transAxes, 
        ha='center', va='center', fontsize=fontsize)

    _, _, w_pxl, h_pxl = txt_tmp.get_tightbbox().bounds
    x_pxl, y_pxl = ax.transAxes.transform(xy)
    # The top-right corner x and y
    x3_pxl = x_pxl + w_pxl + pad_x * fontsize * 2
    y3_pxl = y_pxl + h_pxl + pad_y * fontsize * 2
    x2_pxl, y2_pxl = ax.transAxes.inverted().transform((x3_pxl, y3_pxl))
    width, height = x2_pxl - xy[0], y2_pxl - xy[1]
    txt_tmp.remove()

    # Creating the rectangle patch and text
    ax_topspine = ax.spines['top']
    ax_spinelw = ax_topspine.get_linewidth()
    ax_spineec = ax_topspine.get_edgecolor()
    ax_faceclr = ax.get_facecolor()

    tag_rect = plt.Rectangle(xy, width, height, transform=ax.transAxes, 
        facecolor=ax_faceclr, clip_on=False, edgecolor=ax_spineec, 
        linewidth=ax_spinelw)

    ax.add_patch(tag_rect)
    tag_rx, tag_ry = tag_rect.get_xy()
    tag_cx = tag_rx + 0.5 * tag_rect.get_width()
    tag_cy = tag_ry + 0.5 * tag_rect.get_height()

    ax.text(tag_cx, tag_cy, text, ha='center', va='center', 
        transform=ax.transAxes, fontsize=fontsize)

def plot_mpl(data: dict, fig: any = None, axes: np.ndarray | None = None, 
    mplopts: dict | None = None):
    """
    Plots a stacked bar plot historgram of the numbers or mass of the particles.

    Parameters
    ----------
    data: (dict) The plotting data. The data must be structured according 
        to the following hierarchy system:

            f'{ax_idx}/{lbl_name}/{var_name}': np.ndarray

        For example,

            data = {
                '0/SO4/xy': np.randn(n_pts1, 2),
                ...
                '0/H2O/xy': np.randn(n_pts2, 2),
                ...
                '24/H2O/xy': np.randn(n_pts3, 2),
            } 
    
        For the stacked bar plot and each label:

            * `x` must be provided as the x-axis bin edges. Assuming 
               that there are `k_bins` bins, we should have

                `assert x.shape == (k_bins + 1,)`

            * `y` must be provided as the y-axis values. Assuming that 
                there are `k_bins` bins, we should have

                `assert y.shape == (k_bins,)`
        
        For the scatter plots and each label, there are 3 possibilites 
        for the data keys:
        
            (I) A single `xy` can be provided with the shape of `(n_pnts, 2)`.

            (II) `x` and `y` can be provided, each with a shape of `(n_pnts,)` 
                or `(n_pnts,1)`.
            
            (III) `c` and `r` and `phi` can be provided:

                * `c` and `r` each with a shape of `(n_pnts, 2)`, and

                * `phi` with a shape of `(n_pnts, 1)` or `(n_pnts,)`.

                In this case a set of ellipse blobs centered at `c` with a radii 
                of `r` rotated with `phi` angles in radians will be shown.

    fig: The matplotlib Figure. If `None` is provided, a new figure will
        be created.

    axes: (np.ndarray) The matplotlib axes as a numpy array. It must have a
        shape of `(n_figrows, n_figcols)`.

    Returns
    ----------
    fig: (plt.Figure) The matplotlib Figure.

    axes: (np.ndarray) The matplotlib axes as a numpy array.
    """
    mplopts = dict() if mplopts is None else mplopts
    mplopts, data = mplopts.copy(), data.copy()

    plot_type = mplopts.pop('type')

    colorspec = get_subdict(mplopts, prefix='colorspec', pop=True)
    colors = get_subdict(mplopts, prefix='colors', pop=True)
    pltstyle = mplopts.pop('style', 'default')
    xlim = mplopts.pop('xlim', None)
    ylim = mplopts.pop('ylim', None)
    clim = mplopts.pop('clim', None)
    xbound = mplopts.pop('xbound', None)
    ybound = mplopts.pop('ybound', None)
    cbound = mplopts.pop('cbound', None)
    xscale = mplopts.pop('xscale', None)
    yscale = mplopts.pop('yscale', None)
    cscale = mplopts.pop('cscale', None)
    xticklabels = mplopts.pop('xticklabels', None)
    yticklabels = mplopts.pop('yticklabels', None)
    cticklabels = mplopts.pop('cticklabels', None)
    xsc_kwa = get_subdict(mplopts, 'xscale', pop=True)
    ysc_kwa = get_subdict(mplopts, 'yscale', pop=True)
    csc_kwa = get_subdict(mplopts, 'cscale', pop=True)
    xtl_kwa = get_subdict(mplopts, 'xticklabeler', pop=True)
    ytl_kwa = get_subdict(mplopts, 'yticklabeler', pop=True)
    ctl_kwa = get_subdict(mplopts, 'cticklabeler', pop=True)
    leg_kwa = get_subdict(mplopts, 'leg', pop=True)
    pltrccfg = get_subdict(mplopts, 'rc_context', pop=True)
    leglbls = get_subdict(mplopts, 'leglbls', pop=True)
    sbp_kwa = get_subdict(mplopts, 'plt.subplots', pop=True)
    xtf_kwa = get_subdict(mplopts, 'xtickformatter', pop=True)
    ytf_kwa = get_subdict(mplopts, 'ytickformatter', pop=True)
    ctf_kwa = get_subdict(mplopts, 'ctickformatter', pop=True)
    lay_kwa = get_subdict(mplopts, 'layout', pop=True)
    axb_kwa = get_subdict(mplopts, 'ax.bar', pop=True)
    axs_kwa = get_subdict(mplopts, 'ax.scatter', pop=True)
    axh_kwa = get_subdict(mplopts, 'ax.hist', pop=True)
    axp_kwa = get_subdict(mplopts, 'ax.plot', pop=True)
    axc_kwa = get_subdict(mplopts, 'ax.pcolormesh', pop=True)
    xlblkwa = get_subdict(mplopts, 'xlbl', pop=True)
    ylblkwa = get_subdict(mplopts, 'ylbl', pop=True)
    clblkwa = get_subdict(mplopts, 'clbl', pop=True)
    xtckkwa = get_subdict(mplopts, 'xticks', pop=True)
    ytckkwa = get_subdict(mplopts, 'yticks', pop=True)
    ctckkwa = get_subdict(mplopts, 'cticks', pop=True)
    tckprms = get_subdict(mplopts, 'tckprms', pop=True)
    xtckprms = get_subdict(mplopts, 'xtckprms', pop=True)
    ytckprms = get_subdict(mplopts, 'ytckprms', pop=True)
    ctckprms = get_subdict(mplopts, 'ctckprms', pop=True)
    tagkwa = get_subdict(mplopts, 'tag', pop=True)
    share_xlbl = xlblkwa.pop('share', True)
    assert share_xlbl in (True, False)
    share_ylbl = ylblkwa.pop('share', True)
    assert share_ylbl in (True, False)

    assert ('labels' not in xtckkwa) or (xticklabels is None), dedent(f'''
        I got x tick labels by both `xticks/labels` and `xticklabels`. 
        At most one of them should be provided:
            xtick options: {xtckkwa}
            xticklabels: {xticklabels}''')
    assert ('labels' not in ytckkwa) or (yticklabels is None), dedent(f'''
        I got y tick labels by both `yticks/labels` and `yticklabels`. 
        At most one of them should be provided:
            ytick options: {ytckkwa}
            yticklabels: {yticklabels}''')

    xscale_type = xsc_kwa.pop('type', None)
    xscale_value = xsc_kwa.pop('value', None)
    xscales = [scl for scl in [xscale, xscale_type, xscale_value] if scl is not None]
    assert len(xscales) == 1, dedent(f'''
        Exactly one of the following three must be provided:
            xscale: {xscale}
            xscale/type: {xscale_type}
            xscale/value: {xscale_value}''')
    xsc_kwa['value'] = xscales[0]

    yscale_type = ysc_kwa.pop('type', None)
    yscale_value = ysc_kwa.pop('value', None)
    yscales = [scl for scl in [yscale, yscale_type, yscale_value] if scl is not None]
    assert len(yscales) == 1, dedent(f'''
        Exactly one of the following three must be provided:
            yscale: {yscale}
            yscale/type: {yscale_type}
            yscale/value: {yscale_value}''')
    ysc_kwa['value'] = yscales[0]

    xticklabels = xtckkwa.pop('labels', xticklabels)
    yticklabels = ytckkwa.pop('labels', yticklabels)

    assert 'axis' not in xtckprms, 'axis should not be specified in xtckprms'
    assert 'axis' not in ytckprms, 'axis should not be specified in ytckprms'

    with plt.style.context(pltstyle) as psc, plt.rc_context(pltrccfg) as pltrcctx:
        if fig is None:
            assert axes is None
            n_figrows, n_figcols = sbp_kwa['nrows'], sbp_kwa['ncols']
            if 'axsize' in sbp_kwa:
                assert 'figsize' not in sbp_kwa, 'over-specified both axsize and figsize'
                (ax_w, ax_h) = sbp_kwa.pop('axsize')
                sbp_kwa['figsize'] = (ax_w * n_figcols, ax_h * n_figrows)
            
            # Taking out any axis sharing group formulas
            grpng_axn2frml = dict()
            for xory, share_xory in [('x', 'sharex'), ('y', 'sharey')]:
                if share_xory in sbp_kwa:
                    grpng_frml = sbp_kwa[share_xory]
                    if not (isinstance(grpng_frml, (str, bool)) and 
                        (grpng_frml in (True, False, 'none', 'all', 'row', 'col'))):

                        grpng_frml = sbp_kwa.pop(share_xory)
                        assert isinstance(grpng_frml, str), dedent(f'''
                            The share{xory} should either be one of 
                            plt.subplots/share{xory} options, or a 
                            string grouping formula:
                                share{xory}: {grpng_frml}''')
                        grpng_axn2frml[xory] = grpng_frml

            fig, axes = plt.subplots(**sbp_kwa)
            axes = np.array(axes).reshape(n_figrows, n_figcols)
            
            # Applying the axis sharing group formulas
            for xory, grpng_frml in grpng_axn2frml.items():
                grp_id2axs = defaultdict(list)
                for i_row in range(n_figrows):
                    for i_col in range(n_figcols):
                        try:
                            grp_id = eval_formula(grpng_frml, {'i_row': i_row, 'i_col': i_col}, catch=False)
                            assert isinstance(grp_id, typing.Hashable), f'grp id {grp_id} is not hashable'
                        except Exception as exc:
                            exc.add_note(dedent(f'''
                                This error happened while evaluating the 
                                "{xory}" axes sharing group ids:
                                    grp formula: {grpng_frml}
                                    option: share{xory}
                                    i_row: {i_row}
                                    i_col: {i_col}'''))
                            raise exc
                        grp_id2axs[grp_id].append(axes[i_row, i_col])
                for grp_id, grp_axs in grp_id2axs.items():
                    for ax in grp_axs[1:]:
                        if xory == 'x':
                            ax.sharex(grp_axs[0])
                        elif xory == 'y':
                            ax.sharey(grp_axs[0])
                        else:
                            raise ValueError(f'undefined x/y={xory}')
            
        if isinstance(axes, np.ndarray):
            n_figrows, n_figcols = axes.shape
        else:
            assert isinstance(axes, type(plt.gca()))
            axes = np.array([[axes]])
            n_figrows, n_figcols = 1, 1
        axes1d = axes.ravel()
        n_ax = n_figrows * n_figcols

        assert isinstance(data, dict)

        # Phase 2: Prepping the colors and labels
        if (colors is None) or (len(colors) == 0):
            color_pallst = sns.color_palette('bright', 10)
            lbl2color = dict()
        elif isinstance(colors, (tuple, list)):
            color_pallst = colors
            lbl2color = dict()
        else:
            assert isinstance(colors, dict)
            color_pallst = sns.color_palette('bright', 10)
            lbl2color = dict(colors)
        color_cycler = cycle(color_pallst)

        xtickformatter, ytickformatter, ctickformatter = None, None, None
        for axisname, tckfmttrkwa in [('x', xtf_kwa), ('y', ytf_kwa), ('c', ctf_kwa)]:
            if len(tckfmttrkwa) == 0:
                continue
            tickfmttype = tckfmttrkwa.pop('type')
            if tickfmttype == 'eng':
                engfmttr_ = EngFormatter(unit=tckfmttrkwa.pop('unit', ''),
                    sep=tckfmttrkwa.pop('sep', ' '))
                if 'mul' in tckfmttrkwa:
                    multplr = tckfmttrkwa.pop('mul')
                    def fmttr(x, pos, mul=multplr):
                        return engfmttr_(x * mul)
                else:
                    fmttr = engfmttr_
            elif tickfmttype == 'scalar':
                fmttr = ScalarFormatter(**tckfmttrkwa)
                tckfmttrkwa.clear()
            elif tickfmttype == 'sci':
                fmttr = FuncFormatter(SciFormatter(**tckfmttrkwa))
                tckfmttrkwa.clear()
            else:
                raise ValueError(f'undefined tickformatter type {tickfmttype}')
            if axisname == 'x':
                xtickformatter = fmttr
            if axisname == 'y':
                ytickformatter = fmttr
            if axisname == 'c':
                ctickformatter = fmttr

        xticklabeler, yticklabeler, cticklabeler = None, None, None
        for axisname, tcklblrkwa in [('x', xtl_kwa), ('y', ytl_kwa), ('c', ctl_kwa)]:
            if len(tcklblrkwa) == 0:
                continue
            ticklblrtype = tcklblrkwa.pop('type')
            if ticklblrtype == 'sci':
                ticklabeler = SciLabeler(tcklblrkwa)
            else:
                raise ValueError(f'undefined ticklabeler type {ticklblrtype}')
            if axisname == 'x':
                xticklabeler = ticklabeler
            if axisname == 'y':
                yticklabeler = ticklabeler
            if axisname == 'c':
                cticklabeler = ticklabeler

        if plot_type == 'bar':
            axbar_method = axb_kwa.pop('method', 'fill_between')
            assert axbar_method in ('fill_between', 'bar')
            axbar_bottom = axb_kwa.pop('bottom', None)
            assert (axbar_bottom is None) or isinstance(axbar_bottom, (float, int))
            zero_axb = 0.0 if (axbar_bottom is None) else float(axbar_bottom)
            do_stackbars = axb_kwa.pop('stack', True)
            if not do_stackbars:
                axb_kwa.pop('edgecolor')
        
        # Label-specific keyword arguments
        axb_lblkwas = defaultdict(dict)
        axs_lblkwas = defaultdict(dict)
        axh_lblkwas = defaultdict(dict)
        axp_lblkwas = defaultdict(dict)
        axc_lblkwas = defaultdict(dict)
        for plt_kwa, plt_lblkwas in [(axb_kwa, axb_lblkwas), 
            (axs_kwa, axs_lblkwas), (axh_kwa, axh_lblkwas),
            (axp_kwa, axp_lblkwas), (axc_kwa, axc_lblkwas)]:
            for key in list(plt_kwa):
                key_splt = key.split('/')
                if len(key_splt) >= 2:
                    optn = key_splt[-1]
                    lbl = '/'.join(key_splt[:-1])
                    plt_lblkwas[lbl][optn] = plt_kwa.pop(key)

        ax, cax = None, None
        for ax_idx in range(n_ax):
            i_row, i_col = ax_idx // n_figcols, ax_idx % n_figcols
            # The data dictionary for this particular ax
            ax_data = get_subdict(data, prefix=f'{ax_idx}', sep=':', pop=True)
            
            # No data was provided for this ax, so we will skip this axis
            if len(ax_data) == 0:
                continue

            ax = axes1d[ax_idx]
            # Example: 
            #   ax_data = {
            #       'lbl_1/xy': np.randn(n_pts, 2), 
            #        ..., 
            #       'lbl_N/xy': np.randn(n_pts, 2)}

            # Phase 1: prepping the data
            ax_datadeep = hie2deep(ax_data, sep=':', maxdepth=1)
            if plot_type == 'bar':            
                yax_cumsum, is_ycsnone = axbar_bottom, True
                for lblname, lbl_datadict in ax_datadeep.items():
                    datakeys_set = set(lbl_datadict)

                    if lblname not in lbl2color:
                        lbl2color[lblname] = next(color_cycler)
                    color_lbl = colorspec.get(lbl2color[lblname], lbl2color[lblname])

                    if (datakeys_set == {'x', 'y'}) or (datakeys_set == {'x', 'y/mean', 'y/low', 'y/high'}):
                        if (datakeys_set == {'x', 'y'}):
                            y_axlbl = lbl_datadict['y']
                            assert isinstance(y_axlbl, np.ndarray)
                            draw_ciband, y_axlbllow, y_axlblhigh = False, None, None
                        elif (datakeys_set == {'x', 'y/mean', 'y/low', 'y/high'}):
                            draw_ciband = True
                            assert not do_stackbars, 'cannot stack bars when ci bands are given!'

                            y_axlbl = lbl_datadict['y/mean']
                            assert isinstance(y_axlbl, np.ndarray)

                            y_axlbllow = lbl_datadict['y/low']
                            assert isinstance(y_axlbllow, np.ndarray)

                            y_axlblhigh = lbl_datadict['y/high']
                            assert isinstance(y_axlblhigh, np.ndarray)
                        else:
                            raise ValueError(f'odd keys "{datakeys_set}" for "{lblname}"')
                        
                        k_bins = y_axlbl.shape[0]
                        assert y_axlbl.shape == (k_bins,)

                        x_axlbl = lbl_datadict['x']
                        
                        assert isinstance(x_axlbl, np.ndarray)
                        assert x_axlbl.shape == (k_bins + 1,)

                        x_diff = np.diff(x_axlbl)
                        assert x_diff.shape == (k_bins,)

                        x_aug = np.array([x_axlbl[0]] + x_axlbl.tolist())
                        assert x_aug.shape == (k_bins + 2,)
                    else:
                        raise ValueError(f'odd keys "{datakeys_set}" for "{lblname}"')

                    # Label-specific options
                    color_kwa = (dict(color=color_lbl) if do_stackbars else 
                        dict(color='none', edgecolor=color_lbl))
                    axb_lblkwa = axb_lblkwas.get(lblname, None)
                    axb_lblkwa = axb_kwa if axb_lblkwa is None else {**axb_kwa, **axb_lblkwa}

                    if axbar_method == 'bar':
                        # Note:
                        #     Due to a bug in matplotlib, plotting stacked bar plots with many
                        #     bars and overlaps are slow using the `ax.bar method`. The 
                        #     `ax.fill_between` method is much more efficient for this purpose. 
                        #     Reference:
                        #         https://github.com/matplotlib/matplotlib/issues/15411
                        ax.bar(x_axlbl[:-1], y_axlbl, width=x_diff, label=lblname,
                            bottom=yax_cumsum, **color_kwa, **axb_lblkwa)

                        if draw_ciband:
                            raise NotImplementedError

                        # Updating the y-cumsum
                        if do_stackbars:
                            yax_cumsum = y_axlbl if is_ycsnone else yax_cumsum + y_axlbl 
                            is_ycsnone = False
                    elif axbar_method == 'fill_between':
                        y_low = np.full_like(y_axlbl, zero_axb) if is_ycsnone else yax_cumsum
                        y_high = y_axlbl if is_ycsnone else y_low + y_axlbl

                        ylow_aug = np.array([zero_axb] + y_low.tolist() + [zero_axb])
                        yhigh_aug = np.array([zero_axb] + y_high.tolist() + [zero_axb])

                        ax.fill_between(x_aug, ylow_aug, yhigh_aug, step='post', 
                            label=lblname, **color_kwa, **axb_lblkwa)
                        
                        if draw_ciband:
                            assert not do_stackbars

                            ylow_ciaug = np.array([zero_axb] + y_axlbllow.tolist() + [zero_axb])
                            yhigh_ciaug = np.array([zero_axb] + y_axlblhigh.tolist() + [zero_axb])
                            color_cikwa = dict(color=color_lbl, alpha=0.3, edgecolor='none') 
                            
                            ax.fill_between(x_aug, ylow_ciaug, yhigh_ciaug, step='post', 
                                label=lblname, **color_cikwa, **axb_lblkwa)

                        # Updating the y-cumsum
                        if do_stackbars:
                            yax_cumsum, is_ycsnone = y_high, False
                    else:
                        raise ValueError(f'undefined ax.bar method {axbar_method}')
            elif plot_type == 'scatter':
                for lblname, lbl_datadict in ax_datadeep.items():
                    lbl_datadict = dict(lbl_datadict)
                    if lblname not in lbl2color:
                        lbl2color[lblname] = next(color_cycler)
                    color_lbl = colorspec.get(lbl2color[lblname], lbl2color[lblname])
                    axs_lblkwa = axs_lblkwas.get(lblname, None)
                    axs_lblkwa = axs_kwa if axs_lblkwa is None else {**axs_kwa, **axs_lblkwa}

                    assert len(lbl_datadict) > 0 , f'empty data for scatter of "{lblname}" label'

                    ############################# Plotting Dot Points #############################
                    has_dotpnts = False
                    if ('xy' in lbl_datadict) or ('pnts' in lbl_datadict):
                        has_dotpnts = True
                        key_xy = 'xy' if ('xy' in lbl_datadict) else 'pnts'
                        xy_axlbl = lbl_datadict.pop(key_xy)
                        assert isinstance(xy_axlbl, np.ndarray)
                        n_pts = xy_axlbl.shape[0]
                        assert xy_axlbl.shape == (n_pts, 2)

                        x_axlbl = xy_axlbl[:, 0]
                        assert x_axlbl.shape == (n_pts,)

                        y_axlbl = xy_axlbl[:, 1]
                        assert y_axlbl.shape == (n_pts,)
                    elif ('x' in lbl_datadict) and ('y' in lbl_datadict):
                        has_dotpnts = True
                        x_axlbl = lbl_datadict.pop('x')
                        assert isinstance(x_axlbl, np.ndarray)
                        n_pts = x_axlbl.shape[0]
                        assert x_axlbl.shape == (n_pts,)

                        y_axlbl = lbl_datadict.pop('y')
                        assert isinstance(y_axlbl, np.ndarray)
                        assert y_axlbl.shape == (n_pts,)
                    else:
                        has_dotpnts = False
                    
                    if has_dotpnts:
                        ax.scatter(x_axlbl, y_axlbl, label=lblname, color=color_lbl, **axs_lblkwa)
                    
                    ############################### Plotting Blobs ################################
                    has_blobs = False
                    if all(len(lbl_datadict.keys() & s) == 1 
                        for s in [{'c', 'mu'}, {'sig', 'r'}, {'phi'}]):
                        has_blobs = True
                        c_key, = tuple(lbl_datadict.keys() & {'c', 'mu'})
                        c_axlbl = lbl_datadict.pop(c_key)
                        assert isinstance(c_axlbl, np.ndarray)
                        n_pts = c_axlbl.shape[0]
                        assert c_axlbl.shape == (n_pts, 2)

                        r_key, = tuple(lbl_datadict.keys() & {'sig', 'r'})
                        r_axlbl_ = lbl_datadict.pop(r_key)
                        assert isinstance(r_axlbl_, np.ndarray)
                        if r_axlbl_.shape == (n_pts,):
                            r_axlbl = r_axlbl_[:, None][:, [0, 0]]
                            assert r_axlbl.shape == (n_pts, 2)
                        elif r_axlbl_.shape == (n_pts, 1):
                            r_axlbl = r_axlbl_[:, [0, 0]]
                            assert r_axlbl.shape == (n_pts, 2)
                        elif r_axlbl_.shape == (n_pts, 2):
                            r_axlbl = r_axlbl_
                            assert r_axlbl.shape == (n_pts, 2)
                        else:
                            raise ValueError(f'odd shape {r_axlbl_.shape} for "{lblname}/r"')
                        
                        # These are radian angles
                        phi_axlbl_radian_ = lbl_datadict.pop('phi')
                        assert isinstance(phi_axlbl_radian_, np.ndarray)
                        phi_axlbl_radian = phi_axlbl_radian_.ravel()
                        assert phi_axlbl_radian.shape == (n_pts,)

                        # Converting to degrees, since matplotlib takes degrees
                        phi_axlbl_deg = phi_axlbl_radian * (180.0 / np.pi)
                        assert phi_axlbl_deg.shape == (n_pts,)
                        
                        # EllipseCollection takes "diameter" widths and 
                        # heights, so we need to multiply the radii by two.
                        diam_axlbl = r_axlbl * 2
                        assert diam_axlbl.shape == (n_pts, 2)
                    else:
                        has_blobs = False
                    
                    if has_blobs:
                        # Note: Scattering the center points is a hack to get matplotlib to 
                        # do two things that it cannot do with `EllipseCollection`:
                        #    1. add labels and handles to the legend, and
                        #    2. find the approprate x- and y-axis ranges.
                        axs_kwsct = dict(axs_lblkwa)
                        axs_kwsct['marker'] = 'o'
                        axs_kwsct['lw'] = 0.0
                        axs_kwsct.setdefault('s', 0.1)
                        ax.scatter(c_axlbl[:, 0], c_axlbl[:, 1], label=lblname, color=color_lbl, **axs_kwsct)

                        # We will use the same set of scatter keyword arguments 
                        # except some non-relevant ones for making the blob ellipses.
                        axs_kwelps = dict(axs_lblkwa)
                        axs_kwelps.pop('marker', None)
                        axs_kwelps.pop('s', None)
                        ec = EllipseCollection(widths=diam_axlbl[:, 0], heights=diam_axlbl[:, 1], angles=phi_axlbl_deg, 
                            units='x', offsets=c_axlbl, facecolors=color_lbl, offset_transform=ax.transData, **axs_kwelps)
                        ax.add_collection(ec)

                    assert len(lbl_datadict) == 0 , f'odd data keys for scatter of "{lblname}": {lbl_datadict.keys()}'
            elif plot_type == 'hist':
                for lblname, lbl_datadict in ax_datadeep.items():
                    datakeys_set = set(lbl_datadict)
                    if lblname not in lbl2color:
                        lbl2color[lblname] = next(color_cycler)
                    color_lbl = colorspec.get(lbl2color[lblname], lbl2color[lblname])
                    axh_lblkwa = axh_lblkwas.get(lblname, None)
                    axh_lblkwa = axh_kwa if axh_lblkwa is None else {**axh_kwa, **axh_lblkwa}

                    if datakeys_set == {'y'}:
                        pass
                    elif datakeys_set == {'x', 'y'}:
                        assert 'bins' not in axh_kwa
                        axh_lblkwa['bins'] = lbl_datadict['x']
                    else:
                        raise ValueError(f'odd keys "{datakeys_set}" for "{lblname}"')
                    
                    y_axlbl = lbl_datadict['y']
                    assert isinstance(y_axlbl, np.ndarray)
                    assert y_axlbl.ndim == 1

                    ax.hist(y_axlbl, label=lblname, color=color_lbl, **axh_lblkwa)
            elif plot_type == 'plot':
                for lblname, lbl_datadict in ax_datadeep.items():
                    datakeys_set = set(lbl_datadict)
                    if lblname not in lbl2color:
                        lbl2color[lblname] = next(color_cycler)
                    color_lbl = colorspec.get(lbl2color[lblname], lbl2color[lblname])
                    axp_lblkwa = axp_lblkwas.get(lblname, None)
                    axp_lblkwa = axp_kwa if axp_lblkwa is None else {**axp_kwa, **axp_lblkwa}

                    if datakeys_set == {'x', 'y'}:
                        y_axlbl = lbl_datadict['y']
                        assert isinstance(y_axlbl, np.ndarray)
                        assert y_axlbl.ndim == 1
                        ylow_axlbl, yhigh_axlbl = None, None
                    elif datakeys_set == {'x', 'y/mean', 'y/low', 'y/high'}:
                        y_axlbl = lbl_datadict['y/mean']
                        assert isinstance(y_axlbl, np.ndarray)
                        assert y_axlbl.ndim == 1

                        ylow_axlbl = lbl_datadict['y/low']
                        assert isinstance(y_axlbl, np.ndarray)
                        assert y_axlbl.ndim == 1

                        yhigh_axlbl = lbl_datadict['y/high']
                        assert isinstance(y_axlbl, np.ndarray)
                        assert y_axlbl.ndim == 1
                    else:
                        raise ValueError(f'odd keys "{datakeys_set}" for "{lblname}"')
                    
                    x_axlbl = lbl_datadict['x']
                    assert isinstance(x_axlbl, np.ndarray)
                    assert x_axlbl.ndim == 1

                    ax.plot(x_axlbl, y_axlbl, label=lblname, color=color_lbl, **axp_lblkwa)
                    
                    if ylow_axlbl is not None:
                        ax.fill_between(x_axlbl, ylow_axlbl, 
                            yhigh_axlbl, color=color_lbl, alpha=0.2)
            elif plot_type == 'pcolormesh':
                for lblname, lbl_datadict in ax_datadeep.items():
                    datakeys_set = set(lbl_datadict)
                    axc_lblkwa = axc_lblkwas.get(lblname, None)
                    axc_lblkwa = axc_kwa if axc_lblkwa is None else {**axc_kwa, **axc_lblkwa}
                    cax_kwa = get_subdict(axc_lblkwa, 'cax', sep='.', pop=True)
                    cbar_kwa = get_subdict(axc_lblkwa, 'cbar', sep='.', pop=True)
                    cnrm_kwa = get_subdict(axc_lblkwa, 'norm', sep='.', pop=True)

                    if datakeys_set == {'x', 'y', 'c'}:
                        pass
                    else:
                        raise ValueError(f'odd keys "{datakeys_set}" for "{lblname}"')
                    
                    x_axlbl = lbl_datadict['x']
                    assert isinstance(x_axlbl, np.ndarray)
                    assert x_axlbl.ndim == 1
                    n_x, = x_axlbl.shape

                    y_axlbl = lbl_datadict['y']
                    assert isinstance(y_axlbl, np.ndarray)
                    assert y_axlbl.ndim == 1
                    n_y, = y_axlbl.shape

                    x_axlbl2d, y_axlbl2d = np.meshgrid(x_axlbl, y_axlbl)
                    assert x_axlbl2d.shape == (n_x, n_y)
                    assert y_axlbl2d.shape == (n_x, n_y)

                    shading = axc_lblkwa.pop('shading', 'flat')
                    c_axlbl2d = lbl_datadict['c']
                    assert isinstance(c_axlbl2d, np.ndarray)
                    assert c_axlbl2d.ndim == 2
                    if shading == 'flat':
                        assert c_axlbl2d.shape == (n_x - 1, n_y - 1)
                    else:
                        assert c_axlbl2d.shape == (n_x, n_y)

                    if isinstance(cnrm_kwa, dict) and (len(cnrm_kwa) > 0):
                        cnorm_type = cnrm_kwa.pop('type')
                        cnorm_class = {'log': LogNorm, 'symlog': SymLogNorm, 
                            'power': PowerNorm, 'LogNorm': LogNorm, 
                            'SymLogNorm': SymLogNorm, 'PowerNorm': PowerNorm}[cnorm_type]
                        axc_lblkwa['norm'] = cnorm_class(**cnrm_kwa)
                    elif isinstance(cnrm_kwa, dict) and (len(cnrm_kwa) == 0):
                        pass
                    else:
                        axc_lblkwa['norm'] = cnrm_kwa

                    im = ax.pcolormesh(x_axlbl2d, y_axlbl2d, c_axlbl2d, shading=shading, **axc_lblkwa)

                    if (len(cbar_kwa) + len(cax_kwa)) > 0:
                        divider = make_axes_locatable(ax)
                        cax = divider.append_axes(**cax_kwa)
                        fig.colorbar(im, cax=cax, **cbar_kwa)
            else:
                raise ValueError(f'undefined plot type = {plot_type}')

            if (len(xlblkwa) > 0) and (not(share_xlbl) or (i_row == (n_figrows - 1))):
                assert 'xlabel' in xlblkwa, dedent(f'''
                    Some x label keword arguments were provided, but 
                    'xlabel' was not among them. 'xlabel' is a mandatory 
                    argument:
                        xlblkwa = {xlblkwa}''')
                ax.set_xlabel(**xlblkwa)
            if (len(ylblkwa) > 0) and (not(share_ylbl) or (i_col == 0)):
                assert 'ylabel' in ylblkwa, dedent(f'''
                    Some y label keword arguments were provided, but 
                    'ylabel' was not among them. 'ylabel' is a mandatory 
                    argument:
                        ylblkwa = {ylblkwa}''')
                ax.set_ylabel(**ylblkwa)
            if (len(clblkwa) > 0):
                assert 'clabel' in clblkwa, dedent(f'''
                    Some c label keword arguments were provided, but 
                    'clabel' was not among them. 'ylabel' is a mandatory 
                    argument:
                        clblkwa = {clblkwa}''')
                cax.set_ylabel(**clblkwa)
            if xlim is not None: 
                ax.set_xlim(*xlim)
            if ylim is not None: 
                ax.set_ylim(*ylim)
            if clim is not None: 
                cax.set_ylim(*clim)
            if xbound is not None:
                ax.set_xbound(*xbound)
            if ybound is not None:
                ax.set_ybound(*ybound)
            if cbound is not None:
                cax.set_ybound(*cbound)
            if len(xsc_kwa) > 0: 
                ax.set_xscale(**xsc_kwa)
            if len(ysc_kwa) > 0: 
                ax.set_yscale(**ysc_kwa)
            if len(csc_kwa) > 0: 
                cax.set_yscale(**csc_kwa)
            if len(xtckkwa) > 0: 
                ax.set_xticks(**xtckkwa)
            if len(ytckkwa) > 0: 
                ax.set_yticks(**ytckkwa)
            if len(ctckkwa) > 0: 
                cax.set_yticks(**ctckkwa)
            if xtickformatter is not None: 
                ax.xaxis.set_major_formatter(xtickformatter)
            if ytickformatter is not None: 
                ax.yaxis.set_major_formatter(ytickformatter)
            if ctickformatter is not None: 
                cax.yaxis.set_major_formatter(ctickformatter)
            if xticklabeler is not None:
                ax.set_xticks(ax.get_xticks(), xticklabeler(ax.get_xticks()))
            if yticklabeler is not None:
                ax.set_yticks(ax.get_yticks(), yticklabeler(ax.get_yticks()))
            if cticklabeler is not None:
                cax.set_yticks(cax.get_yticks(), cticklabeler(cax.get_yticks()))
            if xticklabels is not None: 
                ax.set_xticks(ax.get_xticks(), xticklabels)
            if yticklabels is not None: 
                ax.set_yticks(ax.get_yticks(), yticklabels)
            if cticklabels is not None: 
                cax.set_yticks(cax.get_yticks(), cticklabels)
            if len(tckprms) > 0: 
                ax.tick_params(**tckprms)
            if len(xtckprms) > 0: 
                ax.tick_params(axis='x', **xtckprms)
            if len(ytckprms) > 0: 
                ax.tick_params(axis='y', **ytckprms)
            if len(ctckprms) > 0: 
                cax.tick_params(axis='y', **ctckprms)

            if len(tagkwa) > 0:
                tagkwa2 = tagkwa.copy()
                tag_xy = tagkwa2.pop('xy', (0, 1))
                tag_width = tagkwa2.pop('width', 0.12)
                tag_height = tagkwa2.pop('height', 0.12)
                tag_text = tagkwa2.pop('text')
                assert len(tagkwa2) == 0, f'unused options {tagkwa2}'
                tag_axis(ax=ax, text=tag_text, xy=tag_xy, width=tag_width, height=tag_height)

        last_ax = ax
        if len(leg_kwa) > 0:
            assert last_ax is not None, 'None of the axes were plotted in'
            leg_handles, leg_lables = last_ax.get_legend_handles_labels()
            leg_lables2 = [leglbls.get(lbl, lbl) for lbl in leg_lables]
            fig.legend(leg_handles, leg_lables2, **leg_kwa)

        if lay_kwa is not None:
            layengine = fig.get_layout_engine()
            if (layengine is None) or ('layout' in lay_kwa):
                fig.set_layout_engine(**lay_kwa)
            else:
                layengine.set(**lay_kwa)

        assert len(xtf_kwa) == 0, f'unused xtickformatter options: {xtf_kwa.keys()}'
        assert len(ytf_kwa) == 0, f'unused ytickformatter options: {ytf_kwa.keys()}'
        assert len(mplopts) == 0, f'unused mplopts options: {mplopts.keys()}'
        assert len(data) == 0, f'unused data options: {data.keys()}'

    return fig, axes



# +
class Evaluation:
    @staticmethod
    @torch.no_grad()
    def encode(eii, e_split, e_splitidxs, e_pp, e_frq, e_rndmztn,
        ltnt_sigtnsfm, datanp_dict, enc_pp, encoder, 
        e_ppinpspc, e_pprndtrp, xv_dims, uv_dims,
        is_condvae, lblnn_enctype, ey_sigscale, 
        lbl_pp, lblnn, xl_dims, ul_dims, z_dims, y_dims, 
        dec_pp, ud_dims, xd_dims, xedl_dims, 
        n_seeds, n_evlpnts, eval_bsnn, epoch,
        xe_dims, ue_dims, erng, tch_device, tch_dtype):

        if e_split == 'normal':
            renew_eii, ei_mb = False, None
        elif e_split in ('train', 'test', 'snr'):
            n_esplit = e_splitidxs.shape[-1]
            assert e_splitidxs.shape == (n_seeds, n_esplit), f'{e_splitidxs.shape} != {(n_seeds, n_esplit)}'

            # First, we sample random indices to select from `trn_spltidxs`.
            eiistrt, eiiend = (epoch // e_frq) * n_evlpnts, n_evlpnts + (epoch // e_frq) * n_evlpnts
            eii_mbidxs_ = torch.arange(eiistrt, eiiend, device=tch_device).remainder(n_esplit).to(dtype=torch.long)
            assert eii_mbidxs_.shape == (n_evlpnts,)

            renew_eii = (eii_mbidxs_ == 0).any().item()

            if renew_eii and (e_rndmztn == 'none'):
                eii_ = torch.arange(n_esplit, device=tch_device, dtype=torch.long)
                assert eii_.shape == (n_esplit,)

                eii = eii_.reshape(1, n_esplit).expand(n_seeds, n_esplit)
                assert eii.shape == (n_seeds, n_esplit)
            elif renew_eii and (e_rndmztn == 'shuffle'):
                eii_ = erng.rand(n_seeds, n_esplit)
                assert eii_.shape == (n_seeds, n_esplit)

                eii = eii_.argsort(dim=-1).detach()
                assert eii.shape == (n_seeds, n_esplit)
            elif renew_eii and (e_rndmztn == 'iid'):
                eii = (erng.rand(n_seeds, n_esplit) * n_esplit).to(torch.long)
                assert eii.shape == (n_seeds, n_esplit)
            elif renew_eii:
                raise ValueError(f'undefined eval randomization = {e_rndmztn}')
            else:
                assert eii.shape == (n_seeds, n_esplit)

            eii_mbidxs = eii_mbidxs_.reshape(1, n_evlpnts).expand(n_seeds, n_evlpnts)
            assert eii_mbidxs.shape == (n_seeds, n_evlpnts)

            eii_mb = torch.take_along_dim(eii, eii_mbidxs, dim=-1)
            assert eii_mb.shape == (n_seeds, n_evlpnts)

            # Next, we select those from the training split indices.
            ei_mb = torch.take_along_dim(e_splitidxs, eii_mb, dim=1)
            assert ei_mb.shape == (n_seeds, n_evlpnts)
        else:
            raise ValueError(f'undefined e_split = {e_split}')

        n_evliters = math.ceil(n_evlpnts / eval_bsnn)
        emu_mb2lsts = defaultdict(list)
        esigma_mb2lsts = defaultdict(list)
        ey_mb2lsts = defaultdict(list)
        exd_mb2lsts = defaultdict(list)
        
        for i_evliter in range(n_evliters):
            ii1_evl = i_evliter * eval_bsnn
            ii2_evl = min(ii1_evl + eval_bsnn, n_evlpnts)
            n_mbevl = ii2_evl - ii1_evl

            if e_split == 'normal':
                emu_mb2s = dict()
                for z_node, z_dim in z_dims.items():
                    emu_mb2 = torch.zeros(n_seeds, n_mbevl, *z_dim,
                        device=tch_device, dtype=tch_dtype)
                    assert emu_mb2.shape == (n_seeds, n_mbevl, *z_dim)
                    emu_mb2s[z_node] = emu_mb2
                
                esigma_mb2s = dict()
                for z_node, z_dim in z_dims.items():
                    esigma_mb2 = torch.ones(n_seeds, n_mbevl, *z_dim,
                        device=tch_device, dtype=tch_dtype)
                    assert esigma_mb2.shape == (n_seeds, n_mbevl, *z_dim)
                    esigma_mb2s[z_node] = esigma_mb2

                exe_mb2s, eue_mb2s, ey_mb2s = None, None, None
                xue_dims = {**xe_dims, **ue_dims, **y_dims}
                # TODO: uncomment and fix the following
                eu_shaper2 = None
                # eu_shaper2 = e_pp.get_shaper({xu_node: (n_seeds, n_mbevl, n_xuchnls, n_xulen)
                #     for xu_node, (n_xuchnls, n_xulen) in xue_dims.items()})
            elif e_split in ('train', 'test', 'snr'):
                ei_mb2 = ei_mb[:, ii1_evl: ii2_evl]
                assert ei_mb2.shape == (n_seeds, n_mbevl)

                ei_mb2rvld = ei_mb2.ravel()
                assert ei_mb2rvld.shape == (n_seeds * n_mbevl,)
                ei_mb2unqs, ei_mb2invrs, ei_mb2cnts, ei_mb2indx = torch_unique(ei_mb2rvld)
                assert ei_mb2invrs.shape == (n_seeds * n_mbevl,)
                ei_mb2unqsnp = ei_mb2unqs.detach().cpu().numpy()
                ei_mb2invrsnp = ei_mb2invrs.detach().cpu().numpy()
                assert ei_mb2invrsnp.shape == (n_seeds * n_mbevl,)

                # Now, we collect these indices from the full data.
                exedl_mb2s = dict()
                for x_node, (n_xchnls, n_xlen) in xedl_dims.items():
                    ex_mb2_ = datanp_dict[x_node][ei_mb2unqsnp][ei_mb2invrsnp]
                    assert ex_mb2_.shape == (n_seeds * n_mbevl, n_xchnls, n_xlen)
                    if torch.is_tensor(ex_mb2_):
                        ex_mb2th_ = ex_mb2_
                        assert ex_mb2th_.shape == (n_seeds * n_mbevl, n_xchnls, n_xlen)
                    else:
                        ex_mb2th_ = torch.tensor(ex_mb2_).to(device=tch_device, dtype=tch_dtype)
                        assert ex_mb2th_.shape == (n_seeds * n_mbevl, n_xchnls, n_xlen)
                    ex_mb2 = ex_mb2th_.reshape(n_seeds, n_mbevl, n_xchnls, n_xlen)
                    assert ex_mb2.shape == (n_seeds, n_mbevl, n_xchnls, n_xlen)
                    exedl_mb2s[x_node] = ex_mb2

                # The encoder's pp input mini-batches
                exe_mb2s = {x_node: exedl_mb2s[x_node] for x_node in xe_dims}

                # Next, we apply the input pre-processing transformations
                # before feeding it to the encoder.
                eue_mb2s, eu_shaper2 = enc_pp.forward(exe_mb2s, full=False)
                for u_node, (n_uchnls, n_ulen) in ue_dims.items():
                    eu_mb2 = eue_mb2s[u_node]
                    assert eu_mb2.shape == (n_seeds, n_mbevl, n_uchnls, n_ulen)

                #######################################
                #         Applying the Labeler        #
                #######################################
                if is_condvae:
                    # The labeler's pp input mini-batches
                    exl_mb2s = {x_node: exedl_mb2s[x_node] for x_node in xl_dims}

                    # Next, we apply the input pre-processing transformations
                    # before feeding it to the labeler.
                    eul_mb2s, u_shaper2 = lbl_pp.forward(exl_mb2s, full=False)
                    for u_node, (n_uchnls, n_ulen) in ul_dims.items():
                        u_mb2 = eul_mb2s[u_node]
                        assert u_mb2.shape == (n_seeds, n_mbevl, n_uchnls, n_ulen)
                    
                    elblnnkws = dict()
                    if lblnn_enctype == 'vae':
                        elblnnkws = dict(n_seeds=n_seeds, n_mb=-1, rng=erng, sigscale=ey_sigscale)
                    assert lblnn_enctype in ('vae', 'plain')

                    # There is no need for `get_nn_sol` anymore; we're already batching here.
                    # ey_mb2s = get_nn_sol(lblnn, eul_mb2s, n_eval=eval_bsnn, model_kwargs=elblnnkws)
                    # assert isinstance(ey_mb2s, dict)

                    ey_mb2s = lblnn(**eul_mb2s, **elblnnkws)
                    assert isinstance(ey_mb2s, dict)
                else:
                    ey_mb2s = dict()

                for y_node, (n_ychnls, n_ylen) in y_dims.items():
                    ey_mb2 = ey_mb2s[y_node]
                    assert ey_mb2.shape == (n_seeds, n_mbevl, n_ychnls, n_ylen)

                #######################################
                #         Applying the Encoder        #
                #######################################
                # Encoding the pre-processed input
                # There is no need for `get_nn_sol` anymore; we're already batching here.
                # emusig_mb2_ = get_nn_sol(encoder, {**eue_mb2s, **ey_mb2s}, n_eval=eval_bsnn)
                # assert emusig_mb2_.shape == (n_seeds, n_mbevl, 2 * ltnt_dim)

                emusig_mb2s = encoder(**eue_mb2s, **ey_mb2s)

                # Getting the mu values out of the encoder
                emu_mb2s = get_subdict(emusig_mb2s, 'mu', sep='.', pop=True)
                assert set(emu_mb2s) == set(z_dims), dedent(f'''
                    mu nodes are different from the usual:
                        extra nodes: {set(emu_mb2s) - set(z_dims)}
                        missing nodes: {set(z_dims) - set(emu_mb2s)}''')
                for z_node, z_dim in z_dims.items():
                    emu_mb2 = emu_mb2s[z_node]
                    assert emu_mb2.shape == (n_seeds, n_mbevl, *z_dim)

                # Getting the log-sigma values out of the encoder
                elogsigtld_mb2s = get_subdict(emusig_mb2s, 'sig', sep='.', pop=True)
                assert set(elogsigtld_mb2s) == set(z_dims), dedent(f'''
                    sigma nodes are different from the usual:
                        extra nodes: {set(elogsigtld_mb2s) - set(z_dims)}
                        missing nodes: {set(z_dims) - set(elogsigtld_mb2s)}''')
                for z_node, z_dim in z_dims.items():
                    elogsigtld_mb2 = elogsigtld_mb2s[z_node]
                    assert elogsigtld_mb2.shape == (n_seeds, n_mbevl, *z_dim)

                # Making sure the encoder naming convention is followed
                assert len(emusig_mb2s) == 0, dedent(f'''
                    The encoder output keys should be either `mu.*` or `sig.*`:
                        offending items: {emusig_mb2s}''')

                # Applying the latent sigma transformation
                esigma_mb2s = dict()
                for z_node, z_dim in z_dims.items():
                    if ltnt_sigtnsfm == 'exp':
                        elogsigma_mb2 = elogsigtld_mb2s[z_node]
                        assert elogsigma_mb2.shape == (n_seeds, n_mbevl, *z_dim)
                    elif ltnt_sigtnsfm == 'explin':
                        elogsigtlde_mb2 = elogsigtld_mb2s[z_node]
                        assert elogsigtlde_mb2.shape == (n_seeds, n_mbevl, *z_dim)

                        elogsigma_mb2 = torch.where(elogsigtlde_mb2 < 0, elogsigtlde_mb2, 
                            (elogsigtlde_mb2 + 1).log())
                        assert elogsigma_mb2.shape == (n_seeds, n_mbevl, *z_dim)
                    else:
                        raise ValueError(f'undefined ltnt_sigtnsfm={ltnt_sigtnsfm}')
                    
                    esigma_mb2 = elogsigma_mb2.exp()
                    assert esigma_mb2.shape == (n_seeds, n_mbevl, *z_dim)

                    esigma_mb2s[z_node] = esigma_mb2
            else:
                raise ValueError(f'undefined e_split = {e_split}')

            for z_node, emu_mb2 in emu_mb2s.items():
                emu_mb2lsts[z_node].append(emu_mb2)
            for z_node, esigma_mb2 in esigma_mb2s.items():
                esigma_mb2lsts[z_node].append(esigma_mb2)
            
            if ey_mb2s is not None:
                for y_node, ey_mb2 in ey_mb2s.items():
                    ey_mb2lsts[y_node].append(ey_mb2)
            else:
                ey_mb2lsts = None


            # Preparing the decoder evaluation inputs
            if e_split == 'normal':
                exd_mb2s = None
            elif e_split in ('train', 'test', 'snr'):
                # The decoder's pp input mini-batches
                exd_mb2s = {x_node: exedl_mb2s[x_node] for x_node in xd_dims}
            else:
                raise ValueError(f'undefined e_split = {e_split}')

            if exd_mb2s is not None:
                for x_node, exd_mb2 in exd_mb2s.items():
                    exd_mb2lsts[x_node].append(exd_mb2)
            else:
                exd_mb2lsts = None
        
        # Concatenating `emu_mb`, `esigma_mb`, `u_eppmbs`, and `ey_mbs`
        emu_mbs = dict()
        for z_node, z_dim in z_dims.items():
            emu_mb = torch.cat(emu_mb2lsts[z_node], dim=1)
            assert emu_mb.shape == (n_seeds, n_evlpnts, *z_dim)
            emu_mbs[z_node] = emu_mb

        esigma_mbs = dict()
        for z_node, z_dim in z_dims.items():
            esigma_mb = torch.cat(esigma_mb2lsts[z_node], dim=1)
            assert esigma_mb.shape == (n_seeds, n_evlpnts, *z_dim)
            esigma_mbs[z_node] = esigma_mb

        if exd_mb2lsts is not None:
            exd_mbs = dict()
            for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                exd_mb = torch.cat(exd_mb2lsts[x_node], dim=1) 
                assert exd_mb.shape == (n_seeds, n_evlpnts, n_xchnls, n_xlen)
                exd_mbs[x_node] = exd_mb
        else:
            exd_mbs = None
        
        if ey_mb2lsts is not None:
            ey_mbs = dict()
            for y_node, (n_ychnls, n_ylen) in y_dims.items():
                ey_mb = torch.cat(ey_mb2lsts[y_node], dim=1)
                assert ey_mb.shape == (n_seeds, n_evlpnts, n_ychnls, n_ylen)
                ey_mbs[y_node] = ey_mb
        else:
            ey_mbs = None

        return emu_mbs, esigma_mbs, exd_mbs, ey_mbs, ei_mb, renew_eii, eii

    @staticmethod
    @torch.no_grad()
    def mixup(emu_mbs, esigma_mbs, exd_mbs, ey_mbs, ei_mb, 
        e_detspace, ealpha_mixup, e_flipmax, n_seeds, n_evlpnts, n_emixper,
        z_dims, y_dims, xd_dims, erng, tch_dtype, tch_device):

        # the number of inflated/augmented samples
        n_mxdevlpnts = n_evlpnts * n_emixper

        jj_mixup1 = erng.rand(n_seeds, n_evlpnts).argsort(dim=-1)
        assert jj_mixup1.shape == (n_seeds, n_evlpnts)

        ########################################
        ######## Inflating the Samples #########
        ########################################
        emu_iflmbs, esigma_iflmbs = dict(), dict()
        for z_node, z_dim in z_dims.items():
            emu_mb = emu_mbs[z_node]
            assert emu_mb.shape == (n_seeds, n_evlpnts, *z_dim)
            emu_iflmb1 = emu_mb.reshape(n_seeds, n_evlpnts, 1, *z_dim)
            assert emu_iflmb1.shape == (n_seeds, n_evlpnts, 1, *z_dim)
            emu_iflmb2 = emu_iflmb1.expand(n_seeds, n_evlpnts, n_emixper, *z_dim)
            assert emu_iflmb2.shape == (n_seeds, n_evlpnts, n_emixper, *z_dim)
            emu_iflmb = emu_iflmb2.reshape(n_seeds, n_mxdevlpnts, *z_dim)
            assert emu_iflmb.shape == (n_seeds, n_mxdevlpnts, *z_dim)
            emu_iflmbs[z_node] = emu_iflmb

            esigma_mb = esigma_mbs[z_node]
            assert esigma_mb.shape == (n_seeds, n_evlpnts, *z_dim)
            esigma_iflmb1 = esigma_mb.reshape(n_seeds, n_evlpnts, 1, *z_dim)
            assert esigma_iflmb1.shape == (n_seeds, n_evlpnts, 1, *z_dim)
            esigma_iflmb2 = esigma_iflmb1.expand(n_seeds, n_evlpnts, n_emixper, *z_dim)
            assert esigma_iflmb2.shape == (n_seeds, n_evlpnts, n_emixper, *z_dim)
            esigma_iflmb = esigma_iflmb2.reshape(n_seeds, n_mxdevlpnts, *z_dim)
            assert esigma_iflmb.shape == (n_seeds, n_mxdevlpnts, *z_dim)
            esigma_iflmbs[z_node] = esigma_iflmb

        # Let's do the same thing for ey_mbs
        for y_node, y_dim in y_dims.items():
            ey_mb = ey_mbs[y_node]
            assert ey_mb.shape == (n_seeds, n_evlpnts, *y_dim)
            ey_iflmb1 = ey_mb.reshape(n_seeds, n_evlpnts, 1, *y_dim)
            assert ey_iflmb1.shape == (n_seeds, n_evlpnts, 1, *y_dim)
            ey_iflmb2 = ey_iflmb1.expand(n_seeds, n_evlpnts, n_emixper, *y_dim)
            assert ey_iflmb2.shape == (n_seeds, n_evlpnts, n_emixper, *y_dim)
            ey_iflmb = ey_iflmb2.reshape(n_seeds, n_mxdevlpnts, *y_dim)
            assert ey_iflmb.shape == (n_seeds, n_mxdevlpnts, *y_dim)
            ey_mxdmbs[y_node] = ey_iflmb

        # Now, let's do the same thing for exd_mbs
        if exd_mbs is not None:
            exd_iflmbs = dict()
            for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                exd_mb = exd_mbs[x_node]
                assert exd_mb.shape == (n_seeds, n_evlpnts, n_xchnls, n_xlen)
                exd_iflmb1 = exd_mb.reshape(n_seeds, n_evlpnts, 1, n_xchnls, n_xlen)
                assert exd_iflmb1.shape == (n_seeds, n_evlpnts, 1, n_xchnls, n_xlen)
                exd_iflmb2 = exd_iflmb1.expand(n_seeds, n_evlpnts, n_emixper, n_xchnls, n_xlen)
                assert exd_iflmb2.shape == (n_seeds, n_evlpnts, n_emixper, n_xchnls, n_xlen)
                exd_iflmb = exd_iflmb2.reshape(n_seeds, n_mxdevlpnts, n_xchnls, n_xlen)
                assert exd_iflmb.shape == (n_seeds, n_mxdevlpnts, n_xchnls, n_xlen)
                exd_iflmbs[x_node] = exd_iflmb
        else:
            exd_iflmbs = None

        # Now let's do the same thing for jj_mixup1
        jj_mixup2 = jj_mixup1.reshape(n_seeds, n_evlpnts, 1)
        assert jj_mixup2.shape == (n_seeds, n_evlpnts, 1)
        jj_mixup3 = jj_mixup2.expand(n_seeds, n_evlpnts, n_emixper)
        assert jj_mixup3.shape == (n_seeds, n_evlpnts, n_emixper)
        jj_mixup4 = jj_mixup3.reshape(n_seeds, n_mxdevlpnts)
        assert jj_mixup4.shape == (n_seeds, n_mxdevlpnts)

        ########################################
        ###### Determining Mixing Weights ######
        ########################################
        if e_detspace:
            lam_mixup1lst = [scipy.stats.beta.ppf(pcdf, ealpha_mixup, ealpha_mixup) 
                for pcdf in np.linspace(0, 1, n_emixper)]
            lam_mixup1a = torch.tensor(lam_mixup1lst, dtype=tch_dtype, device=tch_device)
            assert lam_mixup1a.shape == (n_emixper,)
            lam_mixup1b = lam_mixup1a.reshape(1, 1, n_emixper).expand(n_seeds, n_evlpnts, n_emixper)
            assert lam_mixup1b.shape == (n_seeds, n_evlpnts, n_emixper)
            lam_mixup1 = lam_mixup1b.reshape(n_seeds, n_mxdevlpnts)
            assert lam_mixup1.shape == (n_seeds, n_mxdevlpnts)
        else:
            # Determining the mixing weights (lambdas)
            lam_mixup1 = erng(beta_genr, n_seeds, n_mxdevlpnts)
            assert lam_mixup1.shape == (n_seeds, n_mxdevlpnts)
        
        if e_flipmax:
            lam_mixup2 = 1.0 - lam_mixup1
            assert lam_mixup2.shape == (n_seeds, n_mxdevlpnts)
            lam_mixup3 = torch.maximum(lam_mixup1, lam_mixup2)
            assert lam_mixup3.shape == (n_seeds, n_mxdevlpnts)
        else:
            lam_mixup3 = lam_mixup1
            assert lam_mixup3.shape == (n_seeds, n_mxdevlpnts)
        
        lam_mixup4 = 1.0 - lam_mixup3
        assert lam_mixup4.shape == (n_seeds, n_mxdevlpnts)

        ########################################
        ########### Performing Mixup ###########
        ########################################
        emu_mxdmbs, esigma_mxdmbs = dict(), dict()
        for z_node, z_dim in z_dims.items():
            z_1dims = [1] * len(z_dim)

            jj_mixup5 = jj_mixup4.reshape(n_seeds, n_mxdevlpnts, *z_1dims)
            assert jj_mixup5.shape == (n_seeds, n_mxdevlpnts, *z_1dims)

            lam_mixup5 = lam_mixup3.reshape(n_seeds, n_mxdevlpnts, *z_1dims)
            assert lam_mixup5.shape == (n_seeds, n_mxdevlpnts, *z_1dims)
            
            lam_mixup6 = lam_mixup4.reshape(n_seeds, n_mxdevlpnts, *z_1dims)
            assert lam_mixup6.shape == (n_seeds, n_mxdevlpnts, *z_1dims)

            emui_mb = emu_iflmbs[z_node]
            assert emui_mb.shape == (n_seeds, n_mxdevlpnts, *z_dim)

            emuj_mb = torch.take_along_dim(emui_mb, jj_mixup5, dim=1)
            assert emuj_mb.shape == (n_seeds, n_mxdevlpnts, *z_dim)

            emu_mxdmb = emui_mb * lam_mixup5 + emuj_mb * lam_mixup6
            assert emu_mxdmb.shape == (n_seeds, n_mxdevlpnts, *z_dim)

            emu_mxdmbs[z_node] = emu_mxdmb

            esigmai_mb = esigma_iflmbs[z_node]
            assert esigmai_mb.shape == (n_seeds, n_mxdevlpnts, *z_dim)

            esigmaj_mb = torch.take_along_dim(esigmai_mb, jj_mixup5, dim=1)
            assert esigmaj_mb.shape == (n_seeds, n_mxdevlpnts, *z_dim)

            esigma_mxdmb = (esigmai_mb.square() * lam_mixup5.square() + 
                esigmaj_mb.square() * lam_mixup6.square()).sqrt()
            assert esigma_mxdmb.shape == (n_seeds, n_mxdevlpnts, *z_dim)

            esigma_mxdmbs[z_node] = esigma_mxdmb

        ey_mxdmbs = dict()
        for y_node, y_dim in y_dims.items():
            y_1dims = [1] * len(y_dim)

            jj_mixup5 = jj_mixup4.reshape(n_seeds, n_mxdevlpnts, *y_1dims)
            assert jj_mixup5.shape == (n_seeds, n_mxdevlpnts, *y_1dims)

            lam_mixup5 = lam_mixup3.reshape(n_seeds, n_mxdevlpnts, *y_1dims)
            assert lam_mixup5.shape == (n_seeds, n_mxdevlpnts, *y_1dims)
            
            lam_mixup6 = lam_mixup4.reshape(n_seeds, n_mxdevlpnts, *y_1dims)
            assert lam_mixup6.shape == (n_seeds, n_mxdevlpnts, *y_1dims)

            eyi_mb = ey_iflmbs[y_node]
            assert eyi_mb.shape == (n_seeds, n_mxdevlpnts, *y_dim)

            eyj_mb = torch.take_along_dim(eyi_mb, jj_mixup5, dim=1)
            assert eyj_mb.shape == (n_seeds, n_mxdevlpnts, *y_dim)

            ey_mxdmb = eyi_mb * lam_mixup5 + eyj_mb * lam_mixup6
            assert ey_mxdmb.shape == (n_seeds, n_mxdevlpnts, *y_dim)

            ey_mxdmbs[y_node] = ey_mxdmb
        
        if exd_iflmbs is not None:
            exd_mxdmbs = dict()
            for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                jj_mixup5 = jj_mixup4.reshape(n_seeds, n_mxdevlpnts, 1, 1)
                assert jj_mixup5.shape == (n_seeds, n_mxdevlpnts, 1, 1)

                lam_mixup5 = lam_mixup3.reshape(n_seeds, n_mxdevlpnts, 1, 1)
                assert lam_mixup5.shape == (n_seeds, n_mxdevlpnts, 1, 1)
                
                lam_mixup6 = lam_mixup4.reshape(n_seeds, n_mxdevlpnts, 1, 1)
                assert lam_mixup6.shape == (n_seeds, n_mxdevlpnts, 1, 1)

                exdi_mb = exd_iflmbs[x_node]
                assert exdi_mb.shape == (n_seeds, n_mxdevlpnts, n_xchnls, n_xlen)

                exdj_mb = torch.take_along_dim(exdi_mb, jj_mixup5, dim=1)
                assert exdj_mb.shape == (n_seeds, n_mxdevlpnts, n_xchnls, n_xlen)

                exd_mxdmb = exdi_mb * lam_mixup5 + exdj_mb * lam_mixup6
                assert exd_mxdmb.shape == (n_seeds, n_mxdevlpnts, n_xchnls, n_xlen)

                exd_mxdmbs[x_node] = exd_mxdmb
        else:
            exd_mxdmbs = None

        if ei_mb is not None:
            # Recovering the j indices from the actual data
            ej_mxdmb = torch.take_along_dim(ei_mb, jj_mixup4, dim=-1)
            assert ej_mxdmb.shape == (n_seeds, n_mxdevlpnts)
            
            # Break the next line into three lines
            ei_mxdmb1 = ei_mb.reshape(n_seeds, n_evlpnts, 1)
            assert ei_mxdmb1.shape == (n_seeds, n_evlpnts, 1)
            ei_mxdmb2 = ei_mxdmb1.expand(n_seeds, n_evlpnts, n_emixper)
            assert ei_mxdmb2.shape == (n_seeds, n_evlpnts, n_emixper)
            ei_mxdmb = ei_mxdmb2.reshape(n_seeds, n_mxdevlpnts)
            assert ei_mxdmb.shape == (n_seeds, n_mxdevlpnts)
        else:
            ei_mxdmb, ej_mxdmb = None, None

        return emu_mxdmbs, esigma_mxdmbs, exd_mxdmbs, ey_mxdmbs, ei_mxdmb, ej_mxdmb, n_mxdevlpnts

    @staticmethod
    @torch.no_grad()
    def decodeorig(exd_mxdmbs, e_split, e_pp, dec_pp, 
        e_ppinpspc, e_pprndtrp, xv_dims, uv_dims, ud_dims,
        n_seeds, n_mxdevlpnts):

        if exd_mxdmbs is not None:
            # Next, we apply the input pre-processing transformations
            # as if it was fed to a perfect decoder.
            eud_mxdmbs, eud_shaper = dec_pp.forward(exd_mxdmbs, full=False)
            for u_node, (n_uchnls, n_ulen) in ud_dims.items():
                u_mb = eud_mxdmbs[u_node]
                assert u_mb.shape == (n_seeds, n_mxdevlpnts, n_uchnls, n_ulen)
        else:
            eud_mxdmbs, eud_shaper = None, None

        # Finding the right input to the evaluation pre-processor:
        #   * `epp_inpmxdmbs`: This is the original data fed to the `e_pp` pre-processor.
        #       * It should be either `eud_mxdmbs` or `ex_tildemxdmbs` depending on `e_ppinpspc`.
        if (e_ppinpspc == 'x') and (e_split in ('train', 'test', 'snr')) and e_pprndtrp:
            ex_tildemxdmbs = dec_pp.inverse(eud_mxdmbs, eud_shaper, strict=True, full=False)
            for x_node, (n_xchnls, n_xlen) in xv_dims.items():
                ex_mb = ex_tildemxdmbs[x_node]
                assert ex_mb.shape == (n_seeds, n_mxdevlpnts, n_xchnls, n_xlen)
            epp_inpmxdmbs = ex_tildemxdmbs
        elif (e_ppinpspc == 'x') and (e_split in ('train', 'test', 'snr')) and (not e_pprndtrp):
            epp_inpmxdmbs, ex_tildemxdmbs = exd_mxdmbs, None
        elif (e_ppinpspc == 'x') and (e_split == 'normal'):
            epp_inpmxdmbs, ex_tildemxdmbs = None, None
        elif (e_ppinpspc == 'u'):
            epp_inpmxdmbs, ex_tildemxdmbs = eud_mxdmbs, None
        else:
            raise ValueError(f'undefined e_ppinpspc={e_ppinpspc}')

        # Applying the evaluation pre-processor to the original data.
        if e_split in ('train', 'test', 'snr'):
            u_eppmxdmbs, u_eppshaper2 = e_pp.forward(epp_inpmxdmbs, full=False)
            for u_node, (n_uchnls, n_ulen) in uv_dims.items():
                eu_mb = u_eppmxdmbs[u_node]
                assert eu_mb.shape == (n_seeds, n_mxdevlpnts, n_uchnls, n_ulen)
        elif e_split  == 'normal':
            assert ex_tildemxdmbs is None
            u_eppmxdmbs = None
        else:
            raise ValueError(f'undefined e_split = {e_split}')
        
        return u_eppmxdmbs

    @staticmethod
    @torch.no_grad()
    def sample_zy(emu_mbs, esigma_mbs, ey_mbs, ei_mb, z_dims, y_dims,
        n_seeds, n_evlpnts, is_condvae, erng, e_encpkgs,
        eid, e_split, e_vrnt, e_vrntcfg, tch_device, tch_dtype):

        e_vrntcfg2 = dict(e_vrntcfg)
        # The variant type
        vrnt_type = e_vrntcfg2.pop('type')

        if vrnt_type == 'orig':
            n_rcns = 1
        elif vrnt_type in ('rcnst', 'genr'):
            n_rcns = e_vrntcfg2.pop('ltnt/n')
        else:
            raise ValueError(f'undefiend vrnt_type={vrnt_type}')

        #######################################################################
        ####################### Determining the Labels  #######################
        #######################################################################
        if is_condvae and (vrnt_type == 'orig'):
            assert ey_mbs is not None
            ey_mbvs1 = ey_mbs
        elif is_condvae and (vrnt_type in ('rcnst', 'genr')):
            y_typvrnt = e_vrntcfg2.pop('lbl/type')
            if y_typvrnt == 'orig':
                assert ey_mbs is not None
                ey_mbvs1 = ey_mbs
            elif y_typvrnt == 'fixed':
                ey_vals = get_subdict(e_vrntcfg2, 'lbl', pop=True)
                ey_mbvs1 = dict()
                for y_node, (n_ychnls, n_ylen) in y_dims.items():
                    assert y_node in ey_vals, dedent(f'''
                        Missing eval label value specification:
                            Missing y node: {y_node}
                            eval name: {eid}
                            split name: {e_split}
                            variant name: {e_vrnt}''')
                    ey_val = ey_vals.pop(y_node)
                    try:
                        y_mbv1 = torch.tensor(ey_val, device=tch_device, dtype=tch_dtype)
                        assert y_mbv1.ndim == 2
                        y_mbv2 = y_mbv1.expand(n_ychnls, n_ylen)
                        assert y_mbv2.shape == (n_ychnls, n_ylen)
                        y_mbv3 = y_mbv2[None, None].expand(n_seeds, n_evlpnts, n_ychnls, n_ylen)
                        assert y_mbv3.shape == (n_seeds, n_evlpnts, n_ychnls, n_ylen)
                    except Exception as exc:
                        exc.add_note(dedent(f'''
                            This happened while defining a fixed label:
                                y node: {y_node}
                                n_ychnls, n_ylen: {(n_ychnls, n_ylen)}
                                eval name: {eid}
                                split name: {e_split}
                                variant name: {e_vrnt}'''))
                        raise exc
                    ey_mbvs1[y_node] = y_mbv3

                assert len(ey_vals) == 0, dedent(f'''
                    Extra eval label specification:
                        extra y nodes: {list(ey_vals)}
                        eval name: {eid}
                        split name: {e_split}
                        variant name: {e_vrnt}''')
            elif y_typvrnt == 'iid':
                y_spltvrnt = e_vrntcfg2.pop('lbl/split')
                assert y_spltvrnt in ('train', 'test')
                assert y_spltvrnt in e_encpkgs

                # Getting the split encoded data
                (emu_mbs2, esigma_mbs2, exd_mbs2, u_eppmbs2, ey_mbs2, ei_mb2, ej_mb2, n_evlpnts2) = e_encpkgs[y_spltvrnt]
                n_evlpnts2 = ei_mb2.shape[1]
                assert ei_mb2.shape == (n_seeds, n_evlpnts2)

                eii_mbv = (erng.rand(n_seeds, n_evlpnts, 1, 1) * n_evlpnts2).to(dtype=torch.long)
                assert eii_mbv.shape == (n_seeds, n_evlpnts, 1, 1)

                ey_mbvs1 = dict()
                for y_node, (n_ychnls, n_ylen) in y_dims.items():
                    y_mbv2 = ey_mbs2[y_node]
                    assert y_mbv2.shape == (n_seeds, n_evlpnts2, n_ychnls, n_ylen)

                    y_mbv3 = torch.take_along_dim(y_mbv2, eii_mbv, dim=1)
                    assert y_mbv3.shape == (n_seeds, n_evlpnts, n_ychnls, n_ylen)

                    ey_mbvs1[y_node] = y_mbv3
            else:
                raise ValueError(f'undefined y type for variant: {y_typvrnt}')
        elif is_condvae:
            raise ValueError(f'unknown vrnt type {vrnt_type}')

        # The labels should be the same across all `n_rcns`
        ey_mbvs = dict()
        for y_node, (n_ychnls, n_ylen) in y_dims.items():
            y_mbv5 = ey_mbvs1[y_node]
            assert y_mbv5.shape == (n_seeds, n_evlpnts, n_ychnls, n_ylen)
            y_mbv6 = y_mbv5.unsqueeze(2)
            assert y_mbv6.shape == (n_seeds, n_evlpnts, 1, n_ychnls, n_ylen)
            y_mbv = y_mbv6.expand(n_seeds, n_evlpnts, n_rcns, n_ychnls, n_ylen)
            assert y_mbv.shape == (n_seeds, n_evlpnts, n_rcns, n_ychnls, n_ylen)
            ey_mbvs[y_node] = y_mbv

        #######################################################################
        ######################## Sampling the Latents  ########################
        #######################################################################
        # The type of sigma multiplication. Either 'mul' or 'matmul'.
        sigma_multype, n_epsdim = 'mul', None
        if vrnt_type == 'orig':
            emu_mbvs, ez_mbvs = dict(), dict()
            for z_node, z_dim in z_dims.items():
                emu_mb = emu_mbs[z_node]
                assert emu_mb.shape == (n_seeds, n_evlpnts, *z_dim)
                emu_mbv =  emu_mb.unsqueeze(2)
                assert emu_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                ez_mbv = emu_mbv
                assert ez_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                emu_mbvs[z_node] = emu_mbv
                ez_mbvs[z_node] = ez_mbv
            
            esigma_mbvs = dict()
            for z_node, z_dim in z_dims.items():
                esigma_mb = esigma_mbs[z_node]
                assert esigma_mb.shape == (n_seeds, n_evlpnts, *z_dim)
                esigma_mbv = esigma_mb.unsqueeze(2)
                assert esigma_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                esigma_mbvs[z_node] = esigma_mbv
            
            ei_mbv = ei_mb.unsqueeze(2)
            assert ei_mbv.shape == (n_seeds, n_evlpnts, n_rcns)
        elif vrnt_type in ('rcnst', 'genr'):
            z_typvrnt = e_vrntcfg2.pop('ltnt/type')
            e_vsigscl = e_vrntcfg2.pop('ltnt/sigscale')

            if z_typvrnt == 'orig':
                emu_mbvs = dict()
                for z_node, z_dim in z_dims.items():
                    emu_mb = emu_mbs[z_node]
                    assert emu_mb.shape == (n_seeds, n_evlpnts, *z_dim)
                    emu_mbv = emu_mb.unsqueeze(2)
                    assert emu_mbv.shape == (n_seeds, n_evlpnts, 1, *z_dim)
                    emu_mbvs[z_node] = emu_mbv

                esigma_mbvs = dict()
                for z_node, z_dim in z_dims.items():
                    esigma_mb = esigma_mbs[z_node]
                    assert esigma_mb.shape == (n_seeds, n_evlpnts, *z_dim)
                    esigma_mbv = esigma_mb.unsqueeze(2)
                    assert esigma_mbv.shape == (n_seeds, n_evlpnts, 1, *z_dim)
                    esigma_mbvs[z_node] = esigma_mbv
            elif z_typvrnt == 'normal':
                mu_spltvrnt = e_vrntcfg2.pop('ltnt/mean/type', 'zero')
                sigma_spltvrnt = e_vrntcfg2.pop('ltnt/cov/type', 'identity')

                if mu_spltvrnt == 'zero':
                    emu_mbvs = {z_node: torch.zeros(n_seeds, n_evlpnts, 1, *z_dim,
                        device=tch_device, dtype=tch_dtype) for z_node, z_dim in z_dims.items()}
                else:
                    (emu_mbs2, esigma_mbs2, exd_mbs2, u_eppmbs2, ey_mbs2, ei_mb2, ej_mb2, n_evlpnts2) = e_encpkgs[mu_spltvrnt]

                    n_evlpnts2 = ei_mb2.shape[1]
                    
                    emu_mbvs = dict()
                    for z_node, z_dim in z_dims.items():
                        emu_mb2 = emu_mbs2[z_node]
                        assert emu_mb2.shape == (n_seeds, n_evlpnts2, *z_dim)
                        emu_mbv3 = emu_mb2.mean(dim=1, keepdim=True).unsqueeze(dim=2)
                        assert emu_mbv3.shape == (n_seeds, 1, 1, *z_dim)
                        emu_mbv = emu_mbv3.expand(n_seeds, n_evlpnts, 1, *z_dim)
                        assert emu_mbv.shape == (n_seeds, n_evlpnts, 1, *z_dim)
                        emu_mbvs[z_node] = emu_mbv
                
                if sigma_spltvrnt == 'identity':
                    esigma_mbvs = {z_node: torch.ones(n_seeds, n_evlpnts, 1, *z_dim, 
                        device=tch_device, dtype=tch_dtype) for z_node, z_dim in z_dims.items()}
                else:
                    # The subsampling size of the covariance matrix data just for 
                    # estimating the slicing directions. A null value will default to using the 
                    # entire covariance data for the related computations.
                    n_sampsig = e_vrntcfg2.pop('ltnt/cov/nsamp', None)

                    (emu_mbs3, esigma_mbs3, exd_mbs3, u_eppmbs3, ey_mbs3, ei_mb3, ej_mb3, n_evlpnts3) = e_encpkgs[sigma_spltvrnt]
                    n_evlpnts3 = ei_mb3.shape[1]

                    esigma_mbvs = dict()
                    for z_node, z_dim in z_dims.items():
                        emu_mb3 = emu_mbs3[z_node]
                        assert emu_mb3.shape == (n_seeds, n_evlpnts3, *z_dim)

                        if (n_sampsig is None) or (n_sampsig >= n_evlpnts3):
                            emu_mb4, n_epsdim = emu_mb3, n_evlpnts3
                            assert emu_mb4.shape == (n_seeds, n_epsdim, *z_dim)
                        else:
                            n_epsdim = n_sampsig
                            
                            i_sigsamp = erng.rand(n_seeds, n_evlpnts3, 1).argsort(dim=1)[:, :n_sampsig, :]
                            assert i_sigsamp.shape == (n_seeds, n_epsdim, 1)

                            emu_mb4 = torch.take_along_dim(emu_mb3, i_sigsamp, dim=1)
                            assert emu_mb4.shape == (n_seeds, n_epsdim, *z_dim)

                        esigma_mbvunscld = emu_mb4 - emu_mb4.mean(dim=1, keepdims=True)
                        assert esigma_mbvunscld.shape == (n_seeds, n_epsdim, *z_dim)

                        # Scaling the cov matrix by the number of samples (i.e., dividing X^T X by N)
                        esigma_mbv = esigma_mbvunscld / math.sqrt(n_epsdim)
                        assert esigma_mbv.shape == (n_seeds, n_epsdim, *z_dim)
                        
                        esigma_mbvs[z_node] = esigma_mbv

                    sigma_multype, n_epsdim = 'matmul', n_epsdim
            elif z_typvrnt == 'knn':
                assert is_condvae
                z_spltvrnt = e_vrntcfg2.pop('ltnt/split')

                # Getting the split encoded data
                (emu_mbs2, esigma_mbs2, exd_mbs2, u_eppmbs2, ey_mbs2, ei_mb2, ej_mb2, n_evlpnts2) = e_encpkgs[z_spltvrnt]
                n_evlpnts2 = ei_mb2.shape[1]
                assert ei_mb2.shape == (n_seeds, n_evlpnts2)

                # Compiling the knn search data
                d_ycat = 0
                src_dataasgnlst, trg_dataasgnlst = [], []
                for y_node, (n_ychnls, n_ylen) in y_dims.items():
                    y_mbv1 = ey_mbvs1[y_node]
                    assert y_mbv1.shape == (n_seeds, n_evlpnts, n_ychnls, n_ylen)
                    y_mbv2 = y_mbv1.flatten(2, -1)
                    assert y_mbv2.shape == (n_seeds, n_evlpnts, n_ychnls * n_ylen)

                    y_mbv3 = ey_mbs2[y_node]
                    assert y_mbv3.shape == (n_seeds, n_evlpnts2, n_ychnls, n_ylen)
                    y_mbv4 = y_mbv3.flatten(2, -1)
                    assert y_mbv4.shape == (n_seeds, n_evlpnts2, n_ychnls * n_ylen)

                    src_dataasgnlst.append(y_mbv2)
                    trg_dataasgnlst.append(y_mbv4)
                    d_ycat += y_mbv2.shape[-1]

                assert d_ycat > 0, dedent(f'''
                    In order to perform a label-knn search to assign 
                    latent variables, the labels cannot be empty:
                        eval name: {eid}
                        split name: {e_split}
                        variant name: {e_vrnt}''')

                src_dataasgn = torch.cat(src_dataasgnlst, dim=-1)
                assert src_dataasgn.shape == (n_seeds, n_evlpnts, d_ycat)

                trg_dataasgn = torch.cat(trg_dataasgnlst, dim=-1)
                assert trg_dataasgn.shape == (n_seeds, n_evlpnts2, d_ycat)

                # Running the KNN search using faiss or torch
                assgn_k = n_rcns
                n_mbsrcasgn, n_mbtrgasgn, dim_db = n_evlpnts, n_evlpnts2, d_ycat
                asgn_cfg = get_subdict(e_vrntcfg2, 'ltnt/asgn', pop=True)

                try:
                    asgn_lib = asgn_cfg.pop('lib')
                    if asgn_lib == 'faiss':
                        trg_idxall = Evaluation.run_faissknn(asgn_cfg, src_dataasgn, trg_dataasgn,
                            n_mbsrcasgn, n_mbtrgasgn, dim_db, assgn_k, n_seeds, tch_device)
                        assert trg_idxall.shape == (n_seeds, n_mbsrcasgn, assgn_k)
                    elif asgn_lib == 'torch':
                        trg_idxall = Evaluation.run_torchknn(asgn_cfg, src_dataasgn, trg_dataasgn,
                            n_mbsrcasgn,  n_mbtrgasgn, dim_db, assgn_k, n_seeds, tch_dtype)
                        assert trg_idxall.shape == (n_seeds, n_mbsrcasgn, assgn_k)
                    else:
                        raise ValueError(f'undefined assignment library = {asgn_lib}')
                except Exception as exc:
                    exc.add_note(dedent(f'''
                        This error happened while running the knn search 
                        to assign latent variables:
                            eval name: {eid}
                            split name: {e_split}
                            variant name: {e_vrnt}'''))
                    raise exc

                trg_idxall2 = trg_idxall.reshape(n_seeds, n_evlpnts * n_rcns, 1)
                assert trg_idxall2.shape == (n_seeds, n_evlpnts * n_rcns, 1)

                emu_mbvs = dict()
                for z_node, z_dim in z_dims.items():
                    emu_mb2 = emu_mbs2[z_node]
                    assert emu_mb2.shape == (n_seeds, n_evlpnts2, *z_dim)
                    emu_mbv1 = torch.take_along_dim(emu_mb2, trg_idxall2, dim=1)
                    assert emu_mbv1.shape == (n_seeds, n_evlpnts * n_rcns, *z_dim)
                    emu_mbv = emu_mbv1.unflatten(1, (n_evlpnts, n_rcns))
                    assert emu_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    emu_mbvs[z_node] = emu_mbv
                
                esigma_mbvs = dict()
                for z_node, z_dim in z_dims.items():
                    esigma_mb2 = esigma_mbs2[z_node]
                    assert esigma_mb2.shape == (n_seeds, n_evlpnts2, *z_dim)
                    esigma_mbv1 = torch.take_along_dim(esigma_mb2, trg_idxall2, dim=1)
                    assert esigma_mbv1.shape == (n_seeds, n_evlpnts * n_rcns, *z_dim)
                    esigma_mbv = esigma_mbv1.unflatten(1, (n_evlpnts, n_rcns))
                    assert esigma_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    esigma_mbvs[z_node] = esigma_mbv
            else:
                raise ValueError(f'undefined ltnt/type: {z_typvrnt}')

            # Decoding the pre-processed input
            if sigma_multype == 'mul':
                assert n_epsdim is None, 'when did this happen?!'

                ez_mbvs = dict()
                for z_node, z_dim in z_dims.items():
                    emu_mbv = emu_mbvs[z_node]
                    assert emu_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    esigma_mbv = esigma_mbvs[z_node]
                    assert esigma_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    eeps_mbv = erng.randn(n_seeds, n_evlpnts, n_rcns, *z_dim)
                    assert eeps_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    ez_mbv = emu_mbv + eeps_mbv * esigma_mbv * e_vsigscl
                    assert ez_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    ez_mbvs[z_node] = ez_mbv
            elif sigma_multype == 'matmul':
                assert n_epsdim is not None, 'this shouldnt have happened!'

                ez_mbvs = dict()
                for z_node, z_dim in z_dims.items():
                    emu_mbv = emu_mbvs[z_node]
                    assert emu_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    esigma_mbv = esigma_mbvs[z_node]
                    assert esigma_mbv.shape == (n_seeds, n_epsdim, *z_dim)
                    eeps_mbv = erng.randn(n_seeds, n_evlpnts * n_rcns, n_epsdim)
                    assert eeps_mbv.shape == (n_seeds, n_evlpnts * n_rcns, n_epsdim)
                    ez_mbv1 = eeps_mbv.matmul(esigma_mbv.flatten(2, -1)).unflatten(2, z_dim)
                    assert ez_mbv1.shape == (n_seeds, n_evlpnts * n_rcns, *z_dim)
                    ez_mbv2 = ez_mbv1.unflatten(1, (n_evlpnts, n_rcns))
                    assert ez_mbv2.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    ez_mbv = emu_mbv + ez_mbv2 * e_vsigscl
                    assert ez_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    ez_mbvs[z_node] = ez_mbv
            else:
                raise ValueError(f'undefined sigma_multype={sigma_multype}')

            # Preventing these useless variables from getting passed around or stored
            emu_mbv, esigma_mbv, ei_mbv = None, None, None
        else:
            raise ValueError(f'unknown vrnt type {vrnt_type}')

        assert len(e_vrntcfg2) == 0, f'unused options: {e_vrntcfg2}'

        return emu_mbvs, esigma_mbvs, ez_mbvs, ey_mbvs, ei_mbv, n_rcns, n_epsdim, sigma_multype, vrnt_type

    @staticmethod
    @torch.no_grad()
    def decode(ez_mbvs, ey_mbvs, dec_pp, decoder, e_pp, e_ppinpspc, lbl_pp, lblnn,
        is_condvae, lblnn_enctype, n_seeds, n_evlpnts, n_rcns, eval_bsnn, 
        xd_dims, ud_dims, uv_dims, xl_dims, ul_dims, z_dims, y_dims):

        #######################################
        #         Applying the Decoder        #
        #######################################
        ez_mbfs = dict()
        for z_node, z_dim in z_dims.items():
            ez_mbv = ez_mbvs[z_node]
            assert ez_mbv.shape == (n_seeds, n_evlpnts, n_rcns, *z_dim)
            ez_mbf = ez_mbv.flatten(1, 2)
            assert ez_mbf.shape == (n_seeds, n_evlpnts * n_rcns, *z_dim)
            ez_mbfs[z_node] = ez_mbf

        ey_mbfs = dict()
        for y_node, (n_ychnls, n_ylen) in y_dims.items():
            ey_mbv = ey_mbvs[y_node]
            assert ey_mbv.shape == (n_seeds, n_evlpnts, n_rcns, n_ychnls, n_ylen)
            ey_mbf = ey_mbv.flatten(1, 2)
            assert ey_mbf.shape == (n_seeds, n_evlpnts * n_rcns, n_ychnls, n_ylen)
            ey_mbfs[y_node] = ey_mbf

        assert 'z' not in ey_mbfs

        eudhat_mbs = get_nn_sol(decoder, {**ez_mbfs, **ey_mbfs}, n_eval=eval_bsnn)
        for u_node, (n_uchnls, n_ulen) in ud_dims.items():
            euhat_mbf = eudhat_mbs[u_node]
            assert euhat_mbf.shape == (n_seeds, n_evlpnts * n_rcns, n_uchnls, n_ulen)

        # This if statement is only added for performance reasons; there is
        # no later need for a `exdhat_mbs` variable if `e_ppinpspc == 'x'`.
        if (e_ppinpspc == 'x') or is_condvae:
            exdhat_mbs = dec_pp.inverse(eudhat_mbs, strict=True, full=False)
            for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                ex_mb = exdhat_mbs[x_node]
                assert ex_mb.shape == (n_seeds, n_evlpnts * n_rcns, n_xchnls, n_xlen)
        else:
            exdhat_mbs = None

        #######################################
        #         Applying the Labeler        #
        #######################################
        if is_condvae:
            # The labeler's pp input mini-batches
            exlhat_mbs = {x_node: exdhat_mbs[x_node] for x_node in xl_dims}

            # Next, we apply the input pre-processing transformations
            # before feeding it to the labeler.
            eulhat_mbs, uhat_shaper = lbl_pp.forward(exlhat_mbs, full=False)
            for u_node, (n_uchnls, n_ulen) in ul_dims.items():
                u_mb = eulhat_mbs[u_node]
                assert u_mb.shape == (n_seeds, n_evlpnts * n_rcns, n_uchnls, n_ulen)
            
            elblhatnnkws = dict()
            if lblnn_enctype == 'vae':
                elblhatnnkws = dict(n_seeds=n_seeds, n_mb=-1, sigscale=0)
            assert lblnn_enctype in ('vae', 'plain')

            eyhat_mbs = get_nn_sol(lblnn, eulhat_mbs, n_eval=eval_bsnn, model_kwargs=elblhatnnkws)
            assert isinstance(eyhat_mbs, dict)
        else:
            eyhat_mbs = dict()

        eyhat_mbvs = dict()
        for y_node, (n_ychnls, n_ylen) in y_dims.items():
            y_mb = eyhat_mbs[y_node]
            assert y_mb.shape == (n_seeds, n_evlpnts * n_rcns, n_ychnls, n_ylen)
            y_mbv = y_mb.unflatten(1, (n_evlpnts, n_rcns))
            assert y_mbv.shape == (n_seeds, n_evlpnts, n_rcns, n_ychnls, n_ylen)
            eyhat_mbvs[y_node] = y_mbv

        # Finding the right input to the evaluation pre-processor:
        #   * `epp_inphatmbs`: This is the reconstructed data fed to the `e_pp` pre-processor.
        #       * It should be either `eudhat_mbs` or `exdhat_mbs` depending on `e_ppinpspc`.
        if e_ppinpspc == 'x':
            epp_inphatmbs = exdhat_mbs
        elif e_ppinpspc == 'u':
            epp_inphatmbs = eudhat_mbs
        else:
            raise ValueError(f'undefined e_ppinpspc={e_ppinpspc}')

        # Applying the evaluation pre-processor to the reconstructed data.
        uhat_eppmbvs = dict()
        uhat_eppmbfs, uhat_eppshaper = e_pp.forward(epp_inphatmbs, full=False)
        for u_node, (n_uchnls, n_ulen) in uv_dims.items():
            eu_mbf = uhat_eppmbfs[u_node]
            assert eu_mbf.shape == (n_seeds, n_evlpnts * n_rcns, n_uchnls, n_ulen)
            eu_mbv = eu_mbf.unflatten(1, (n_evlpnts, n_rcns))
            assert eu_mbv.shape == (n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen)
            uhat_eppmbvs[u_node] = eu_mbv

        return uhat_eppmbvs, eyhat_mbvs

    @staticmethod
    @torch.no_grad()
    def assemble(eid, e_split, eis_vrntdatas, n_seeds, n_evlpnts, uv_dims, z_dims, y_dims):
        # Example:
        #   e_vrnt in ('orig', 'rcnst', 'genr', ...)
        #   e_space in ('x', 'z', 'i')
        #   e_node in ('z', 'm_chmprtnrm', ...)
        #   e_repr in ('pnts', 'mu', 'sig', 'idxs')
        
        mtrc_keyfmt = '{eid}/{e_split}/{e_vrnt}/{e_space}/{e_node}/{e_repr}'
        viz_keyfmt = '{eid}:{e_space}:{e_node}/{e_split}/{e_vrnt}/{e_repr}'

        key_infos = []

        for e_vrnt, eis_vrntdata in eis_vrntdatas.items():
            (emu_mbvs, esigma_mbvs, ez_mbvs, ey_mbvs2, uhat_eppmbvs, ei_mbv, n_rcns, n_epsdim, sigma_multype) = eis_vrntdata

            # Adding the decoding variables in the `z` space
            if ez_mbvs is not None:
                for z_node, ez_mbv in ez_mbvs.items():
                    z_dim = z_dims[z_node]
                    key_infos += [
                        (eid, e_split, e_vrnt, 'z', z_node, 'pnts', ez_mbv,     
                        (n_seeds, n_evlpnts, n_rcns, *z_dim))]
            if emu_mbvs is not None:
                for z_node, emu_mbv in emu_mbvs.items():
                    z_dim = z_dims[z_node]
                    key_infos += [
                        (eid, e_split, e_vrnt, 'z', z_node, 'mu', emu_mbv,    
                        (n_seeds, n_evlpnts, n_rcns, *z_dim))]
            if esigma_mbvs is not None:
                for z_node, esigma_mbv in esigma_mbvs.items():
                    z_dim = z_dims[z_node]
                    if sigma_multype == 'mul':
                        e_shape = (n_seeds, n_evlpnts, n_rcns, *z_dim)
                    elif sigma_multype == 'matmul':
                        e_shape = (n_seeds, n_epsdim, *z_dim)
                        # TODO: Figure out a way to add to key_infos without mds(...) complaining 
                    else:
                        raise ValueError(f'undefined sigma_multype={sigma_multype}')
                    if sigma_multype != 'matmul':
                        key_infos += [
                            (eid, e_split, e_vrnt, 'z', z_node, 'sig', esigma_mbv, e_shape)]
            if ei_mbv is not None:
                key_infos += [
                    (eid, e_split, e_vrnt, 'i', 'z', 'idxs', ei_mbv,
                    (n_seeds, n_evlpnts, n_rcns))]

            # Adding the variables in the `y` space
            if ey_mbvs2 is not None:
                for y_node, ey_mbv in ey_mbvs2.items():
                    (n_ychnls, n_ylen) = y_dims[y_node]
                    key_infos += [
                        (eid, e_split, e_vrnt, 'y', y_node, 'pnts', ey_mbv, 
                        (n_seeds, n_evlpnts, n_rcns, n_ychnls, n_ylen))]

            # Adding the decoded variables in the `x` space
            if uhat_eppmbvs is not None:
                for u_node, uhat_eppmbv in uhat_eppmbvs.items():
                    (n_uchnls, n_ulen) = uv_dims[u_node]
                    key_infos += [
                        (eid, e_split, e_vrnt,  'x', u_node, 'pnts', uhat_eppmbv,
                        (n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen))]

        eis_mtrcinputs, eis_vizinputs = dict(), dict()
        for keyfmt, eis_inputs in ((mtrc_keyfmt, eis_mtrcinputs), (viz_keyfmt, eis_vizinputs)):
            for eid, e_split, e_vrnt, e_space, e_node, e_repr, e_tnsr, e_shape in key_infos:
                assert e_tnsr.shape == e_shape
                key = keyfmt.format(eid=eid, e_split=e_split, e_vrnt=e_vrnt, 
                    e_space=e_space, e_node=e_node, e_repr=e_repr)
                eis_inputs[key] = e_tnsr

        return eis_mtrcinputs, eis_vizinputs

    @staticmethod
    def make_faissdb(asgn_cfg, dim_db, tch_device):
        asgn_cfgcp = dict(asgn_cfg)

        # The faiss index type
        indextype_faiss = asgn_cfgcp.pop('index')

        if indextype_faiss == 'IndexFlatL2':
            # Brute-force search against all the database
            db_faisscpu = faiss.IndexFlatL2(dim_db)

            # n_probe has no meaning for exact search
            nprobe_faiss = None
        elif indextype_faiss in ('IndexIVFFlat', 'IndexIVFPQ'):
            # The quantized input to the high-level k-means
            qntzrtype_faiss = asgn_cfgcp.pop('quantizer')

            # Constructing the quantizer
            assert qntzrtype_faiss == 'IndexFlatL2', dedent('''
                Only quantizer = IndexFlatL2 is supported''')
            qntzr_faiss = faiss.IndexFlatL2(dim_db)

            # The high-level index metric (not the quantizer)
            metrictype_faiss = asgn_cfgcp.pop('metric')
            assert metrictype_faiss == 'METRIC_L2'
            metric_faiss = faiss.METRIC_L2

            # The number of clusters/centroids for the high-level k-means
            nlist_faiss = asgn_cfgcp.pop('nlist')

            if indextype_faiss == 'IndexIVFFlat':
                # No quantization before k-means is needed and a flat index is good enough
                db_faisscpu = faiss.IndexIVFFlat(qntzr_faiss, dim_db, nlist_faiss, metric_faiss)
            elif indextype_faiss == 'IndexIVFPQ':
                # Number of segments for product quantization (e.g., if you
                # have a 128-d vector data and want to perform PQ on each 2
                # dimensions, you will have M = 128 / 2 = 64 segments).
                nsegs_faiss = asgn_cfgcp.pop('M')

                # The number of bits to encode each segment.
                nbits_faiss = asgn_cfgcp.pop('nbits')

                db_faisscpu = faiss.IndexIVFPQ(qntzr_faiss, dim_db, nlist_faiss,
                                               nsegs_faiss, nbits_faiss, metric_faiss)
            else:
                raise ValueError(f'undefined indextype_faiss={indextype_faiss}')

            # The number of closest clusters to search for each query
            nprobe_faiss = asgn_cfgcp.pop('nprobe')
            # Setting the probe size
            db_faisscpu.nprobe = nprobe_faiss
        else:
            raise ValueError(f'undefined faiss index={indextype_faiss}')

        if tch_device.type == 'cuda':
            faiss_rsrc = faiss.StandardGpuResources()
            db_faiss = faiss.index_cpu_to_gpu(faiss_rsrc, tch_device.index, db_faisscpu)
        else:
            assert tch_device.type == 'cpu', f'faiss for {tch_device.type} is undefined'
            db_faiss = db_faisscpu

        assert len(asgn_cfgcp) == 0, f'unused assignment options: {asgn_cfg}'

        return db_faiss, indextype_faiss
    
    @staticmethod
    def run_faissknn(asgn_cfg, src_dataasgn, trg_dataasgn, n_mbsrcasgn, 
        n_mbtrgasgn, dim_db, assgn_k, n_seeds, tch_device):
        
        # Making sure the input data has no NaN or inf entries.
        assert not trg_dataasgn.isnan().any()
        assert not trg_dataasgn.isinf().any()
        assert not src_dataasgn.isnan().any()
        assert not src_dataasgn.isinf().any()
        
        trg_indxslst = []
        # The faiss lookup can only be done one seed at a time.
        for i_seed in range(n_seeds):
            # Making an empty fass index
            db_faiss, indextype_faiss = Evaluation.make_faissdb(asgn_cfg, dim_db, tch_device)

            # Note:
            #   To be able to add/search pytorch cuda tensors directly
            #   to faiss, you must import the following line:
            #       `import faiss.contrib.torch_utils`
            #   source: https://github.com/facebookresearch/faiss/issues/3127
            db_trgdata = trg_dataasgn[i_seed].reshape(n_mbtrgasgn, dim_db)
            db_trgdata = db_trgdata.to(dtype=torch.float32)
            assert db_trgdata.shape == (n_mbtrgasgn, dim_db)

            # Training the faiss database
            if indextype_faiss == 'IndexFlatL2':
                # no training is necessary for exact brute-force search
                pass
            elif indextype_faiss in ('IndexIVFFlat', 'IndexIVFPQ'):
                # The k-means in the `IndexIVF*` needs training
                assert not db_faiss.is_trained
                db_faiss.train(db_trgdata)
                assert db_faiss.is_trained
            else:
                raise ValueError(f'undefined training for "{indextype_faiss}"')

            # Adding the target data
            db_faiss.add(db_trgdata)

            # Preparing the target data
            db_srcdata = src_dataasgn[i_seed].reshape(n_mbsrcasgn, dim_db)
            db_srcdata = db_srcdata.to(dtype=torch.float32)
            assert db_srcdata.shape == (n_mbsrcasgn, dim_db)

            # Performing the search
            srctrgdsts, trg_indxsseed = db_faiss.search(db_srcdata, assgn_k)
            assert trg_indxsseed.shape == (n_mbsrcasgn, assgn_k)
            trg_indxslst.append(trg_indxsseed)
            
            # Freeing the memory of faiss index
            db_faiss.reset()
            del db_faiss

        trg_idxall = torch.stack(trg_indxslst, dim=0)
        assert trg_idxall.shape == (n_seeds, n_mbsrcasgn, assgn_k)

        return trg_idxall

    @staticmethod
    def run_torchknn(asgn_cfg, src_dataasgn, trg_dataasgn, n_mbsrcasgn, 
        n_mbtrgasgn, dim_db, assgn_k, n_seeds, tch_dtype):
        ram_bytes = int(asgn_cfg.pop('ram/mb', 400) * 1e6)
        tch_dtypebytes = torch.finfo(tch_dtype).bits // 8
        n_mbsrch_ = max(1, ram_bytes // (n_mbtrgasgn * dim_db * tch_dtypebytes))
        n_itersrch = math.ceil(n_mbsrcasgn / n_mbsrch_)

        db_trgdata = trg_dataasgn.reshape(n_seeds, 1, n_mbtrgasgn, dim_db)
        assert db_trgdata.shape == (n_seeds, 1, n_mbtrgasgn, dim_db)

        trg_indxslst = []
        for i_itersrch in range(n_itersrch):
            i1_srch = i_itersrch * n_mbsrch_
            i2_srch = min(n_mbsrcasgn, i1_srch + n_mbsrch_)
            n_mbsrch = i2_srch - i1_srch

            db_srcdatamb = src_dataasgn[:, i1_srch:i2_srch, ...].reshape(n_seeds, n_mbsrch, 1, dim_db)
            assert db_srcdatamb.shape == (n_seeds, n_mbsrch, 1, dim_db)

            delta_srctrgmb = db_srcdatamb - db_trgdata
            assert delta_srctrgmb.shape == (n_seeds, n_mbsrch, n_mbtrgasgn, dim_db)

            deltasq_srctrgmb = delta_srctrgmb.square_().sum(dim=-1)
            assert deltasq_srctrgmb.shape == (n_seeds, n_mbsrch, n_mbtrgasgn)

            trg_indxsmb = deltasq_srctrgmb.topk(k=assgn_k, dim=-1, sorted=True, largest=False).indices
            assert trg_indxsmb.shape == (n_seeds, n_mbsrch, assgn_k)
            
            trg_indxslst.append(trg_indxsmb)
        
        trg_idxall = torch.cat(trg_indxslst, dim=-2)
        assert trg_idxall.shape == (n_seeds, n_mbsrcasgn, assgn_k)

        assert len(asgn_cfg) == 0, f'unused asgn cfg options: {asgn_cfg}'
        
        return trg_idxall

    @staticmethod
    def check_mengereval(mtrc_id, src_eid, src_splt, src_lbl, trg_eid, 
        trg_splt, trg_lbl, asgn_type, eparams, split_vars):

        # Making sure there is no assignment process
        assert asgn_type == 'same', dedent(f'''
            The menger curvature cannot accept any type of 
            assignment other than "same":
                asgn_type: {asgn_type}
                metric: {mtrc_id}
        ''')

        # Making sure there is not a target data
        assert ((src_eid, src_splt, src_lbl) == 
                (trg_eid, trg_splt, trg_lbl)), dedent(f'''
            The menger curvature requires identical source 
            and target data (the target data is ignored for now):
                source: {src_eid}/{src_splt}/{src_lbl}
                target: {trg_eid}/{trg_splt}/{trg_lbl}
                metric: {mtrc_id}
        ''')

        # Making sure the source data is not shuffled or sampled in an iid manner
        src_eopts = eparams[src_eid]
        src_erndmztn = src_eopts['rndmztn']
        assert src_erndmztn == 'none', dedent(f'''
            The menger curvature calculation requires the source data 
            evaluation ordering to be "none"; there should be no shuffling or 
            iid sample applied to the trajectory points:
                src/eid: {src_eid}
                rndmztn: {src_erndmztn}
                metric: {mtrc_id}
        ''')

        # Making sure the split strategy was based on scenario and time
        assert split_vars == ['snr', 't'], dedent(f'''
            The menger curvature calculation requires the split variables to be 
            `["snr", "t"]`. This is because we need the trajectory orderings to 
            be intact:
                split_vars: {split_vars}
                metric: {mtrc_id}
        ''')

    @staticmethod
    def metric(mtrc_cfg, mtrc_id, src_mtrcdatas, trg_mtrcdatas, aero_cstinfo, 
        n_seeds, n_t, mrng, tch_device, tch_dtype, tch_cdtype):
        # The metric type (e.g., pnorm, wass, slcdwass, etc.)
        mtrc_type = mtrc_cfg.pop('type')

        # Metric name format. This is usually the same as the
        # metric id. However, some metric "id"s produce multiple
        # outputs and thus they will create multiple "names".
        mtrc_namefmt = mtrc_cfg.pop('name/fmt', mtrc_id)
        assert '/' not in mtrc_namefmt, dedent(f'''
            metric name format cannot include a "/":
                mtrc_namefmt = {mtrc_namefmt}''')

        # A mapping of metric names to the calculated pre-stat
        # metric data. For example:
        #    mtrc_pntsdata = {
        #       'wass_1^1': torch.randn(n_seeds, n_pnts, n_rcns),
        #       'wass_2^1': torch.randn(n_seeds, n_pnts, n_rcns),
        #       'wass_1^2': torch.randn(n_seeds, n_pnts, n_rcns),
        #       'wass_2^2': torch.randn(n_seeds, n_pnts, n_rcns)}
        mtrc_pntsdata = dict()
        n_pnts, n_rcns = None, None

        # Whether to be strict about the extra data keys
        ignore_xtrareprs = True

        # The mathematical metrics can only work with a single node
        if mtrc_type in ('pnorm', 'rel', 'mahalanobis', 'tv', 'wass', 'hellinger', 
            'kl', 'icc', 'slcdwass', 'menger'):
            assert (len(src_mtrcdatas) == 1) and (len(trg_mtrcdatas) == 1), dedent(f'''
                The metric type "{mtrc_type}" can only take a single 
                source node as input.
                    provided src nodes: {list(src_mtrcdatas)}
                    provided trg nodes: {list(trg_mtrcdatas)}''')
            
            src_mtrcdata, = list(src_mtrcdatas.values())
            trg_mtrcdata, = list(trg_mtrcdatas.values())
        else:
            src_mtrcdata, trg_mtrcdata = None, None

        if mtrc_type in ('pnorm', 'rel'):
            if not ignore_xtrareprs:
                assert tuple(src_mtrcdata.keys()) == ('pnts',), dedent(f'''
                    The pnorm metric can only be applied to points data:
                        src_mtrcdata.keys = {src_mtrcdata.keys()}''')

                assert tuple(trg_mtrcdata.keys()) == ('pnts',), dedent(f'''
                The pnorm metric can only be applied to points data:
                    trg_mtrcdata.keys = {trg_mtrcdata.keys()}''')

            src_pnts = src_mtrcdata['pnts'].flatten(3, -1)
            n_mbsrc, n_srcrcns, dim_src = src_pnts.shape[1:]
            assert src_pnts.shape == (n_seeds, n_mbsrc, n_srcrcns, dim_src)

            trg_pnts = trg_mtrcdata['pnts'].flatten(3, -1)
            n_mbtrg, n_trgrcns, dim_trg = trg_pnts.shape[1:]
            assert trg_pnts.shape == (n_seeds, n_mbtrg, n_trgrcns, dim_trg)

            n_pnts = n_mbsrc
            assert n_mbsrc == n_mbtrg, dedent(f'''
                The pnorm metric requires the source and target data 
                to have the same number of points:
                    n_mbsrc = {n_mbsrc}
                    n_mbtrg = {n_mbtrg}''')

            n_rcns = get_nbcst(n_srcrcns, n_trgrcns)
            assert n_rcns is not None, dedent(f'''
                The pnorm metric requires the source and target data 
                to have broadcastable n_rcns:
                    n_srcrcns = {n_srcrcns}
                    n_trgrcns = {n_trgrcns}''')

            if mtrc_type == 'pnorm':
                err_srctrgabs = (src_pnts - trg_pnts).abs()
                assert err_srctrgabs.shape == (n_seeds, n_pnts, n_rcns, dim_src)

                # The metric's p-norm
                mtrc_plst_ = mtrc_cfg.pop('pnorm/p')
                mtrc_plst = (mtrc_plst_,) if isinstance(mtrc_plst_, (int, float)) else mtrc_plst_
                assert isinstance(mtrc_plst, (tuple, list))

                # The metric's exponent
                mtrc_explst_ = mtrc_cfg.pop('pnorm/exp')
                mtrc_explst = (mtrc_explst_,) if isinstance(mtrc_explst_, (int, float)) else mtrc_explst_
                assert isinstance(mtrc_explst, (tuple, list))
                
                # The metric's reduction method
                mtrc_rdctn = mtrc_cfg.pop('pnorm/rdc')
                assert mtrc_rdctn in ('sum', 'mean')

                for mtrc_p in mtrc_plst:
                    err_srctrgpowfull = err_srctrgabs.pow(mtrc_p)
                    assert err_srctrgpowfull.shape == (n_seeds, n_pnts, n_rcns, dim_src)

                    if mtrc_rdctn == 'sum':
                        err_srctrgpow = err_srctrgpowfull.sum(dim=-1)
                        assert err_srctrgpow.shape == (n_seeds, n_pnts, n_rcns)
                    elif mtrc_rdctn == 'mean':
                        err_srctrgpow = err_srctrgpowfull.mean(dim=-1)
                        assert err_srctrgpow.shape == (n_seeds, n_pnts, n_rcns)
                    else:
                        raise ValueError(f'undefined mtrc_rdctn={mtrc_rdctn}')

                    for mtrc_exp in mtrc_explst:
                        mtrc_pnts = err_srctrgpow.pow(mtrc_exp / mtrc_p)
                        assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                        mtrc_name = mtrc_namefmt.format(p=mtrc_p, exp=mtrc_exp)
                        assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                            The metric name format produced clashing names for 
                            different settings. Make sure you make the name format 
                            specific to each setting.
                                name/fmt = {mtrc_namefmt}
                                pnorm/p = {mtrc_p}
                                pnorm/exp = {mtrc_exp}
                        ''')

                        mtrc_pntsdata[mtrc_name] = mtrc_pnts
            elif mtrc_type == 'rel':
                # The metric's p-norm
                mtrc_plst_ = mtrc_cfg.pop('pnorm/p')
                mtrc_plst = (mtrc_plst_,) if isinstance(mtrc_plst_, (int, float)) else mtrc_plst_
                assert isinstance(mtrc_plst, (tuple, list))

                err_srctrgabs = (src_pnts - trg_pnts).abs()
                assert err_srctrgabs.shape == (n_seeds, n_pnts, n_rcns, dim_src)

                src_pntsabs = src_pnts.abs()
                assert src_pntsabs.shape == (n_seeds, n_pnts, n_srcrcns, dim_src)

                trg_pntsabs = trg_pnts.abs()
                assert trg_pntsabs.shape == (n_seeds, n_pnts, n_trgrcns, dim_src)

                for mtrc_p in mtrc_plst:
                    mtrc_pntsnumer = err_srctrgabs.pow(mtrc_p).sum(dim=-1).pow(1.0 / mtrc_p)
                    assert mtrc_pntsnumer.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_pntsdenomsrc = src_pntsabs.pow(mtrc_p).sum(dim=-1).pow(1.0 / mtrc_p)
                    assert mtrc_pntsdenomsrc.shape == (n_seeds, n_pnts, n_srcrcns)

                    mtrc_pntsdenomtrg = trg_pntsabs.pow(mtrc_p).sum(dim=-1).pow(1.0 / mtrc_p)
                    assert mtrc_pntsdenomtrg.shape == (n_seeds, n_pnts, n_trgrcns)

                    mtrc_pntsdenom1 = mtrc_pntsdenomsrc + mtrc_pntsdenomtrg
                    assert mtrc_pntsdenom1.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_pntsdenom2 = torch.where(mtrc_pntsdenom1 == 0, 1.0, mtrc_pntsdenom1)
                    assert mtrc_pntsdenom2.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_pnts = mtrc_pntsnumer / mtrc_pntsdenom2
                    assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_name = mtrc_namefmt.format(p=mtrc_p)
                    assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                        The metric name format produced clashing names for 
                        different settings. Make sure you make the name format 
                        specific to each setting.
                            name/fmt = {mtrc_namefmt}
                            pnorm/p = {mtrc_p}
                            pnorm/exp = {mtrc_exp}
                    ''')
                    mtrc_pntsdata[mtrc_name] = mtrc_pnts
            else:
                raise ValueError(f'undefined mtrc_type={mtrc_type}')
        elif mtrc_type == 'mahalanobis':
            if not ignore_xtrareprs:
                assert tuple(src_mtrcdata.keys()) == ('pnts',), dedent(f'''
                    The mahalanobis metric can only be applied to 
                    "points" source data:
                        src_mtrcdata.keys = {src_mtrcdata.keys()}''')

                assert set(trg_mtrcdata.keys()) == {'mu', 'sig'}, dedent(f'''
                The mahalanobis metric can only be applied to 
                "blob" target data:
                    trg_mtrcdata.keys = {trg_mtrcdata.keys()}''')

            src_pnts = src_mtrcdata['pnts'].flatten(3, -1)
            n_mbsrc, n_srcrcns, dim_src = src_pnts.shape[1:]
            assert src_pnts.shape == (n_seeds, n_mbsrc, n_srcrcns, dim_src)

            trg_mus = trg_mtrcdata['mu'].flatten(3, -1)
            n_mbtrg, n_trgrcns, dim_trg = trg_mus.shape[1:]
            assert trg_mus.shape == (n_seeds, n_mbtrg, n_trgrcns, dim_trg)

            trg_sigs = trg_mtrcdata['sig'].flatten(3, -1)
            assert trg_sigs.shape == (n_seeds, n_mbtrg, n_trgrcns, dim_trg)

            n_pnts = n_mbsrc
            assert n_mbsrc == n_mbtrg, dedent(f'''
                The mahalanobis metric requires the source and target data 
                to have the same number of points:
                    n_mbsrc = {n_mbsrc}
                    n_mbtrg = {n_mbtrg}''')

            n_rcns = get_nbcst(n_srcrcns, n_trgrcns)
            assert n_rcns is not None, dedent(f'''
                The mahal metric requires the source and target data 
                to have broadcastable n_rcns:
                    n_srcrcns = {n_srcrcns}
                    n_trgrcns = {n_trgrcns}''')

            assert dim_src == dim_trg, dedent(f'''
                The mahalanobis metric requires the source and target data 
                to have the same data dimensions:
                    src_pnts.shape = {src_mtrcdata['pnts'].shape}
                    trg_mus.shape = {trg_mtrcdata['mu'].shape}
                    trg_sigs.shape = {trg_mtrcdata['sig'].shape}''')

            diff_srctrg = (src_pnts - trg_mus)
            assert diff_srctrg.shape == (n_seeds, n_pnts, n_rcns, dim_src)

            err_srctrgabs = (diff_srctrg / trg_sigs).abs()
            assert err_srctrgabs.shape == (n_seeds, n_pnts, n_rcns, dim_src)

            # The metric's p-norm
            mtrc_plst_ = mtrc_cfg.pop('pnorm/p')
            mtrc_plst = (mtrc_plst_,) if isinstance(mtrc_plst_, (int, float)) else mtrc_plst_
            assert isinstance(mtrc_plst, (tuple, list))

            # The metric's exponent
            mtrc_explst_ = mtrc_cfg.pop('pnorm/exp')
            mtrc_explst = (mtrc_explst_,) if isinstance(mtrc_explst_, (int, float)) else mtrc_explst_
            assert isinstance(mtrc_explst, (tuple, list))

            # The metric's reduction method
            mtrc_rdctn = mtrc_cfg.pop('pnorm/rdc')
            assert mtrc_rdctn in ('sum', 'mean')

            for mtrc_p in mtrc_plst:
                err_srctrgpowfull = err_srctrgabs.pow(mtrc_p)
                assert err_srctrgpowfull.shape == (n_seeds, n_pnts, n_rcns, dim_src)

                if mtrc_rdctn == 'sum':
                    err_srctrgpow = err_srctrgpowfull.sum(dim=-1)
                    assert err_srctrgpow.shape == (n_seeds, n_pnts, n_rcns)
                elif mtrc_rdctn == 'mean':
                    err_srctrgpow = err_srctrgpowfull.mean(dim=-1)
                    assert err_srctrgpow.shape == (n_seeds, n_pnts, n_rcns)
                else:
                    raise ValueError(f'undefined mtrc_rdctn={mtrc_rdctn}')

                for mtrc_exp in mtrc_explst:
                    mtrc_pnts = err_srctrgpow.pow(mtrc_exp / mtrc_p)
                    assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_name = mtrc_namefmt.format(p=mtrc_p, exp=mtrc_exp)
                    assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                        The metric name format produced clashing names for 
                        different settings. Make sure you make the name format 
                        specific to each setting.
                            name/fmt = {mtrc_namefmt}
                            pnorm/p = {mtrc_p}
                            pnorm/exp = {mtrc_exp}
                    ''')

                    mtrc_pntsdata[mtrc_name] = mtrc_pnts
        elif mtrc_type in ('tv', 'wass', 'hellinger', 'kl', 'icc'):
            if not ignore_xtrareprs:
                assert set(src_mtrcdata.keys()) == {'mu', 'sig'}, dedent(f'''
                    The {mtrc_type} metric can only be applied to 
                    "blob" data:
                        src_mtrcdata.keys = {src_mtrcdata.keys()}''')

                assert set(trg_mtrcdata.keys()) == {'mu', 'sig'}, dedent(f'''
                    The {mtrc_type} metric can only be applied to 
                    "blob" data:
                        trg_mtrcdata.keys = {trg_mtrcdata.keys()}''')

            src_mus = src_mtrcdata['mu'].flatten(3, -1)
            n_mbsrc, n_srcrcns, dim_src = src_mus.shape[1:]
            assert src_mus.shape == (n_seeds, n_mbsrc, n_srcrcns, dim_src)

            src_sigs = src_mtrcdata['sig'].flatten(3, -1)
            assert src_sigs.shape == (n_seeds, n_mbsrc, n_srcrcns, dim_src)

            trg_mus = trg_mtrcdata['mu'].flatten(3, -1)
            n_mbtrg, n_trgrcns, dim_trg = trg_mus.shape[1:]
            assert trg_mus.shape == (n_seeds, n_mbtrg, n_trgrcns, dim_trg)

            trg_sigs = trg_mtrcdata['sig'].flatten(3, -1)
            assert trg_sigs.shape == (n_seeds, n_mbtrg, n_trgrcns, dim_trg)

            n_pnts = n_mbsrc
            assert n_mbsrc == n_mbtrg, dedent(f'''
                The mahalanobis metric requires the source and target data 
                to have the same number of points:
                    n_mbsrc = {n_mbsrc}
                    n_mbtrg = {n_mbtrg}''')

            n_rcns = get_nbcst(n_srcrcns, n_trgrcns)
            assert n_rcns is not None, dedent(f'''
                The mahalanobis metric requires the source and target data 
                to have broadcastable n_rcns:
                    n_srcrcns = {n_srcrcns}
                    n_trgrcns = {n_trgrcns}''')

            assert dim_src == dim_trg, dedent(f'''
                The mahalanobis metric requires the source and target data 
                to have the same data dimensions:
                    src_mus.shape = {src_mtrcdata['mu'].shape}
                    src_sigs.shape = {src_mtrcdata['sig'].shape}
                    trg_mus.shape = {trg_mtrcdata['mu'].shape}
                    trg_sigs.shape = {trg_mtrcdata['sig'].shape}''')


            delta_sigs = (src_sigs - trg_sigs)
            assert delta_sigs.shape == (n_seeds, n_pnts, n_rcns, dim_src)

            delta_mus = (src_mus - trg_mus)
            assert delta_mus.shape == (n_seeds, n_pnts, n_rcns, dim_src)

            if mtrc_type  == 'tv':
                # The only implemented process for approximating KL
                # for now is to use importance sampling.
                tv_approxmthd = mtrc_cfg.pop('approx')
                assert tv_approxmthd == 'is', dedent(f'''
                    Only importance sampling ("is") is supported 
                    for TV approximation:
                            tv_approxmthd = {tv_approxmthd}''')

                src_logdetsig = src_sigs.log().sum(dim=-1, keepdims=True)
                assert src_logdetsig.shape == (n_seeds, n_pnts, n_rcns, 1)

                trg_logdetsig = trg_sigs.log().sum(dim=-1, keepdims=True)
                assert trg_logdetsig.shape == (n_seeds, n_pnts, n_rcns, 1)

                # The number of points for importance sampling
                n_isall = mtrc_cfg.pop('n')
                # The mini-batch of points that fit in the memory at one time
                n_ismb = mtrc_cfg.pop('mb')
                assert n_isall % (2 * n_ismb) == 0

                # Here's the total variation (TV) definition:
                #   $$ TV(p, q) = 1/2 \int |p(x) - q(x)| dx $$
                #
                # source:
                #   https://en.wikipedia.org/wiki/Total_variation_distance_of_probability_measures
                #
                # The idea is to use the following estimator:
                #   $$ TV(p, q) = E_{x\sim (p+q)/2} [ |p(x) - q(x)| / |p(x) + q(x)| ] $$
                #
                # To sample from $(p+q)/2$, we will sample twice: once from $p$ and once from $q$.
                is_ratioslst = []
                for i_ismb in range(n_isall // (2 * n_ismb)):
                    for is_sampdstr in ('src', 'trg'):
                        if is_sampdstr == 'src':
                            # The first set of points are defined as
                            #     is_pnts1 = src_mus + src_eps * src_sigs
                            src_eps = mrng.randn(n_seeds, n_pnts, n_rcns, n_ismb, dim_src)
                            assert src_eps.shape == (n_seeds, n_pnts, n_rcns, n_ismb, dim_src)

                            src_ispnts = src_eps * src_sigs.unsqueeze(-2) + src_mus.unsqueeze(-2)
                            assert src_ispnts.shape == (n_seeds, n_pnts, n_rcns, n_ismb, dim_src)
                            
                            trg_eps = (src_ispnts - trg_mus.unsqueeze(-2)) / trg_sigs.unsqueeze(-2)
                            assert trg_eps.shape == (n_seeds, n_pnts, n_rcns, n_ismb, dim_src)
                        else:
                            # The second set of points are defined as
                            #     is_pnts2 = trg_mus + trg_eps * trg_sigs
                            trg_eps = mrng.randn(n_seeds, n_pnts, n_rcns, n_ismb, dim_src)
                            assert trg_eps.shape == (n_seeds, n_pnts, n_rcns, n_ismb, dim_src)

                            trg_ispnts = trg_eps * trg_sigs.unsqueeze(-2) + trg_mus.unsqueeze(-2)
                            assert trg_ispnts.shape == (n_seeds, n_pnts, n_rcns, n_ismb, dim_trg)
                            
                            src_eps = (trg_ispnts - src_mus.unsqueeze(-2)) / src_sigs.unsqueeze(-2)
                            assert src_eps.shape == (n_seeds, n_pnts, n_rcns, n_ismb, dim_trg)

                        src_mahalsq = src_eps.square().sum(dim=-1)
                        assert src_mahalsq.shape == (n_seeds, n_pnts, n_rcns, n_ismb)

                        trg_mahalsq = trg_eps.square().sum(dim=-1)
                        assert trg_mahalsq.shape == (n_seeds, n_pnts, n_rcns, n_ismb)

                        src_logprob = -0.5 * src_mahalsq - src_logdetsig - 0.5 * dim_src * math.log(2 * math.pi)
                        assert src_logprob.shape == (n_seeds, n_pnts, n_rcns, n_ismb)

                        trg_logprob = -0.5 * trg_mahalsq - trg_logdetsig - 0.5 * dim_src * math.log(2 * math.pi)
                        assert trg_logprob.shape == (n_seeds, n_pnts, n_rcns, n_ismb)

                        is_ratio = torch.tanh((src_logprob - trg_logprob) / 2.0).abs()
                        assert is_ratio.shape == (n_seeds, n_pnts, n_rcns, n_ismb)

                        is_ratioslst.append(is_ratio)

                is_ratios = torch.cat(is_ratioslst, dim=-1)
                assert is_ratios.shape == (n_seeds, n_pnts, n_rcns, n_isall)

                mtrc_pnts = is_ratios.mean(dim=-1)
                assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                mtrc_name = mtrc_namefmt
                mtrc_pntsdata[mtrc_name] = mtrc_pnts
            elif mtrc_type  == 'wass':
                # The Wasserstein distance between two normal distributions is:
                #
                #   $$ W_2(\mathcal{N}(m_1,\Sigma_1); \mathcal{N}(m_2,\Sigma_2))^2 =
                #       \| m_1 - m_2 \|_2^2 +
                #       \| \Sigma_1^{1/2} - \Sigma_2^{1/2} \|_{Frobenius}^2.$$
                #
                # source:
                #   https://djalil.chafai.net/blog/2010/04/30/wasserstein-distance-between-two-gaussians/

                # The metric's p-norm
                mtrc_p = mtrc_cfg.pop('pnorm/p')
                assert mtrc_p == 2, dedent(f'''
                    Only the l2-norm version of the Wasserstein 
                    distance is implemented:
                        pnorm/p = {mtrc_p}
                ''')

                delta_musq = delta_mus.square().sum(dim=-1)
                assert delta_musq.shape == (n_seeds, n_pnts, n_rcns)

                delta_sigsq = delta_sigs.square().sum(dim=-1)
                assert delta_sigsq.shape == (n_seeds, n_pnts, n_rcns)

                mtrc_sqpntssum = delta_musq + delta_sigsq
                assert mtrc_sqpntssum.shape == (n_seeds, n_pnts, n_rcns)

                # The metric's reduction method
                mtrc_rdctn = mtrc_cfg.pop('pnorm/rdc')
                assert mtrc_rdctn in ('sum', 'mean')

                if mtrc_rdctn == 'sum':
                    mtrc_sqpnts = mtrc_sqpntssum
                    assert mtrc_sqpnts.shape == (n_seeds, n_pnts, n_rcns)
                elif mtrc_rdctn == 'mean':
                    mtrc_sqpnts = mtrc_sqpntssum / dim_src
                    assert mtrc_sqpnts.shape == (n_seeds, n_pnts, n_rcns)
                else:
                    raise ValueError(f'undefined mtrc_rdctn={mtrc_rdctn}')
                
                # The metric's exponent
                mtrc_explst_ = mtrc_cfg.pop('pnorm/exp')
                mtrc_explst = (mtrc_explst_,) if isinstance(mtrc_explst_, (int, float)) else mtrc_explst_
                assert isinstance(mtrc_explst, (tuple, list))

                for mtrc_exp in mtrc_explst:
                    mtrc_pnts = mtrc_sqpnts.pow(mtrc_exp / mtrc_p)
                    assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_name = mtrc_namefmt.format(p=mtrc_p, exp=mtrc_exp)
                    assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                        The metric name format produced clashing names for 
                        different settings. Make sure you make the name format 
                        specific to each setting.
                            name/fmt = {mtrc_namefmt}
                            pnorm/p = {mtrc_p}
                            pnorm/exp = {mtrc_exp}
                    ''')

                    mtrc_pntsdata[mtrc_name] = mtrc_pnts.pow(mtrc_exp)
            elif mtrc_type  == 'hellinger':
                # Assuming
                #   $p_{i} = \mathcal{N}(\mu_i, \Sigma_i)$
                #
                # We have the Bhattacharya distance as
                #   $D_{B}(p_{1},p_{2}) =
                #     {1 \over 8} (\mu_{1}-\mu_{2})^{T}\Sigma^{-1}(\mu_{1}-\mu_{2}) +
                #     {1 \over 2} \ln {\det \Sigma \over {\sqrt {\det \Sigma_{1}\,\det \Sigma_{2}}}}$
                #
                # source:
                #   https://en.wikipedia.org/wiki/Bhattacharyya_distance

                scrtrgavg_sig = ((src_sigs.square() + trg_sigs.square()) / 2).sqrt()
                assert scrtrgavg_sig.shape == (n_seeds, n_pnts, n_rcns, dim_src)

                logbc_term1 = (delta_mus / scrtrgavg_sig).square().sum(dim=-1) / 8.0
                assert logbc_term1.shape == (n_seeds, n_pnts, n_rcns)

                logbc_term2 = scrtrgavg_sig.log().sum(dim=-1)
                assert logbc_term2.shape == (n_seeds, n_pnts, n_rcns)

                logbc_term3 = (src_sigs.log().sum(dim=-1) + trg_sigs.log().sum(dim=-1)) / 2.0
                assert logbc_term3.shape == (n_seeds, n_pnts, n_rcns)

                # The log Bhattacharya coefficient
                log_bc = logbc_term3 - logbc_term1 - logbc_term2
                assert log_bc.shape == (n_seeds, n_pnts, n_rcns)

                # The Bhattacharya coefficient
                bhattachar_ = log_bc.exp()
                assert bhattachar_.shape == (n_seeds, n_pnts, n_rcns)

                # bc_eps = 1e-3
                # assert (bhattachar_ >= (0 - bc_eps)).all()
                # assert (bhattachar_ <= (1 + bc_eps)).all()

                bhattachar = torch.clamp(bhattachar_, 0.0, 1.0)
                assert bhattachar.shape == (n_seeds, n_pnts, n_rcns)

                # The squared Hellinger distance
                hllngrsq = 1 - bhattachar
                assert hllngrsq.shape == (n_seeds, n_pnts, n_rcns)

                # The Hellinger distance
                hllngr = hllngrsq.sqrt()
                assert hllngr.shape == (n_seeds, n_pnts, n_rcns)

                # The TV-approximation based on the hellinger distance
                # according to the following inequality:
                #   $ H^{2}(P,Q) \leq TV(P,Q) \leq \min(1, \sqrt{2} H(P,Q)) $
                tvaprx = (hllngrsq + (hllngr * math.sqrt(2)).clamp(0, 1)) / 2.0
                assert tvaprx.shape == (n_seeds, n_pnts, n_rcns)

                mtrc_varnames = mtrc_cfg.pop('vars')
                assert isinstance(mtrc_varnames, (list, tuple))

                mtrc_vars = {'bc': bhattachar, 'h1': hllngr, 'h2': hllngrsq, 'tv': tvaprx}

                for mtrc_varname in mtrc_varnames:
                    mtrc_name = mtrc_namefmt.format(var=mtrc_varname)

                    assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                        The metric name format produced clashing names for 
                        different settings. Make sure you make the name format 
                        specific to each setting.
                            name/fmt = {mtrc_namefmt}
                    ''')

                    assert mtrc_varname in mtrc_vars, dedent(f'''
                        The hellinger distance does not produce a "{mtrc_varname}"
                        variable:
                            offending var name = {mtrc_varname}
                            list of requested vars = {mtrc_varnames}
                            available hellinger vars = {mtrc_vars.keys()}
                    ''')

                    mtrc_pntsdata[mtrc_name] = mtrc_vars[mtrc_varname]
            elif mtrc_type  == 'kl':
                kl_dirlst = mtrc_cfg.pop('dir')
                for kl_dir in kl_dirlst:
                    if kl_dir == 'fwd':
                        left_sigs, right_sigs = src_sigs, trg_sigs
                    elif kl_dir == 'inv':
                        left_sigs, right_sigs = trg_sigs, src_sigs
                    else:
                        raise ValueError(f'undefined kl direction = {kl_dir}')
                    assert left_sigs.shape == (n_seeds, n_pnts, n_rcns, dim_src)
                    assert right_sigs.shape == (n_seeds, n_pnts, n_rcns, dim_src)

                    kl_term1 = (delta_mus / right_sigs).square().sum(dim=-1)
                    assert kl_term1.shape == (n_seeds, n_pnts, n_rcns)

                    kl_term2 = right_sigs.log().sum(dim=-1) - left_sigs.log().sum(dim=-1)
                    assert kl_term2.shape == (n_seeds, n_pnts, n_rcns)

                    kl_term3 = (left_sigs / right_sigs).square().sum(dim=-1)
                    assert kl_term3.shape == (n_seeds, n_pnts, n_rcns)

                    kl_pnts = kl_term2 + (kl_term1 + kl_term3 - dim_src) / 2.0
                    assert kl_pnts.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_name = mtrc_namefmt.format(dir=kl_dir)
                    assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                        The metric name format produced clashing names for 
                        different settings. Make sure you make the name format 
                        specific to each setting.
                            name/fmt = {mtrc_namefmt}
                    ''')
                    mtrc_pntsdata[mtrc_name] = kl_pnts
            elif mtrc_type  == 'icc':
                # The intra-class correlation coefficient ratio between multiple clusters with the means
                #
                #   $$ \forall 1 <= c <= C: \bar{y}_c $$
                #
                # can be calculated as
                #
                #   $$ \eta^{2} = 
                #       \sum_{c} n_c(\bar{y}_c - \bar{y})^{2} 
                #       / 
                #       \sum_{c,i}(y_{c,i}-\bar{y})^{2} $$
                #
                # where $\bar{y}_c$ is the mean of the category $c$ and $\bar{y}$ is the mean of 
                # the whole population.
                #
                # When we have two multi-variate normal distributions 
                #
                #   $\mathcal(N)(\mu_1, \Sigma_1)$ and $\mathcal(N)(\mu_2, \Sigma_2)$ 
                # 
                # the ICC can be calculated as
                #
                #   $$ \eta^{2} = 
                #       \| \mu_1 - \mu_2 \|_2^2
                #       /
                #       (\| \mu_1 - \mu_2 \|_2^2 + 2 * Tr(\Sigma_1) + 2 * Tr(\Sigma_2))
                #
                # It is evident that $0 <= \eta < 1$; having $\eta=0$ means the means are the same, 
                # while $\eta=1$ means the clusters are dispersed.
                #
                # source:
                #   https://en.wikipedia.org/wiki/Correlation_ratio
                #   https://www.itl.nist.gov/div898/software/dataplot/refman2/auxillar/icc.htm
                delta_musq = delta_mus.square().sum(dim=-1)
                assert delta_musq.shape == (n_seeds, n_pnts, n_rcns)

                sum_sigsq = src_sigs.square().sum(dim=-1) + trg_sigs.square().sum(dim=-1)
                assert sum_sigsq.shape == (n_seeds, n_pnts, n_rcns)

                mtrc_sqpnts = delta_musq / (delta_musq + 2 * sum_sigsq)
                assert mtrc_sqpnts.shape == (n_seeds, n_pnts, n_rcns)

                # The metric's exponent
                mtrc_explst_ = mtrc_cfg.pop('exp')
                mtrc_explst = (mtrc_explst_,) if isinstance(mtrc_explst_, (int, float)) else mtrc_explst_
                assert isinstance(mtrc_explst, (tuple, list))

                for mtrc_exp in mtrc_explst:
                    mtrc_pnts = mtrc_sqpnts.pow(mtrc_exp)
                    assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                    mtrc_name = mtrc_namefmt.format(exp=mtrc_exp)
                    assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                        The metric name format produced clashing names for 
                        different settings. Make sure you make the name format 
                        specific to each setting.
                            name/fmt = {mtrc_namefmt}
                            exp = {mtrc_exp}
                    ''')

                    mtrc_pntsdata[mtrc_name] = mtrc_pnts.pow(mtrc_exp)
            else:
                raise ValueError(f'undefined metric type = {mtrc_type}')
        elif mtrc_type == 'slcdwass':
            if not ignore_xtrareprs:
                assert tuple(src_mtrcdata.keys()) == ('pnts',), dedent(f'''
                    The pnorm metric can only be applied to points data:
                        src_mtrcdata.keys = {src_mtrcdata.keys()}''')

                assert tuple(trg_mtrcdata.keys()) == ('pnts',), dedent(f'''
                    The pnorm metric can only be applied to points data:
                        trg_mtrcdata.keys = {trg_mtrcdata.keys()}''')

            src_pnts1 = src_mtrcdata['pnts'].flatten(3, -1)
            n_mbsrc1, n_srcrcns, dim_src = src_pnts1.shape[1:]
            assert src_pnts1.shape == (n_seeds, n_mbsrc1, n_srcrcns, dim_src)
            src_pnts, n_mbsrc = src_pnts1.flatten(1, 2), n_mbsrc1 * n_srcrcns
            assert src_pnts.shape == (n_seeds, n_mbsrc, dim_src)

            trg_pnts1 = trg_mtrcdata['pnts'].flatten(3, -1)
            n_mbtrg1, n_trgrcns, dim_trg = trg_pnts1.shape[1:]
            assert trg_pnts1.shape == (n_seeds, n_mbtrg1, n_trgrcns, dim_trg)
            trg_pnts, n_mbtrg = trg_pnts1.flatten(1, 2), n_mbtrg1 * n_trgrcns
            assert trg_pnts.shape == (n_seeds, n_mbtrg, dim_trg)

            n_rcnstmp = get_nbcst(n_srcrcns, n_trgrcns)
            assert n_rcnstmp is not None, dedent(f'''
                The source and target data fed to the slcdwass metric 
                do not have broadcastable n_rcns. This is odd, even 
                though the slcdwass metric could be computed:
                
                Disclaimer: this is not a real issue. it's just that 
                we have never thought to use the function this way! :)
                    n_srcrcns = {n_srcrcns}
                    n_trgrcns = {n_trgrcns}''')

            assert dim_src == dim_trg, dedent(f'''
                The sliced wasserstein metric requires the source and target
                data to have the same data dimensions:
                    src_pnts.shape = {src_mtrcdata['pnts'].shape}
                    trg_pnts.shape = {trg_mtrcdata['pnts'].shape}''')

            # The number of slices
            n_slc = mtrc_cfg.pop('slc/n')
            # For the sliced-wasserstein metric, the output number of points for 
            # statistical aggregation is the same as the number of slices.
            n_pnts, n_rcns = n_slc, 1

            # The mini-batch of slices that fit in the memory at one time
            n_mbslc = mtrc_cfg.pop('slc/mb')

            # The distribution of slicing initial random variables
            dstr_slc = mtrc_cfg.pop('slc/dstr')
            assert dstr_slc == 'normal', dedent(f'''
                Only sampling random slices from a normal 
                distribution is supported for now:
                    slc/dstr = {dstr_slc}''')

            # The type of slicing covariance matrix
            type_slccov = mtrc_cfg.pop('slc/cov/type')
            
            if type_slccov in ('src', 'trg'):
                # The approximations for the slice sampling covariance
                aprx_slccov = mtrc_cfg.pop('slc/cov/approx')

                # We will use the source/target data's singular values to sample the 
                # slicing epsilons.
                if type_slccov == 'src':
                    cov_pntsfull, n_mbcovdatafull = src_pnts, n_mbsrc
                elif type_slccov == 'trg':
                    cov_pntsfull, n_mbcovdatafull = trg_pnts, n_mbtrg
                else:
                    raise ValueError(f'undefined type_slccov={type_slccov}')
                assert cov_pntsfull.shape == (n_seeds, n_mbcovdatafull, dim_src)

                # The subsampling size of the covariance matrix data. 
                # This is only used for estimating the slicing directions and 
                # not computing the sliced wasserstein distance values.
                n_covsubsamp = mtrc_cfg.pop('slc/cov/nsamp', None)
                if (n_covsubsamp is None) or (n_covsubsamp >= n_mbcovdatafull):
                    cov_pnts, n_mbcovdata = cov_pntsfull, n_mbcovdatafull
                else:
                    n_mbcovdata = n_covsubsamp
                    
                    i_covsubsamp = mrng.rand(n_seeds, n_mbcovdatafull, 1).argsort(dim=-1)[:, :n_covsubsamp]
                    assert i_covsubsamp.shape == (n_seeds, n_covsubsamp, 1)

                    cov_pnts = torch.take_along_dim(cov_pntsfull, i_covsubsamp, dim=-2)
                    assert cov_pnts.shape == (n_seeds, n_mbcovdata, dim_src)
            else:
                aprx_slccov, cov_pnts, n_mbcovdata = None, None, None

            if (type_slccov in ('src', 'trg')) and (aprx_slccov == 'pcalr'):
                # The covariance exponent, i.e., the singular values exponent. Should default to 1.
                exp_slccovlst_ = mtrc_cfg.pop('slc/cov/exp')
                exp_slccovlst = (exp_slccovlst_,) if isinstance(exp_slccovlst_, (int, float)) else exp_slccovlst_
                assert isinstance(exp_slccovlst, (tuple, list))

                # With a non-unit covariance exponent, we have to use a low-rank pca.
                slc_covpcalrniter = mtrc_cfg.pop('slc/cov/pcalr/niter')
                slc_covpcalrq = mtrc_cfg.pop('slc/cov/pcalr/q')

                assert not cov_pnts.isnan().any(), f'{cov_pnts.isnan().sum().item()} elementas are NaN.'
                slc_covpcalrq2 = slc_covpcalrq if slc_covpcalrq < min(cov_pnts.shape[-2:]) else min(cov_pnts.shape[-2:])
                u_slccov, s_slccov, v_slccov = torch.pca_lowrank(cov_pnts, q=slc_covpcalrq2, 
                    niter=slc_covpcalrniter)
                q_slccov = s_slccov.shape[-1]
                assert u_slccov.shape == (n_seeds, n_mbcovdata, q_slccov)
                assert s_slccov.shape == (n_seeds, q_slccov)
                assert v_slccov.shape == (n_seeds, dim_src, q_slccov)
            elif (type_slccov in ('src', 'trg')) and (aprx_slccov == 'exact'):
                exp_slccovlst, u_slccov, s_slccov, v_slccov = (None,), None, None, None
            elif (type_slccov in ('src', 'trg')):
                raise ValueError(f'undefined slc/cov/approx = {aprx_slccov}')
            else:
                exp_slccovlst, u_slccov, s_slccov, v_slccov = (None,), None, None, None

            # The slice vectors are nromalized after sampling w.r.t. a pnorm.
            slc_pnorm = mtrc_cfg.pop('slc/pnorm')

            # The metric's p-norm
            mtrc_plst_ = mtrc_cfg.pop('pnorm/p')
            mtrc_plst = (mtrc_plst_,) if isinstance(mtrc_plst_, (int, float)) else mtrc_plst_
            assert isinstance(mtrc_plst, (tuple, list))

            # The metric's exponent
            mtrc_explst_ = mtrc_cfg.pop('pnorm/exp')
            mtrc_explst = (mtrc_explst_,) if isinstance(mtrc_explst_, (int, float)) else mtrc_explst_
            assert isinstance(mtrc_explst, (tuple, list))
            
            for exp_slccov in exp_slccovlst:
                err_srctrglst = []
                assert n_slc % n_mbslc == 0
                for i_slcmb in range(n_slc // n_mbslc):
                    #################### Step 1: Sampling the Slicing Random Variables ####################
                    if type_slccov == 'identity':
                        # We will use the identity covariance matrix to sample the slicing epsilons.
                        slc_epsilons = mrng.randn(n_seeds, n_mbslc, dim_src)
                        assert slc_epsilons.shape == (n_seeds, n_mbslc, dim_src)
                    elif (type_slccov in ('src', 'trg')) and (aprx_slccov == 'exact'):
                        # Since the slicing covariance exponent is 1, we don't need to perform a PCA on 
                        # the src/trg data to find the scalings; A simple data matrix multiplication will 
                        # do the trick in this case.
                        cov_pntsnomean = cov_pnts - cov_pnts.mean(dim=-2, keepdims=True)
                        assert cov_pntsnomean.shape == (n_seeds, n_mbcovdata, dim_src)

                        slc_randns = mrng.randn(n_seeds, n_mbslc, n_mbcovdata)
                        assert slc_randns.shape == (n_seeds, n_mbslc, n_mbcovdata)

                        slc_epsilons = slc_randns.matmul(cov_pntsnomean)
                        assert slc_epsilons.shape == (n_seeds, n_mbslc, dim_src)
                    elif (type_slccov in ('src', 'trg')) and (aprx_slccov == 'pcalr'):
                        # With a non-unit covariance exponent, we have to use the low-rank pca.
                        slc_randns = mrng.randn(n_seeds, n_mbslc, q_slccov)
                        assert slc_randns.shape == (n_seeds, n_mbslc, q_slccov)

                        stilde_slccov = (1e-30 + s_slccov.unsqueeze(-2)).pow(exp_slccov) / math.sqrt(n_mbcovdata)
                        assert stilde_slccov.shape == (n_seeds, 1, q_slccov)

                        slc_randnsscld = slc_randns * stilde_slccov
                        assert slc_randnsscld.shape == (n_seeds, n_mbslc, q_slccov)

                        slc_epsilons = torch.einsum('nsq,ndq->nsd', slc_randnsscld, v_slccov)
                        assert slc_epsilons.shape == (n_seeds, n_mbslc, dim_src)
                    else:
                        raise ValueError(dedent(f'''
                            undefined type of slice sampling covariance:
                                slc/cov/type = {type_slccov}
                                slc/cov/exp = {exp_slccov}'''))

                    ####################### Step 2: Normalizing the Slice Directions ######################
                    if slc_pnorm == 2:
                        slc_epspnorms = slc_epsilons.square().sum(dim=-1, keepdims=True).sqrt()
                        assert slc_epspnorms.shape == (n_seeds, n_mbslc, 1)
                    else:
                        slc_epspnorms = slc_epsilons.abs().pow(slc_pnorm).sum(dim=-1, keepdims=True).pow(1.0 / slc_pnorm)
                        assert slc_epspnorms.shape == (n_seeds, n_mbslc, 1)

                    slc_dirs = slc_epsilons / slc_epspnorms
                    assert slc_dirs.shape == (n_seeds, n_mbslc, dim_src)

                    ################### Step 3: Projecting the Data to Sliced Directions ##################
                    # Projecting to the sliced directions
                    src_slcdpnts = torch.einsum('sid,sjd->sji', src_pnts, slc_dirs)
                    assert src_slcdpnts.shape == (n_seeds, n_mbslc, n_mbsrc)

                    trg_slcdpnts = torch.einsum('sid,sjd->sji', trg_pnts, slc_dirs)
                    assert trg_slcdpnts.shape == (n_seeds, n_mbslc, n_mbtrg)

                    ########################### Step 4: Sorting the Sliced Data ###########################
                    # Sorting the projected values
                    src_srtdslcds = src_slcdpnts.sort(dim=-1).values
                    assert src_srtdslcds.shape == (n_seeds, n_mbslc, n_mbsrc)

                    trg_srtdslcds = trg_slcdpnts.sort(dim=-1).values
                    assert trg_srtdslcds.shape == (n_seeds, n_mbslc, n_mbtrg)

                    ################# Step 5: Unifying the Source and Target Sample Sizes #################
                    # Next, we align the smaller sample size (i.e., min of `n_mbsrc` and
                    # `n_mbtrg`) to their largest by oversampling the smaller set of points.
                    # By the end of this phase, both the source and target tensors will have
                    # a shape of `(n_seeds, n_mbslc, n_mbalgn)`.
                    n_mbalgn = max(n_mbsrc, n_mbtrg)

                    # Aligning the source projections
                    if n_mbsrc >= n_mbtrg:
                        src_algndsrtd = src_srtdslcds
                        assert src_algndsrtd.shape == (n_seeds, n_mbslc, n_mbalgn)
                    else:
                        i_algn = (torch.arange(n_mbtrg, device=tch_device) * ((n_mbsrc - 1) / n_mbtrg))
                        i_algn = i_algn.round().to(dtype=torch.long)
                        assert i_algn.shape == (n_mbtrg,)

                        src_algndsrtd = src_srtdslcds[..., i_algn]
                        assert src_algndsrtd.shape == (n_seeds, n_mbslc, n_mbalgn)

                    # Aligning the target projections
                    if n_mbtrg >= n_mbsrc:
                        trg_algndsrtd = trg_srtdslcds
                        assert trg_algndsrtd.shape == (n_seeds, n_mbslc, n_mbalgn)
                    else:
                        i_algn = (torch.arange(n_mbsrc, device=tch_device) * ((n_mbtrg - 1) / n_mbsrc))
                        i_algn = i_algn.round().to(dtype=torch.long)
                        assert i_algn.shape == (n_mbsrc,)

                        trg_algndsrtd = trg_srtdslcds[..., i_algn]
                        assert trg_algndsrtd.shape == (n_seeds, n_mbslc, n_mbalgn)

                    ############### Step 6: Finding the Projected and Sorted Error's P-Norm ###############
                    # The aligned error
                    err_srctrgmb = src_algndsrtd - trg_algndsrtd
                    assert err_srctrgmb.shape == (n_seeds, n_mbslc, n_mbalgn)

                    err_srctrglst.append(err_srctrgmb)

                err_srctrg = torch.cat(err_srctrglst, dim=-2)
                assert err_srctrg.shape == (n_seeds, n_slc, n_mbalgn)

                for mtrc_p in mtrc_plst:
                    err_slcs = err_srctrg.abs().pow(mtrc_p).mean(dim=-1)
                    assert err_slcs.shape == (n_seeds, n_pnts)

                    for mtrc_exp in mtrc_explst:
                        mtrc_pnts1 = err_slcs.pow(mtrc_exp / mtrc_p)
                        assert mtrc_pnts1.shape == (n_seeds, n_pnts)
                        mtrc_pnts = mtrc_pnts1.unsqueeze(-1)
                        assert mtrc_pnts.shape == (n_seeds, n_pnts, n_rcns)

                        mtrc_name = mtrc_namefmt.format(pnorm_p=mtrc_p,
                            pnorm_exp=mtrc_exp, cov_exp=exp_slccov)
                        assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                            The metric name format produced clashing names for 
                            different settings. Make sure you make the name format 
                            specific to each setting.
                                name/fmt = {mtrc_namefmt}
                                pnorm/p = {mtrc_p}
                                pnorm/exp = {mtrc_exp}
                        ''')
                        mtrc_pntsdata[mtrc_name] = mtrc_pnts
        elif mtrc_type == 'menger':
            # The Menger curvature of a triple of points in d-dimensional 
            # Euclidean space $R^d$ is the reciprocal of the radius of the 
            # circle that passes through the three points.
            #
            #       $ c(x,y,z) = 1 / R_{x,y,z}
            #                  = 4 A_{x,y,z} / (|x-y| |y-z| |z-x|)$,
            #
            # where $A_{x,y,z}$ denotes the area of the triangle spanned by $x$, 
            # $y$ and $z$.
            #
            # Another way of computing Menger curvature is the identity
            #
            #       $ c(x,y,z) = 2 \sin(\angle_{x,y,z}) / |x-z| $
            #
            # To integrate the curvature over an entire trajectory, we take all 
            # the triplets $(x_i, y_i, z_i)$ within a trajectory and compute their 
            # weighted sum:
            #
            #       $ c_1({x_i, y_i, z_i}_{i=1}^{n}) = 
            #           \sum_{x_i,y_i,z_i} |x_i - y_i| / R_{x_i,y_i,z_i} $
            #
            # The $c_1$ statistic is scaling invariant:
            #      
            #       $ c_1({x_i, y_i, z_i}_{i=1}^{n}) = c_1({2 x_i, 2 y_i, 2 z_i}_{i=1}^{n}) $
            #
            # It is also statistically consistent:
            #
            #       $ c_1 = \lim_{n\rightarrow \infty} c_1({x_i, y_i, z_i}_{i=1}^{n})$
            # 
            # To have these properties, we cannot normalize $c_1$ with respect to the 
            # number of triplets $n$:
            #
            #       $ 0 \leq c_1({x_i, y_i, z_i}_{i=1}^{n}) \leq 2n$.
            #
            # Since the menger curvature is a sin function, it cannot differentiate 
            # between 10 a 170 degrees! To alleviate this, we will report the following 
            # curvature statistic as well:
            #
            #       $ c_2({x_i, y_i, z_i}_{i=1}^{n}) = 
            #           (\sum_{x_i,y_i,z_i} [|x_i - y_i| * \angle_{x,y,z}]) 
            #           /  
            #           (\sum_{x_i,y_i,z_i} [|x_i - y_i|]) $
            #
            # Because of the denominator, the $c_2$ statistic is also scaling invariant:
            #      
            #       $ c_2({x_i, y_i, z_i}_{i=1}^{n}) = c_2({2 x_i, 2 y_i, 2 z_i}_{i=1}^{n}) $
            #
            # Similar to $c_1$, $c_2$ is also statistically consistent and has the following range
            #
            #       $ 0 \leq c_2({x_i, y_i, z_i}_{i=1}^{n}) \leq n \pi$.
            #
            # Here is a mapping of variables to mathematical definitions:
            #    src_xy:         y - x
            #    src_xynorm:     \|y - x\|
            #    src_xynrmlzd:   (y - x) / \|y - x\|
            #    src_yz:         z - y
            #    src_yznorm:     \|z - y\|
            #    src_yznrmlzd:   (z - y) / \|z - y\|
            #    src_xz:         z - x
            #    src_xznorm:     \|z - x\|
            #    src_xznrmlzd:   (z - x) / \|z - x\|
            #    src_xyzcos:     \cos(\angle_{x,y,z})
            #       = (z - y)^T (y - x) / (\|z - y\| * \|y - x\|)
            #    src_xyzsin:     \sin(\angle_{x,y,z})
            #    src_xyztheta:   \angle_{x,y,z}
            #    src_menger:     1 / R_{x,y,z} 
            #       = 2 * (\sin \angle_{x,y,z}) / |x - z|$
            #    src_trajmenger: 
            #       \sum_{x,y,z} |x - y| / R_{x,y,z}
            #    src_trajangle:  
            #       (\sum_{x,y,z} |x - y| * \theta_{x,y,z}) / (\sum_{x,y,z} |x - y|)

            assert n_t >= 3, dedent(f'''
                The menger curvature needs at least 3 time-steps: 
                    n_t={n_t}''')
            if not ignore_xtrareprs:
                assert tuple(src_mtrcdata.keys()) == ('pnts',), dedent(f'''
                    The menger metric can only be applied to "points" 
                    source data:
                        src_mtrcdata.keys = {src_mtrcdata.keys()}''')

            src_pnts = src_mtrcdata['pnts'].flatten(3, -1)
            n_mbsrc, n_srcrcns, dim_src = src_pnts.shape[1:]
            assert src_pnts.shape == (n_seeds, n_mbsrc, n_srcrcns, dim_src)

            if not ignore_xtrareprs:
                assert tuple(trg_mtrcdata.keys()) == ('pnts',), dedent(f'''
                    The menger metric can only be applied to "points" 
                    target data:
                        trg_mtrcdata.keys = {trg_mtrcdata.keys()}''')

            trg_pnts = trg_mtrcdata['pnts'].flatten(3, -1)
            n_mbtrg, n_trgrcns, dim_trg = trg_pnts.shape[1:]
            assert trg_pnts.shape == (n_seeds, n_mbtrg, n_trgrcns, dim_trg)

            assert src_pnts.shape == trg_pnts.shape, dedent(f'''
                The menger metric requires the source and target data 
                to be identical:
                    src_pnts.shape = {src_pnts}
                    trg_pnts.shape = {trg_pnts}''')

            assert (src_pnts == trg_pnts).all(), dedent('''
                For now, the menger metric can only be evaluated when the 
                source and target data are identical''')

            assert n_mbsrc % n_t == 0, dedent(f'''
                The menger metric can only be evaluated when the source 
                mini-batch size (n_mbsrc={n_mbsrc}) is a multiple of 
                the number of time-steps (n_t={n_t}). Unfortunately, that 
                is not the case''')

            # Here, the number of trajectories is the same as the number of points
            n_trajs = n_mbsrc // n_t
            n_pnts = n_trajs
            n_rcns = n_srcrcns
            eps_norm = 1e-200

            src_trajs1 = src_pnts.reshape(n_seeds, n_trajs, n_t, n_rcns, dim_src)
            assert src_trajs1.shape == (n_seeds, n_trajs, n_t, n_rcns, dim_src)

            src_trajs = src_trajs1.transpose(2, 3)
            assert src_trajs.shape == (n_seeds, n_pnts, n_rcns, n_t, dim_src)

            src_x = src_trajs[:, :, :, 0 : (n_t - 2), :]
            assert src_x.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            src_y = src_trajs[:, :, :, 1 : (n_t - 1), :]
            assert src_y.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)
            
            src_z = src_trajs[:, :, :, 2 : (n_t - 0), :]
            assert src_z.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            # Processing the x-y difference and normalizing it
            src_xy = src_y - src_x
            assert src_xy.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            src_xynorm = eps_norm + src_xy.square().sum(dim=-1).sqrt()
            assert src_xynorm.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)
            
            src_xynrmlzd = src_xy / src_xynorm.unsqueeze(dim=-1)
            assert src_xynrmlzd.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            # Processing the y-z difference and normalizing it
            src_yz = src_z - src_y
            assert src_yz.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            src_yznorm = eps_norm + src_yz.square().sum(dim=-1).sqrt()
            assert src_yznorm.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)

            src_yznrmlzd = src_yz / src_yznorm.unsqueeze(dim=-1)
            assert src_yznrmlzd.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            # Processing the x-z difference and finding its norm
            src_xz = src_z - src_x
            assert src_xz.shape == (n_seeds, n_pnts, n_rcns, n_t - 2, dim_src)

            src_xznorm = eps_norm + src_xz.square().sum(dim=-1).sqrt()
            assert src_xznorm.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)

            # Computing the Menger angle: $\theta$ and $sin(\theta)$
            src_xyzcos = torch.linalg.vecdot(src_xynrmlzd, src_yznrmlzd, dim=-1)
            assert src_xyzcos.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)

            src_xyzsin = torch.sqrt(1 - src_xyzcos.square())
            assert src_xyzsin.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)

            src_xyztheta = src_xyzsin.arcsin()
            assert src_xyztheta.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)

            # Computing the Menger curvature: 
            #   $1 / R_{x,y,z} = 2 * (\sin \theta_{x,y,z}) / |x - z|$
            src_menger = 2 * src_xyzsin / src_xznorm
            assert src_menger.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)
            
            # Computing the integral Menger curvature: 
            #   $ \sum_{x,y,z} |x - y| / R_{x,y,z} $
            # Note that the $|x - y| / R_{x,y,z}$ value is strictly between 0 and 2.
            src_trajmenger = torch.linalg.vecdot(src_menger, src_xynorm, dim=-1)
            assert src_trajmenger.shape == (n_seeds, n_pnts, n_rcns)

            # Computing the integral Menger curvature: 
            #   $ (\sum_{x,y,z} |x - y| * \theta_{x,y,z}) / (\sum_{x,y,z} |x - y|) $
            src_xynorm_w = src_xynorm / src_xynorm.sum(dim=-1, keepdims=True)
            assert src_xynorm_w.shape == (n_seeds, n_pnts, n_rcns, n_t - 2)

            src_trajangle = torch.linalg.vecdot(src_xyztheta, src_xynorm_w, dim=-1)
            assert src_trajangle.shape == (n_seeds, n_pnts, n_rcns)


            mtrc_varnames = mtrc_cfg.pop('vars')
            assert isinstance(mtrc_varnames, (list, tuple))

            mtrc_vars = {'theta': src_trajangle, 'curve': src_trajmenger}

            for mtrc_varname in mtrc_varnames:
                mtrc_name = mtrc_namefmt.format(var=mtrc_varname)

                assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                    The metric name format produced clashing names for 
                    different settings. Make sure you make the name format 
                    specific to each setting.
                        name/fmt = {mtrc_namefmt}
                ''')

                assert mtrc_varname in mtrc_vars, dedent(f'''
                    The menger curvature does not produce a "{mtrc_varname}"
                    variable:
                        offending var name = {mtrc_varname}
                        list of requested vars = {mtrc_varnames}
                        available hellinger vars = {mtrc_vars.keys()}
                ''')

                mtrc_pntsdata[mtrc_name] = mtrc_vars[mtrc_varname]
        elif mtrc_type in ('aeroccn', 'aeroinp', 'aeroopt', 'aerorel'):

            raise NotImplementedError('''
                this case has not been adapted to particle data yet, and 
                is a legacy form the histogram vae code.''')
            ########################### Extracting the Constants ##########################
            # n_chem: (int) The total number of chemical species including water.
            n_chem = aero_cstinfo['n_chem']
            # n_bins: (int) The number of diameter histogram bins.
            n_bins = aero_cstinfo['n_bins']
            # n_waveref: (int) The number of refrence wave-lengths.
            n_waveref = aero_cstinfo['n_waveref']
            # rho_chm: (torch.tensor) The physical density of the chemical species.
            #     `assert rho_chm.shape == (n_chem,)`
            rho_chm = aero_cstinfo['rho_chm']
            assert rho_chm.shape == (n_chem,)
            # kappa_chm: (torch.tensor) The Kappa of the chemical species.
            #     `assert kappa_chm.shape == (n_chem,)`
            kappa_chm = aero_cstinfo['kappa_chm']
            assert kappa_chm.shape == (n_chem,)
            # chem_species: (list) The chemical species names. This will be used to detect 
            #     and remove the water (H2O) chemical.
            chem_species = aero_cstinfo['chem_species']
            assert len(chem_species) == n_chem
            # refr_wvchmref: (torch.tensor) The complex refraction indices of the chemical 
            #     species at the reference wave lengths.
            refr_wvchmref = aero_cstinfo['refr_wvchmref']
            assert refr_wvchmref.shape == (n_waveref, n_chem)
            # len_wvref: (torch.tensor) The wave lengths for optical property calculations.
            len_wvref = aero_cstinfo['len_wvref']
            assert len_wvref.shape == (n_waveref,)

            # Extracting the input source data
            src_mtrcdatashie = deep2hie(src_mtrcdatas)
            m_chmprthstsrc1 = src_mtrcdatashie.pop('x/m_chmprthst/pnts')
            n_mbsrc, n_srcrcns = m_chmprthstsrc1.shape[1:3]
            assert m_chmprthstsrc1.shape == (n_seeds, n_mbsrc, n_srcrcns, n_chem, n_bins)
            n_prthstsrc1 = src_mtrcdatashie.pop('x/n_prthst/pnts')
            assert n_prthstsrc1.shape == (n_seeds, n_mbsrc, n_srcrcns, 1, n_bins)
            if not ignore_xtrareprs:
                assert len(src_mtrcdatashie) == 0, f'unused src data: {src_mtrcdatashie}'

            # Extracting the input target data
            trg_mtrcdatashie = deep2hie(trg_mtrcdatas)
            m_chmprthsttrg1 = trg_mtrcdatashie.pop('x/m_chmprthst/pnts')
            n_mbtrg, n_trgrcns = m_chmprthsttrg1.shape[1:3]
            assert m_chmprthsttrg1.shape == (n_seeds, n_mbtrg, n_trgrcns, n_chem, n_bins)
            n_prthsttrg1 = trg_mtrcdatashie.pop('x/n_prthst/pnts')
            assert n_prthsttrg1.shape == (n_seeds, n_mbtrg, n_trgrcns, 1, n_bins)
            if not ignore_xtrareprs:
                assert len(trg_mtrcdatashie) == 0, f'unused trg data: {trg_mtrcdatashie}'

            # Making sure they match in number of samples.
            n_pnts = n_mbsrc
            assert n_mbsrc == n_mbtrg, dedent(f'''
                The aerosol metrics require the source and target data 
                to have the same number of points:
                    n_mbsrc = {n_mbsrc}
                    n_mbtrg = {n_mbtrg}''')

            n_rcns = get_nbcst(n_srcrcns, n_trgrcns)
            assert n_rcns is not None, dedent(f'''
                The mahal metric requires the source and target data 
                to have broadcastable n_rcns:
                    n_srcrcns = {n_srcrcns}
                    n_trgrcns = {n_trgrcns}''')

            n_pntsrcns = n_pnts * n_rcns

            m_chmprthstsrc = m_chmprthstsrc1.expand(n_seeds, n_pnts, n_rcns, n_chem, n_bins).flatten(1, 2)
            assert m_chmprthstsrc.shape == (n_seeds, n_pntsrcns, n_chem, n_bins)
            n_prthstsrc = n_prthstsrc1.expand(n_seeds, n_pnts, n_rcns, 1, n_bins).flatten(1, 2)
            assert n_prthstsrc.shape == (n_seeds, n_pntsrcns, 1, n_bins)

            m_chmprthsttrg = m_chmprthsttrg1.expand(n_seeds, n_pnts, n_rcns, n_chem, n_bins).flatten(1, 2)
            assert m_chmprthsttrg.shape == (n_seeds, n_pntsrcns, n_chem, n_bins)
            n_prthsttrg = n_prthsttrg1.expand(n_seeds, n_pnts, n_rcns, 1, n_bins).flatten(1, 2)
            assert n_prthsttrg.shape == (n_seeds, n_pntsrcns, 1, n_bins)

            ######################### Extracting the Metric Configs ########################
            # n_mbevl: (int) The evaluation mini-batch size for computing the metric
            n_mbevl = mtrc_cfg.pop('eval/bs')
            if mtrc_type == 'aeroccn':
                # eps_histmin: (float) the minimum epsilon value for the CDF interval.
                eps_histmin = mtrc_cfg.pop('aeroccn/eps_histmin')
                # eps_histmax: (float) the maximum epsilon value for the CDF interval.
                eps_histmax = mtrc_cfg.pop('aeroccn/eps_histmax')
                # n_epshist: (int) the number of bins to divide the log-epsilon interval into.
                n_epshist = mtrc_cfg.pop('aeroccn/n_epshist')
                # tmprtr_sclr: (float) The temperature used to compute the critical relative 
                # humidity values; The `A` factor in the computation relies on this temperature.
                tmprtr_sclr = mtrc_cfg.pop('aeroccn/tmprtr_sclr')
                # void_kappa: (float) The kappa value for empty particles.
                void_kappa = mtrc_cfg.pop('aeroccn/void_kappa')
                # n_rfiters: (int) The number of Newtonian root finding iterations to compute 
                # the critical diameters.
                n_rfiters = mtrc_cfg.pop('aeroccn/n_rfiters')
                # e_rfiters: (float) The relative update size stopping threshold for the 
                # newton root finder.
                e_rfiters = mtrc_cfg.pop('aeroccn/e_rfiters')

                # All metric variable names produced by the `get_ccnerr` function.
                mtrc_allvarnames = ('ccn_err',)
            elif mtrc_type == 'aeroopt':
                # wave_length: (list[float] | float) The optical wave length for measuring 
                #     the optical efficiency properties.
                wave_length = mtrc_cfg.pop('aeroopt/wave_length')
                # n_max: (int) The number of iterations needed for the mie sequence 
                #     summations to converge.
                n_max = mtrc_cfg.pop('aeroopt/n_max')
                # void_refr: (complex) The optical refraction coefficient for particles
                #     with zero volume.
                void_refr = mtrc_cfg.pop('aeroopt/void_refr')
                void_refr = eval(void_refr) if isinstance(void_refr, str) else void_refr
                assert isinstance(void_refr, complex)

                # Computing the wave-length related parameters
                refr_wvchm, len_wv, n_wave = interp_refr(refr_wvchmref, len_wvref, 
                    wave_length, chem_species=chem_species)
                assert refr_wvchm.shape == (n_wave, n_chem)
                assert len_wv.shape == (n_wave,)

                # All metric variable names produced by the `get_opticalerr` function.
                mtrc_allvarnames = ('opt_err', 'qs_relerr', 'qa_relerr')
            elif mtrc_type == 'aeroinp':
                # tmprtr_inpmin: (float) The lower end of the the temperature range in Celsius.
                #     Usually, this value should be around -40 degrees.
                tmprtr_inpmin = mtrc_cfg.pop('aeroinp/tmprtr_inpmin')
                # tmprtr_inpmax: (float) The higher end of the the temperature range in Celsius.
                #     Usually, this value should be around 0 degrees.
                tmprtr_inpmax = mtrc_cfg.pop('aeroinp/tmprtr_inpmax')
                # n_tmprtr: (int) The number of temperature bins.
                n_tmprtr = mtrc_cfg.pop('aeroinp/n_tmprtr')

                # All metric variable names produced by the `get_inperr` function.
                mtrc_allvarnames = ('inp_err', 'loginp_err')
            elif mtrc_type == 'aerorel':
                # pnorm: (int) The p-norm of the relative error metric
                pnorm = mtrc_cfg.pop('aerorel/pnorm')

                # All metric variable names produced by the `get_histrelerrs` function.
                mtrc_allvarnames = ('n_relerr', 'm_chmrelerr', 'm_relerr', 'm_perchmrelerr')
            else:
                raise ValueError(f'undefined mtrc_type={mtrc_type}')
            
            ######################### Running the Metric Functions #########################
            mtrc_varlsts = defaultdict(list)
            x_mbs = {'m_chmprthst': m_chmprthstsrc, 'n_prthst': n_prthstsrc}
            xhat_mbs = {'m_chmprthst': m_chmprthsttrg, 'n_prthst': n_prthsttrg}
            for i_evliter in range(math.ceil(n_pntsrcns / n_mbevl)):
                ii1, ii2 = i_evliter * n_mbevl, min(n_pntsrcns, i_evliter * n_mbevl + n_mbevl)
                n_mbevl_ = ii2 - ii1

                x_mbsmini = {key: val[:, ii1: ii2, :, :] for key, val in x_mbs.items()}
                xhat_mbsmini = {key: val[:, ii1: ii2, :, :] for key, val in xhat_mbs.items()}

                if mtrc_type == 'aeroccn':
                    mtrc_varmbs = get_ccnerr(x_mbsmini, xhat_mbsmini, eps_histmin, eps_histmax, 
                        n_epshist, tmprtr_sclr, void_kappa, n_rfiters, e_rfiters, rho_chm, kappa_chm, 
                        chem_species, n_seeds, n_mbevl_, n_chem, n_bins, tch_device, tch_dtype, 
                        input_type='hst', strict=True)
                elif mtrc_type == 'aeroopt':
                    mtrc_varmbs = get_opticalerr(x_mbsmini, xhat_mbsmini, rho_chm, refr_wvchm, 
                        len_wv, n_max, void_refr, chem_species, n_seeds, n_mbevl_, n_chem, 
                        n_bins, n_wave, tch_device, tch_dtype, tch_cdtype, input_type='hst')
                elif mtrc_type == 'aeroinp':
                    mtrc_varmbs = get_inperr(x_mbsmini, xhat_mbsmini, tmprtr_inpmin, tmprtr_inpmax, 
                        n_tmprtr, rho_chm, chem_species, n_seeds, n_mbevl_, n_chem, n_bins, 
                        input_type='hst')
                elif mtrc_type == 'aerorel':
                    mtrc_varmbs = get_histrelerrs(x_mbsmini, xhat_mbsmini, n_seeds, n_mbevl_, 
                        n_chem, n_bins, pnorm=pnorm)
                else:
                    raise ValueError(f'undefined mtrc_type={mtrc_type}')

                for mtrc_varname in mtrc_allvarnames:
                    mtrc_varmb = mtrc_varmbs[mtrc_varname]
                    assert mtrc_varmb.shape == (n_seeds, n_mbevl_)
                    mtrc_varlsts[mtrc_varname].append(mtrc_varmb)
            
            mtrc_vars = dict()
            for mtrc_varname, metrc_varlst in mtrc_varlsts.items():
                # Example:
                #    mtrc_varname = 'ccn_err'
                #    metrc_varlst = [torch.randn(n_seeds, 32), ..., torch.randn(n_seeds, 32)]
                #    mtrc_var = 'ccn_err'
                mtrc_var1 = torch.cat(metrc_varlst, dim=-1)
                assert mtrc_var1.shape == (n_seeds, n_pntsrcns)

                mtrc_var = mtrc_var1.unflatten(1, (n_pnts, n_rcns))
                assert mtrc_var.shape == (n_seeds, n_pnts, n_rcns)
                
                mtrc_vars[mtrc_varname] = mtrc_var

            # The list of variables to output
            mtrc_varnames = mtrc_cfg.pop('vars')
            mtrc_varnames = mtrc_varnames if mtrc_varnames is not None else list(mtrc_vars)
            assert isinstance(mtrc_varnames, (list, tuple))

            for mtrc_varname in mtrc_varnames:
                mtrc_name = mtrc_namefmt.format(var=mtrc_varname)

                assert mtrc_name not in mtrc_pntsdata, dedent(f'''
                    The metric name format produced clashing names for 
                    different settings. Make sure you make the name format 
                    specific to each setting.
                        name/fmt = {mtrc_namefmt}''')

                assert mtrc_varname in mtrc_vars, dedent(f'''
                    The hellinger distance does not produce a "{mtrc_varname}"
                    variable:
                        offending var name = {mtrc_varname}
                        list of requested vars = {mtrc_varnames}
                        available vars = {mtrc_vars.keys()}''')

                mtrc_pntsdata[mtrc_name] = mtrc_vars[mtrc_varname]
        else:
            raise ValueError(f'undefined metric type = {mtrc_type}')

        assert n_pnts is not None
        assert len(mtrc_cfg) == 0, f'unused metric options: {mtrc_cfg}'

        return mtrc_pntsdata, n_pnts, n_rcns

    @staticmethod
    @torch.no_grad()
    def performance(perfcfgs, e_mtrcinputs, aero_cstinfo, asgn_mdlscfg, metric_mdlscfg, eparams, 
        split_vars, n_seeds, n_t, mrng, timer, tch_device, tch_dtype, tch_cdtype):
        # The performance output key formatter
        prf_keyfmt  = '{prf_id}/{src_eid}/{src_splt}/{src_lbl}/{trg_eid}/{trg_splt}/{trg_lbl}'
        prf_keyfmt += '/{asgn_id}/{asgn_idx}/{prf_spc}/{prf_node}/{mtrc_name}/{stat_name}'

        # The performance statistics output dictionary
        e_perfstats, e_perfhists = dict(), dict()
        for prf_id, prf_cartcfg in perfcfgs.items():
            for srctrg in prf_cartcfg['srctrg']:
                # Example:
                #    srctrg = 'mediumscale/train/orig::mediumscale/train/rcnst'
                #    src_eidspltlbl = 'mediumscale/train/orig'
                #    trg_eidspltlbl = 'mediumscale/train/rcnst'
                #    src_eid, src_splt, src_lbl = 'mediumscale', 'train', 'orig'
                #    trg_eid, trg_splt, trg_lbl = 'mediumscale', 'train', 'rcnst'
                #
                # Example 2:
                #    srctrg = 'mediumscale//orig::/train/rcnst'
                #    src_eidspltlbl = 'mediumscale//orig'
                #    trg_eidspltlbl = '/train/rcnst'
                #    src_eid_, src_splt_, src_lbl_ = 'mediumscale', '', 'orig'
                #    trg_eid_, trg_splt_, trg_lbl_ = '', 'train', 'rcnst'
                #    src_eid, src_splt, src_lbl = 'mediumscale', 'train', 'orig'
                #    trg_eid, trg_splt, trg_lbl = 'mediumscale', 'train', 'rcnst'
                src_eidspltlbl, trg_eidspltlbl = srctrg.split('::')
                src_eid_, src_splt_, src_lbl_ = src_eidspltlbl.split('/')
                trg_eid_, trg_splt_, trg_lbl_ = trg_eidspltlbl.split('/')

                # Filling out any missing values
                src_eid = trg_eid_ if src_eid_ == '' else src_eid_
                trg_eid = src_eid_ if trg_eid_ == '' else trg_eid_

                src_splt = trg_splt_ if src_splt_ == '' else src_splt_
                src_lbl = trg_lbl_ if src_lbl_ == '' else src_lbl_

                trg_splt = src_splt_ if trg_splt_ == '' else trg_splt_
                trg_lbl = src_lbl_ if trg_lbl_ == '' else trg_lbl_

                # Example:
                #   src_data = {
                #       'x/m_chmprtnrm/pnts': torch.randn(n_seeds, n_pnts, n_chem, n_bins)
                #       'x/m_prthstmag/pnts': torch.randn(n_seeds, n_pnts, 1, 1)}
                src_data = get_subdict(e_mtrcinputs, prefix=f'{src_eid}/{src_splt}/{src_lbl}', pop=False)
                trg_data = get_subdict(e_mtrcinputs, prefix=f'{trg_eid}/{trg_splt}/{trg_lbl}', pop=False)

                assert (len(src_data) > 0) == (len(trg_data) > 0), dedent(f'''
                    One of the source or target data is empty while the other is not. This most likely 
                    means the source and target data may not have the same frequency of evaluation and 
                    sampling:
                        src_data: {src_data.keys()}
                        trg_data: {trg_data.keys()}
                        perf: {prf_id}
                        source: {src_eid}/{src_splt}/{src_lbl}
                        target: {trg_eid}/{trg_splt}/{trg_lbl}''')
                
                # Looks like this epoch is not the evaluation time.
                if len(src_data) == 0:
                    continue

                # Example:
                #    prf_cartcfg['asgn'] = ['nearstx/0', 'nearstx/1', 'nearstx/2', 'same/0']
                #    asgnididx = 'nearstx/1'
                #    asgn_id = 'nearstx'
                #    asgnidxstr = '1'
                #    asgn_id2idxs = {'nearstx': [0, 1, 2], 'same': [0]}
                asgn_id2idxs = defaultdict(list)
                for asgnididx in prf_cartcfg['asgn']:
                    asgn_id, asgnidxstr = asgnididx.split('/')
                    asgn_id2idxs[asgn_id].append(int(asgnidxstr))

                for asgn_id, asgnidxs in asgn_id2idxs.items():
                    ###############################################################################
                    ####################### Applying the Assignment Process #######################
                    ###############################################################################
                    # Example:
                    #   asgn_id = 'nearstz'
                    #   asgn_cfg = {'type': 'knn', 'k': 2,
                    #       'node': 'x/m_chmprtnrm1/pnts::x/m_chmprtnrm2/pnts'}
                    #   asgn_type = 'knn'
                    #   assgn_k = 2
                    #   assgn_spcnoderepr = 'x/m_chmprtnrm1/pnts::x/m_chmprtnrm2/pnts'
                    #   assgn_srcspc, asgn_srcnode, asgn_srcrepr = 'x', 'm_chmprtnrm1', 'pnts'
                    #   assgn_trgspc, asgn_trgnode, asgn_trgrepr = 'x', 'm_chmprtnrm2', 'pnts'
                    #   src_dataasgn = {'pnts': torch.randn(n_seeds, n_pnts, n_chem, n_bins)}
                    #   trg_dataasgn = {'pnts': torch.randn(n_seeds, n_pnts, n_chem, n_bins)}
                    asgn_cfg = get_subdict(asgn_mdlscfg, prefix=asgn_id, pop=False)
                    asgn_type = asgn_cfg.pop('type')

                    timer.tic(f'perf/asgn/{prf_id}/{asgn_id}')
                    if asgn_type == 'same':
                        trg_idxall, n_mbsrcasgn, n_mbtrgasgn = None, None, None
                    elif asgn_type == 'knn':
                        assgn_k = asgn_cfg.pop('k')
                        assgn_spcnoderepr = asgn_cfg.pop('node')

                        if assgn_spcnoderepr.count('::') == 1:
                            assgn_srcspcnoderepr, assgn_trgspcnoderepr = assgn_spcnoderepr.split('::')
                        else:
                            assert assgn_spcnoderepr.count('::') == 0
                            assgn_srcspcnoderepr = assgn_trgspcnoderepr = assgn_spcnoderepr

                        assgn_srcspc, asgn_srcnode, asgn_srcrepr = assgn_srcspcnoderepr.split('/')
                        assgn_trgspc, asgn_trgnode, asgn_trgrepr = assgn_trgspcnoderepr.split('/')

                        src_dataasgn = get_subdict(src_data, 
                            prefix=f'{assgn_srcspc}/{asgn_srcnode}/{asgn_srcrepr}', pop=False)
                        assert torch.is_tensor(src_dataasgn)
                        n_mbsrcasgn, *dims_asgnsrc = src_dataasgn.shape[1:]
                        assert src_dataasgn.shape == (n_seeds, n_mbsrcasgn, *dims_asgnsrc)

                        trg_dataasgn = get_subdict(trg_data, 
                            prefix=f'{assgn_trgspc}/{asgn_trgnode}/{asgn_trgrepr}', pop=False)
                        assert torch.is_tensor(trg_dataasgn)
                        n_mbtrgasgn, *dims_asgntrg = trg_dataasgn.shape[1:]
                        assert trg_dataasgn.shape == (n_seeds, n_mbtrgasgn, *dims_asgntrg)

                        dim_db = math.prod(dims_asgnsrc)
                        assert dim_db == np.prod(dims_asgntrg), dedent(f'''
                            The source and target data do not share the same dimensions:
                                performance id = {prf_id}
                                source split = {src_eidspltlbl}
                                target split = {trg_eidspltlbl}
                                assignment node = {assgn_srcspcnoderepr}
                                source dims = {dims_asgnsrc}
                                target dims = {dims_asgntrg}''')

                        asgn_lib = asgn_cfg.pop('lib')
                        if asgn_lib == 'faiss':
                            trg_idxall = Evaluation.run_faissknn(asgn_cfg, src_dataasgn, trg_dataasgn, 
                                n_mbsrcasgn, n_mbtrgasgn, dim_db, assgn_k, n_seeds, tch_device)
                            assert trg_idxall.shape == (n_seeds, n_mbsrcasgn, assgn_k)
                        elif asgn_lib == 'torch':
                            trg_idxall = Evaluation.run_torchknn(asgn_cfg, src_dataasgn, trg_dataasgn, 
                               n_mbsrcasgn,  n_mbtrgasgn, dim_db, assgn_k, n_seeds, tch_dtype)
                            assert trg_idxall.shape == (n_seeds, n_mbsrcasgn, assgn_k)
                        else:
                            raise ValueError(f'undefined assignment library = {asgn_lib}')
                    else:
                        raise ValueError(f'undefined assignment type = {asgn_type}')
                    timer.toc()

                    for asgn_idx in asgnidxs:
                        # Next, we use the `trg_idxall` variable to align the target to the source data.

                        # We don't need to re-arrange the src data
                        src_dataasgnd = src_data
                        if trg_idxall is None:
                            assert asgn_idx == 0, dedent(f'''
                                only "same/0" is an acceptable "same" assignment:
                                    asgn_id = {asgn_id}
                                    asgn_idx = {asgn_idx}''')
                            # There is no need to re-arrange the target points in this case.
                            trg_dataasgnd = trg_data
                        else:
                            # We'll use the `trg_idxs` to align the source with the target.
                            trg_idxs = trg_idxall[:, :, asgn_idx]
                            assert trg_idxs.shape == (n_seeds, n_mbsrcasgn)

                            trg_dataasgnd = dict()
                            for node_name, node_tnsr in trg_data.items():
                                bcast_ones = (1,) * len(node_tnsr.shape[2:])
                                trg_idxsbcast = trg_idxs.reshape(n_seeds, n_mbsrcasgn, *bcast_ones)
                                trg_dataasgnd[node_name] = torch.take_along_dim(node_tnsr, trg_idxsbcast, dim=1)

                        for prf_spcnodes in prf_cartcfg['node']:
                            # Example:
                            #    prf_spcnodes = 'x/m_chmprthst,x/n_prthst'
                            #    prf_spcnode = 'x/m_chmprthst'
                            #    prf_spc, prf_node = 'x', 'm_chmprthst'
                            #    src_mtrcdata = {'pts': torch.randn(n_seeds, n_mbsrc, n_chem, n_bins)}
                            #    src_mtrcdatas = {
                            #       'x/m_chmprthst': {'pts': torch.randn(n_seeds, n_mbsrc, n_chem, n_bins),
                            #       'x/n_prthst':    {'pts': torch.randn(n_seeds, n_mbsrc,      1, n_bins)}
                            src_mtrcdatas, trg_mtrcdatas = dict(), dict()
                            for prf_spcnode in prf_spcnodes.split(','):
                                prf_spc, prf_node = prf_spcnode.split('/')
                                src_mtrcdata = get_subdict(src_dataasgnd, prefix=f'{prf_spc}/{prf_node}', pop=False)
                                assert len(src_mtrcdata) > 0, dedent(f'''
                                    Found empty source metric data:
                                        perf: {prf_id}
                                        source: {src_eid}/{src_splt}/{src_lbl}
                                        target: {trg_eid}/{trg_splt}/{trg_lbl}
                                        assignment: {asgn_id}/{asgn_idx}
                                        node: {prf_spc}/{prf_node}''')
                                src_mtrcdatas[f'{prf_spc}/{prf_node}'] = src_mtrcdata
                                trg_mtrcdata = get_subdict(trg_dataasgnd, prefix=f'{prf_spc}/{prf_node}', pop=False)
                                assert len(trg_mtrcdata) > 0, dedent(f'''
                                    Found empty target metric data:
                                        perf: {prf_id}
                                        source: {src_eid}/{src_splt}/{src_lbl}
                                        target: {trg_eid}/{trg_splt}/{trg_lbl}
                                        assignment: {asgn_id}/{asgn_idx}
                                        node: {prf_spc}/{prf_node}''')
                                trg_mtrcdatas[f'{prf_spc}/{prf_node}'] = trg_mtrcdata
                            prf_spc, prf_node = None, None

                            ###############################################################################
                            #################### Calculating the Performance Metrics ######################
                            ###############################################################################
                            for mtrc_id in prf_cartcfg['metric']:
                                timer.tic(f'perf/metric/{prf_id}/{mtrc_id}')
                                # This is where most of the computational heavy-lifting happens
                                mtrc_cfg = deep2hie(metric_mdlscfg[mtrc_id])

                                if mtrc_cfg.get('type', None) == 'menger':
                                    Evaluation.check_mengereval(mtrc_id, src_eid, src_splt, src_lbl, 
                                        trg_eid, trg_splt, trg_lbl, asgn_type, eparams, split_vars)

                                mtrc_pntsdata, n_pnts, n_rcns = Evaluation.metric(mtrc_cfg, mtrc_id, src_mtrcdatas,
                                    trg_mtrcdatas, aero_cstinfo, n_seeds, n_t, mrng, tch_device, tch_dtype, 
                                    tch_cdtype)

                                timer.toc()
                                ###############################################################################
                                #################### Applying the Aggregation Statistics ######################
                                ###############################################################################
                                for mtrc_name, mtrc_pnts1 in mtrc_pntsdata.items():
                                    assert mtrc_pnts1.shape == (n_seeds, n_pnts, n_rcns)
                                    mtrc_pnts = mtrc_pnts1.flatten(1, 2)
                                    assert mtrc_pnts.shape == (n_seeds, n_pnts * n_rcns)

                                    for stat_name in prf_cartcfg['stat']:
                                        mtrc_hist, mtrc_agg = None, None
                                        if stat_name == 'mean':
                                            mtrc_agg = mtrc_pnts.mean(dim=-1)
                                            assert mtrc_agg.shape == (n_seeds,)
                                        elif stat_name == 'median':
                                            mtrc_agg = mtrc_pnts.median(dim=-1).values
                                            assert mtrc_agg.shape == (n_seeds,)
                                        elif stat_name == 'max':
                                            mtrc_agg = mtrc_pnts.max(dim=-1).values
                                            assert mtrc_agg.shape == (n_seeds,)
                                        elif stat_name == 'min':
                                            mtrc_agg = mtrc_pnts.min(dim=-1).values
                                            assert mtrc_agg.shape == (n_seeds,)
                                        elif stat_name.startswith('q'):
                                            qntl_str = stat_name[1:]
                                            assert qntl_str.replace('.', '').isdigit(), f'bad quantile: {qntl_str}'
                                            assert qntl_str.count('.') <= 1, f'bad quantile: {qntl_str}'
                                            qntl_q = float(qntl_str)
                                            assert qntl_q >= 0.0, f'bad quantile: {qntl_str} < 0'
                                            assert qntl_q <= 100.0, f'bad quantile: {qntl_str} > 100'
                                            mtrc_agg = mtrc_pnts.quantile(qntl_q / 100.0, dim=-1)
                                            assert mtrc_agg.shape == (n_seeds,)
                                        elif stat_name.startswith('Q'):
                                            nqntls_str = stat_name[1:]
                                            assert nqntls_str.isdigit(), f'bad number of bins: {nqntls_str}'
                                            n_qntls = int(nqntls_str) + 1
                                            qntls = torch.linspace(0.0, 1.0, n_qntls, device=tch_device, dtype=tch_dtype)
                                            mtrc_histT = mtrc_pnts.quantile(q=qntls, dim=-1)
                                            assert mtrc_histT.shape == (n_qntls, n_seeds)
                                            mtrc_hist = mtrc_histT.transpose(-1, -2)
                                            assert mtrc_hist.shape == (n_seeds, n_qntls)
                                        else:
                                            raise ValueError(f'undefined metric aggregation stat = {stat_name}')

                                        prf_spc_ = '_'.join([prf_spcnode.split('/')[0] for prf_spcnode in prf_spcnodes.split(',')])
                                        prf_node_ = '_'.join([prf_spcnode.split('/')[1] for prf_spcnode in prf_spcnodes.split(',')])
                                        naming_vars = dict(prf_id=prf_id, src_eid=src_eid, src_splt=src_splt,
                                            src_lbl=src_lbl, trg_eid=trg_eid, trg_splt=trg_splt, trg_lbl=trg_lbl,
                                            asgn_id=asgn_id, asgn_idx=asgn_idx, prf_spc=prf_spc_, prf_node=prf_node_,
                                            mtrc_name=mtrc_name, stat_name=stat_name)

                                        prf_key = prf_keyfmt.format(**naming_vars)

                                        assert ((prf_key not in e_perfstats) and 
                                                (prf_key not in e_perfhists)), dedent(f'''
                                            The metric name format produced clashing names for 
                                            different settings. Make sure you make the name format 
                                            specific to each setting.
                                                performance key = {prf_key}
                                                key formatter = {prf_keyfmt}
                                                naming variables = {naming_vars}
                                        ''')
                                        if mtrc_agg is not None:
                                            e_perfstats[prf_key] = mtrc_agg
                                        if mtrc_hist is not None:
                                            e_perfhists[prf_key] = mtrc_hist

        return e_perfstats, e_perfhists

class Visualization:
    @staticmethod
    def mds(e_vizinputs, eparams, vizcfgs, vrng, n_seeds, z_dims, y_dims, epoch, timer, tch_device):
        # e_vizinputs = {
        #     '{eid}:{e_space}:{e_node}/{e_split}/{e_vrnt}/{e_repr}': torch.randn(n_seeds, n_evlpnts, *)}
        # where we have
        #     e_vrnt in ('orig', 'rcnst', 'genr', ...)
        #     e_space in ('x', 'y', 'z')
        #     e_node in ('z', 'm_chmprtnrm', ...)
        #     e_repr in ('pnts', 'mu', 'sig')
        #
        # Example:
        #   e_vizinputs = {
        #       f'{eid}:x:m_chmprthst/train/orig/pnts':  torch.randn(n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen),
        #       f'{eid}:x:m_chmprthst/train/rcnst/pnts': torch.randn(n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen),
        #       f'{eid}:x:m_chmprthst/test/orig/pnts':   torch.randn(n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen),
        #       f'{eid}:x:m_chmprthst/test/rcnst/pnts':  torch.randn(n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen),
        #       f'{eid}:x:m_chmprthst/normal/genr/pnts': torch.randn(n_seeds, n_evlpnts, n_rcns, n_uchnls, n_ulen),
        #       f'{eid}:z:z/train/ltnt/pnts':            torch.randn(n_seeds, n_evlpnts, n_rcns, ltnt_dim),
        #       f'{eid}:z:z/train/blob/mu':              torch.randn(n_seeds, n_evlpnts, n_rcns, ltnt_dim),
        #       f'{eid}:z:z/train/blob/sig':             torch.randn(n_seeds, n_evlpnts, n_rcns, ltnt_dim),
        #       f'{eid}:z:z/test/ltnt/pnts':             torch.randn(n_seeds, n_evlpnts, n_rcns, ltnt_dim),
        #       f'{eid}:z:z/test/blob/mu':               torch.randn(n_seeds, n_evlpnts, n_rcns, ltnt_dim),
        #       f'{eid}:z:z/test/blob/sig':              torch.randn(n_seeds, n_evlpnts, n_rcns, ltnt_dim),
        # ...}

        e_vizoutputs = dict()
        for vid, vcfg_ in vizcfgs.items():
            vcfg = dict(vcfg_)
            eid = vcfg.pop('eval')
            v_space = vcfg.pop('space')
            v_node = vcfg.pop('node')
            v_type = vcfg.pop('type')
            v_store = vcfg.pop('store')
            v_seedidxs = vcfg.pop('seeds')
            v_splits = vcfg.pop('splits')
            v_mplopts = vcfg.pop('mplopts')
            v_fig = vcfg.pop('fig', None)
            v_axes = vcfg.pop('axes', None)
            v_chnlnames = vcfg.pop('chnlnames', None)
            assert v_node is not None

            eopts = eparams[eid]
            # The evaluation frequency
            e_frq = eopts['frq']
            # Checking if the samples were evaluated this epoch
            if epoch % e_frq > 0:
                continue 
            # The split configs
            e_splitcfgs = eopts['splits']
            # Example:
            #   e_splitcfgs = {
            #       'train/type': 'train',
            #       'train/n': 20000,
            #       ...
            #       'test/type': 'test',
            #       'test/n': 5000,
            #       ...}

            evizdict = get_subdict(e_vizinputs, prefix=f'{eid}:{v_space}:{v_node}', pop=False)
            # Example:
            #   eid, vid, v_space, v_node = 'identity', '1x1xtsne', 'x', 'm_chmprthst'
            #   v_type in ('bar', 'scatter')
            #   v_space in ('x', 'z')
            #   evizdict = {
            #       'train/orig/pnts': np.randn(n_seeds, n_evlpnts, n_rcns, n_chnls * k_bins),
            #       'train/rcnst/pnts':  np.randn(n_seeds, n_evlpnts, n_rcns, n_chnls * k_bins),
            #       'test/orig/pnts':  np.randn(n_seeds, n_evlpnts, n_rcns, n_chnls * k_bins),
            #        ...}

            # The nodes and their dimensions after applying the `e_pp` pre-processor
            uv_dims = eopts['udims']

            # The viz input data dimensions before applying any MDS, etc.
            lookup1 = {f'scatter/x/{u_node}': (n_uchnls, n_ulen)
                for u_node, (n_uchnls, n_ulen) in uv_dims.items()}
            lookup2 = {f'scatter/y/{y_node}': (n_ychnls, n_ylen)
                for y_node, (n_ychnls, n_ylen) in y_dims.items()}
            lookup3 = {f'bar/x/{u_node}': (n_uchnls, n_ulen)
                for u_node, (n_uchnls, n_ulen) in uv_dims.items()}
            lookup4 = {f'bar/y/{y_node}': (n_ychnls, n_ylen)
                for y_node, (n_ychnls, n_ylen) in y_dims.items()}
            lookup5 = {f'scatter/z/{z_node}': z_dim 
                for z_node, z_dim in z_dims.items()}
            lookup6 = {**lookup1, **lookup2, **lookup3, **lookup4, **lookup5}
            shp_vizinp = lookup6[f'{v_type}/{v_space}/{v_node}']

            # The viz output data dimensions after applying any MDS, etc.
            lookup7 = {f'scatter/x/{u_node}': (2,)
                for u_node, (n_uchnls, n_ulen) in uv_dims.items()}
            lookup8 = {f'scatter/y/{y_node}': (2,)
                for y_node, (n_ychnls, n_ylen) in y_dims.items()}
            lookup9 = {f'bar/x/{u_node}': (n_uchnls, n_ulen)
                for u_node, (n_uchnls, n_ulen) in uv_dims.items()}
            lookup10 = {f'bar/y/{y_node}': (2,)
                for y_node, (n_ychnls, n_ylen) in y_dims.items()}
            lookup11 = {f'scatter/z/{z_node}': (2,)
                for z_node, z_dim in z_dims.items()}
            lookup12 = {**lookup7, **lookup8, **lookup9, **lookup10, **lookup11}
            shp_vizout = lookup12[f'{v_type}/{v_space}/{v_node}']

            d_vizinp, d_vizout = np.prod(shp_vizinp), np.prod(shp_vizout)

            # Let's collect the points data (i.e., exclude data such as `mu` and `sigma` tensors)
            eviz_pntsdict, eviz_musdict, eviz_sigsdict = dict(), dict(), dict()
            spltnpntslst, spltnmuslst, spltnsigslst = [], [], []
            for v_svr, v_tensor in evizdict.items():
                # Example:
                #  `e_split in ('train', 'test', 'normal', 'snr')
                #   e_vrnt in ('orig', 'rcnst', 'genr', 'ltnt', 'blob')
                #   v_repr in ('pnts', 'mu', 'sig')`
                #   v_svr in ('train/orig/pnts', 'test/rcnst/pnts', ...)
                e_split, e_vrnt, v_repr = v_svr.split('/')
                n_evlpnts = e_splitcfgs[e_split]['n']
                e_vrntcfgs = e_splitcfgs[e_split]['vrnts']
                n_rcns = e_vrntcfgs[e_vrnt].get('ltnt/n', 1)
                assert v_tensor.shape == (n_seeds, n_evlpnts, n_rcns, *shp_vizinp)
                if v_repr == 'pnts':
                    eviz_pntsdict[v_svr] = v_tensor.flatten(1, 2)
                    spltnpntslst.append((v_svr, n_evlpnts, n_rcns))
                elif v_repr == 'mu':
                    eviz_musdict[v_svr] = v_tensor.flatten(1, 2)
                    spltnmuslst.append((v_svr, n_evlpnts, n_rcns))
                elif v_repr == 'sig':
                    eviz_sigsdict[v_svr] = v_tensor.flatten(1, 2)
                    spltnsigslst.append((v_svr, n_evlpnts, n_rcns))
                else:
                    raise RuntimeError(f'undefined e_vrnt={e_vrnt}')

            # The total number of visualization points
            n_vizpnts = sum(n_evlpnts * n_rcns for v_svr, n_evlpnts, n_rcns in spltnpntslst)
            # The total number of visualization mu points
            n_vizmus = sum(n_evlmus * n_rcns for v_svr, n_evlmus, n_rcns in spltnmuslst)
            # The total number of visualization sig points
            n_vizsigs = sum(n_evlsigs * n_rcns for v_svr, n_evlsigs, n_rcns in spltnsigslst)

            # These four variables are the product of the following if-else case.
            # In order not to leave any dangling pointers from previous iterations, 
            # I'll nullify them here, even though I tried to be careful when coding 
            # and re-defining them in all cases.
            v_pntsemb, v_muemb, v_sigemb, v_sigrots = None, None, None, None

            timer.tic(f'mds/{eid}/{vid}')
            if v_type == 'scatter':
                # the multi-dimensional scaling used for scatter plots
                v_mds = vcfg.pop('mds')
                # Whether we should apply the tsne/pca on CPU or GPU ('')
                v_device = vcfg.pop('device') if v_mds in ('tsne',) else None

                with torch.no_grad():
                    # Concatenating all points
                    vall_epp = torch.cat([v_tensor 
                        for v_svr, v_tensor in eviz_pntsdict.items()], dim=1).flatten(2, -1)
                    assert vall_epp.shape == (n_seeds, n_vizpnts, d_vizinp)

                    # Concatenating all mus
                    if n_vizmus > 0:
                        vmuall_epp = torch.cat([v_tensor 
                            for v_svr, v_tensor in eviz_musdict.items()], dim=1).flatten(2, -1)
                        assert vmuall_epp.shape == (n_seeds, n_vizmus, d_vizinp)

                    # Concatenating all sigmas
                    if n_vizsigs > 0:
                        vsigall_epp = torch.cat([v_tensor 
                            for v_svr, v_tensor in eviz_sigsdict.items()], dim=1).flatten(2, -1)
                        assert vsigall_epp.shape == (n_seeds, n_vizsigs, d_vizinp)

                    # Making sure the tsne input data is not too large, since faiss crashes in it.
                    assert not vall_epp.isnan().any()
                    assert not vall_epp.isinf().any()
                    if (v_mds == 'tsne') and (vall_epp.abs() > 1e30).any().item():
                        vall_epp[vall_epp > 1e30] = 1e30
                        vall_epp[vall_epp < -1e30] = -1e30

                ####### Applying Multi-Dimensional Scaling #######
                ########### to the Visualization Data ############
                if v_mds == 'tsne':
                    if (v_device == 'cuda') and (tch_device.type == 'cuda'):
                        TSNEClass = TSNECuda
                        TSNEDeviceArgs = dict(device=tch_device.index, random_seed=12345)
                    else:
                        TSNEClass, TSNEDeviceArgs = TSNESK, dict(random_state=12345)

                    v_emb_lst = []
                    for i_seed in range(n_seeds):
                        tsne = TSNEClass(n_components=d_vizout, **TSNEDeviceArgs)
                        tsne_idata = vall_epp[i_seed].detach().cpu().numpy()
                        assert tsne_idata.shape == (n_vizpnts, d_vizinp)

                        v_iemb_ = tsne.fit_transform(tsne_idata)
                        assert v_iemb_.shape == (n_vizpnts, d_vizout)

                        v_iemb = v_iemb_.reshape(n_vizpnts, d_vizout)
                        assert v_iemb.shape == (n_vizpnts, d_vizout)
                        v_emb_lst.append(v_iemb)

                    v_pntsemb_ = np.stack(v_emb_lst, axis=0)
                    assert v_pntsemb_.shape == (n_seeds, n_vizpnts, d_vizout)

                    # blob transformation are undefined for tsne
                    v_muemb_, v_sigemb_, v_sigrots_ = None, None, None
                elif v_mds == 'pcalr':
                    pcalr_niter = vcfg.pop('niter')
                    pcalr_q = vcfg.pop('q')
                    with torch.no_grad():
                        # First we find the right set of projection directions 
                        # (i.e., principal components) using the points data alone
                        pca_u, pca_s, pca_v = torch.pca_lowrank(vall_epp, q=pcalr_q, center=True, niter=pcalr_niter)
                        d_pcalrinfered = pca_v.shape[-1]
                        assert pca_v.shape == (n_seeds, d_vizinp, d_pcalrinfered)
                        pca_ldproj = pca_v[..., :d_vizout]
                        assert pca_ldproj.shape == (n_seeds, d_vizinp, d_vizout), f'{pca_ldproj.shape} != {(n_seeds, d_vizinp, d_vizout)}'
                        
                        # Now, we project the points data back using `pca_ldproj`
                        v_embb = vall_epp.matmul(pca_ldproj).detach().cpu().numpy()
                        assert v_embb.shape == (n_seeds, n_vizpnts, d_vizout)
                        v_pntsemb_ = v_embb.reshape(n_seeds, n_vizpnts, d_vizout)
                        assert v_pntsemb_.shape == (n_seeds, n_vizpnts, d_vizout)

                        # Next, we project the mu data back using `pca_ldproj`
                        if n_vizmus > 0:
                            v_muembb_ = vmuall_epp.matmul(pca_ldproj).detach().cpu().numpy()
                            assert v_muembb_.shape == (n_seeds, n_vizmus, d_vizout)

                            v_muemb_ = v_muembb_.reshape(n_seeds, n_vizmus, d_vizout)
                            assert v_muemb_.shape == (n_seeds, n_vizmus, d_vizout)
                        else:
                            v_muemb_ = None
                        
                        # Next, we project the sig data back using `pca_ldproj`
                        if n_vizsigs > 0:
                            sig_datarot = (vsigall_epp.reshape(n_seeds, n_vizsigs, d_vizinp,        1) * 
                                           pca_ldproj.reshape( n_seeds, 1        , d_vizinp, d_vizout))
                            assert sig_datarot.shape == (n_seeds, n_vizsigs, d_vizinp, d_vizout)

                            _, v_sigemb_tch, rot_matout = torch.linalg.svd(sig_datarot, full_matrices=False)
                            assert rot_matout.shape == (n_seeds, n_vizsigs, d_vizout, d_vizout)
                            assert v_sigemb_tch.shape == (n_seeds, n_vizsigs, d_vizout)

                            v_sigemb_ = v_sigemb_tch.detach().cpu().numpy()
                            assert v_sigemb_.shape == (n_seeds, n_vizsigs, d_vizout)

                            assert d_vizout == 2
                            # `rot_matout` is a 2d rotation matrix: 
                            #     rot_matout = [[cos(phi), -sin(phi)], [sin(phi), cos(phi)]]
                            # This matrix can be summarized in a single `phi` angle.
                            # To recover phi unambigiously, we use the 
                            #   `phi = acos(cos(phi)) * sign(sin(phi))` 
                            # formula. `v_sigrots_` will hold the `phi` values
                            v_sigrots_tch = torch.acos(rot_matout[..., 0, 0]) * torch.sign(rot_matout[..., 1, 0])
                            assert v_sigrots_tch.shape == (n_seeds, n_vizsigs)

                            v_sigrots_ = v_sigrots_tch.reshape(n_seeds, n_vizsigs, 1).detach().cpu().numpy()
                            assert v_sigrots_.shape == (n_seeds, n_vizsigs, 1)
                        else:
                            v_sigemb_, v_sigrots_ = None, None
                elif (v_mds == 'pca'):
                    with torch.no_grad():
                        pcadata_nomean = vall_epp - vall_epp.mean(dim=-2, keepdims=True)
                        assert pcadata_nomean.shape == (n_seeds, n_vizpnts, d_vizinp)

                        pca_u, pca_s, pca_vt = torch.linalg.svd(pcadata_nomean, full_matrices=True)
                        pca_v = pca_vt[..., :d_vizout, :].transpose(-1, -2)
                        v_pntsemb_ = pcadata_nomean.matmul(pca_v).detach().cpu().numpy()
                        assert v_pntsemb_.shape == (n_seeds, n_vizpnts, d_vizout)

                        # blob transformation are undefined for pca (use pcalr instead)
                        v_muemb_, v_sigemb_, v_sigrots_ = None, None, None
                elif (v_mds == 'none'):
                    assert d_vizinp == d_vizout
                    v_pntsemb_ = vall_epp.detach().cpu().numpy()
                    assert v_pntsemb_.shape == (n_seeds, n_vizpnts, d_vizout)

                    if n_vizmus > 0:
                        v_muemb_ = vmuall_epp.detach().cpu().numpy()
                        assert v_muemb_.shape == (n_seeds, n_vizmus, d_vizout)
                    else:
                        v_muemb_ = None
                    
                    if n_vizsigs > 0:
                        v_sigemb_ = vsigall_epp.detach().cpu().numpy()
                        assert v_sigemb_.shape == (n_seeds, n_vizsigs, d_vizout)

                        assert d_vizout == 2
                        v_sigrots_ = np.zeros((n_seeds, n_vizsigs, 1))
                        assert v_sigrots_.shape == (n_seeds, n_vizsigs, 1)
                    else:
                        v_sigemb_, v_sigrots_ = None, None
                else:
                    raise ValueError(f'undefined viz mds = {v_mds}')

                # Reshaping the MDS embedding data to the output shape
                v_pntsemb = v_pntsemb_.reshape(n_seeds, n_vizpnts, *shp_vizout)
                assert v_pntsemb.shape == (n_seeds, n_vizpnts, *shp_vizout)

                if v_muemb_ is not None:
                    v_muemb = v_muemb_.reshape(n_seeds, n_vizmus, *shp_vizout)
                    assert v_muemb.shape == (n_seeds, n_vizmus, *shp_vizout)
                else:
                    v_muemb = None
                
                if v_sigemb_ is not None:
                    v_sigemb = v_sigemb_.reshape(n_seeds, n_vizsigs, *shp_vizout)
                    assert v_sigemb.shape == (n_seeds, n_vizsigs, *shp_vizout)
                else:
                    v_sigemb = None

                if v_sigrots_ is not None:
                    v_sigrots = v_sigrots_
                    assert v_sigrots.shape == (n_seeds, n_vizsigs, 1)
                else:
                    v_sigrots = None                
            elif v_type == 'bar':
                v_mds = None
                
                assert v_space == 'x', f'bar plot on {v_space} space makes no sense'
                v_mds, v_device = None, None
                # `v_node` in this case is the same as `x_node`
                (n_uchnls, n_ulen) = uv_dims[v_node]

                with torch.no_grad():
                    vall_epp = torch.cat([v_tensor for v_svr, v_tensor in eviz_pntsdict.items()], dim=1)
                    assert vall_epp.shape == (n_seeds, n_vizpnts, n_uchnls, n_ulen)

                # Converting pytorch tensors to numpy
                v_pntsemb = vall_epp.detach().cpu().numpy()
                assert v_pntsemb.shape == (n_seeds, n_vizpnts, *shp_vizout)

                # blob transformation meaningless for bar plots
                v_muemb, v_sigemb, v_sigrots = None, None, None
            else:
                raise ValueError(f'undefined viz type = {v_type}')
            timer.toc()

            # By here, all `v_pntsemb`, `v_muemb`, `v_sigemb`, and `v_sigrots` must be defined.

            # Inserting all the calculated data in the `e_vizoutputs` dictionary
            for  v_repr, v_nrepritms,  n_vizrepr, v_repremb, v_reprshp in (
                ('pnts', spltnpntslst, n_vizpnts, v_pntsemb, shp_vizout),
                ('mu',   spltnmuslst,  n_vizmus,  v_muemb,   shp_vizout),
                ('sig',  spltnsigslst, n_vizsigs, v_sigemb,  shp_vizout),
                ('phi',  spltnsigslst, n_vizsigs, v_sigrots, (1,)      )):

                # If a particular `v_repr` is not available, e.g., 
                #   `mu, sig, rot` for bar plots or tsne scatter plots,
                # Then we will skip inserting them into the output
                if v_repremb is None:
                    continue
                assert v_repremb.shape == (n_seeds, n_vizrepr, *v_reprshp)

                i_mbsofar = 0
                for _, (v_svr, n_evlpnts, n_rcns) in enumerate(v_nrepritms):
                    # Example:
                    #   e_split in ('train', 'test', 'normal', 'snr')
                    #   e_vrnt in ('orig', 'rcnst', 'genr', 'ltnt', 'blob')
                    #   v_repr in ('pnts', 'mu', 'sig', 'phi')
                    #   v_svr in ('train/orig/pnts', 'test/rcnst/pnts', ...)
                    #   v_repr_ in ('pnts', 'mu', 'sig')
                    e_split, e_vrnt, v_repr_ = v_svr.split('/')
                    assert (v_repr == v_repr_) or ((v_repr, v_repr_) == ('phi', 'sig'))

                    v_repremblbl1 = v_repremb[:, i_mbsofar:(i_mbsofar + n_evlpnts * n_rcns), ...]
                    assert v_repremblbl1.shape == (n_seeds, n_evlpnts * n_rcns, *v_reprshp)
                    v_repremblbl = v_repremblbl1.reshape(n_seeds, n_evlpnts, n_rcns, *v_reprshp)
                    assert v_repremblbl.shape == (n_seeds, n_evlpnts, n_rcns, *v_reprshp)
                    e_vizoutputs[f'{eid}:{vid}:{v_space}:{v_node}/{e_split}/{e_vrnt}/{v_repr}'] = v_repremblbl

                    i_mbsofar += n_evlpnts * n_rcns
                assert i_mbsofar == n_vizrepr

            assert len(vcfg) == 0, f'unused viz options for {eid}/{vid}: {vcfg}'

        return e_vizoutputs

    @staticmethod
    @torch.no_grad()
    def draw(e_vizoutputs, vizcfgs, epoch, n_seeds,
        chem_species, d_histbins, fig_dir, timer, tbwriter):

        e_vizoutputsdeep = hie2deep(e_vizoutputs, maxdepth=1, sep='/')
        # Example: 
        #   e_vizoutputsdeep == {
        #       f'{eid}:{vid}:x:m_chmprthst': {
        #           'train/orig/pnts': x_epp1,
        #           'train/rcnst/pnts': xhat_epp2,
        #           'test/orig/pnts': x_epp3,
        #           'test/rcnst/pnts': xhat_epp4,
        #           'normal/genr/pnts': xhat_epp5},
        #   ...}

        if (fig_dir is None) and (tbwriter is None):
            return
        
        for eidvidvspc, v_embdict in e_vizoutputsdeep.items():
            # `eidvidvspc` contains the `eid`, `vid`, `v_space`, and `v_node`.
            # `v_embdict` contains the visualization data
            # Example:
            #      eidvidvspc = f'{eid}:{vid}:x:m_chmprthst'
            #      v_embdict = {
            #          'train/orig/pnts':  np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
            #          'train/rcnst/pnts': np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
            #          'test/orig/pnts':   np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
            #          'test/rcnst/pnts':  np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
            #          ...}
            eid, vid, v_space, v_node = eidvidvspc.split(':')
            vcfg = vizcfgs[vid]

            v_splitslst = vcfg['splits']
            v_splitslst = [['*']] if v_splitslst is None else v_splitslst
            v_splitslst = [[v_splitslst]] if isinstance(v_splitslst, str) else v_splitslst
            assert isinstance(v_splitslst, (list, tuple))

            for i_vsplt, v_splits in enumerate(v_splitslst):
                v_splits = ['*'] if v_splits is None else v_splits
                v_splits = [v_splits] if isinstance(v_splits, str) else v_splits
                assert isinstance(v_splits, (list, tuple))
                assert all(isinstance(pat, str) for pat in v_splits)

                all_vmplbls = list(v_embdict.keys())
                v_mpllbls = set(xx for pat in v_splits for xx in
                fnmatch.filter(all_vmplbls, pat))
                assert len(v_mpllbls) > 0, dedent(f'''
                    None of the vizualization data labels match 
                    the given viz split patterns:
                        viz subsetting pattern: {v_splits}
                        all data labels: {list(all_vmplbls)}''')
                v_embdictsub = {v_mpllbl: v_mpldat 
                    for v_mpllbl, v_mpldat in v_embdict.items()
                    if v_mpllbl in v_mpllbls}

                # `v_mplopts` contains the matplotlib options dictionary
                v_mplopts = vcfg['mplopts'].copy()
                v_seedidxs = vcfg['seeds']
                v_seedidxs = np.arange(v_seedidxs) if isinstance(v_seedidxs, int) else v_seedidxs

                # Note: `n_vseedidxs` should be the same as `n_ax` for plotting!
                n_ax = len(v_seedidxs)

                # `v_type` is the visualization type ('scatter', 'bar', etc.)
                v_type = v_mplopts.get('type')
                timer.tic(f'plt/{eid}/{vid}/{i_vsplt}')
                if v_type == 'scatter':
                    v_mpldata = dict()
                    for v_svr, v_emblbl in v_embdictsub.items():
                        # Example:
                        #  `e_split in ('train', 'test', 'normal', 'snr')
                        #   e_vrnt in ('orig', 'rcnst', 'genr', 'ltnt', 'blob')
                        #   v_repr in ('pnts', 'mu', 'sig', 'phi')`
                        #   v_svr in ('train/orig/pnts', 'test/rcnst/pnts', ...)
                        e_split, e_vrnt, v_repr = v_svr.split('/')

                        d_repr = 1 if v_repr == 'phi' else 2
                        
                        n_evlpnts, n_rcns = v_emblbl.shape[1: 3]
                        assert v_emblbl.shape == (n_seeds, n_evlpnts, n_rcns, d_repr)

                        # We will only visualize the specified seeds
                        v_emblbl2 = v_emblbl[v_seedidxs].reshape(n_ax, n_evlpnts * n_rcns, d_repr)
                        assert v_emblbl2.shape == (n_ax, n_evlpnts * n_rcns, d_repr)
                        
                        e_split, e_vrnt, v_repr = v_svr.split('/')

                        # There is no point drawing the big normal blob!
                        if (e_split == 'normal') and (v_repr in ('mu', 'sig', 'phi')):
                            continue

                        # The kind of the plotting argument
                        for ax_idx in range(n_ax):
                            if v_repr == 'pnts':
                                v_mpldata[f'{ax_idx}:{e_split}/{e_vrnt}:xy'] = v_emblbl2[ax_idx]
                            elif v_repr == 'mu':
                                v_mpldata[f'{ax_idx}:{e_split}/{e_vrnt}:c'] = v_emblbl2[ax_idx]
                            elif v_repr == 'sig':
                                # We need to show `2*sigma` as the radius
                                v_mpldata[f'{ax_idx}:{e_split}/{e_vrnt}:r'] = 2 * v_emblbl2[ax_idx]
                            elif v_repr == 'phi':
                                v_mpldata[f'{ax_idx}:{e_split}/{e_vrnt}:phi'] = v_emblbl2[ax_idx]
                            else:
                                raise ValueError(f'undefined v_repr={v_repr}')

                    # The full id of this particular figure in tesnorboard, etc.
                    viz_fullid = f'{v_type}/{eid}/{vid}/{v_space}/{v_node}/{i_vsplt:02d}'

                    # Obtaining any existing figure and axes for plotting
                    e_fig, e_axes = vcfg.get('fig', None), vcfg.get('axes', None)
                    if e_fig is not None: 
                        clear_figaxes(e_fig, e_axes)

                    # Calling the matplotlib routines to draw the figure
                    e_fig, e_axes = plot_mpl(data=v_mpldata,
                        fig=e_fig, axes=e_axes, mplopts=v_mplopts)

                    # Saving the figure and axes objects for later re-use
                    assert e_fig is not None
                    assert e_axes is not None
                    vcfg['fig'] = e_fig
                    vcfg['axes'] = e_axes

                    # Saving the figure into the disk
                    if fig_dir is not None:
                        fig_name = viz_fullid.replace("_", "").replace("/", "_")
                        fig_name = f'{epoch:04d}_{fig_name}'
                        fig_path = f'{fig_dir}/{fig_name}.pdf'
                        e_fig.savefig(fig_path, dpi=144, bbox_inches="tight")

                    # Publishing the figure to tensorboard
                    if tbwriter is not None:
                        tbwriter.add_figure(viz_fullid, e_fig, epoch)
                elif v_type == 'bar':
                    # Example:
                    #      v_embdictsub = {
                    #          'train/orig/pnts':  np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
                    #          'train/rcnst/pnts': np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
                    #          'test/orig/pnts':   np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
                    #          'test/rcnst/pnts':  np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout),
                    #          ...}
                    for v_seedidx in v_seedidxs:
                        for v_lbl, v_ydatafull1 in v_embdictsub.items():
                            # Example: 
                            #   `v_lbl == 'train/orig/pnts'`
                            #   `v_ydatafull1 == np.randn(n_seeds, n_evlpnts, n_rcns, *shp_vizout)`
                            # Note that in this context, we have
                            #      shp_vizout == (n_chnls, n_len)
                            #      (n_chnls, n_len) == (n_chem, k_bins)
                            #      n_ax == n_evlpnts * n_rcns
                            # Therefore, we have
                            #      v_ydatafull.shape == (n_seeds, n_ax, n_chnls, n_len)
                            e_splt, e_vrnt, v_repr = v_lbl.split('/')

                            n_evlpnts, n_rcns, n_chnls, n_len = v_ydatafull1.shape[1:]
                            assert v_ydatafull1.shape == (n_seeds, n_evlpnts, n_rcns, n_chnls, n_len)

                            n_ax = n_rcns * n_evlpnts
                            v_ydatafull = np.moveaxis(v_ydatafull1, 1, 2).reshape(n_seeds, n_ax, n_chnls, n_len)
                            assert v_ydatafull.shape == (n_seeds, n_ax, n_chnls, n_len)

                            # We will only visualize the specified seeds
                            v_ydata = v_ydatafull[v_seedidx]
                            assert v_ydata.shape == (n_ax, n_chnls, n_len)

                            # The full id of this particular figure in tesnorboard, etc.
                            viz_fullid = f'{v_type}/{eid}/{vid}/{v_space}/{v_node}/{e_splt}/{e_vrnt}/{v_seedidx:02d}_seed/{i_vsplt:02d}'

                            # Obtaining any existing figure and axes for plotting
                            e_fig, e_axes = vcfg.get('fig', None), vcfg.get('axes', None)
                            if e_fig is not None: 
                                clear_figaxes(e_fig, e_axes)

                            if e_axes is not None:
                                assert e_axes.size == n_ax

                            # Over-riding the channel names from the dataset
                            chnlnames = vcfg['chnlnames']
                            chnlvars = {'chem_species': chem_species}
                            lblnames = eval_formula(chnlnames, chnlvars, catch=True)
                            assert len(lblnames) == n_chnls

                            # Compiling the plotting data
                            v_mpldata = dict()
                            for ax_idx in range(n_ax):
                                for i_lbl, lblname in enumerate(lblnames):
                                    v_mpldata[f'{ax_idx}:{lblname}:x'] = d_histbins
                                    v_mpldata[f'{ax_idx}:{lblname}:y'] = v_ydata[ax_idx, i_lbl]

                            # Calling the matplotlib routines to draw the figure
                            e_fig, e_axes = plot_mpl(data=v_mpldata,
                                fig=e_fig, axes=e_axes, mplopts=v_mplopts)

                            # Saving the figure and axes objects for later re-use
                            assert e_fig is not None
                            assert e_axes is not None
                            vcfg['fig'] = e_fig
                            vcfg['axes'] = e_axes

                            # Saving the figure into the disk
                            if fig_dir is not None:
                                fig_name = viz_fullid.replace("_", "").replace("/", "_")
                                fig_name = f'{epoch:04d}_{fig_name}'
                                fig_path = f'{fig_dir}/{fig_name}.pdf'
                                e_fig.savefig(fig_path, dpi=144, bbox_inches="tight")

                            # Publishing the figure to tensorboard
                            if tbwriter is not None:
                                tbwriter.add_figure(viz_fullid, e_fig, epoch)
                else:
                    raise ValueError(f'viz type == {v_type} is undefined')
                timer.toc()

class Timer:
    def __init__(self):
        self.tics = defaultdict(list)
        self.tocs = defaultdict(list)
        self.elapsed = defaultdict(list)
        self.last = None

    def tic(self, name):
        assert self.last is None, dedent(f'''
            Tic was pressed twice:
              * old tic was made for {self.last}
              * new tic was made for {name}''')
        self.last = name
        self.tics[name].append(time.perf_counter())
    
    def toc(self):
        name = self.last
        assert name is not None, 'Toc was over-pressed'
        self.tocs[name].append(time.perf_counter())
        dt = self.tocs[name][-1] - self.tics[name][-1]
        self.elapsed[name].append(dt)
        self.last = None

    def toctic(self, name):
        self.toc()
        self.tic(name)
    
    def get_rawdf(self):
        df_dict = defaultdict(list)
        for name, dt_list in self.elapsed.items():
            n_calls = len(dt_list)
            s_calls = sum(dt_list)
            df_dict['name'].append(name)
            df_dict['calls'].append(n_calls)
            df_dict['total'].append(s_calls)
            df_dict['mean'].append(s_calls / n_calls)
        df = pd.DataFrame(df_dict)
        return df
    
    def report(self, df=None, sep='/', tablevel=0, time_total=None, report_lst=None):
        df = self.get_rawdf() if df is None else df
        time_total = df['total'].sum() if time_total is None else time_total
        report_lst = list() if report_lst is None else report_lst
        phase_cat2subs = defaultdict(list)
        for phase in df['name'].tolist():
            phase_splt = phase.split(sep)
            phase_cat2subs[phase_splt[0]].append(phase)
        
        df = df.set_index('name')
        phase_cat2info = dict()
        for phase_ctgry, phase_subs in phase_cat2subs.items():
            phase_catnet = df.loc[phase_subs]['total'].sum()
            phase_catcalls = df.loc[phase_subs]['calls'].sum()
            phase_catmean = phase_catnet / phase_catcalls
            phase_cat2info[phase_ctgry] = (phase_catnet, phase_catcalls, phase_catmean, phase_subs)
        
        sort_key = lambda phase_ctgry: phase_cat2info[phase_ctgry][0]
        phase_catssrtd = sorted(phase_cat2info, key=sort_key, reverse=True)
        for phase_ctgry in phase_catssrtd:
            (phase_catnet, phase_catcalls, phase_catmean, phase_subs) = phase_cat2info[phase_ctgry]
            df_ctgry = df.loc[phase_subs]
            df_ctgry = df_ctgry.reset_index()
            df_ctgry['name'] = [sep.join(x.split(sep)[1:]) for x in df_ctgry['name']]
            sign = '+' if len(df_ctgry) > 1 else '-'
            phase_prnct = 100.0 * phase_catnet / time_total

            report_lst.append({'name': phase_ctgry, 'total': phase_catnet, 'percent': phase_prnct, 
                'calls': phase_catcalls, 'mean': phase_catmean, 'depth': tablevel, 'sign': sign})

            if len(df_ctgry) > 1:
                self.report(df_ctgry, sep, tablevel+1, time_total, report_lst)
        
        return pd.DataFrame(report_lst) if tablevel == 0 else None
    
    def print(self):
        df = self.report()
        for i_row, row in df.iterrows():
            depth = row['depth']
            sign = row['sign']
            name = row['name']
            total = row['total']
            percent = row['percent']
            calls = row['calls']
            mean = row['mean']
            msg = '    ' * depth + f'{sign} "{name}" took {total:.1f} seconds '
            msg += f'({percent:.2f}%, {calls} calls, {mean:.3f} s/call).'
            print(msg, flush=True)
    

# +
class DataPrepper:
    def __init__(self, n_seeds, rng_seed_list, io_avgfrq, chkpnt_period, iomon_period, 
                 device_name, dtype_name, dtnow, cfg_tree, cfg_name):

        assert iomon_period % io_avgfrq == 0, dedent(f'''
            "io/mon/frq" % "io/avg/frq" != 0''')
        assert chkpnt_period % io_avgfrq == 0, dedent(f'''
            "io/ckpt/frq" % "io/avg/frq" != 0''')
        assert len(rng_seed_list) == n_seeds

        self.n_seeds = n_seeds
        self.rng_seed_list = rng_seed_list
        self.io_avgfrq = io_avgfrq
        self.chkpnt_period = chkpnt_period
        self.iomon_period = iomon_period
        self.device_name = device_name
        self.dtype_name = dtype_name
        self.dtnow = dtnow
        self.cfg_tree = cfg_tree
        self.cfg_name = cfg_name
        self.trn_sttime = None
        self.avg_history = dict()
        self.last_avgstats = dict()
        self.static_dtups = []

    @staticmethod
    def make_hpetcopts(cfg_dict_input, hppats, etcpats, n_seeds, device_name, dtnow):
        #########################################################
        #### Collecting the Config Columns in the Dataframe #####
        #########################################################
        all_options = set(cfg_dict_input.keys())
        etcopts = [x for pat in etcpats for x in
                   fnmatch.filter(all_options, pat)]
        all_options -= set(etcopts)
        hpopts = [x for pat in hppats for x in
                  fnmatch.filter(all_options, pat)]
        all_options -= set(hpopts)
        assert len(all_options) == 0, dedent(f'''
            Some input config options were neither 
            designated as hpopts nor etcopts members:
                left out options: {list(all_options)}''')

        # Converting the list and tuples to strings
        hp_dict_ = dict()
        etc_dict_ = dict()
        for optn, val in cfg_dict_input.items():
            val = cfg_dict_input[optn]
            if isinstance(val, (int, float, str, bool, type(None))):
                srlval = val
            elif isinstance(val, (list, tuple)):
                srlval = repr(val)
            else:
                msg_  = f'Not sure how to log "{optn}" with '
                msg_ += f'a value type of "{type(val)}"'
                raise RuntimeError(msg_)

            if optn in hpopts:
                hp_dict_[optn] = srlval
            elif optn in etcopts:
                etc_dict_[optn] = srlval
            else:
                raise RuntimeError(f'Not sure how to log "{optn}"')

        # Few exceptions for the etc directory
        hostname = socket.gethostname()
        commit_hash = get_git_commit()
        if device_name.startswith('cuda'):
            tch_device = torch.device(device_name)
            tch_dvcmdl = torch.cuda.get_device_name(tch_device)
        else:
            tch_dvcmdl = device_name

        etc_dict_['hostname'] = hostname
        etc_dict_['commit'] = commit_hash
        etc_dict_['date/run'] = dtnow
        etc_dict_['io/dvc_mdl'] = tch_dvcmdl
        etc_dict_['date/cfg'] = etc_dict_.pop('date')
        etc_dict_.pop('io/results_dir')
        etc_dict_.pop('io/storage_dir')

        # Repeating the values by n_seeds
        hp_dict = dict()
        for optn, val in hp_dict_.items():
            hp_dict[optn] = [val] * n_seeds
        etc_dict = dict()
        for optn, val in etc_dict_.items():
            etc_dict[optn] = [val] * n_seeds

        return hp_dict, etc_dict

    @staticmethod
    def get_mondict(device_name, trn_sttime, n_seeds):
        s_rsrc = resource.getrusage(resource.RUSAGE_SELF)
        c_rsrc = resource.getrusage(resource.RUSAGE_CHILDREN)
        psmem = psutil.virtual_memory()
        pscpu = psutil.cpu_times()
        pscpuload = psutil.getloadavg()

        mon_dict = {'cpu/mem/tot': [psmem.total] * n_seeds,
                    'cpu/mem/avail': [psmem.available] * n_seeds,
                    'cpu/mem/used': [psmem.used] * n_seeds,
                    'cpu/mem/free': [psmem.free] * n_seeds,
                    'cpu/time/user/ps': [pscpu.user] * n_seeds,
                    'cpu/time/sys/ps': [pscpu.system] * n_seeds,
                    'cpu/time/idle/ps': [pscpu.idle] * n_seeds,
                    'cpu/load/1m': [pscpuload[0]] * n_seeds,
                    'cpu/load/5m': [pscpuload[1]] * n_seeds,
                    'cpu/load/15m': [pscpuload[2]] * n_seeds,
                    'cpu/time/train': [time.time() - trn_sttime] * n_seeds,
                    'cpu/time/sys/py':   [s_rsrc.ru_stime  + c_rsrc.ru_stime] * n_seeds,
                    'cpu/time/user/py':  [s_rsrc.ru_utime  + c_rsrc.ru_utime] * n_seeds,
                    'n_seeds': [n_seeds] * n_seeds}

        if 'cuda' in device_name:
            tch_device = torch.device(device_name)
            t_gpumem = torch.cuda.get_device_properties(tch_device).total_memory
            r_gpumem = torch.cuda.memory_reserved(tch_device)
            a_gpumem = torch.cuda.memory_allocated(tch_device)
            f_gpumem = r_gpumem - a_gpumem
            mon_dict.update({
                'gpu/mem/tot':   [t_gpumem] * n_seeds,
                'gpu/mem/res':   [r_gpumem] * n_seeds,
                'gpu/mem/alloc': [a_gpumem] * n_seeds,
                'gpu/mem/free':  [f_gpumem] * n_seeds})

        return mon_dict

    @staticmethod
    def get_timrrprt(df, n_seeds):
        cols = df.columns.tolist()
        assert 'name' in cols
        cols.remove('name')

        rprt_dct = dict()
        for i_row, row in df.iterrows():
            name = row['name']
            for col in cols:
                rprt_dct[f'{name}/{col}'] = [row[col]] * n_seeds
        
        return rprt_dct

    @staticmethod
    def get_np(value, name=None):
        if isinstance(value, np.ndarray):
            valnp = value
        elif torch.is_tensor(value):
            valnp = value.detach().cpu().numpy()
        else:
            raise ValueError(dedent(f'''
                Undefined data type for the item "{name}".
                It has a type of "{type(value)}"
            '''))
        return valnp

    @staticmethod
    def filter_vizstrg(strg_dict: dict, eparams: dict, vizcfgs: dict):
        """
        Removes the visualization data where the evaluation 
        options in the config specified `store` as `False`.

        Parameters
        ----------
        strg_dict (dict): The storage dictionary, where the 
            keys follow the following example format:

            Example:
            ```
            key = 'var/eval/mds/01_trn:01_xtsne:x:m_chmprthst/train/orig/pnts'
            key_prfxs = ('var', 'eval')
            v_dtype, e_split, e_vrnt = 'raw', 'train', 'orig'
            eid, vid, v_space, v_node = '01_trn', '01_xtsne', 'x', 'm_chmprthst'
            assert v_store in (True, False, 'raw', 'mds', ('raw',), 
                                ('mds',), ('raw', 'mds'))
            ```
        
        eparams (dict): The dictionary containing all the 
            evaluation profile configs.

        vizcfgs (dict): The dictionary containing all the 
            vizualization profile configs.
        """
        for key in list(strg_dict.keys()):
            #  Example:
            #     key = 'var/eval/mds/01_trn:01_xtsne:x:m_chmprthst/train/orig/pnts'
            #     key_prfxs = ('var', 'eval')
            #     v_dtype, e_split, e_vrnt, v_repr = 'raw', 'train', 'orig', 'pnts'
            #     eid, vid, v_space, v_node = '01_trn', '01_xtsne', 'x', 'm_chmprthst'
            #     assert v_store in (True, False, 'raw', 'mds', ('raw',), 
            #                        ('mds',), ('raw', 'mds'))

            *key_prfxs, v_dtype, key_mid, e_split, e_vrnt, v_repr = key.split('/')
            if v_dtype == 'mds':
                # The mds keys come out of the Evaluation.output method and 
                # have the following format: 
                #   key -> f'var/eval/mds/{eid}:{vid}:{v_space}:{v_node}/{e_split}/{e_vrnt}/{v_repr}'
                #
                #  Example:
                #     key = 'var/eval/mds/01_trn:01_xtsne:x:m_chmprthst/train/orig/pnts'
                #     key_mid = f'{eid}:{vid}:{v_space}:{v_node}'
                #     key_prfxs = ('var', 'eval')
                #     v_dtype, e_split, e_vrnt, v_repr = 'mds', 'train', 'orig', 'pnts'
                #     eid, vid, v_space, v_node = '01_trn', '01_xtsne', 'x', 'm_chmprthst'
                eid, vid, v_space, v_node = key_mid.split(':')
                space, node = v_space, v_node
                store_info = vizcfgs[vid]['store']
            elif v_dtype == 'raw':
                # The mds keys come out of the Evaluation.output method and 
                # have the following format: 
                #   key -> f'var/eval/raw/{eid}:{v_space}:{v_node}/{e_split}/{e_vrnt}/{v_repr}'
                #
                #  Example:
                #     key = 'var/eval/mds/01_trn:01_xtsne:x:m_chmprthst/train/orig/pnts'
                #     key_mid = f'{eid}:{v_space}:{v_node}'
                #     key_prfxs = ('var', 'eval')
                #     v_dtype, e_split, e_vrnt, v_repr = 'raw', 'train', 'orig', 'pnts'
                #     eid, e_space, e_node = '01_trn', 'x', 'm_chmprthst'
                eid, e_space, e_node = key_mid.split(':')
                space, node = e_space, e_node
                # Getting the evaluation profile config
                store_info = eparams[eid]['store']
            else:
                raise ValueError(f'undefined v_dtype = {v_dtype}')

            if isinstance(store_info, bool):
                do_store = store_info
            elif isinstance(store_info, dict):
                row_dict = dict(split=e_split, vrnt=e_vrnt, repr=v_repr, space=space, node=node)
                rd_fltrd = filter_rowdicts((row_dict,), **store_info)
                do_store = len(rd_fltrd) > 0
            else:
                raise ValueError(f'undefined store_info={store_info}')
            
            # Popping out the key from the storage
            if not do_store:
                strg_dict.pop(key)

    def push_static(self, hp_dict, etc_dict, cfg_dict_input, shapevars, pp_statics):
        pd_dicttype = 'pd.yml'
        n_seeds = self.n_seeds
        cfg_dict2 = {key: [val] * n_seeds for key, val in cfg_dict_input.items()}
        shapevars2 = {key: [val] * n_seeds for key, val in shapevars.items()}
        self.static_dtups += [
            ('hp',       hp_dict,      pd_dicttype),
            ('etc',      etc_dict,     pd_dicttype),
            ('cfg',      cfg_dict2,    pd_dicttype),
            ('shp',      shapevars2,    pd_dicttype)]

        # Adding the pre-processing stage
        for pp_id, pp_io in pp_statics.items():
            ppio_sd = pp_io.save()

            ppio_params = dict()
            ppio_sdp = ppio_sd.pop('params')
            # Manually broadcasting the shapes to (n_seeds,)
            for p_name, param_tch in ppio_sdp.items():
                assert torch.is_tensor(param_tch)
                param_np = param_tch.detach().cpu().numpy()
                if (param_np.ndim == 0) or (param_np.size == 1):
                    ppio_params[p_name] = np.repeat(param_np, n_seeds)
                elif param_np.shape[0] == n_seeds:
                    ppio_params[p_name] = param_np
                else:
                    raise ValueError(dedent(f'''
                        I am not sure how to deal with pp parameter shape:
                            param name = {p_name}
                            shape = {param_np.shape}'''))
                assert len(ppio_params[p_name]) == n_seeds, f'{p_name}: {ppio_params[p_name].shape}'

            self.static_dtups += [
                (f'pp/{pp_id}/params', ppio_params, 'np.arr')]

            for name in ('layers', 'hpdefs'):
                ppio_sdname = ppio_sd.pop(name)
                sd_dictname = {key: [val] * n_seeds for key, val in ppio_sdname.items()}
                self.static_dtups += [
                    (f'pp/{pp_id}/{name}', sd_dictname, pd_dicttype)]
            
            assert len(ppio_sd) == 0, f'unused keys: {ppio_sd.keys()}'

    def prep(self, strg_dict, stat_dict, nn_dict, timer, epoch):
        avg_history = self.avg_history
        last_avgstats = self.last_avgstats
        static_dtups = self.static_dtups
        n_seeds = self.n_seeds
        rng_seed_list = self.rng_seed_list
        io_avgfrq = self.io_avgfrq
        iomon_period = self.iomon_period
        device_name = self.device_name
        chkpnt_period = self.chkpnt_period
        trn_sttime = self.trn_sttime

        dtups = []
        if epoch % io_avgfrq == 0:
            dtups += [(key, rows_dict if epoch == 0 else None, col_t) 
                for (key, rows_dict, col_t) in static_dtups]

        # pushing the results to the data writer
        statdict_hie = deep2hie(stat_dict, dict)
        for stat_name, stat_vals in statdict_hie.items():
            avg_history.setdefault(stat_name, [])
            stat_valsnp = self.get_np(stat_vals, stat_name)
            assert stat_valsnp.shape[0] == n_seeds
            avg_history[stat_name].append(stat_valsnp)

        if epoch % io_avgfrq == 0:
            avg_statlst  = [('epoch',       [epoch] * n_seeds),
                            ('rng_seed',    rng_seed_list)]
            avg_statlst += [(name, np.stack(svl, axis=0).mean(axis=0).tolist())
                            for name, svl in avg_history.items()]
            
            # Defaulting missing keys to the last average values
            avg_statdict = dict(last_avgstats)
            avg_statdict.update(dict(avg_statlst))
            
            # Updating the latest average stats
            last_avgstats.update(avg_statdict)
            
            # Pushing the data out
            dtups += [('stat',  avg_statdict, 'pd.qnt')]

            # Clearing the averaging history queue
            avg_history.clear()

        # pusshing the storage items to the data writer
        strg_dicthie = deep2hie(strg_dict)
        for strg_name, strg_arr in strg_dicthie.items():
            assert epoch % io_avgfrq == 0
            strg_np = self.get_np(strg_arr, strg_name)
            dtups += [(strg_name, {'data': strg_np}, 'np.arr')]

        # monitoring the resource utilization
        if epoch % iomon_period == 0:
            assert epoch % io_avgfrq == 0
            mon_dict = self.get_mondict(device_name, trn_sttime, n_seeds)
            dtups += [('mon', mon_dict, 'pd.qnt')]

            df_timer = timer.report()
            rprt_dct = self.get_timrrprt(df_timer, n_seeds)
            dtups += [('timer', rprt_dct, 'pd.qnt')]

        # checkpointing the neural parameters
        with torch.no_grad():
            if epoch % chkpnt_period == 0:
                assert epoch % io_avgfrq == 0
                for mdl_name, mdl_nn in nn_dict.items():
                    mdl_sdnp = dict()
                    for p_name, param in mdl_nn.state_dict().items():
                        p_shape = param.shape
                        n_pseeds = p_shape[0] if len(p_shape) > 0 else None
                        if (param.numel() == 1) and p_name.endswith('.num_batches_tracked'):
                            p_valid = param.reshape(1).expand(n_seeds)
                        elif (n_pseeds is not None) and (n_pseeds == n_seeds):
                            p_valid = param
                        elif ('.convs.' in p_name) or ('.norms.' in p_name) or ('.bn_layer.' in p_name):
                            assert n_pseeds is not None
                            assert n_pseeds % n_seeds == 0
                            # Note: I have verified that CNN layers do not require any transpositions 
                            # between the `n_seeds` and `n_inplay` dimensions.
                            n_inplay = n_pseeds // n_seeds
                            p_rshape = (n_seeds, n_inplay, *p_shape[1:])
                            p_valid = param.reshape(*p_rshape)
                        else:
                            raise ValueError(dedent(f'''
                                checkpointing parameter name 
                                    "{p_name}"
                                has an odd shape:
                                    "{p_shape}"
                                I dont have a rule to make it (n_seeds,*)-compatible.
                            '''))
                        mdl_sdnp[p_name] = p_valid.detach().cpu().numpy()
                    dtups += [(mdl_name,   mdl_sdnp, 'np.arr')]

        return dtups

# +

# TODO: Clean this up

def get_attrshiedict(obj):
    obj_attrs = deep2hie(vars(obj))
    nnprmdctkeys = [key for key, val in obj_attrs.items() 
        if isinstance(val, nn.ParameterDict)]
    for key in nnprmdctkeys:
        val = obj_attrs.pop(key)
        obj_attrs.update(deep2hie(dict(val)))
    return obj_attrs

def check_deepeq(obj1, obj2):
    obj_attrs1 = get_attrshiedict(obj1)
    obj_attrs2 = get_attrshiedict(obj2)

    obj_attrsset1 = set(obj_attrs1)
    obj_attrsset2 = set(obj_attrs2)
    assert obj_attrsset1 == obj_attrsset2, dedent(f'''
        There are missing/extra object attributes:
            missing attributes: {obj_attrsset1 - obj_attrsset2}
            extra attributes: {obj_attrsset2 - obj_attrsset1}''')

    for key, val1 in obj_attrs1.items():
        val2 = obj_attrs2[key]
        val_eq = val1 == val2
        if hasattr(val_eq, 'all'):
            val_eq = val_eq.all()
        assert val_eq, dedent(f'''
            Non-equal attributes were found:
                attribute name: {key}
                value 1: {val1}
                value 2: {val2}''')

def load_riopp(rio, device_name, dtype_name, tch_device, tch_dtype, tch_cdtype, 
    pp_type='enc', method_loadpp='make', check_mthdeq=False, seed_idx=None):
    """
    Loads a static PP from an HDF file using the result io (rio) interface.

    Args:
        rio: Result IO object
        device_name: Device name
        dtype_name: Data type name
        tch_device: torch.device object
        tch_dtype: torch dtype
        tch_cdtype: torch complex dtype
        pp_type: Preprocessor type - 'enc', 'dec', or 'cri' (default: 'enc')
        method_loadpp: Loading method - 'make' or 'init' (default: 'make')
        check_mthdeq: Check method equality (default: False)
        seed_idx: If provided, load only this seed index (extracts single seed parameters)

    Returns:
        dict: Dictionary containing 'pp' (preprocessor) and 'rng_seed_list'
    """
    cfg_dictld1 = rio.load_key('cfg')
    rng_seed_listld = cfg_dictld1['rng_seed/list'][0]
    n_seedsld = len(rng_seed_listld)
    for key, val in cfg_dictld1.items():
        assert isinstance(val, list) and len(val) == n_seedsld
    cfg_dictld2 = {key: val[0] for key, val in cfg_dictld1.items()}

    shapevars_ld1 = rio.load_key('shp')
    for key, val in shapevars_ld1.items():
        assert isinstance(val, list) and len(val) == n_seedsld
    shapevars_ld2 = {key: val[0] for key, val in shapevars_ld1.items()}
    shapevars_ld3 = dict(shapevars_ld2)
    for key, val in [('dtype', dtype_name), ('device', device_name), ('cdtype', None)]:
        if key in shapevars_ld3:
            shapevars_ld3[key] = val

    rio_dtypes = rio.dtypes(full=False)
    assert pp_type in ('enc', 'dec', 'cri'), dedent(f'''
        pp_type value "{pp_type}" is invalid:
            must be either 'enc', 'dec', or 'cri'.''')
    assert method_loadpp in ('make', 'init'), dedent(f'''
        method_loadpp value "{method_loadpp}" is invalid:
            must be either 'make' or 'init'.''')
    assert (seed_idx is None) or (0 <= seed_idx < n_seedsld), dedent(f'''
        seed_idx value {seed_idx} is out of bounds:
            n_seedsld = {n_seedsld}''')

    nnpp_paramsld1 = dict()
    for key, dtype in get_subdict(rio_dtypes, f'pp/mdl/{pp_type}/params').items():
        val = rio.load_key(f'pp/mdl/{pp_type}/params/{key}', dtype=dtype)[:]
        assert val.shape[0] == n_seedsld
        # Extract single seed if requested
        if seed_idx is not None:
            val2 = val[seed_idx:seed_idx+1]
        else:
            # TODO: Fix this "expand" issue at storage and loading.
            # The PP should be able to handle non-scalar `(n_seeds,)` add parameters.
            val2 = val[0] if (val == val[0]).all() else val
        nnpp_paramsld1[key] = val2

    nnpp_ld = None
    if (method_loadpp == 'make') or check_mthdeq:
        # Loading the main elements of the nn pp
        nnpp_hpdefsld1 = rio.load_key(f'pp/mdl/{pp_type}/hpdefs')
        nnpp_layersld1 = rio.load_key(f'pp/mdl/{pp_type}/layers')

        # Taking the first element
        for key, val in nnpp_hpdefsld1.items():
            assert isinstance(val, list) and (len(val) == n_seedsld)
        nnpp_hpdefsld2 = {key: val[0] for key, val in nnpp_hpdefsld1.items()}

        for key, val in nnpp_layersld1.items():
            assert isinstance(val, list) and (len(val) == n_seedsld)
        nnpp_layersld2 = {key: val[0] for key, val in nnpp_layersld1.items()}

        nnpp_stateld = {
            'hpdefs': nnpp_hpdefsld2,
            'layers': nnpp_layersld2,
            'params': nnpp_paramsld1}

        nnpp_ld1 = PreProcessor.make(nnpp_stateld, device=tch_device, dtype=tch_dtype, cdtype=tch_cdtype)
        nnpp_ld = nnpp_ld1 if method_loadpp == 'make' else nnpp_ld

    if (method_loadpp == 'init') or check_mthdeq:
        # Determine config key based on pp_type
        pp_cfgkey = {'cri': 'cri/rcnst/pp', 'enc': 'nn/pp/enc', 'dec': 'nn/pp/dec'}[pp_type]

        ppnn_hparamsrawld = get_subdict(cfg_dictld2, pp_cfgkey)
        ppnn_typeld = ppnn_hparamsrawld.pop('type')
        pp_mdlscfgld = get_subdict(cfg_dictld2, 'pp/modules')

        ppnn_hparamsld = {hparam: eval_formula(val, shapevars_ld3, catch=True)
            for hparam, val in ppnn_hparamsrawld.items()}
        nnpp_ld2 = make_pp(ppnn_typeld, ppnn_hparamsld, pp_mdlscfgld,
            tch_device, tch_dtype, tch_cdtype)
        nnpp_ld2.load_params(nnpp_paramsld1, inplace=True)
        nnpp_ld = nnpp_ld2 if method_loadpp == 'make' else nnpp_ld

    if check_mthdeq:
        check_deepeq(nnpp_ld1, nnpp_ld2)

    assert nnpp_ld is not None

    # Adjust rng_seed_list for output
    rng_seed_listld2 = rng_seed_listld if seed_idx is None else [rng_seed_listld[seed_idx]]

    outdict = {
        'pp': nnpp_ld,
        'rng_seed_list': rng_seed_listld2}

    return outdict

def load_riomodel(rio, model_type, device_name, dtype_name, tch_device, tch_dtype, seed_idx=None):
    """
    Generalized function to load encoder or decoder from HDF file using the result io (rio) interface.

    Args:
        rio: Result IO object
        model_type: Either 'enc' or 'dec'
        device_name: Device name (e.g., 'cuda:0', 'cpu')
        dtype_name: Data type name (e.g., 'float32', 'float64')
        tch_device: torch.device object
        tch_dtype: torch dtype
        seed_idx: If provided, load only this seed index (creates model with n_seeds=1)

    Returns:
        dict: Dictionary containing:
            - 'model': The loaded model (encoder or decoder)
            - 'ltnt_dim': Latent dimension
            - 'ltnt_sigtnsfm': Latent sigma transformation
            - 'rng_seed_list': List of RNG seeds used
    """
    assert model_type in ('enc', 'dec'), f"model_type must be 'enc' or 'dec', got {model_type}"

    cfg_dictld1 = rio.load_key('cfg')
    rng_seed_listld = cfg_dictld1['rng_seed/list'][0]
    n_seedsld = len(rng_seed_listld)

    assert (seed_idx is None) or (0 <= seed_idx < n_seedsld), dedent(f'''
        seed_idx value {seed_idx} is out of bounds:
            n_seedsld = {n_seedsld}''')

    for key, val in cfg_dictld1.items():
        assert isinstance(val, list) and len(val) == n_seedsld
    cfg_dictld2 = {key: val[0] for key, val in cfg_dictld1.items()}

    shapevars_ld1 = rio.load_key('shp')
    for key, val in shapevars_ld1.items():
        assert isinstance(val, list) and len(val) == n_seedsld
    shapevars_ld2 = {key: val[0] for key, val in shapevars_ld1.items()}
    shapevars_ld3 = dict(shapevars_ld2)
    for key, val in [('dtype', dtype_name), ('device', device_name), ('cdtype', None)]:
        if key in shapevars_ld3:
            shapevars_ld3[key] = val

    rio_dtypes = rio.dtypes(full=False)

    model_hparamsrawld = get_subdict(cfg_dictld2, f'nn/{model_type}')
    model_typeld = model_hparamsrawld.pop('type')
    nn_mdlscfgld = get_subdict(cfg_dictld2, 'nn/modules')

    # Initializing the model
    model_hparamsld = dict()
    for hparam, val in model_hparamsrawld.items():
        model_hparamsld[hparam] = eval_formula(val, shapevars_ld3, catch=True)

    # Determine n_seeds for model instantiation
    n_seedsmdl = n_seedsld if seed_idx is None else 1
    rng_seedsld = np.array(rng_seed_listld if seed_idx is None else [rng_seed_listld[seed_idx]])

    # Making a dummy rng for initializing the neural model upon construction
    rngld = BatchRNG(shape=(n_seedsmdl,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0, norm_cache_cols=0)
    rngld.seed(np.broadcast_to(rng_seedsld, rngld.shape))

    # Constructing the neural network with potentially overridden n_seeds
    model_ld = make_nn(name=model_typeld, hparams=model_hparamsld, config=nn_mdlscfgld,
        n_seeds=n_seedsmdl, rng=rngld, device=tch_device, dtype=tch_dtype)

    # Load parameters from HDF5
    model_paramsnpld = dict()
    for key, dtype in get_subdict(rio_dtypes, f'mdl/{model_type}').items():
        h5grp = rio.load_key(f'mdl/{model_type}/{key}', dtype=dtype)
        assert h5grp.shape[0] % n_seedsld == 0
        # Get the last n_seedsld parameters
        params_all = h5grp[-n_seedsld:]
        # Extract single seed if requested
        if seed_idx is not None:
            model_paramsnpld[key] = params_all[seed_idx:seed_idx+1]
        else:
            model_paramsnpld[key] = params_all

    # Converting the nn state dict to torch tensors
    model_paramsld = {key: torch.from_numpy(val) for key, val in model_paramsnpld.items()}

    # Loading the state dict into the network
    model_ld.load_state_dict(model_paramsld)

    # Extracting the VAE properties (for both encoder and decoder)
    ltnt_dimrawld = cfg_dictld2['nn/ltnt/dim']
    ltnt_dimld = eval_formula(ltnt_dimrawld, shapevars_ld3, catch=True)
    ltnt_sigtnsfmld = cfg_dictld2.get('nn/ltnt/sig/tnsfm', 'exp')

    # Adjust rng_seed_list for output
    rng_seed_listld2 = rng_seed_listld if seed_idx is None else [rng_seed_listld[seed_idx]]

    outdict = {
        'model': model_ld,
        'ltnt_dim': ltnt_dimld,
        'ltnt_sigtnsfm': ltnt_sigtnsfmld,
        'rng_seed_list': rng_seed_listld2}

    return outdict

class VAELabelNN(nn.Module):
    def __init__(self, encoder, ltnt_dim, ltnt_sigtnsfm, y_node):
        super().__init__()
        self.encoder = encoder
        self.ltnt_dim = ltnt_dim
        self.ltnt_sigtnsfm = ltnt_sigtnsfm
        self.y_node = y_node
        self.z_dims = None
    
    def forward(self, *args, **kwargs):
        y_node = self.y_node
        z_dims = self.z_dims
        ltnt_dim = self.ltnt_dim
        ltnt_sigtnsfm = self.ltnt_sigtnsfm
        rng = kwargs.pop('rng', None)
        n_mb = kwargs.pop('n_mb', None)
        assert n_mb is not None, 'Need to know n_mb for shape checkings.'
        assert isinstance(n_mb, int), f'invalid n_mb = {n_mb}'
        assert n_mb >= -1, f'invalid n_mb = {n_mb}'
        n_seeds = kwargs.pop('n_seeds', None)
        assert n_seeds is not None, 'Need to know n_seeds for shape checkings.'
        sigscale = kwargs.pop('sigscale', None)
        assert isinstance(sigscale, (int, float)), f'Only scalar sigscale is acceptable: {sigscale}'

        # Encoding the pre-processed input
        ymusig_mbs = self.encoder(*args, **kwargs)
        assert isinstance(ymusig_mbs, dict), dedent(f'''
            The encoder should output a dictionary: {ymusig_mbs}''')
        
        # Getting the mu values out of the encoder
        ymu_mbs = get_subdict(ymusig_mbs, 'mu', sep='.', pop=True)
        z_dims = {z_node: tuple(mu_mb.shape[2:]) for z_node, mu_mb 
            in ymu_mbs.items()} if (z_dims is None) else z_dims
        assert set(ymu_mbs) == set(z_dims), dedent(f'''
            mu nodes are different from the usual:
                extra nodes: {set(ymu_mbs) - set(z_dims)}
                missing nodes: {set(z_dims) - set(ymu_mbs)}''')
        for z_node, z_dim in z_dims.items():
            ymu_mb = ymu_mbs[z_node]
            n_mb = ymu_mb.shape[1] if (n_mb == -1) else n_mb
            assert ymu_mb.shape == (n_seeds, n_mb, *z_dim)

        # Getting the log-sigma values out of the encoder
        ylogsigtld_mbs = get_subdict(ymusig_mbs, 'sig', sep='.', pop=True)
        assert set(ylogsigtld_mbs) == set(z_dims), dedent(f'''
            sigma nodes are different from the usual:
                extra nodes: {set(ylogsigtld_mbs) - set(z_dims)}
                missing nodes: {set(z_dims) - set(ylogsigtld_mbs)}''')
        for z_node, z_dim in z_dims.items():
            ylogsigtld_mb = ylogsigtld_mbs[z_node]
            assert ylogsigtld_mb.shape == (n_seeds, n_mb, *z_dim)
        
        # Making sure the encoder naming convention is followed
        assert len(ymusig_mbs) == 0, dedent(f'''
            The encoder output keys should be either `mu.*` or `sig.*`:
                offending items: {ymusig_mbs}''')

        # Sanity checking the total latent dimensionality
        assert sum(math.prod(z_dim) for z_dim in z_dims.values()) == ltnt_dim, dedent(f'''
            The total number of latent dimensions does not match:
                actual dims: {sum(math.prod(z_dim) for z_dim in z_dims.values())}
                expected dims: {ltnt_dim}''')

        if sigscale == 0:
            y_mbs = dict()
            for z_node, z_dim in z_dims.items():
                ymu_mb = ymu_mbs[z_node]
                assert ymu_mb.shape == (n_seeds, n_mb, *z_dim)
                y_mb = ymu_mb.unsqueeze(-2)
                assert y_mb.shape == (n_seeds, n_mb, 1, *z_dim)
                y_mbs[z_node] = y_mb
        else:
            assert rng is not None, 'Need an RNG for sampling a vae label nn.'

            # Applying the latent sigma transformation
            ylogsigma_mbs = dict()
            for z_node, z_dim in z_dims.items():
                if ltnt_sigtnsfm == 'exp':
                    ylogsigma_mb = ylogsigtld_mbs[z_node]
                    assert ylogsigma_mb.shape == (n_seeds, n_mb, *z_dim)
                elif ltnt_sigtnsfm == 'explin':
                    ylogsigtlde_mb = ylogsigtld_mbs[z_node]
                    assert ylogsigtlde_mb.shape == (n_seeds, n_mb, *z_dim)

                    ylogsigma_mb = torch.where(ylogsigtlde_mb < 0, ylogsigtlde_mb, 
                        (ylogsigtlde_mb + 1).log())
                    assert ylogsigma_mb.shape == (n_seeds, n_mb, *z_dim)
                else:
                    raise ValueError(f'undefined ltnt_sigtnsfm={ltnt_sigtnsfm}')
                ylogsigma_mbs[z_node] = ylogsigma_mb

            y_mbs = dict()
            for z_node, z_dim in z_dims.items():
                ymu_mb = ymu_mbs[z_node]
                assert ymu_mb.shape == (n_seeds, n_mb, *z_dim)
                ylogsigma_mb = ylogsigma_mbs[z_node]
                assert ylogsigma_mb.shape == (n_seeds, n_mb, *z_dim)
                yeps_mb = rng.randn(n_seeds, n_mb, *z_dim)
                assert yeps_mb.shape == (n_seeds, n_mb, *z_dim)
                y_mb_ = ymu_mb + sigscale * yeps_mb * ylogsigma_mb.exp()
                assert y_mb_.shape == (n_seeds, n_mb, *z_dim)
                y_mb = y_mb_.unsqueeze(-2)
                assert y_mb.shape == (n_seeds, n_mb, 1, *z_dim)
                y_mbs[z_node] = y_mb
        
        self.z_dims = z_dims if (self.z_dims is None) else self.z_dims

        return y_mbs

class IIDLossMaker:
    def __init__(self, n_hist, mtrc_cfg, mrng, device, dtype):
        self.n_hist = n_hist
        self.device = device
        self.dtype = dtype
        self.mtrc_cfg = dict(mtrc_cfg)
        self.mtrc_type = self.mtrc_cfg.pop('type', None)
        self.mrng = mrng
        
        self.z_data = None
        self.z_idxs = None
        self.z_dim = None

        self.y_data = None
        self.y_idxs = None
        self.y_dim = None

        self.is_filled = False
        self.n_mb = None
        self.n_seeds = None
        self.i_mb = 0
        self.n_calcs = 0

    @staticmethod
    @torch.no_grad()
    def init_data(u_mbs, n_seeds, n_mb, n_hist, device, dtype):
        # Initializing the first index
        i_ustrt = 0

        # A mapping of each 
        u_idxs = odict()
        for u_node, u_mb in u_mbs.items():
            # The spatial dimensions of u
            u_dims = u_mb.shape[2:]
            # The total number of data dimensions in u
            d_unode = math.prod(u_dims)
            # Making sure the shape matches our expectation
            assert u_mb.shape == (n_seeds, n_mb, *u_dims), dedent(f'''
                (n_seeds, n_mb, *dims) shape form mismatch:
                    node: {u_node}
                    actual shape: {u_mb.shape}
                    n_seeds: {n_seeds}
                    n_''')
            # The end index for the node in the concatenated data
            i_uend = i_ustrt + d_unode
            # Storing the start and 
            u_idxs[u_node] = (i_ustrt, i_uend, u_dims, d_unode)
            # the start index for the next node
            i_ustrt = i_uend
        
        # The total number of u dimensions should be `i_ustrt` by now
        u_dim = sum(math.prod(u_mb.shape[2:]) for u_node, u_mb in u_mbs.items())
        assert u_dim == i_ustrt
        
        # Making sure the idxs covert the entire data dimensions
        all_uidxs = []
        for u_node, (i_ustrt, i_uend, u_dims, d_unode) in u_idxs.items():
            all_uidxs += list(range(i_ustrt, i_uend))
        assert all_uidxs == list(range(u_dim))

        # Initializing the data buffer
        u_data = torch.full((n_seeds, n_hist, u_dim), torch.nan, device=device, dtype=dtype)
        assert u_data.shape == (n_seeds, n_hist, u_dim)

        return u_idxs, u_dim, u_data

    @staticmethod
    @torch.no_grad()
    def push_data(u_mbs, i_mb, u_data, u_idxs, u_dim, n_seeds, n_mb, n_hist):
        # TODO: use a `tensor.slice_scatter_` implementation to relieve 
        # the `n_hist % n_mb == 0` restriction.
        if n_hist == 0:
            return

        for u_node, (i_ustrt, i_uend, u_dims, d_unode) in u_idxs.items():
            u_mb = u_mbs[u_node]
            assert u_mb.shape == (n_seeds, n_mb, *u_dims)
            
            u_mb2 = u_mb.flatten(2, -1)
            assert u_mb2.shape == (n_seeds, n_mb, d_unode)
            
            i_mb2 = i_mb + n_mb
            assert (i_mb2 <= n_hist) and (i_ustrt <= i_uend <= u_dim), dedent(f'''
                something bad happened:
                    node: {u_node}
                    shape: {u_shape}
                    n_mb: {n_mb}
                    flat dim: {d_unode}
                    node dim start index: {i_ustrt}
                    node dim end index: {i_uend}
                    buffer data dim: {u_dim}
                    data insertion start index: {i_mb}
                    data insertion end index: {i_mb2}
                    n_hist = {n_hist}''')
            
            assert u_data.shape == (n_seeds, n_hist, u_dim)
            u_data[:, i_mb: i_mb2, i_ustrt: i_uend] = u_mb2
    
    @staticmethod
    def cat_data(u_mbs, u_idxs, u_dim, n_seeds, n_mb):
        u_catlst = []
        for u_node, (i_ustrt, i_uend, u_dims, d_unode) in u_idxs.items():
            u_mb = u_mbs[u_node]
            assert u_mb.shape == (n_seeds, n_mb, *u_dims)
            
            u_mb2 = u_mb.flatten(2, -1)
            assert u_mb2.shape == (n_seeds, n_mb, d_unode)

            u_catlst.append(u_mb2)
        
        u_mbcat = torch.cat(u_catlst, dim=-1)
        assert u_mbcat.shape == (n_seeds, n_mb, u_dim)
        
        return u_mbcat

    @staticmethod
    def get_slcdirs(src_pnts, mtrc_cfg, n_seeds, n_slc, n_mbsrc, dim_src, mrng, pop=False):
        mtrc_cfg = mtrc_cfg.copy() if not pop else mtrc_cfg
        assert src_pnts.shape == (n_seeds, n_mbsrc, dim_src)

        # The distribution of slicing initial random variables
        dstr_slc = mtrc_cfg.pop('slc/dstr')
        assert dstr_slc == 'normal', dedent(f'''
            Only sampling random slices from a normal 
            distribution is supported for now:
                slc/dstr = {dstr_slc}''')

        # The type of slicing covariance matrix
        type_slccov = mtrc_cfg.pop('slc/cov/type')
        
        if type_slccov in ('src',):
            # The approximations for the slice sampling covariance
            aprx_slccov = mtrc_cfg.pop('slc/cov/approx')

            # We will use the source/target data's singular values to sample the 
            # slicing epsilons.
            if type_slccov == 'src':
                cov_pntsfull, n_mbcovdatafull = src_pnts, n_mbsrc
            else:
                raise ValueError(f'undefined type_slccov={type_slccov}')
            assert cov_pntsfull.shape == (n_seeds, n_mbcovdatafull, dim_src)

            # The subsampling size of the covariance matrix data. 
            # This is only used for estimating the slicing directions and 
            # not computing the sliced wasserstein distance values.
            n_covsubsamp = mtrc_cfg.pop('slc/cov/nsamp', None)
            if (n_covsubsamp is None) or (n_covsubsamp >= n_mbcovdatafull):
                cov_pnts, n_mbcovdata = cov_pntsfull, n_mbcovdatafull
            else:
                n_mbcovdata = n_covsubsamp
                
                i_covsubsamp = mrng.rand(n_seeds, n_mbcovdatafull, 1).argsort(dim=-1)[:, :n_covsubsamp]
                assert i_covsubsamp.shape == (n_seeds, n_covsubsamp, 1)

                cov_pnts = torch.take_along_dim(cov_pntsfull, i_covsubsamp, dim=-2)
                assert cov_pnts.shape == (n_seeds, n_mbcovdata, dim_src)
        else:
            aprx_slccov, cov_pnts, n_mbcovdata = None, None, None

        if (type_slccov == 'src') and (aprx_slccov == 'pcalr'):
            # The covariance exponent, i.e., the singular values exponent. Should default to 1.
            exp_slccov = mtrc_cfg.pop('slc/cov/exp')
            assert isinstance(exp_slccov, (int, float))

            # With a non-unit covariance exponent, we have to use a low-rank pca.
            slc_covpcalrniter = mtrc_cfg.pop('slc/cov/pcalr/niter')
            slc_covpcalrq = mtrc_cfg.pop('slc/cov/pcalr/q')

            assert not cov_pnts.isnan().any(), f'{cov_pnts.isnan().sum().item()} elementas are NaN.'
            slc_covpcalrq2 = slc_covpcalrq if slc_covpcalrq < min(cov_pnts.shape[-2:]) else min(cov_pnts.shape[-2:])
            u_slccov, s_slccov, v_slccov = torch.pca_lowrank(cov_pnts, q=slc_covpcalrq2, 
                niter=slc_covpcalrniter)
            q_slccov = s_slccov.shape[-1]
            assert u_slccov.shape == (n_seeds, n_mbcovdata, q_slccov)
            assert s_slccov.shape == (n_seeds, q_slccov)
            assert v_slccov.shape == (n_seeds, dim_src, q_slccov)
        elif (type_slccov == 'src') and (aprx_slccov == 'exact'):
            # The covariance exponent, i.e., the singular values exponent. Should default to 1.
            exp_slccov = mtrc_cfg.pop('slc/cov/exp', 1)
            assert isinstance(exp_slccov, int), 'only int slc/cov/exp is supported for x'
            assert exp_slccov >= 0, f'only non-negative slc/cov/exp supported -> {exp_slccov}'
            exp_slccov, u_slccov, s_slccov, v_slccov = exp_slccov, None, None, None
        elif (type_slccov == 'src'):
            raise ValueError(f'undefined slc/cov/approx = {aprx_slccov}')
        else:
            exp_slccov, u_slccov, s_slccov, v_slccov = None, None, None, None

        #################### Step 1: Sampling the Slicing Random Variables ####################
        if type_slccov == 'identity':
            # We will use the identity covariance matrix to sample the slicing epsilons.
            slc_epsilons = mrng.randn(n_seeds, n_slc, dim_src)
            assert slc_epsilons.shape == (n_seeds, n_slc, dim_src)
        elif (type_slccov == 'src') and (aprx_slccov == 'exact'):
            # Since the slicing covariance exponent is 1, we don't need to perform a PCA on 
            # the src/trg data to find the scalings; A simple data matrix multiplication will 
            # do the trick in this case.
            cov_pntsnomean = cov_pnts - cov_pnts.mean(dim=-2, keepdims=True)
            assert cov_pntsnomean.shape == (n_seeds, n_mbcovdata, dim_src)

            # Here is the idea:
            #   if exp_slccov == 1:
            #     \eps_{n_slc \times dim_src} = \mathcal{N}_{n_slc \times n_mb} @ X_{n_mb \times dim_src}
            #     E[\eps^T \eps] = (X^T @ X)
            #   if exp_slccov == 2:
            #     \eps_{n_slc \times dim_src} = \mathcal{N}_{n_slc \times dim_src} @ (X^T @ X)
            #     E[\eps^T \eps] =  (X^T @ X) @ (X^T @ X)
            #   if exp_slccov == 3:
            #      \eps_{n_slc \times dim_src} = \mathcal{N}_{n_slc \times n_mb} @ X_{n_mb \times dim_src} @ (X^T @ X)
            #      E[\eps^T \eps] = (X^T @ X) @ (X^T @ X) @ (X^T @ X)
            #   if exp_slccov == 4:
            #     \eps_{n_slc \times dim_src} = \mathcal{N}_{n_slc \times dim_src} @ (X^T @ X) @ (X^T @ X)
            #     E[\eps^T \eps] =  (X^T @ X) @ (X^T @ X) @ (X^T @ X) @ (X^T @ X)
            #   ...

            assert exp_slccov >= 0
            # The number of data covariance multiplications
            n_covmults = exp_slccov // 2

            cov_nomean = None
            if n_covmults > 0:
                # `cov_nomean` is basically `(X^T @ X)`.
                cov_nomean = torch.einsum('sij,sik->sjk', cov_pntsnomean, cov_pntsnomean)
                assert cov_nomean.shape == (n_seeds, dim_src, dim_src)

            # For efficiency reasons, it may be better to multiply cov by itself 
            # before mutliplying it by the samples.
            cov_nomeanpow = None
            if (n_covmults > 0) and (dim_src <= n_slc):
                # Multiplying for as many times as needed.
                cov_nomeanpow = cov_nomean
                assert cov_nomeanpow.shape == (n_seeds, dim_src, dim_src)

                for i_covmult in range(1, n_covmults):
                    cov_nomeanpow = cov_nomeanpow.matmul(cov_nomean)
                    assert cov_nomeanpow.shape == (n_seeds, dim_src, dim_src)

            if (exp_slccov % 2) == 1:
                slc_randns = mrng.randn(n_seeds, n_slc, n_mbcovdata)
                assert slc_randns.shape == (n_seeds, n_slc, n_mbcovdata)

                slc_epsilons = slc_randns.matmul(cov_pntsnomean)
                assert slc_epsilons.shape == (n_seeds, n_slc, dim_src)
            elif (exp_slccov % 2) == 0:
                slc_epsilons = mrng.randn(n_seeds, n_slc, dim_src)
                assert slc_epsilons.shape == (n_seeds, n_slc, dim_src)
            else:
                raise ValueError(f'undefined slc/cov/exp={exp_slccov}')
            
            if (n_covmults > 0) and (dim_src <= n_slc):
                slc_epsilons = slc_epsilons.matmul(cov_nomeanpow)
                assert slc_epsilons.shape == (n_seeds, n_slc, dim_src)
            elif (n_covmults > 0) and (dim_src > n_slc):
                for i_covmult in range(n_covmults):
                    slc_epsilons = slc_epsilons.matmul(cov_nomean)
                    assert slc_epsilons.shape == (n_seeds, n_slc, dim_src)
            elif (n_covmults > 0):
                raise ValueError('case should have never happened')
        elif (type_slccov == 'src') and (aprx_slccov == 'pcalr'):
            # With a non-unit covariance exponent, we have to use the low-rank pca.
            slc_randns = mrng.randn(n_seeds, n_slc, q_slccov)
            assert slc_randns.shape == (n_seeds, n_slc, q_slccov)

            stilde_slccov = (1e-30 + s_slccov.unsqueeze(-2)).pow(exp_slccov) / math.sqrt(n_mbcovdata)
            assert stilde_slccov.shape == (n_seeds, 1, q_slccov)

            slc_randnsscld = slc_randns * stilde_slccov
            assert slc_randnsscld.shape == (n_seeds, n_slc, q_slccov)

            slc_epsilons = torch.einsum('nsq,ndq->nsd', slc_randnsscld, v_slccov)
            assert slc_epsilons.shape == (n_seeds, n_slc, dim_src)
        else:
            raise ValueError(dedent(f'''
                undefined type of slice sampling covariance:
                    slc/cov/type = {type_slccov}
                    slc/cov/exp = {exp_slccov}'''))


        ####################### Step 2: Normalizing the Slice Directions ######################
        slc_epspnorms = slc_epsilons.square().sum(dim=-1, keepdims=True).sqrt()
        assert slc_epspnorms.shape == (n_seeds, n_slc, 1)

        slc_dirs = slc_epsilons / slc_epspnorms
        assert slc_dirs.shape == (n_seeds, n_slc, dim_src)

        return slc_dirs

    @staticmethod
    def get_iidscldwass(z_data, z_mbs, z_idxs, y_data, y_mbs, y_idxs, 
        n_seeds, n_mb, n_hist, z_dim, y_dim, mtrc_cfg, mrng):

        mtrc_cfg = dict(mtrc_cfg)

        assert z_data.shape == (n_seeds, n_hist, z_dim)
        assert not z_data.isnan().any()

        assert y_data.shape == (n_seeds, n_hist, y_dim)
        assert not y_data.isnan().any()

        n_mb2 = n_mb + n_hist

        # Converting `z_mbs` into a single tensor `z_mbcat`
        z_mbcat = IIDLossMaker.cat_data(z_mbs, z_idxs, z_dim, n_seeds, n_mb)
        assert z_mbcat.shape == (n_seeds, n_mb, z_dim)

        # Concatenating the buffer with the new z data
        z_data2 = torch.cat([z_mbcat, z_data], dim=-2)
        assert z_data2.shape == (n_seeds, n_mb2, z_dim)

        # Whether to normalize z before any computations
        nrmlz_z = mtrc_cfg.pop('scale/z/nrmlz')
        if nrmlz_z:
            use_zscalar = mtrc_cfg.pop('scale/z/scalar')
            z_rdcdims = [1, 2] if use_zscalar else 1

            z_data3 = z_data2 - z_data2.mean(dim=1, keepdims=True)
            assert z_data3.shape == (n_seeds, n_mb2, z_dim)

            z_data4 = z_data3 / z_data3.square().mean(dim=z_rdcdims, keepdims=True).sqrt()
            assert z_data4.shape == (n_seeds, n_mb2, z_dim)
        else:
            z_data4 = z_data2
            assert z_data4.shape == (n_seeds, n_mb2, z_dim)
        
        # Converting `y_mbs` into a single tensor `y_mbcat`
        y_mbcat = IIDLossMaker.cat_data(y_mbs, y_idxs, y_dim, n_seeds, n_mb)
        assert y_mbcat.shape == (n_seeds, n_mb, y_dim)

        # Concatenating the buffer with the new y data
        y_data2 = torch.cat([y_mbcat, y_data], dim=-2)
        assert y_data2.shape == (n_seeds, n_mb2, y_dim)

        # Whether to normalize y before any computations
        nrmlz_y = mtrc_cfg.pop('scale/y/nrmlz')
        if nrmlz_y:
            use_yscalar = mtrc_cfg.pop('scale/y/scalar')
            y_rdcdims = [1, 2] if use_yscalar else 1

            y_data3 = y_data2 - y_data2.mean(dim=1, keepdims=True)
            assert y_data3.shape == (n_seeds, n_mb2, y_dim)

            y_data4 = y_data3 / y_data3.square().mean(dim=y_rdcdims, keepdims=True).sqrt()
            assert y_data4.shape == (n_seeds, n_mb2, y_dim)
        else:
            y_data4 = y_data2
            assert y_data4.shape == (n_seeds, n_mb2, y_dim)

        # The number of slices
        n_slc = mtrc_cfg.pop('slc/n')

        ##################### Step 1: Getting the Sliced Driections for z #####################
        detach_slcdirs = mtrc_cfg.pop('slc/detach')
        with torch.no_grad() if detach_slcdirs else nullcontext():
            # Getting the slice directions for z
            z_slcdirs = IIDLossMaker.get_slcdirs(z_data4, mtrc_cfg, n_seeds, 
                n_slc, n_mb2, z_dim, mrng, pop=True)
            assert z_slcdirs.shape == (n_seeds, n_slc, z_dim)

        # Projecting to the sliced directions
        z_slcd = torch.einsum('sid,sjd->sji', z_data4, z_slcdirs)
        assert z_slcd.shape == (n_seeds, n_slc, n_mb2)

        ##################### Step 2: Getting the Sliced Driections for y #####################
        with torch.no_grad():
            ymtrc_cfg = {'slc/dstr': 'normal', 'slc/cov/type': 'identity'}

            # Getting the slice directions for y
            y_slcdirs = IIDLossMaker.get_slcdirs(y_data4, ymtrc_cfg, n_seeds, 
                n_slc, n_mb2, y_dim, mrng, pop=True)
            assert y_slcdirs.shape == (n_seeds, n_slc, y_dim)

        # Projecting to the sliced directions
        y_slcd = torch.einsum('sid,sjd->sji', y_data4, y_slcdirs)
        assert y_slcd.shape == (n_seeds, n_slc, n_mb2)

        ######################## Step 3: Shuffling the y sliced points ########################
        trg_type = mtrc_cfg.pop('trg/type')
        trg_nrmsamepnts = False
        if trg_type == 'shuff':
            unif_yshuff1 = mrng.rand(n_seeds, n_slc, n_mb2)
            assert unif_yshuff1.shape == (n_seeds, n_slc, n_mb2)

            i_yshuff1 = unif_yshuff1.argsort(dim=-1)
            assert i_yshuff1.shape == (n_seeds, n_slc, n_mb2)
            
            y_slcdtrg = torch.take_along_dim(y_slcd, i_yshuff1, dim=-1)
            assert y_slcdtrg.shape == (n_seeds, n_slc, n_mb2)

            z_slcdtrg = z_slcd
            assert z_slcdtrg.shape == (n_seeds, n_slc, n_mb2)
        elif (trg_type == 'normal') and trg_nrmsamepnts:
            z_data5 = mrng.randn(n_seeds, n_mb2, z_dim)
            assert z_data5.shape == (n_seeds, n_mb2, z_dim)

            z_slcdtrg = torch.einsum('sid,sjd->sji', z_data5, z_slcdirs)
            assert z_slcdtrg.shape == (n_seeds, n_slc, n_mb2)

            y_slcdtrg = y_slcd
            assert y_slcdtrg.shape == (n_seeds, n_slc, n_mb2)
        elif (trg_type == 'normal') and (not trg_nrmsamepnts):
            # Here we are assuming the `z_slcdirs` have unit l2 norm.
            z_slcdtrg = mrng.randn(n_seeds, n_slc, n_mb2)
            assert z_slcdtrg.shape == (n_seeds, n_slc, n_mb2)

            y_slcdtrg = y_slcd
            assert y_slcdtrg.shape == (n_seeds, n_slc, n_mb2)
        else:
            raise ValueError(f'undefined trg/type={trg_type}')

        ################ Step 4: Linearly combining z and y sliced projections ################
        zyorig_slcd = (z_slcd / 2) + (y_slcd / 2)
        assert zyorig_slcd.shape == (n_seeds, n_slc, n_mb2)

        zy_slcdtrg = (z_slcdtrg / 2) + (y_slcdtrg / 2)
        assert zy_slcdtrg.shape == (n_seeds, n_slc, n_mb2)
        
        # Whether to deatch the shuffled target points or not
        detach_trg = mtrc_cfg.pop('trg/detach')
        trg_slcd = zy_slcdtrg.detach() if detach_trg else zy_slcdtrg

        mtrc_scltype = mtrc_cfg.pop('scale/out/type')

        src_pntsdct = {'orig': zyorig_slcd}
        if mtrc_scltype == 'abs':
            pass
        elif mtrc_scltype in ('rel', 'rpd', 'ptt'):
            detach_relref = mtrc_cfg.pop('scale/out/rfr/detach')
            with torch.no_grad() if detach_relref else nullcontext():
                unif_yshuff2 = mrng.rand(n_seeds, n_slc, n_mb2)
                assert unif_yshuff2.shape == (n_seeds, n_slc, n_mb2)
                
                i_yshuff2 = unif_yshuff2.argsort(dim=-1)
                assert i_yshuff2.shape == (n_seeds, n_slc, n_mb2)
                
                y_slcdtrg2 = torch.take_along_dim(y_slcd, i_yshuff2, dim=-1)
                assert y_slcdtrg2.shape == (n_seeds, n_slc, n_mb2)

                z_slcdtrg2 = z_slcdtrg
                assert z_slcdtrg2.shape == (n_seeds, n_slc, n_mb2)
                
                zy_slcdtrg2 = (z_slcdtrg2 / 2) + (y_slcdtrg2 / 2)
                assert zy_slcdtrg2.shape == (n_seeds, n_slc, n_mb2)
            
            src_pntsdct['iid'] = zy_slcdtrg2
        else:
            raise ValueError(f'undefined scale/out/type={mtrc_scltype}')

        ########################### Step 4: Sorting the Sliced Data ###########################
        method_wassalgn = mtrc_cfg.pop('algn/method', 'reindextrg')

        # Whether to filter the part of the wasserstein distance belonging to the historical source points.
        fltr_hist = mtrc_cfg.pop('hist/filter') if (method_wassalgn == 'reindextrg') else None

        # The metric's p-norm
        mtrc_p = mtrc_cfg.pop('pnorm/p')
        assert isinstance(mtrc_p, (int, float))

        # The metric's exponent
        mtrc_exp = mtrc_cfg.pop('pnorm/exp')
        assert isinstance(mtrc_exp, (int, float))

        mtrc_pntsdct = dict()
        for src_id, src_slcd in src_pntsdct.items():
            if method_wassalgn == 'reindextrg':
                src_algndsrtd = src_slcd
                assert src_algndsrtd.shape == (n_seeds, n_slc, n_mb2)

                # Aligning the target values
                trg_algnidxs = get_srtalgnidxs(src_algndsrtd, trg_slcd, dim=-1)
                assert trg_algnidxs.shape == (n_seeds, n_slc, n_mb2)

                trg_algndsrtd = trg_slcd.take_along_dim(trg_algnidxs, dim=-1)
                assert trg_algndsrtd.shape == (n_seeds, n_slc, n_mb2)

                # The sorce to target error
                if fltr_hist:
                    assert detach_trg, dedent(f'''
                        It does not make sense to filter the history without detaching 
                        the targets. Are you sure you know what you are doing?!:
                            hist/filter: {fltr_hist}
                            trg/detach: {detach_trg}''')
                    n_errmb, scl_err = n_mb, 1

                    src_algndsrtdfltrd = src_algndsrtd[..., :n_mb]
                    assert src_algndsrtdfltrd.shape == (n_seeds, n_slc, n_mb)

                    trg_algndsrtdfltrd = trg_algndsrtd[..., :n_mb]
                    assert trg_algndsrtdfltrd.shape == (n_seeds, n_slc, n_mb)

                    err_srctrg = src_algndsrtdfltrd - trg_algndsrtdfltrd
                    assert err_srctrg.shape == (n_seeds, n_slc, n_errmb)
                else:
                    n_errmb, scl_err = n_mb2, n_mb2 / n_mb

                    err_srctrg = src_algndsrtd - trg_algndsrtd
                    assert err_srctrg.shape == (n_seeds, n_slc, n_errmb)
            elif method_wassalgn == 'sort':
                # Sorting the projected values
                src_algndsrtd = src_slcd.sort(dim=-1).values
                assert src_algndsrtd.shape == (n_seeds, n_slc, n_mb2)

                trg_algndsrtd = trg_slcd.sort(dim=-1).values
                assert trg_algndsrtd.shape == (n_seeds, n_slc, n_mb2)

                # The sorce to target error
                n_errmb, scl_err = n_mb2, n_mb2 / n_mb

                err_srctrg = src_algndsrtd - trg_algndsrtd
                assert err_srctrg.shape == (n_seeds, n_slc, n_errmb)
            else:
                raise ValueError(f'undefined wass alignment method {method_wassalgn}')

            ############### Step 6: Finding the Projected and Sorted Error's P-Norm ###############
            err_srctrgabs = err_srctrg.abs() * scl_err
            assert err_srctrgabs.shape == (n_seeds, n_slc, n_errmb)
            
            err_slcs = err_srctrgabs.pow(mtrc_p).mean(dim=-1)
            assert err_slcs.shape == (n_seeds, n_slc)

            mtrc_pnts = err_slcs.pow(mtrc_exp / mtrc_p)
            assert mtrc_pnts.shape == (n_seeds, n_slc)

            mtrc_pntsdct[src_id] = mtrc_pnts

        if mtrc_scltype == 'abs':
            mtrc_pnts = mtrc_pntsdct.pop('orig')
            assert mtrc_pnts.shape == (n_seeds, n_slc)
        elif mtrc_scltype in ('rel', 'rpd', 'ptt'):
            mtrc_pntsorig = mtrc_pntsdct.pop('orig')
            assert mtrc_pntsorig.shape == (n_seeds, n_slc)

            mtrc_pntsiid1 = mtrc_pntsdct.pop('iid')
            assert mtrc_pntsiid1.shape == (n_seeds, n_slc)

            with torch.no_grad() if detach_relref else nullcontext():
                mtrc_pntsiid2 = mtrc_pntsiid1.mean(dim=-1, keepdims=True)
                assert mtrc_pntsiid2.shape == (n_seeds, 1)

            if mtrc_scltype == 'rel':
                # Relative error definition:
                #   d(x, y) = (x - y) / y
                mtrc_pnts = (mtrc_pntsorig / mtrc_pntsiid2) - 1
                assert mtrc_pnts.shape == (n_seeds, n_slc)
            elif mtrc_scltype == 'rpd':
                # Relative Percentile Difference (RPD) definition:
                #   d(x, y) = (x - y) / (x + y)
                mtrc_pntsnumer = (mtrc_pntsorig - mtrc_pntsiid2)
                assert mtrc_pntsnumer.shape == (n_seeds, n_slc)
                
                mtrc_pntsdenom = (mtrc_pntsorig + mtrc_pntsiid2)
                assert mtrc_pntsdenom.shape == (n_seeds, n_slc)
                
                mtrc_pnts = mtrc_pntsnumer / mtrc_pntsdenom
                assert mtrc_pnts.shape == (n_seeds, n_slc)
            elif mtrc_scltype == 'ptt':
                # Paired t-Test for equal sample means:
                #   d({x}, {y}) = (E[x] - E[y]) / sqrt(V[x - y])
                dlt_mtrcpnts = (mtrc_pntsorig - mtrc_pntsiid2)
                assert dlt_mtrcpnts.shape == (n_seeds, n_slc)

                dlt_std = dlt_mtrcpnts.std(dim=-1, correction=1, keepdims=True)
                assert dlt_std.shape == (n_seeds, 1)

                mtrc_pnts = dlt_mtrcpnts / dlt_std
                assert mtrc_pnts.shape == (n_seeds, n_slc)
            else:
                raise ValueError(f'undefined scale/out/type={mtrc_scltype}')
        else:
            raise ValueError(f'undefined scale/out/type={mtrc_scltype}')

        assert len(mtrc_pntsdct) == 0
        assert not mtrc_pnts.isnan().any()

        # The final reduced loss
        rdctn_slcs = mtrc_cfg.pop('slc/rdc')
        if rdctn_slcs == 'mean':
            loss_indzy = mtrc_pnts.mean(dim=-1)
            assert loss_indzy.shape == (n_seeds,)
        elif rdctn_slcs == 'max':
            loss_indzy = mtrc_pnts.max(dim=-1).values
            assert loss_indzy.shape == (n_seeds,)
        else:
            raise ValueError(f'undefined slc/rdc={rdctn_slcs}')

        assert len(mtrc_cfg) == 0, f'unused options: {mtrc_cfg}'

        return loss_indzy

    def __call__(self, z_mbs, y_mbs, n_seeds, n_mb):
        # Initializing the z and y buffers
        if self.z_data is None:
            assert self.z_idxs is None
            assert self.z_dim is None

            assert self.y_data is None
            assert self.y_idxs is None
            assert self.y_dim is None

            self.z_idxs, self.z_dim, self.z_data = self.init_data(z_mbs, 
                n_seeds, n_mb, self.n_hist, self.device, self.dtype)
            
            self.y_idxs, self.y_dim, self.y_data = self.init_data(y_mbs, 
                n_seeds, n_mb, self.n_hist, self.device, self.dtype)
            
            assert self.n_mb is None
            self.n_mb = n_mb
            assert self.n_seeds is None
            self.n_seeds = n_seeds
        
        # A few sanity checks
        assert (self.n_hist % self.n_mb) == 0, dedent(f'''
            Due to lack of an inplace `tensor.slice_scatter_` 
            function, we have to restrict the history size to 
            be an integer multiple of n_mb
                n_hist: {self.n_hist}
                n_mb: {self.n_mb}''')
        
        assert n_seeds == self.n_seeds, dedent(f'''
            unmatchin n_seeds:
                self n_seeds: {self.n_seeds}
                input n_seeds: {n_seeds}''')
        
        assert n_mb == self.n_mb, dedent(f'''
            The mini-batch sizes should stay the same due to the update 
            lack of an inplace `tensor.slice_scatter_` implementation 
            for storing the new variables:
                self  n_mb = {self.n_mb}
                input n_mb = {n_mb}''')

        assert set(self.z_idxs) == set(z_mbs), dedent(f'''
            Unmatching z nodes:
                extra nodes: {z_mbs.keys() - self.z_idxs.keys()}
                missing nodes: {self.z_idxs.keys() - z_mbs.keys()}''')
        
        assert set(self.y_idxs) == set(y_mbs), dedent(f'''
            Unmatching y nodes:
                extra nodes: {y_mbs.keys() - self.y_idxs.keys()}
                missing nodes: {self.y_idxs.keys() - y_mbs.keys()}''')
        
        # Getting the sliced wasserstein losses
        assert self.mtrc_type == 'slcdwass', dedent(f'''
            Only the sliced wasserstein iid loss is implemented 
            at this point:
                mtrc_type = {self.mtrc_type}''')

        # Can only compute the loss if the buffers were filled before.
        if self.is_filled or (self.n_hist == 0):
            loss_indzy = self.get_iidscldwass(self.z_data, z_mbs, 
                self.z_idxs, self.y_data, y_mbs, self.y_idxs, n_seeds, n_mb, 
                self.n_hist, self.z_dim, self.y_dim, self.mtrc_cfg, self.mrng)
            assert loss_indzy.shape == (n_seeds,)

            self.n_calcs += 1
        else:
            loss_indzy = torch.zeros(n_seeds, device=self.device, dtype=self.dtype)
            assert loss_indzy.shape == (n_seeds,)
        
        # Inserting the new data
        with torch.no_grad():
            self.push_data(z_mbs, self.i_mb, self.z_data, self.z_idxs, self.z_dim, 
                n_seeds, n_mb, self.n_hist)
            
            self.push_data(y_mbs, self.i_mb, self.y_data, self.y_idxs, self.y_dim, 
                n_seeds, n_mb, self.n_hist)
        
        # Incrementing the mb index pointer    
        self.i_mb += n_mb
        if self.i_mb >= self.n_hist:
            self.i_mb = (self.i_mb % self.n_hist) if (self.n_hist > 0) else 0

            # Remembering that the buffer was already filled once
            self.is_filled = True

        return loss_indzy
    

# +
###############################################################################
###################### Unit Tests for the Pre-Processor #######################
###############################################################################

class PPTestRef1:
    def __init__(self, pp_id: str, data: dict, split: any, sizes: dict, 
        hparams: dict = None, device: any = None, dtype: any = None):
        hparams = dict() if hparams is None else hparams

        m_chmprthst = data['m_chmprthst']
        n_snr, n_t = sizes['n_snr'], sizes['n_t']
        n_chem, k_bins = sizes['n_chem'], sizes['k_bins']
        n_seeds, n_trn = sizes['n_seeds'], sizes['n_trn']
        if pp_id in ('01_massnrmlzd', '02_massnrmlzd'):
            eps = hparams.pop('eps', 1e-25)
            if pp_id == '02_massnrmlzd':
                exponent = hparams.pop('exponent', 0.5)
            
            with torch.no_grad():
                if torch.is_tensor(m_chmprthst):
                    mchmprthst0 = m_chmprthst.detach().cpu().numpy()
                else:
                    mchmprthst0 = np.array(m_chmprthst)
                assert mchmprthst0.shape == (n_snr * n_t, n_chem, k_bins)
                mchmprthst1 = torch.from_numpy(mchmprthst0 + eps)
                mchmprthst1 = mchmprthst1.to(device=device, dtype=dtype)
                assert mchmprthst1.shape == (n_snr * n_t, n_chem, k_bins)

                if pp_id == '01_massnrmlzd':
                    mchmprthst1b = mchmprthst1
                    assert mchmprthst1b.shape == (n_snr * n_t, n_chem, k_bins)
                elif pp_id == '02_massnrmlzd':
                    mchmprthst1b = mchmprthst1.pow(exponent)
                    assert mchmprthst1b.shape == (n_snr * n_t, n_chem, k_bins)
                else:
                    raise ValueError(f'undefined pp_id = {pp_id}')

                assert split.shape == (n_seeds, n_trn)
                mchmprthst2 = mchmprthst1b[split.ravel()].reshape(n_seeds, n_trn, n_chem, k_bins)
                assert mchmprthst2.shape == (n_seeds, n_trn, n_chem, k_bins)
                
                mchmprthst3 = mchmprthst2.square().sum(dim=[-1, -2], keepdim=True).sqrt()
                assert mchmprthst3.shape == (n_seeds, n_trn, 1, 1)

                mchmprthst4 = mchmprthst2 / mchmprthst3
                assert mchmprthst4.shape == (n_seeds, n_trn, n_chem, k_bins)
                mchmprthst5 = mchmprthst4.mean(dim=1, keepdim=True)
                assert mchmprthst5.shape == (n_seeds, 1, n_chem, k_bins)
                mchmprthst6 = mchmprthst4.std(dim=1, correction=0, keepdim=True)
                assert mchmprthst6.shape == (n_seeds, 1, n_chem, k_bins)
                
                if pp_id == '01_massnrmlzd':
                    mchmprthst7 = mchmprthst3.log()
                    assert mchmprthst7.shape == (n_seeds, n_trn, 1, 1)
                elif pp_id == '02_massnrmlzd':
                    mchmprthst7 = mchmprthst3.pow(exponent)
                    assert mchmprthst7.shape == (n_seeds, n_trn, 1, 1)
                else:
                    raise ValueError(f'undefined pp_id = {pp_id}')

                mchmprthst8 = mchmprthst7.mean(dim=1, keepdim=True)
                assert mchmprthst8.shape == (n_seeds, 1, 1, 1)
                mchmprthst9 = mchmprthst7.std(dim=1, correction=0, keepdim=True)
                assert mchmprthst9.shape == (n_seeds, 1, 1, 1)

                mchmprthst1c = mchmprthst1[split.ravel()].reshape(n_seeds, n_trn, n_chem, k_bins)
                assert mchmprthst1c.shape == (n_seeds, n_trn, n_chem, k_bins)
                
                mchmprthst10 = mchmprthst1c.sum(dim=-1, keepdim=True)
                assert mchmprthst10.shape == (n_seeds, n_trn, n_chem, 1)

                mchmprthst11 = mchmprthst10.mean(dim=[1, 2], keepdim=True)
                assert mchmprthst11.shape == (n_seeds, 1, 1, 1)

                mchmprthst12 = (mchmprthst10 - mchmprthst11).clamp(min=-1e-009, max=1e-009)
                assert mchmprthst12.shape == (n_seeds, n_trn, n_chem, 1)

                mchmprthst13 = mchmprthst12.square().mean(dim=[1, 2], keepdim=True)
                mchmprthst13 = (mchmprthst13 + ((1e-00015)**2)).sqrt()
                assert mchmprthst13.shape == (n_seeds, 1, 1, 1)

                params = dict(mchmprthst5=mchmprthst5, mchmprthst6=mchmprthst6, 
                    mchmprthst8=mchmprthst8, mchmprthst9=mchmprthst9, 
                    mchmprthst11=mchmprthst11, mchmprthst13=mchmprthst13, eps=eps)
                if pp_id == '02_massnrmlzd':
                    params['exponent'] = exponent
                
                x_names = ['m_chmprthst']
                u_names = ['m_chmprtnrm', 'm_prthsttns']
        else:
            raise ValueError(f'undefined pp_id = {pp_id}')
        
        # assert len(hparams) == 0, f'unused hparams = {hparams}'

        self.pp_id = pp_id
        self.params = params
        self.sizes = sizes
        self.x_names = x_names
        self.u_names = u_names

    def forward(self, x_mbs, full=False):
        pp_id = self.pp_id
        assert pp_id in ('01_massnrmlzd', '02_massnrmlzd')
        eps = self.params['eps']
        if pp_id == '02_massnrmlzd':
            exponent = self.params['exponent']
        mchmprthst5 = self.params['mchmprthst5']
        mchmprthst6 = self.params['mchmprthst6']
        mchmprthst8 = self.params['mchmprthst8']
        mchmprthst9 = self.params['mchmprthst9']
        mchmprthst11 = self.params['mchmprthst11']
        mchmprthst13 = self.params['mchmprthst13']
        n_snr = self.sizes['n_snr']
        n_t = self.sizes['n_t']
        n_chem = self.sizes['n_chem']
        k_bins = self.sizes['k_bins']
        n_seeds = self.sizes['n_seeds']

        m_chmprthst = x_mbs['m_chmprthst']
        n_mb = m_chmprthst.shape[1]
        assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, k_bins)

        # calculating `m_chmprtnrm`
        l01_nonzero = m_chmprthst + eps
        assert l01_nonzero.shape == (n_seeds, n_mb, n_chem, k_bins)

        if pp_id == '01_massnrmlzd':
            l01b_nonzero = l01_nonzero
            assert l01b_nonzero.shape == (n_seeds, n_mb, n_chem, k_bins)
        elif pp_id == '02_massnrmlzd':
            l01b_nonzero = l01_nonzero.pow(exponent)
            assert l01b_nonzero.shape == (n_seeds, n_mb, n_chem, k_bins)
        else:
            raise ValueError(f'undefined pp_id = {pp_id}')

        l05_norm = (l01b_nonzero ** 2).sum(dim=[-1, -2], keepdim=True).sqrt()
        assert l05_norm.shape == (n_seeds, n_mb, 1, 1)

        l02_nrmlzd = l01b_nonzero / l05_norm
        assert l02_nrmlzd.shape == (n_seeds, n_mb, n_chem, k_bins)

        l03_nrmlzd = l02_nrmlzd - mchmprthst5
        assert l03_nrmlzd.shape == (n_seeds, n_mb, n_chem, k_bins)
        
        l04_nrmlzd = l03_nrmlzd / mchmprthst6
        assert l04_nrmlzd.shape == (n_seeds, n_mb, n_chem, k_bins)

        # Calculating `m_prthsttns`
        if pp_id == '01_massnrmlzd':
            l06_tnsnorm = l05_norm.log()
            assert l06_tnsnorm.shape == (n_seeds, n_mb, 1, 1)
        elif pp_id == '02_massnrmlzd':
            l06_tnsnorm = l05_norm.pow(exponent)
            assert l06_tnsnorm.shape == (n_seeds, n_mb, 1, 1)
        else:
            raise ValueError(f'undefined pp_id = {pp_id}')

        l07_tnsnorm = l06_tnsnorm - mchmprthst8
        assert l07_tnsnorm.shape == (n_seeds, n_mb, 1, 1)

        l08_tnsnorm = l07_tnsnorm / mchmprthst9
        assert l08_tnsnorm.shape == (n_seeds, n_mb, 1, 1)

        # Calculating `m_chmprtwrd`
        l09_mchm = l01_nonzero.sum(dim=-1, keepdim=True)
        assert l09_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        l10_mchm = l09_mchm - mchmprthst11
        assert l10_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        l11_mchm = l10_mchm.clamp(min=-1e-009, max=1e-009)
        assert l11_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        l12_mchm = l11_mchm / mchmprthst13
        assert l12_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        # Output renamings
        m_chmprtnrm = l04_nrmlzd
        assert m_chmprtnrm.shape == (n_seeds, n_mb, n_chem, k_bins)

        m_prthsttns = l08_tnsnorm
        assert m_prthsttns.shape == (n_seeds, n_mb, 1, 1)

        m_chmprtwrd = l12_mchm
        assert m_chmprtwrd.shape == (n_seeds, n_mb, n_chem, 1)

        if full:
            output = dict(m_chmprthst=m_chmprthst, l01_nonzero=l01_nonzero, 
                l02_nrmlzd=l02_nrmlzd, l03_nrmlzd=l03_nrmlzd, l04_nrmlzd=l04_nrmlzd, 
                l05_norm=l05_norm, l06_tnsnorm=l06_tnsnorm, l07_tnsnorm=l07_tnsnorm, 
                l08_tnsnorm=l08_tnsnorm, l09_mchm=l09_mchm, l10_mchm=l10_mchm, 
                l11_mchm=l11_mchm, l12_mchm=l12_mchm)
            if pp_id == '02_massnrmlzd':
                output['l01b_nonzero'] = l01b_nonzero
        else:
            output = dict()
        output.update(dict(m_chmprtnrm=m_chmprtnrm, m_prthsttns=m_prthsttns, 
            m_chmprtwrd=m_chmprtwrd))
            
        return output, None

    def inverse(self, u_mbs, u_shaper, strict=False, full=False):
        assert not strict

        pp_id = self.pp_id
        assert pp_id in ('01_massnrmlzd', '02_massnrmlzd')
        eps = self.params['eps']
        if pp_id == '02_massnrmlzd':
            exponent = self.params['exponent']
        mchmprthst5 = self.params['mchmprthst5']
        mchmprthst6 = self.params['mchmprthst6']
        mchmprthst8 = self.params['mchmprthst8']
        mchmprthst9 = self.params['mchmprthst9']
        mchmprthst11 = self.params['mchmprthst11']
        mchmprthst13 = self.params['mchmprthst13']
        n_snr = self.sizes['n_snr']
        n_t = self.sizes['n_t']
        n_chem = self.sizes['n_chem']
        k_bins = self.sizes['k_bins']
        n_seeds = self.sizes['n_seeds']

        m_prthsttns = u_mbs['m_prthsttns']
        n_mb = m_prthsttns.shape[1]
        assert m_prthsttns.shape == (n_seeds, n_mb, 1, 1)

        l08_tnsnorm = m_prthsttns
        assert l08_tnsnorm.shape == (n_seeds, n_mb, 1, 1)

        l07_tnsnorm = l08_tnsnorm * mchmprthst9
        assert l07_tnsnorm.shape == (n_seeds, n_mb, 1, 1)

        l06_tnsnorm = l07_tnsnorm + mchmprthst8
        assert l06_tnsnorm.shape == (n_seeds, n_mb, 1, 1)

        if pp_id == '01_massnrmlzd':
            l05_norm = l06_tnsnorm.exp()
            assert l05_norm.shape == (n_seeds, n_mb, 1, 1)
        elif pp_id == '02_massnrmlzd':
            l05_norm = l06_tnsnorm.pow(1.0 / exponent)
            assert l05_norm.shape == (n_seeds, n_mb, 1, 1)
        else:
            raise ValueError(f'undefined pp_id = {pp_id}')

        m_chmprtnrm = u_mbs['m_chmprtnrm']
        assert m_chmprtnrm.shape == (n_seeds, n_mb, n_chem, k_bins)

        l04_nrmlzd = m_chmprtnrm
        assert l04_nrmlzd.shape == (n_seeds, n_mb, n_chem, k_bins)

        l03_nrmlzd = l04_nrmlzd * mchmprthst6
        assert l03_nrmlzd.shape == (n_seeds, n_mb, n_chem, k_bins)

        l02_nrmlzd = l03_nrmlzd + mchmprthst5
        assert l02_nrmlzd.shape == (n_seeds, n_mb, n_chem, k_bins)
        
        l01b_nonzero = (l02_nrmlzd / (l02_nrmlzd ** 2).sum(dim=[-1, -2], 
            keepdim=True).sqrt()) * l05_norm
        assert l01b_nonzero.shape == (n_seeds, n_mb, n_chem, k_bins)

        if pp_id == '01_massnrmlzd':
            l01_nonzero = l01b_nonzero
            assert l01_nonzero.shape == (n_seeds, n_mb, n_chem, k_bins)
        elif pp_id == '02_massnrmlzd':
            l01_nonzero = l01b_nonzero.pow(1.0 / exponent)
            assert l01_nonzero.shape == (n_seeds, n_mb, n_chem, k_bins)
        else:
            raise ValueError(f'undefined pp_id = {pp_id}')

        m_chmprthst = l01_nonzero - eps
        assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, k_bins)

        # Calculating `m_chmprtwrd`
        l09_mchm = l01_nonzero.sum(dim=-1, keepdim=True)
        assert l09_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        l10_mchm = l09_mchm - mchmprthst11
        assert l10_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        l11_mchm = l10_mchm.clamp(min=-1e-009, max=1e-009)
        assert l11_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        l12_mchm = l11_mchm / mchmprthst13
        assert l12_mchm.shape == (n_seeds, n_mb, n_chem, 1)

        m_chmprtwrd = u_mbs['m_chmprtwrd']
        assert m_chmprtwrd.shape == (n_seeds, n_mb, n_chem, 1)
        
        if full:
            output = dict(m_prthsttns=m_prthsttns, m_chmprtnrm=m_chmprtnrm, 
                m_chmprtwrd=m_chmprtwrd, l12_mchm=l12_mchm, l11_mchm=l11_mchm, 
                l10_mchm=l10_mchm, l09_mchm=l09_mchm, l08_tnsnorm=l08_tnsnorm, 
                l07_tnsnorm=l07_tnsnorm, l06_tnsnorm=l06_tnsnorm, 
                l05_norm=l05_norm, l04_nrmlzd=l04_nrmlzd, l03_nrmlzd=l03_nrmlzd, 
                l02_nrmlzd=l02_nrmlzd, l01_nonzero=l01_nonzero)
            if pp_id == '02_massnrmlzd':
                output['l01b_nonzero'] = l01b_nonzero
        else:
            output = dict()
        output.update(dict(m_chmprthst=m_chmprthst))

        return output

class PPTestRef2:
    def __init__(self, pp_id: str, data: dict, split: any, sizes: dict, 
        hparams: dict = None, device: any = None, dtype: any = None):
        hparams = dict() if hparams is None else hparams

        m_chmprthst = data['m_chmprthst']
        n_snr, n_t = sizes['n_snr'], sizes['n_t']
        n_chem, k_bins = sizes['n_chem'], sizes['k_bins']
        n_seeds, n_trn = sizes['n_seeds'], sizes['n_trn']
        with torch.no_grad():
            magdim = hparams['magdim']
            nrmeps = hparams['nrmeps']
            magpnrm = hparams['magpnrm']
            magexp = hparams['magexp']
            nrmexp = hparams['nrmexp']
            cdfbin = hparams['cdfbin']
            cdfdim = hparams['cdfdim']
            cdfdmn = hparams['cdfdmn']
            cdffrm = hparams['cdffrm']
            domain_min, domain_max = cdfdmn

            # The magnitude reduction dimensions
            tnsltn_dict2 = {'n_chem': [-1], 'n_chnls': [-1], 'n_bins': [-2], 'n_len': [-2], 'one': [-1, -2]}
            mag_rdcdims = tnsltn_dict2[magdim]

            if torch.is_tensor(m_chmprthst):
                mchmprthst0th = m_chmprthst
                assert mchmprthst0th.shape == (n_snr * n_t, n_chem, k_bins)
            else:
                mchmprthst0np = np.array(m_chmprthst)
                assert mchmprthst0np.shape == (n_snr * n_t, n_chem, k_bins)
                mchmprthst0th = torch.from_numpy(mchmprthst0np)
                assert mchmprthst0th.shape == (n_snr * n_t, n_chem, k_bins)
            
            mchmprthst1 = mchmprthst0th.to(device=device, dtype=dtype)
            assert mchmprthst1.shape == (n_snr * n_t, n_chem, k_bins)

            assert split.shape == (n_seeds, n_trn)
            mchmprthst2 = mchmprthst1[split.ravel()].reshape(n_seeds, n_trn, n_chem, k_bins)
            assert mchmprthst2.shape == (n_seeds, n_trn, n_chem, k_bins)

            m01 = mchmprthst2 + nrmeps
            assert m01.shape == (n_seeds, n_trn, n_chem, k_bins)

            m02 = m01.abs().pow(magpnrm).sum(dim=mag_rdcdims, keepdim=True).pow(1. / magpnrm)
            assert set(mag_rdcdims).issubset({-1, -2})
            n_chnlsmag = 1 if (-2 in mag_rdcdims) else n_chem
            n_binsmag = 1 if (-1 in mag_rdcdims) else k_bins
            assert m02.shape == (n_seeds, n_trn, n_chnlsmag, n_binsmag)

            m03 = m02.sign() * m02.abs().pow(magexp)
            assert m03.shape == (n_seeds, n_trn, n_chnlsmag, n_binsmag)

            m04 = m01 / m02
            assert m04.shape == (n_seeds, n_trn, n_chem, k_bins)

            m05 = m04.sign() * m04.abs().pow(nrmexp)
            assert m05.shape == (n_seeds, n_trn, n_chem, k_bins)
            
            n_qnts = cdfbin + 1
            q = torch.linspace(0, 1, n_qnts, device=device, dtype=dtype)
            if set(cdfdim) == {-1}:
                m05b = m05.unsqueeze(dim=-2)
                assert m05b.shape == (n_seeds, n_trn, n_chem, 1, k_bins)
                m05c = m05b.transpose(1, -2)
                assert m05c.shape == (n_seeds, 1, n_chem, n_trn, k_bins)
                m05d = m05c.flatten(-2, -1)
                assert m05d.shape == (n_seeds, 1, n_chem, n_trn * k_bins)
                m05e = m05d.quantile(q, dim=-1, keepdim=True)
                assert m05e.shape == (n_qnts, n_seeds, 1, n_chem, 1)
                m05f = m05e.transpose(0, 2).squeeze(0)
                assert m05f.shape == (n_seeds, n_qnts, n_chem, 1)
            elif set(cdfdim) == {-1, -2}:
                m05b = m05.flatten(1, -1)
                assert m05b.shape == (n_seeds, n_trn * n_chem * k_bins)
                m05c = m05b.quantile(q, dim=-1, keepdim=False)
                assert m05c.shape == (n_qnts, n_seeds)
                m05f = m05c.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
                assert m05f.shape == (n_seeds, n_qnts, 1, 1)
            elif set(cdfdim) == {-2}:
                m05b = m05.flatten(1, -2)
                assert m05b.shape == (n_seeds, n_trn * n_chem, k_bins)
                m05c = m05b.quantile(q, dim=-2, keepdim=False)
                assert m05c.shape == (n_qnts, n_seeds, k_bins)
                m05f = m05c.transpose(0, 1).unsqueeze(-2)
                assert m05f.shape == (n_seeds, n_qnts, 1, k_bins)
            else:
                raise NotImplementedError(f'undefined cdfdim={cdfdim}')
            
            m06qntls = torch.cat([
                torch.full((n_seeds, 1, *m05f.shape[-2:]), domain_min, device=device, dtype=dtype),
                m05f,
                torch.full((n_seeds, 1, *m05f.shape[-2:]), domain_max, device=device, dtype=dtype)], dim=1)
            assert m06qntls.shape == (n_seeds, n_qnts+2, *m05f.shape[-2:])

            params = dict(m06qntls=m06qntls, magdim=magdim, nrmeps=nrmeps, 
                magpnrm=magpnrm, magexp=magexp, nrmexp=nrmexp, cdfbin=cdfbin, 
                cdfdim=cdfdim, domain_min=domain_min, domain_max=domain_max, 
                mag_rdcdims=mag_rdcdims, n_chnlsmag=n_chnlsmag, n_binsmag=n_binsmag,
                cdffrm=cdffrm)
            x_names = ['m_chmprthst']
            u_names = ['m_chmprtnrm', 'm_prthstmag']

        self.pp_id = pp_id
        self.params = params
        self.sizes = sizes
        self.x_names = x_names
        self.u_names = u_names

    def forward(self, x_mbs, full=False):
        pp_id = self.pp_id
        assert pp_id in ('03_massnrmlzd',)
        
        m06qntls = self.params['m06qntls']
        magdim = self.params['magdim']
        nrmeps = self.params['nrmeps']
        magpnrm = self.params['magpnrm']
        magexp = self.params['magexp']
        nrmexp = self.params['nrmexp']
        cdfbin = self.params['cdfbin']
        cdfdim = self.params['cdfdim']
        domain_min = self.params['domain_min']
        domain_max = self.params['domain_max']
        mag_rdcdims = self.params['mag_rdcdims']
        n_chnlsmag = self.params['n_chnlsmag']
        n_binsmag = self.params['n_binsmag']
        mag_rdcdims = self.params['mag_rdcdims']
        cdffrm = self.params['cdffrm']
        cdf_maxerr = 1e-5
        
        n_snr = self.sizes['n_snr']
        n_t = self.sizes['n_t']
        n_chem = self.sizes['n_chem']
        k_bins = self.sizes['k_bins']
        n_seeds = self.sizes['n_seeds']
        n_qnts = cdfbin + 1
        n_qnts2 = n_qnts + 2

        m_chmprthst = x_mbs['m_chmprthst']
        n_mb = m_chmprthst.shape[1]
        assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, k_bins)

        m01 = m_chmprthst + nrmeps
        assert m01.shape == (n_seeds, n_mb, n_chem, k_bins)

        m02 = m01.abs().pow(magpnrm).sum(dim=mag_rdcdims, keepdim=True).pow(1. / magpnrm)
        assert m02.shape == (n_seeds, n_mb, n_chnlsmag, n_binsmag)

        m03 = m02.sign() * m02.abs().pow(magexp)
        assert m03.shape == (n_seeds, n_mb, n_chnlsmag, n_binsmag)

        m04 = m01 / m02
        assert m04.shape == (n_seeds, n_mb, n_chem, k_bins)

        m05 = m04.sign() * m04.abs().pow(nrmexp)
        assert m05.shape == (n_seeds, n_mb, n_chem, k_bins)

        ############################ Finding the CDF of m05 ############################
        if set(cdfdim) == {-1}:
            assert m06qntls.shape == (n_seeds, n_qnts2, n_chem, 1)
        elif set(cdfdim) == {-1, -2}:
            assert m06qntls.shape == (n_seeds, n_qnts2, 1, 1)
        elif set(cdfdim) == {-2}:
            assert m06qntls.shape == (n_seeds, n_qnts2, 1, k_bins)
        else:
            raise NotImplementedError(f'undefined cdfdim={cdfdim}')

        srtdseq = m06qntls.expand(n_seeds, n_qnts2, n_chem, k_bins
            ).unsqueeze(-1).transpose(1, -1).contiguous()
        assert srtdseq.shape == (n_seeds, 1, n_chem, k_bins, n_qnts2)
        srchvals = m05.unsqueeze(-1).transpose(1, -1).contiguous()
        assert srchvals.shape == (n_seeds, 1, n_chem, k_bins, n_mb)
        srch_idxs1 = torch.searchsorted(srtdseq, srchvals, right=True)
        assert srch_idxs1.shape == (n_seeds, 1, n_chem, k_bins, n_mb)
        srch_idxs = srch_idxs1.transpose(1, -1).squeeze(-1)
        assert srch_idxs.shape == (n_seeds, n_mb, n_chem, k_bins)

        # Finding the search bins low value
        srch_idxslo = (srch_idxs - 1).clamp(min=0)
        assert srch_idxslo.shape == (n_seeds, n_mb, n_chem, k_bins)
        srch_valslo = torch.take_along_dim(m06qntls, srch_idxslo, dim=1)
        assert srch_valslo.shape == (n_seeds, n_mb, n_chem, k_bins)
        # Finding the search bins high value
        srch_idxshi = srch_idxs.clamp(max=n_qnts2 - 1)
        assert srch_idxshi.shape == (n_seeds, n_mb, n_chem, k_bins)
        srch_valshi = torch.take_along_dim(m06qntls, srch_idxshi, dim=1)
        assert srch_valshi.shape == (n_seeds, n_mb, n_chem, k_bins)

        assert (m05 <= srch_valshi).all()
        assert (m05 >= srch_valslo).all()
        srch_idxshilodiff = (srch_idxshi - srch_idxslo)
        assert (srch_idxshilodiff <= 1).all()
        assert (srch_idxshilodiff >= 0).all()

        alpha1 = (m05 - srch_valslo) / (srch_valshi - srch_valslo)
        assert alpha1.shape == (n_seeds, n_mb, n_chem, k_bins)
        
        alpha2 = torch.where(srch_valshi == srch_valslo, 0.0, alpha1)
        assert alpha2.shape == (n_seeds, n_mb, n_chem, k_bins)
        assert (alpha2 >= 0).all()
        assert (alpha2 <= 1).all()

        m05_cdfs1 = (srch_idxs - 1).clamp(min=0) / (n_qnts2 - 1) + alpha2 / (n_qnts2 - 1)
        assert m05_cdfs1.shape == (n_seeds, n_mb, n_chem, k_bins)
        assert (m05_cdfs1 <= (1 + cdf_maxerr)).all()
        assert (m05_cdfs1 >= (0 - cdf_maxerr)).all()

        if cdffrm == 'domain':
            m05_cdfs = m05_cdfs1
            assert m05_cdfs.shape == (n_seeds, n_mb, n_chem, k_bins)
        elif cdffrm == 'data':
            # Pushing the domain min and max out of the [0, 1] cdf region
            q0, q1 = 1 / (n_qnts2 - 1), (n_qnts2 - 2) / (n_qnts2 - 1)
            m05_cdfs = (m05_cdfs1 - q0) / (q1 - q0)
            assert m05_cdfs.shape == (n_seeds, n_mb, n_chem, k_bins)
        else:
            raise ValueError(f'undefined cdffrm={cdffrm}')
        
        m06 = m05_cdfs
        assert m06.shape == (n_seeds, n_mb, n_chem, k_bins)

        # Output renamings
        m_chmprtnrm = m06
        assert m_chmprtnrm.shape == (n_seeds, n_mb, n_chem, k_bins)

        m_prthstmag = m03
        assert m_prthstmag.shape == (n_seeds, n_mb, n_chnlsmag, n_binsmag)

        if full:
            output = dict(m_chmprthst=m_chmprthst, m01=m01, m02=m02, m03=m03, 
                m04=m04, m05=m05, m06=m06)
        else:
            output = dict()
        output.update(dict(m_chmprtnrm=m_chmprtnrm, m_prthstmag=m_prthstmag))
            
        return output, None

    def inverse(self, u_mbs, u_shaper, strict=False, full=False):
        assert not strict

        pp_id = self.pp_id
        assert pp_id in ('03_massnrmlzd',)
        
        m06qntls = self.params['m06qntls']
        magdim = self.params['magdim']
        nrmeps = self.params['nrmeps']
        magpnrm = self.params['magpnrm']
        magexp = self.params['magexp']
        nrmexp = self.params['nrmexp']
        cdfbin = self.params['cdfbin']
        cdfdim = self.params['cdfdim']
        domain_min = self.params['domain_min']
        domain_max = self.params['domain_max']
        mag_rdcdims = self.params['mag_rdcdims']
        n_chnlsmag = self.params['n_chnlsmag']
        n_binsmag = self.params['n_binsmag']
        mag_rdcdims = self.params['mag_rdcdims']
        cdffrm = self.params['cdffrm']
        cdf_maxerr = 1e-5
        
        n_snr = self.sizes['n_snr']
        n_t = self.sizes['n_t']
        n_chem = self.sizes['n_chem']
        k_bins = self.sizes['k_bins']
        n_seeds = self.sizes['n_seeds']
        n_qnts = cdfbin + 1
        n_qnts2 = n_qnts + 2

        m_prthstmag = u_mbs['m_prthstmag']
        n_mb = m_prthstmag.shape[1]
        assert m_prthstmag.shape == (n_seeds, n_mb, n_chnlsmag, n_binsmag)

        m_chmprtnrm = u_mbs['m_chmprtnrm']
        assert m_chmprtnrm.shape == (n_seeds, n_mb, n_chem, k_bins)

        m06 = m_chmprtnrm
        assert m06.shape == (n_seeds, n_mb, n_chem, k_bins)

        ######### Finding m05
        m05_cdfs = m06
        assert m05_cdfs.shape == (n_seeds, n_mb, n_chem, k_bins)

        if cdffrm == 'domain':
            m05_cdfs2 = m05_cdfs
            assert m05_cdfs2.shape == m05_cdfs.shape
        elif cdffrm == 'data':
            # Shifting the domain min and max out of the [0, 1] quantile region
            q0, q1 = 1 / (n_qnts2 - 1), (n_qnts2 - 2) / (n_qnts2 - 1)
            m05_cdfs2 = q0 + (q1 - q0) * m05_cdfs
            assert m05_cdfs2.shape == m05_cdfs.shape
        else:
            raise ValueError(f'undefined cdffrm={cdffrm}')

        m05_cdfs3 = (m05_cdfs2 * (n_qnts2 - 1))
        assert m05_cdfs3.shape == m05_cdfs.shape
        
        alpha3 = m05_cdfs3 - m05_cdfs3.floor()
        assert alpha3.shape == m05_cdfs.shape

        srch_idxslo2 = (m05_cdfs3 - alpha3).round().long()
        assert srch_idxslo2.shape == m05_cdfs.shape
        assert (srch_idxslo2 < n_qnts2).all()
        assert (srch_idxslo2 >= 0).all()

        srch_valslo2 = torch.take_along_dim(m06qntls, srch_idxslo2, dim=1)
        assert srch_valslo2.shape == m05_cdfs.shape

        srch_idxshi2 = (srch_idxslo2 + 1).clamp(max=n_qnts2-1)
        assert srch_idxshi2.shape == m05_cdfs.shape
        assert (srch_idxshi2 < n_qnts2).all()
        assert (srch_idxshi2 >= 0).all()

        srch_valshi2 = torch.take_along_dim(m06qntls, srch_idxshi2, dim=1)
        assert srch_valshi2.shape == m05_cdfs.shape

        m05 = alpha3 * srch_valshi2 + (1 - alpha3) * srch_valslo2
        assert m05.shape == m05_cdfs.shape

        ############## Finding the rest
        m04 = m05.sign() * m05.abs().pow(1. / nrmexp)
        assert m04.shape == (n_seeds, n_mb, n_chem, k_bins)

        m04 = m04 / m04.abs().pow(magpnrm).sum(dim=mag_rdcdims, keepdim=True).pow(1. / magpnrm)
        assert m04.shape == (n_seeds, n_mb, n_chem, k_bins)

        m03 = m_prthstmag
        assert m03.shape == (n_seeds, n_mb, n_chnlsmag, n_binsmag)

        m02 = m03.sign() * m03.abs().pow(1.0 / magexp)
        assert m02.shape == (n_seeds, n_mb, n_chnlsmag, n_binsmag)

        m01 = m02 * m04
        assert m01.shape == (n_seeds, n_mb, n_chem, k_bins)

        m_chmprthst = m01 - nrmeps
        assert m_chmprthst.shape == (n_seeds, n_mb, n_chem, k_bins)

        if full:
            output = dict(m_chmprtnrm=m_chmprtnrm, m_prthstmag=m_prthstmag, m01=m01, m02=m02, 
                m03=m03, m04=m04, m05=m05, m06=m06)
        else:
            output = dict()
        output.update(dict(m_chmprthst=m_chmprthst))

        return output

class PPTest:
    def __init__(self, pp_id, data_id, split_id, n_seeds, n_mb, hparams, 
        device=None, dtype=None, cdtype=None):
        self.device = torch.device('cpu' if device is None else device)
        self.dtype = torch.float64 if dtype is None else dtype
        self.cdtype = torch.complex128 if cdtype is None else cdtype

        self.configs = self.make_configs(hparams)

        self.pp = self.make_pp(self.configs, pp_id, self.device, self.dtype, self.cdtype)
        self.data, self.sizes = self.load_data(data_id, self.device, self.dtype)
        self.sizes['n_seeds'], self.sizes['n_mb'] = n_seeds, n_mb
        
        self.n_snr, self.n_t = self.sizes['n_snr'], self.sizes['n_t']
        self.n_chem, self.k_bins = self.sizes['n_chem'], self.sizes['k_bins']
        self.n_seeds, self.n_mb = n_seeds, n_mb
        
        self.rng = self.make_rng(n_seeds, self.device, self.dtype)
        self.split = self.make_split(self.configs, split_id, self.rng, self.n_snr, self.n_t, 
            n_seeds, self.device, self.dtype)
        self.n_trn = self.sizes['n_trn'] = self.split.shape[1]
        self.pp.infer(self.data, self.split, n_seeds, n_mb)

        self.refpp = self.make_refpp(self.configs, pp_id, self.data, self.split, 
            self.sizes, self.device, self.dtype)
    
    @staticmethod
    def make_configs(hparams=None):
        hparams = hparams if hparams is not None else dict()
        eps = hparams.get('eps', 1e-25)
        exponent = hparams.get('exponent', 0.5)
        
        magdim = hparams.get('magdim', 'n_len')
        nrmeps = hparams.get('nrmeps', 1e-25)
        magpnrm = hparams.get('magpnrm', 2)
        magexp = hparams.get('magexp', 0.3)
        nrmexp = hparams.get('nrmexp', 0.5)
        cdfbin = hparams.get('cdfbin', 100)
        cdfdim = hparams.get('cdfdim', [-1])
        cdfdmn = hparams.get('cdfdmn', [-1, 100])
        cdffrm = hparams.get('cdffrm', 'data')

        cfgstrs = f'''
        pp:
            01_massnrmlzd:
                input: 
                    m_chmprthst: [n_seeds, n_mb, n_chem, k_bins]
                output: 
                    m_chmprtnrm: l04_nrmlzd
                    m_prthsttns: l08_tnsnorm
                    m_chmprtwrd: l12_mchm
                hparams:
                    eps: {eps}
                layers:
                    l01_nonzero:
                        type: add
                        value: eps
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    # The normalized mass values
                    l02_nrmlzd:
                        type: normalize
                        dim: [-1, -2]
                        pnorm: 2
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    l03_nrmlzd:
                        type: shift
                        dim: null
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    l04_nrmlzd:
                        type: scale
                        dim: null
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    # The log norm of the mass values
                    l05_norm:
                        type: pnorm
                        dim: [-1, -2]
                        pnorm: 2
                        input: l01_nonzero
                        shape: [n_seeds, n_mb, 1, 1]

                    l06_tnsnorm:
                        type: log
                        shape: [n_seeds, n_mb, 1, 1]

                    l07_tnsnorm:
                        type: shift
                        dim: null
                        shape: [n_seeds, n_mb, 1, 1]

                    l08_tnsnorm:
                        type: scale
                        dim: null
                        shape: [n_seeds, n_mb, 1, 1]
                    
                    # The per-chemical total mass values
                    l09_mchm:
                        type: torch.sum
                        input: l01_nonzero
                        dim: -1
                        keepdim: true
                        shape: [n_seeds, n_mb, n_chem, 1]

                    l10_mchm:
                        type: shift
                        dim: [-1, -2]
                        shape: [n_seeds, n_mb, n_chem, 1]
                    
                    l11_mchm:
                        type: clamp
                        min: -1e-009
                        max: 1e-009
                        shape: [n_seeds, n_mb, n_chem, 1]

                    l12_mchm:
                        type: scale
                        eps: 1e-00015
                        dim: [-1, -2]
                        shape: [n_seeds, n_mb, n_chem, 1]
        
            02_massnrmlzd:
                input: 
                    m_chmprthst: [n_seeds, n_mb, n_chem, k_bins]
                output: 
                    m_chmprtnrm: l04_nrmlzd
                    m_prthsttns: l08_tnsnorm
                    m_chmprtwrd: l12_mchm
                hparams:
                    eps: {eps}
                    exponent: {exponent}
                layers:
                    l01_nonzero:
                        type: add
                        value: eps
                        shape: [n_seeds, n_mb, n_chem, k_bins]
                    
                    l01b_nonzero:
                        type: sgnpow
                        exponent: exponent
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    # The normalized mass values
                    l02_nrmlzd:
                        type: normalize
                        dim: [-1, -2]
                        pnorm: 2
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    l03_nrmlzd:
                        type: shift
                        dim: null
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    l04_nrmlzd:
                        type: scale
                        dim: null
                        shape: [n_seeds, n_mb, n_chem, k_bins]

                    # The log norm of the mass values
                    l05_norm:
                        type: pnorm
                        dim: [-1, -2]
                        pnorm: 2
                        input: l01b_nonzero
                        shape: [n_seeds, n_mb, 1, 1]

                    l06_tnsnorm:
                        type: sgnpow
                        exponent: exponent
                        shape: [n_seeds, n_mb, 1, 1]

                    l07_tnsnorm:
                        type: shift
                        dim: null
                        shape: [n_seeds, n_mb, 1, 1]

                    l08_tnsnorm:
                        type: scale
                        dim: null
                        shape: [n_seeds, n_mb, 1, 1]
                    
                    # The per-chemical total mass values
                    l09_mchm:
                        type: torch.sum
                        input: l01_nonzero
                        dim: -1
                        keepdim: true
                        shape: [n_seeds, n_mb, n_chem, 1]

                    l10_mchm:
                        type: shift
                        dim: [-1, -2]
                        shape: [n_seeds, n_mb, n_chem, 1]
                    
                    l11_mchm:
                        type: clamp
                        min: -1e-009
                        max: 1e-009
                        shape: [n_seeds, n_mb, n_chem, 1]

                    l12_mchm:
                        type: scale
                        eps: 1e-00015
                        dim: [-1, -2]
                        shape: [n_seeds, n_mb, n_chem, 1]
            
            03_massnrmlzd:
                input:
                    m_chmprthst: [n_seeds, n_mb, n_chem, n_bins]

                output: 
                    m_chmprtnrm: m06
                    m_prthstmag: m03

                hparams:
                    magdim: {magdim}
                    nrmeps: {nrmeps}
                    magpnrm: {magpnrm}
                    magexp: {magexp}
                    nrmexp: {nrmexp}
                    cdfbin: {cdfbin}
                    cdfdim: {cdfdim}
                    cdfdmn: {cdfdmn}
                    cdffrm: {cdffrm}
                    
                defs:
                    # The magnitude reduction dimensions
                    tnsltn_dict2: 
                        n_chem: [-1]
                        n_chnls: [-1]
                        n_bins: [-2]
                        n_len: [-2]
                        one: [-1, -2]
                    mag_rdcdims: tnsltn_dict2[magdim]
                
                layers:
                    #######################################
                    ######## M1: Adding an Epsilon ########
                    #######################################
                    # M1 = m0 + nrmeps
                    m01:
                        type: add
                        value: nrmeps
                        input: m_chmprthst
                        shape: [n_seeds, n_mb, n_chem, n_bins]

                    #######################################
                    #### M2: Taking the Lp-Norm of M1 #####
                    #######################################
                    # M2 = \| M1 \| 
                    m02:
                        type: pnorm
                        dim: mag_rdcdims
                        pnorm: magpnrm
                        shape: [n_seeds, n_mb, n_chnlsmag, n_binsmag]

                    #######################################
                    ## M3: Exponentiating the magnitude ###
                    #######################################
                    # M3 = M2 ^ alpha
                    m03:
                        type: sgnpow
                        exponent: magexp
                        shape: [n_seeds, n_mb, n_chnlsmag, n_binsmag]

                    #######################################
                    ######### M4: Normalization ###########
                    #######################################
                    # M4 = M1 / M2
                    m04:
                        type: normalize
                        dim: mag_rdcdims
                        pnorm: magpnrm
                        input: m01
                        shape: [n_seeds, n_mb, n_chem, n_bins]

                    #######################################
                    ## M5: Exponentiating the normalztn ###
                    #######################################
                    # M5 = M4 ^ alpha
                    m05:
                        type: sgnpow
                        exponent: nrmexp
                        shape: [n_seeds, n_mb, n_chem, n_bins]
                    
                    #######################################
                    ###### M6: Taking the histogram #######
                    #######################################
                    m06:
                        type: cdf
                        dim: cdfdim
                        bins: cdfbin
                        frame: cdffrm
                        domain: cdfdmn
                        shape: [n_seeds, n_mb, n_chem, n_bins]
        
        split:
            01_snrsplt:
                vars: [snr,t]
                snr:
                    shuffle: true
                    low: 0
                    high: 0.8
                t:
                    shuffle: false
                    low: 0
                    high: 1
            
            02_fulltrn:
                vars: [snr,t]
                snr:
                    shuffle: false
                    low: 0
                    high: 1
                t:
                    shuffle: false
                    low: 0
                    high: 1
        
        data:
            01_uniform:
                none: null
            
            02_masshist:01_bwchisamp.nc:
                none: null
        '''
        return dict(ruyaml.safe_load(dedent(cfgstrs)))

    @staticmethod
    def make_pp(configs, pp_id, device, dtype, cdtype):
        pp_config = configs['pp'][pp_id]
        pp = PreProcessor(pp_config, device=device, dtype=dtype, cdtype=cdtype)
        return pp
    
    @staticmethod
    def make_refpp(configs, pp_id, data, split, sizes, device, dtype):
        if pp_id in ('01_massnrmlzd', '02_massnrmlzd', '03_massnrmlzd'):
            hparams = configs['pp'][pp_id]['hparams']
            PPTestRef = {'01_massnrmlzd': PPTestRef1, '02_massnrmlzd': PPTestRef1, 
                '03_massnrmlzd': PPTestRef2}[pp_id]
            refpp = PPTestRef(pp_id, data, split, sizes, hparams, device, dtype)
        else:
            raise ValueError(f'undefined pp_id = {pp_id}')
        return refpp

    @staticmethod
    def load_data(data_id, device, dtype):
        if data_id == '02_masshist:01_bwchisamp.nc':
            ################ Loading the Test Data ################
            data_path = f'{data_dir}/02_masshist/01_bwchisamp.nc'
            assert data_path.endswith('.nc')
            with netCDF4.Dataset(data_path, "r") as fp:
                ################# The Array Dimensions #################
                # The number of scenarios
                n_snr = fp.dimensions['n_snr'].size
                # The number of time steps in each scenario
                n_t = fp.dimensions['n_t'].size
                # The number of diameter histogram bins
                k_bins = fp.dimensions['k_bins'].size
                # The number of chemical specices
                n_chem = fp.dimensions['n_chem'].size

                #################### The Data Arrays ####################
                # The number of particles within each diameter bin
                n_prthst = fp.variables['n_prthst'][:]
                assert n_prthst.shape == (n_snr, n_t, k_bins)

                # The mass within each diameter bin (the mass histogram)
                m_prthst = fp.variables['m_prthst'][:]
                assert m_prthst.shape == (n_snr, n_t, k_bins)

                # The mass of each chemical species within the diameter bins
                m_chmprthst = np.moveaxis(fp.variables['m_chmprthst'][:], -1, -2)
                assert m_chmprthst.shape == (n_snr, n_t, n_chem, k_bins)
            
            # `datanp_dict` is an input variable (str) to array (np.ndarray) mapping.
            datanp_dict = dict(
                n_prthst=n_prthst.reshape(n_snr * n_t, 1, k_bins),
                m_prthst=m_prthst.reshape(n_snr * n_t, 1, k_bins),
                m_chmprthst=m_chmprthst.reshape(n_snr * n_t, n_chem, k_bins))

            # `data_dict` is an input variable (str) to tensor (torch.tensor) mapping.
            data_dict = {key: torch.from_numpy(varnp).to(device=device, dtype=dtype)
                for key, varnp in datanp_dict.items()}

            sizes = dict(n_snr=n_snr, n_t=n_t, n_chem=n_chem, k_bins=k_bins)
        elif data_id == '01_uniform':
            n_snr = 1000
            # The number of time steps in each scenario
            n_t = 25
            # The number of diameter histogram bins
            k_bins = 20
            # The number of chemical specices
            n_chem = 21

            np_random = np.random.RandomState(seed=12345)
             # The number of particles within each diameter bin
            n_prthst = np_random.rand(n_snr, n_t, k_bins) * 1e5
            assert n_prthst.shape == (n_snr, n_t, k_bins)

            # The mass of each chemical species within the diameter bins
            m_chmprthst = np_random.rand(n_snr, n_t, n_chem, k_bins) * 1e-5
            assert m_chmprthst.shape == (n_snr, n_t, n_chem, k_bins)

            # The mass within each diameter bin (the mass histogram)
            m_prthst = m_chmprthst.sum(axis=-2)
            assert m_prthst.shape == (n_snr, n_t, k_bins)

            datanp_dict = dict(
                n_prthst=n_prthst.reshape(n_snr * n_t, 1, k_bins),
                m_prthst=m_prthst.reshape(n_snr * n_t, 1, k_bins),
                m_chmprthst=m_chmprthst.reshape(n_snr * n_t, n_chem, k_bins))

            # `data_dict` is an input variable (str) to tensor (torch.tensor) mapping.
            data_dict = {key: torch.from_numpy(varnp).to(device=device, dtype=dtype)
                for key, varnp in datanp_dict.items()}

            sizes = dict(n_snr=n_snr, n_t=n_t, n_chem=n_chem, k_bins=k_bins)
        else:
            raise ValueError(f'undefined data id = {data_id}')
        return data_dict, sizes
    
    @staticmethod
    def make_rng(n_seeds, device, dtype):
        # Making the random generator
        rng_seed_list = list(range(0, n_seeds * 1000, 1000))
        rng_seeds = np.array(rng_seed_list)
        rng = BatchRNG(shape=(n_seeds,), lib='torch',
            device=device, dtype=dtype,
            unif_cache_cols=1_000, norm_cache_cols=5_000)
        rng.seed(np.broadcast_to(rng_seeds, rng.shape))
        return rng
    
    @staticmethod
    def make_split(configs, split_id, rng, n_snr, n_t, n_seeds, device, dtype):
        # Making the split configurations
        split_config = deep2hie(configs['split'][split_id])
        split_idxdict = get_snrtspltidxs(split_config, n_snr, n_t, n_seeds,
            rng, device, pop_opts=True)
        trn_spltidxs = split_idxdict['split_idxs']
        return trn_spltidxs

    @staticmethod
    def calc_errs(a_key, a_mb, a_mbref, n_seeds, n_mb, max_diffs):
        # Calclates the max abs and relative errors and inserts them into max_diffs
        abs_diff0, rel_diff0 = max_diffs.get(a_key, (0, 0))
        a_absdiff = (a_mb - a_mbref).abs()
        abs_diff1 = a_absdiff.max().item()
        abs_diff2 = max(abs_diff0, abs_diff1)

        d_a = math.prod(a_mb.shape[2:])
        a_mb2 = a_mb.flatten(2, -1) 
        a_mbref2 = a_mbref.flatten(2, -1)
        assert a_mb2.shape == (n_seeds, n_mb, d_a)
        assert a_mbref2.shape == (n_seeds, n_mb, d_a)

        a_reldiffnum = (a_mb2 - a_mbref2).square().sum(dim=-1).sqrt()
        a_reldiffden = a_mb2.square().sum(dim=-1).sqrt() + a_mbref2.square().sum(dim=-1).sqrt()
        a_reldiff = a_reldiffnum / a_reldiffden
        rel_diff1 = a_reldiff.max().item()
        rel_diff2 = max(rel_diff0, rel_diff1)
        max_diffs[a_key] = (abs_diff2, rel_diff2)

    def test(self, rtol=1e-05, atol=1e-08, equal_nan=False, do_assert=True):
        n_totrnds = int(np.ceil(self.n_trn / self.n_mb))
        isclose = torch.isclose if do_assert else lambda *a, **kw: torch.tensor(True)
        
        max_diffs = dict()
        for i_rnd in range(n_totrnds):
            i1 = i_rnd * self.n_mb
            i2 = min(self.n_trn, i1 + self.n_mb)
            n_mb = i2 - i1

            i_mb = self.split[:, i1:i2]
            assert i_mb.shape == (self.n_seeds, n_mb)

            with torch.no_grad():
                x_mbs, x_mbsref = dict(), dict()
                for varname, vardata in self.data.items():
                    mb_var1 = vardata[i_mb.ravel()]
                    mb_var2 = mb_var1.unflatten(0, (self.n_seeds, n_mb))
                    mb_var3 = nn.Parameter(mb_var2)
                    mb_var4 = deepcopy(mb_var2.detach().clone())
                    mb_var5 = nn.Parameter(mb_var4)
                    x_mbs[varname] = mb_var3
                    x_mbsref[varname] = mb_var5
            
            u_mbs, u_shaper = self.pp.forward(x_mbs, full=True)
            u_mbsref, _ = self.refpp.forward(x_mbsref, full=True)

            for u_name, u_mb in u_mbs.items():
                if u_name.endswith('__NRMLZD'): continue
                u_mbref = u_mbsref[u_name]
                assert u_mb.shape == u_mbref.shape
                assert isclose(u_mb, u_mbref, 
                    rtol=rtol, atol=atol, equal_nan=equal_nan).all()
                self.calc_errs(f'forward/{u_name}', u_mb, u_mbref, n_seeds, n_mb, max_diffs)
            
            xhat_mbs = self.pp.inverse(u_mbs, u_shaper, full=True)
            xhat_mbsref = self.refpp.inverse(u_mbsref, None, full=True)

            for x_name, xhat_mb in xhat_mbs.items():
                if x_name.endswith('__NRMLZD'): continue
                xhat_mbref = xhat_mbsref[x_name]
                assert xhat_mb.shape == xhat_mbref.shape
                assert isclose(xhat_mb, xhat_mbref, 
                    rtol=rtol, atol=atol, equal_nan=equal_nan).all()
                self.calc_errs(f'inverse/{x_name}', xhat_mb, xhat_mbref, n_seeds, n_mb, max_diffs)
            
            for x_name in self.refpp.x_names:
                x_mb = x_mbs[x_name]
                xhat_mb = xhat_mbs[x_name]
                assert xhat_mb.shape == x_mb.shape
                assert isclose(xhat_mb, x_mb, 
                    rtol=rtol, atol=atol, equal_nan=equal_nan).all()
                self.calc_errs(f'roundtrip/{x_name}', x_mb, xhat_mb, n_seeds, n_mb, max_diffs)
            
            pp_loss = torch.cat([xhat_mbs[x_name].sum().reshape(1) 
                for x_name in self.refpp.x_names], dim=0).sum()
            ppref_loss = torch.cat([xhat_mbsref[x_name].sum().reshape(1) 
                for x_name in self.refpp.x_names], dim=0).sum()
            
            pp_loss.backward()
            ppref_loss.backward()
            for x_name in self.refpp.x_names:
                xgrad_mb = x_mbs[x_name].grad
                xgrad_mbref = x_mbsref[x_name].grad
                assert xgrad_mb.shape == xgrad_mbref.shape
                assert isclose(xgrad_mbref, xgrad_mb, 
                    rtol=rtol, atol=atol, equal_nan=equal_nan).all()
                self.calc_errs(f'gradref/{x_name}', xgrad_mbref, xgrad_mb, n_seeds, n_mb, max_diffs)
                self.calc_errs(f'gradunit/{x_name}', torch.ones_like(xgrad_mb), xgrad_mb, n_seeds, 
                    n_mb, max_diffs)

        return max_diffs

class MetricTest:
    def __init__(self, n_seeds=16, n_mbsrc=512, device=None, dtype=None, cdtype=None):
        assert device is not None
        assert dtype is not None
        rng_seeds = (np.arange(n_seeds) * 1000).tolist()
        rng = BatchRNG(shape=(n_seeds,), lib='torch',
            device=device, dtype=dtype,
            unif_cache_cols=0,
            norm_cache_cols=0)
        rng.seed(np.broadcast_to(rng_seeds, rng.shape))
        self.rng = rng
        self.n_mbsrc = n_mbsrc
        self.n_seeds = n_seeds
        self.device = device
        self.dtype = dtype
        self.cdtype = cdtype

    def test(self, rtol=1e-02, atol=1e-02, do_assert=True):
        isclose = np.isclose if do_assert else lambda *a, **kw: True
        n_mbsrc, n_seeds, rng = self.n_mbsrc, self.n_seeds, self.rng
        device, dtype, cdtype = self.device, self.dtype, self.cdtype
        
        n_t, n_rcns, d_ltnt = None, 1, 1
        mu2 = rng.randn(n_seeds, n_mbsrc, n_rcns, d_ltnt)
        sig2 = mu2 * 0.0 + 1.0
        src_mtrcdatas = {'x/x': {'mu': mu2, 'sig': sig2}}

        ref_tv = {0: 0.000, 1: 0.383, 2: 0.683, 3: 0.866, 4: 0.954, 5: 0.987, 6: 0.997}
        ref_hlngr = {0: 0.000, 1: 0.343, 2: 0.627, 3: 0.822, 4: 0.93, 5: 0.978, 6: 0.994}

        for delta_mu in (0, 1, 2, 3, 4, 5, 6):
            trg_mtrcdatas = {'x/x': {'mu': mu2 + delta_mu, 'sig': sig2}}

            for mtrc_id in ('tvd', 'hlngr'):
                if mtrc_id == 'tvd':
                    mtrc_cfg = {'type': 'tv', 'approx': 'is', 'n': 1024, 'mb': 32} 
                    ref_val = ref_tv[delta_mu]
                elif mtrc_id == 'hlngr':
                    mtrc_cfg = {'type': 'hellinger', 'vars': ['h1'],
                        'name/fmt': 'hlngr'}
                    ref_val = ref_hlngr[delta_mu]
                else:
                    raise ValueError(f'undefined mtrc_id={mtrc_id}')

                aero_cstinfo = dict()
                mtrc_val = Evaluation.metric(mtrc_cfg, mtrc_id, src_mtrcdatas, trg_mtrcdatas,
                    aero_cstinfo, n_seeds, n_t, rng, device, dtype, cdtype)[0][mtrc_id].mean().item()

                assert isclose(mtrc_val, ref_val, rtol=rtol, atol=atol, equal_nan=False).all(), dedent(f'''
                    The metric evaluation error is too high:
                        metric id = {mtrc_id}
                        delta mu = {delta_mu}
                        estimated value = {mtrc_val}
                        reference value = {ref_val}
                        error = {np.abs(mtrc_val - ref_val).item()}''')
 

if __name__ == '__main__':
    import netCDF4
    from copy import deepcopy
    from ruamel import yaml as ruyaml
    from partnn.io_cfg import data_dir

    #########################################################
    ################ Metric Evaluation Tests ################
    #########################################################
    device_name = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)
    dtype = torch.float64
    cdtype = torch.complex128

    mtrictester = MetricTest(n_seeds=16, n_mbsrc=512, device=device, dtype=dtype)
    mtrictester.test(rtol=1e-03, atol=1e-03, do_assert=True)
    print('The metric evaluation test was successful.')

    #########################################################
    ################## Pre-Processor Tests ##################
    #########################################################
    pp_id = '02_massnrmlzd'
    data_id = '02_masshist:01_bwchisamp.nc'
    split_id = '01_snrsplt'
    n_seeds = 5
    n_mb = 1024
    eps = 1e-25
    power = 0.25

    tester = PPTest(pp_id=pp_id, data_id=data_id, split_id=split_id, 
        n_seeds=n_seeds, n_mb=n_mb, hparams=dict(eps=eps, power=power), 
        device=device, dtype=dtype, cdtype=cdtype)
    
    diff_dict = tester.test(rtol=1e-03, atol=1e-03, equal_nan=True, 
        do_assert=True)
    
    print('The pre-processing max differences are as follows:\n')
    for oprtn, op_diffdict in hie2deep(diff_dict).items():
        print(f'  {oprtn}:')
        for key, (abserr, relerr) in op_diffdict.items():
            print(f'    {key}{" "* (15-len(key))} -> Abs: {abserr:.4e}, Rel: {relerr:.4e}')
        print('')
