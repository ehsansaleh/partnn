# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=["active-ipynb"]
# import matplotlib.pyplot as plt
# import importlib
# %matplotlib inline
# mplbackend = 'agg'
# assert mplbackend in ('agg', 'retina')
# if (mplbackend == 'agg'):
#     import matplotlib
#     matplotlib.use('Agg')
# elif (mplbackend == 'retina') and (importlib.util.find_spec("matplotlib_inline") is not None):
#     import matplotlib_inline
#     matplotlib_inline.backend_inline.set_matplotlib_formats('retina')
# elif (mplbackend == 'retina'):
#     from IPython.display import set_matplotlib_formats
#     set_matplotlib_formats('retina')
# else:
#     raise ValueError(f'undefined matplotlib backend {mplbackend}')
# plt.rcParams.update({'figure.max_open_warning': 0})
# plt.ioff();

# %% editable=true slideshow={"slide_type": ""}
import os
import torch
import numpy as np
import json
import yaml
import time
import math
import h5py
import fnmatch
import pathlib
import warnings
import datetime
import tensorboardX
from torch import nn
from os.path import isdir
from os.path import exists
import torch.distributions
from textwrap import dedent
import matplotlib.pyplot as plt
from pyinstrument import Profiler

from partnn.io_cfg import PROJPATH
from partnn.io_cfg import data_dir
from partnn.io_cfg import configs_dir
from partnn.io_cfg import results_dir
from partnn.io_cfg import storage_dir
from partnn.io_utils import DataWriter
from partnn.io_utils import eval_formula
from partnn.io_utils import read_ymlhdf
from partnn.io_utils import preproc_cfgdict
from partnn.io_utils import hie2deep, deep2hie
from partnn.io_utils import parse_refs, get_subdict, resio
from partnn.tch_utils import BatchRNG, EMA, profmem, make_nn
from partnn.tch_utils import BatchParamGrouper, BatchLRScheduler, torch_unique
from partnn.aerometrics import AeroMeasures

from ruamel import yaml as ruyaml
from partnn.notebooks.n22_utils import unflatten_cfg, get_snrtspltidxs, load_histdata, Timer
from partnn.notebooks.n22_utils import make_pp, Evaluation, Visualization, DataPrepper, get_spltcfgs 
from partnn.notebooks.n22_utils import get_lbldims, load_riopp, load_riomodel, VAELabelNN, IIDLossMaker


# %% [markdown]
# # Loading Config

# %% tags=["active-ipynb"]
# json_cfgtreeid = '03_prtcls/02_mlpprt.yml'
# # ! rm -rf "./22_vaepart/results/02_mlpprt.h5"
# # ! rm -rf "./22_vaepart/storage/02_mlpprt"
# json_cfgpath = f'{configs_dir}/{json_cfgtreeid}'
# if json_cfgpath.endswith('.json'):
#     with open(json_cfgpath, 'r') as fp:
#         json_cfgdict = json.load(fp, object_pairs_hook=dict)
# elif json_cfgpath.endswith('.yml'):
#     with open(json_cfgpath, "r") as fp:
#         json_cfgdict = dict(ruyaml.safe_load(fp))
# else:
#     raise RuntimeError(f'unknown config extension: {json_cfgpath}')
#
# json_cfgdict['io/config_id'] = '02_mlpprt'
# json_cfgdict['io/results_dir'] = './22_vaepart/results'
# json_cfgdict['io/storage_dir'] = './22_vaepart/storage'
# json_cfgdict['io/tch/device'] = 'cuda:0' if torch.cuda.is_available() else 'cpu'
#
# # Applying the looping processes
# all_cfgdicts1 = preproc_cfgdict(json_cfgdict)[:10]
# # Parsing all the "/ref"-, "/def"-, and "/pop"-ending keys
# all_cfgdicts = [parse_refs(cfgdict, trnsfrmtn='hie', pathsep=' -> ', 
#     cfg_path=json_cfgtreeid) for cfgdict in all_cfgdicts1]
# # Dropping the trailing references section
# all_refdicts = [get_subdict(cfgdict, prefix='refs', pop=True) 
#     for cfgdict in all_cfgdicts]
#
# cfg_dict_input = all_cfgdicts[0]

# %% tags=["active-py"]
def main(cfg_dict_input):


    # %%
    ###############################################################################
    ###################### Processing the Config Dictionary #######################
    ###############################################################################
    cfg_dict = cfg_dict_input.copy()

    #########################################################
    #################### Ignored Options ####################
    #########################################################
    cfgdesc = cfg_dict.pop('desc', None)
    cfgdate = cfg_dict.pop('date', None)

    #########################################################
    ################### Mandatory Options ###################
    #########################################################
    prob_type = cfg_dict.pop('problem')
    rng_seed_list = cfg_dict.pop('rng_seed/list')

    opt_type = cfg_dict.pop('opt/type')
    n_epochs = cfg_dict.pop('opt/epoch')
    n_mb = cfg_dict.pop('opt/mb/n')
    schdlr_type = cfg_dict.pop('opt/schdlr/type', None)
    schdlr_hparams = get_subdict(cfg_dict, 'opt/schdlr', pop=True)
    mb_rndmztn = cfg_dict.pop('opt/rndmztn')
    assert mb_rndmztn in ('shuffle', 'iid', 'none')
    opt_hparams = get_subdict(cfg_dict, 'opt', pop=True)

    #########################################################
    ###################### Data Options #####################
    #########################################################
    data_cfg = get_subdict(cfg_dict, 'data', pop=False)
    data_tree = cfg_dict.pop('data/tree')

    split_cfg = get_subdict(cfg_dict, 'split', pop=True)
    #########################################################
    ################## Neural Spec Options ##################
    #########################################################
    nn_mdlscfg = get_subdict(cfg_dict, 'nn/modules', pop=True)
    pp_mdlscfg = get_subdict(cfg_dict, 'pp/modules', pop=True)
    asgn_mdlscfg = get_subdict(cfg_dict, 'asgn/modules', pop=True)
    metric_mdlscfg = hie2deep(get_subdict(cfg_dict, 'metric/modules', pop=True), maxdepth=1)

    enc_type = cfg_dict.pop('nn/enc/type')
    enc_hparamsraw = get_subdict(cfg_dict, 'nn/enc', pop=True)

    dec_type = cfg_dict.pop('nn/dec/type')
    dec_hparamsraw = get_subdict(cfg_dict, 'nn/dec', pop=True)

    ltnt_dimraw = cfg_dict.pop('nn/ltnt/dim')
    ltnt_sigtnsfm = cfg_dict.pop('nn/ltnt/sig/tnsfm', 'exp')

    ppenc_type = cfg_dict.pop('nn/pp/enc/type')
    ppenc_hparamsraw = get_subdict(cfg_dict, 'nn/pp/enc', pop=True)

    ppdec_type = cfg_dict.pop('nn/pp/dec/type')
    ppdec_hparamsraw = get_subdict(cfg_dict, 'nn/pp/dec', pop=True)

    lblpp_type = cfg_dict.pop('nn/lbl/pp/type', None)
    lblpp_hparamsraw = get_subdict(cfg_dict, 'nn/lbl/pp', pop=True)

    lblnn_type = cfg_dict.pop('nn/lbl/nn/type', None)
    lblnn_hparamsraw = get_subdict(cfg_dict, 'nn/lbl/nn', pop=True)

    #########################################################
    ################### Training Criteria ###################
    #########################################################
    w_kl = cfg_dict.pop('cri/kl/w', None)
    w_klmu = cfg_dict.pop('cri/kl/mu/w', w_kl)
    w_klsig = cfg_dict.pop('cri/kl/sig/w', w_kl)
    rdctn_kl = cfg_dict.pop('cri/kl/rdctn', 'sum')

    rcnst_cri = cfg_dict.pop('cri/rcnst/metric')
    ppcri_inpspc = cfg_dict.pop('cri/rcnst/ppinp/space', 'x')
    ppcri_rndtrp = cfg_dict.pop('cri/rcnst/ppinp/rndtrp', True)
    ppcri_type = cfg_dict.pop('cri/rcnst/pp/type')
    ppcri_hparamsraw = get_subdict(cfg_dict, 'cri/rcnst/pp', pop=True)

    w_indzy = cfg_dict.pop('cri/indzy/w', None)
    n_histindzy = cfg_dict.pop('cri/indzy/hist/n', None)
    cri_indzymtrccfg = get_subdict(cfg_dict, 'cri/indzy/mtrc', pop=True)

    w_mixup = cfg_dict.pop('cri/mixup/w', 0.0)
    alpha_mixup = cfg_dict.pop('cri/mixup/alpha', None)
    has_mixup = (w_mixup != 0.0) and (alpha_mixup is not None)

    #########################################################
    ################## Evaluation Profiles ##################
    #########################################################
    evalcfgs = hie2deep(get_subdict(cfg_dict, 'eval', pop=True), maxdepth=1)
    perfcfgs = hie2deep(get_subdict(cfg_dict, 'perf', pop=True), maxdepth=1)
    vizcfgs = hie2deep(get_subdict(cfg_dict, 'viz', pop=True), maxdepth=1)

    mplopts_cfg = get_subdict(cfg_dict, 'mplopts', pop=True)
    mplopts_cfg = hie2deep(mplopts_cfg, maxdepth=1)

    #########################################################
    ################# I/O Logistics Options #################
    #########################################################
    config_id = cfg_dict.pop('io/config_id')
    results_dir = cfg_dict.pop('io/results_dir')
    storage_dir = cfg_dict.pop('io/storage_dir', None)
    io_avgfrq = cfg_dict.pop('io/avg/frq')
    ioflsh_period = cfg_dict.pop('io/flush/frq')
    chkpnt_period = cfg_dict.pop('io/ckpt/frq')
    device_name = cfg_dict.pop('io/tch/device')
    dtype_name = cfg_dict.pop('io/tch/dtype')
    iomon_period = cfg_dict.pop('io/mon/frq')
    io_cmprssnlvl = cfg_dict.pop('io/cmprssn_lvl')
    eval_bs = cfg_dict.pop('io/eval/bs', None)
    eval_bsnn = cfg_dict.pop('io/eval/bs/nn', eval_bs)
    eval_bspp = cfg_dict.pop('io/eval/bs/pp', eval_bs)
    eval_enbl = cfg_dict.pop('io/eval/enbl', True)
    do_logtb_ = cfg_dict.pop('io/strg/logtb', None)
    do_profile_ = cfg_dict.pop('io/strg/profile', None)
    do_savefigs_ = cfg_dict.pop('io/strg/savefigs', None)

    # disabling the evaluation if necessary.
    evalcfgs = evalcfgs if eval_enbl else dict()
    perfcfgs = perfcfgs if eval_enbl else dict()
    vizcfgs = vizcfgs if eval_enbl else dict()

    dtnow = datetime.datetime.now().isoformat(timespec='seconds')
    cfg_tree = '/'.join(config_id.split('/')[:-1])
    cfg_name = config_id.split('/')[-1]

    # Making sure no other options are left unused.
    if len(cfg_dict) > 0:
        msg_ = 'The following settings were left unused:\n'
        for key, val in cfg_dict.items():
            msg_ += f'  {key}: {val}'
        raise RuntimeError(msg_)

    # %%
    ###############################################################################
    ################ Initializing the Training Classes and Objects ################
    ###############################################################################
    # Derived options and assertions
    eval_bsnn = n_mb if eval_bsnn is None else eval_bsnn
    eval_bspp = n_mb if eval_bspp is None else eval_bspp

    assert (lblnn_type is None) == (lblpp_type is None), dedent(f'''
        A label {"pp" if (lblpp_type is not None) else "nn"} was provided, 
        but a label {"nn" if (lblpp_type is not None) else "pp"} was not 
        specified. Either both or neither of them have to be provided.''')

    is_condvae = (lblpp_type is not None)
    if is_condvae:
        assert lblnn_type is not None

    if is_condvae and (lblnn_type == 'pretrained'):
        lblnn_hparams = lblnn_hparamsraw.copy()
        lblnn_enctype = lblnn_hparams.pop('enc/type')
        assert lblnn_enctype == 'vae', f'unsupported lbl/nn/enc/type = {lblnn_enctype}'
        y_sigscale = lblnn_hparams.pop('enc/sigscale')
        y_nodename = lblnn_hparams.pop('enc/nodename')
    elif is_condvae:
        lblnn_enctype, lblnn_hparams, y_sigscale = 'plain', None, 0
    else:
        lblnn_enctype, lblnn_hparams, y_sigscale = None, None, None

    #########################################################
    ########### I/O-Related Options and Operations ##########
    #########################################################
    name2dtype = dict(float64=(torch.double, torch.complex128), 
        float32=(torch.float32, torch.complex64),
        float16=(torch.float16, torch.complex32))
    tch_device = torch.device(device_name)
    tch_dtype, tch_cdtype = name2dtype[dtype_name]

    has_storage = storage_dir is not None
    do_logtb = bool(do_logtb_) if do_logtb_ is not None else has_storage
    do_profile = bool(do_profile_) if do_profile_ is not None else has_storage
    do_savefigs = bool(do_savefigs_) if do_savefigs_ is not None else has_storage

    do_logtb = do_logtb and has_storage
    do_profile = do_profile and has_storage
    do_savefigs = do_savefigs and has_storage

    assert not(do_logtb) or (storage_dir is not None)
    assert not(do_profile) or (storage_dir is not None)
    assert not(do_savefigs) or (storage_dir is not None)

# %% [markdown]
# # Data Loading

    # %%
    #########################################################
    ################ Defining the VAE Problem ###############
    #########################################################
    assert prob_type == 'vaepart'

    data_path = f'{data_dir}/{data_tree}'
    assert data_path.endswith('.h5')

    h5fp = h5py.File(data_path, "r")

    # The shape variables, Example: 
    #     shapevars = dict(
    #         n_chem=15, n_part=1000, n_bins=19, n_wave=8, n_epshist=100, 
    #         n_tmprtr=100, n_snr=1000, n_snr_all=1000, n_t=25, n_pop=25000, 
    #         n_grid_x=1000, n_grid_y=1, n_grid_z=1, one=1, 
    #         chem_species='SO4,NO3,Cl,NH4,ARO1,ARO2,ALK1,OLE1,API1,Na,OIN,OC,BC,MOC,H2O')
    shapevars = read_ymlhdf(h5fp, 'dims')
    n_snr = shapevars['n_snr']
    n_t = shapevars['n_t']

    chem_species_str = shapevars['chem_species']
    chem_species = chem_species_str.split(',')
    shapevars['chem_species'] = chem_species
    shapevars['device'] = device_name
    shapevars['dtype'] = dtype_name

    data_attrs = read_ymlhdf(h5fp, 'attrs')

    # Example:
    #   data_dims == {
    #     'i_snr': (), i_t': (), i_x': (), i_y': (), i_z': (),
    #     'm_prtchm': (1000, 15), 'n_prt': (1000, 1), 'ccn_cdf': (1, 100), 
    #     'm_chmprthst': (15, 19), 'm_prthst': (1, 19), 'n_prthst': (1, 19),
    #     'qa_pop': (1, 8), 'qa_prt': (1000, 8), 'qacs_prt': (1000, 8),
    #     'qs_pop': (1, 8), 'qs_prt': (1000, 8), 'qscs_prt': (1000, 8),
    #     'frznfrac_tmp': (1, 100), 'logfrznfrac_tmp': (1, 100), 
    #     'surf_bc': (1000, 1), 'surf_oin': (1000, 1)}
    data_dims = {key: tuple(varnp.shape)[1:] for key, varnp in h5fp.items()
        if data_attrs.get(f'{key}/shape/str', (None,))[0] == 'n_pop'}

    # Preparing the data hash for later caching
    hash_data = {key: data_cfg for key in h5fp}

    # Computing `d_histbins` for the purposes of axis ticks in plots
    n_bins = shapevars['n_bins']
    diam_low = h5fp['diam_low'][:].item()
    diam_high = h5fp['diam_high'][:].item()
    diam_logbase = h5fp['diam_logbase'][:].item()
    logdiam_low, logdiam_high = np.log(diam_low) / diam_logbase, np.log(diam_high) / diam_logbase
    logdiams = np.linspace(logdiam_low, logdiam_high, n_bins + 1, endpoint=True)
    d_histbins = np.exp(logdiams * diam_logbase)
    assert d_histbins.shape == (n_bins + 1,)

    # `aero_cstinfo` contains the aerosol constants used within ccn, inp, and optical evaluation metrics
    n_chem = shapevars['n_chem']
    n_part = shapevars['n_part']
    n_wave = shapevars['n_wave']

    rho_chmnp = h5fp['chem_rho'][:]
    assert rho_chmnp.shape == (n_chem,)
    kappa_chmnp = h5fp['chem_kappa'][:]
    assert kappa_chmnp.shape == (n_chem,)
    len_wvnp = h5fp['wave_len'][:]
    assert len_wvnp.shape == (n_wave,)
    refr_wvchmrealnp = h5fp['chem_refr_real'][:]
    assert refr_wvchmrealnp.shape == (n_wave, n_chem)
    refr_wvchmimagnp = h5fp['chem_refr_imag'][:]
    assert refr_wvchmimagnp.shape == (n_wave, n_chem)
    refr_wvchmrefnp = refr_wvchmrealnp + 1j * refr_wvchmimagnp
    assert refr_wvchmrefnp.shape == (n_wave, n_chem)
                
    aero_cstinfo = dict(chem_species=chem_species, n_chem=n_chem, n_bins=n_bins, n_part=n_part)
    aero_cstinfo['rho_chm'] = torch.from_numpy(rho_chmnp).to(device=tch_device, dtype=tch_dtype)
    aero_cstinfo['kappa_chm'] = torch.from_numpy(kappa_chmnp).to(device=tch_device, dtype=tch_dtype)
    aero_cstinfo['len_wvref'] = torch.from_numpy(len_wvnp).to(device=tch_device, dtype=tch_dtype)
    aero_cstinfo['refr_wvchmref'] = torch.from_numpy(refr_wvchmrefnp).to(device=tch_device, dtype=tch_cdtype)

    # Adding the aeromeasures pp config
    pp_mdlscfg['aeromeasures'] = {'class': AeroMeasures, 'hparams/aerocst': aero_cstinfo}

    # For all intents and purposes, h5fp serves the same way as a key to numpy array dictionary
    datanp_dict = h5fp

    # %%
    datanp_dict = {kkk: val[:] for kkk, val in datanp_dict.items()}

    # %%
    #########################################################
    ########### Constructing the Batch RNG Object ###########
    #########################################################
    n_seeds = len(rng_seed_list)
    rng_seeds = np.array(rng_seed_list)
    rng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=1_000_000,
        norm_cache_cols=5_000_000)
    rng.seed(np.broadcast_to(rng_seeds, rng.shape))

    erng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0,
        norm_cache_cols=0)
    erng.seed(np.broadcast_to(rng_seeds, erng.shape))

    vrng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0,
        norm_cache_cols=0)
    vrng.seed(np.broadcast_to(rng_seeds, vrng.shape))

    mrng = BatchRNG(shape=(n_seeds,), lib='torch',
        device=tch_device, dtype=tch_dtype,
        unif_cache_cols=0,
        norm_cache_cols=0)
    mrng.seed(np.broadcast_to(rng_seeds, mrng.shape))

    #########################################################
    ######## Instantiating Train/Test Split Indecis #########
    #########################################################
    split_vars = split_cfg['vars']
    split_idxdict = get_snrtspltidxs(split_cfg, n_snr, n_t, n_seeds,
        rng, tch_device, pop_opts=True)
    trn_spltidxs = split_idxdict['split_idxs']
    n_trn = trn_spltidxs.shape[-1]
    n_tst = n_snr * n_t - n_trn
    assert trn_spltidxs.shape == (n_seeds, n_trn)
    tst_spltidxs = split_idxdict['negsplit_idxs']
    assert tst_spltidxs.shape == (n_seeds, n_tst)

    # %%
    #######################################
    #  Constructing the NN pre-processor  #
    #######################################
    ppenc_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in ppenc_hparamsraw.items()}
    enc_pp = make_pp(ppenc_type, ppenc_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
    enc_pp.infer(datanp_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
    ue_dims = enc_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
    xe_dims = enc_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

    ppdec_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in ppdec_hparamsraw.items()}
    dec_pp = make_pp(ppdec_type, ppdec_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
    dec_pp.infer(datanp_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
    ud_dims = dec_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
    xd_dims = dec_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

    datannpp_dims = {**data_dims, **ud_dims}
    ppcri_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in ppcri_hparamsraw.items()}
    cri_pp = make_pp(ppcri_type, ppcri_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
    cri_pp.infer(datanp_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
    u_cridims = cri_pp.get_dims(datannpp_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
    x_cridims = cri_pp.get_dims(datannpp_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

    lbl_pp, ul_dims, xl_dims = None, dict(), dict()
    if is_condvae and (lblpp_type == 'pretrained'):
        fpidx_lblpp = lblpp_hparamsraw.pop('fpidx')
        resdir_lblpp = lblpp_hparamsraw.pop('resdir', None)
        rio_lbl = resio(fpidx=fpidx_lblpp, resdir=resdir_lblpp)
        lbl_ppinfo = load_riopp(rio_lbl, device_name, dtype_name, tch_device, tch_dtype, tch_cdtype, pp_type='enc')
        rng_seed_list_lblpp, lbl_pp = lbl_ppinfo.pop('rng_seed_list'), lbl_ppinfo.pop('pp')
        assert rng_seed_list == rng_seed_list_lblpp, f'loaded label pp rng seeds do not match'
        xl_dims = lbl_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)
        ul_dims = lbl_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
        assert len(lblpp_hparamsraw) == 0, f'unused options: {lblpp_hparamsraw}'
    elif is_condvae:
        lblpp_hparams = {hparam: eval_formula(val, shapevars, catch=True) for hparam, val in lblpp_hparamsraw.items()}
        lbl_pp = make_pp(lblpp_type, lblpp_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)
        lbl_pp.infer(datanp_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
        xl_dims = lbl_pp.get_dims(data_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)
        ul_dims = lbl_pp.get_dims(data_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)

    # The dimensions of [ecnoder's inputs + decoder's inputs + labels]
    xedl_dims = {**xe_dims, **xd_dims, **xl_dims}
    # The dimensions of [ecnoder's inputs + labels]
    xel_dims = {**xe_dims, **xl_dims}
    # The dimensions of [decoder's inputs + labels]
    xdl_dims = {**xe_dims, **xl_dims}

    for x_node, x_dtup in xedl_dims.items():
        x_dtupenc = xe_dims.get(x_node, x_dtup)
        assert x_dtup == x_dtupenc
        x_dtupdec = xd_dims.get(x_node, x_dtup)
        assert x_dtup == x_dtupdec
        x_dtuplbl = xl_dims.get(x_node, x_dtup)
        assert x_dtup == x_dtuplbl


    # %%
    #########################################################
    ########### Evaluation Point Sampling Options ###########
    #########################################################
    eparams = dict()
    for eid, eopts in evalcfgs.items():
        epp_cfg = dict()

        # TODO: The inferred parameters must be run on $f^-1(f(x))$ and
        # not $x$ directly for the baseline shifts and scales to be most
        # accurate in practice. That being said, this seems good enough.
        epp_type = eopts.pop('pp/type')
        epp_inpspc = eopts.pop('ppinp/space', 'x')
        epp_rndtrp = eopts.pop('ppinp/rndtrp', True)
        epp_hparamsraw = get_subdict(eopts, 'pp', pop=True)
        
        epp_hparams = {hparam: eval_formula(val, shapevars, catch=True) 
            for hparam, val in epp_hparamsraw.items()}
        epp = make_pp(epp_type, epp_hparams, pp_mdlscfg, tch_device, tch_dtype, tch_cdtype)

        epp.infer(datanp_dict, trn_spltidxs, n_seeds, eval_bspp, hash_data=hash_data)
        uv_dims = epp.get_dims(datannpp_dims, types='output', n_seeds=n_seeds, n_mb=1, n_dims=2)
        xv_dims = epp.get_dims(datannpp_dims, types='input', n_seeds=n_seeds, n_mb=1, n_dims=2)

        # Getting the split configuration dictionary
        n_evlpnts = eopts.pop('n', None)
        e_sigscale = eopts.pop('sigscale', None)
        e_spltcfgsraw = get_subdict(eopts, 'splits', pop=True)
        e_spltcfgs = get_spltcfgs(e_spltcfgsraw, eid, n_evlpnts, e_sigscale, is_condvae, lblnn_enctype)

        # Determining the split data indecis `e_spltidxs`
        for e_split, e_spltcfg in e_spltcfgs.items():
            if e_split == 'train':
                e_spltidxs = trn_spltidxs
            elif e_split == 'test':
                e_spltidxs = tst_spltidxs
            elif e_split == 'snr':
                with torch.no_grad():
                    e_snridxlst_ = eopts.pop('snr/idxs')
                    n_esnridxlst = len(e_snridxlst_)

                    assert all(ii_snr < n_snr for ii_snr in e_snridxlst_)
                    assert all(ii_snr >= -n_snr for ii_snr in e_snridxlst_)

                    # Translating negative scenario indices
                    e_snridxlst = [ii_snr % n_snr for ii_snr in e_snridxlst_]
                    n_esplit = n_esnridxlst * n_t

                    e_splitidxs1 = torch.tensor(e_snridxlst).to(device=tch_device, dtype=torch.long)
                    assert e_splitidxs1.shape == (n_esnridxlst,)

                    ii_t = torch.arange(n_t, device=tch_device, dtype=torch.long)
                    assert ii_t.shape == (n_t,)

                    e_splitidxs2 = n_t * e_splitidxs1.reshape(n_esnridxlst, 1) + ii_t.reshape(1, n_t)
                    assert e_splitidxs2.shape == (n_esnridxlst, n_t)

                    e_splitidxs3 = e_splitidxs2.reshape(n_esplit)
                    assert e_splitidxs3.shape == (n_esplit,)

                    e_spltidxs = e_splitidxs3.reshape(1, n_esplit).expand(n_seeds, n_esplit)
                    assert e_spltidxs.shape == (n_seeds, n_esplit)
            elif e_split == 'normal':
                e_spltidxs = None
            else:
                raise ValueError(f'undefined e_split={e_split}')
            
            assert e_spltcfg['idxs'] is None
            e_spltcfg['idxs'] = e_spltidxs
            assert e_spltcfg['iidxs'] is None
            e_spltcfg['iidxs'] = None

        eparams[eid] = dict()
        eparams[eid]['pp'] = epp
        eparams[eid]['inpspc'] = epp_inpspc
        eparams[eid]['rndtrp'] = epp_rndtrp
        eparams[eid]['udims'] = uv_dims
        eparams[eid]['xdims'] = xv_dims
        eparams[eid]['frq'] = eopts.pop('frq')
        eparams[eid]['rndmztn'] = eopts.pop('rndmztn')
        eparams[eid]['store'] = get_subdict(eopts, 'store', pop=True)
        eparams[eid]['splits'] = e_spltcfgs

        assert len(eopts) == 0, f'unused eval options for {eid}: {eopts}'


    # %%
    # Piping the matplotlib-related options dictionary
    for vid, vcfg in vizcfgs.items():
        mplopt_tmplts = vcfg.pop('mplopts/plan', [])
        if isinstance(mplopt_tmplts, str):
            mplopt_tmplts = [mplopt_tmplts]
        v_mplopts = dict()
        for mpo_tmplt in mplopt_tmplts:
            assert mpo_tmplt in mplopts_cfg, dedent(f'''
                The Matplotlib option template "{mpo_tmplt}" 
                visualization "{vid}". However, this template
                was never defined:
                    available templates: {mplopts_cfg.keys()}
            ''')
            v_mplopts.update(mplopts_cfg[mpo_tmplt])
        mpo_ovrrd = get_subdict(vcfg, 'mplopts', pop=True)
        mpo_ovrrd = dict() if mpo_ovrrd is None else mpo_ovrrd.copy()
        v_mplopts.update(mpo_ovrrd)
        v_mplopts['type'] = vcfg['type']
        vcfg['mplopts'] = v_mplopts
        vcfg.setdefault('splits', None)

    # Converting str values to tuples
    for prf_id in perfcfgs:
        prf_cartcfg = perfcfgs[prf_id]
        assert set(prf_cartcfg) == {'srctrg', 'asgn', 'node', 'metric', 'stat'}
        prf_cartcfg = {optn: (val,) if isinstance(val, str) else val 
            for optn, val in prf_cartcfg.items()}
        perfcfgs[prf_id] = prf_cartcfg

    # %%
    # The latent dimension
    ltnt_dim = eval_formula(ltnt_dimraw, shapevars, catch=True)

    # Initializing the encoder
    enc_hparams = dict()
    for hparam, val in enc_hparamsraw.items():
        enc_hparams[hparam] = eval_formula(val, shapevars, catch=True)

    encoder = make_nn(name=enc_type, hparams=enc_hparams, config=nn_mdlscfg, 
        n_seeds=n_seeds, rng=rng, device=tch_device, dtype=tch_dtype)

    z_dims = None

    # Initializing the decoder
    dec_hparams = dict()
    for hparam, val in dec_hparamsraw.items():
        dec_hparams[hparam] = eval_formula(val, shapevars, catch=True)

    decoder = make_nn(name=dec_type, hparams=dec_hparams, config=nn_mdlscfg, 
        n_seeds=n_seeds, rng=rng, device=tch_device, dtype=tch_dtype)

    # Setting the optimizer
    model_parameters = list(encoder.parameters()) + list(decoder.parameters())

    # Getting the seed-splitted parameters
    if schdlr_type is not None:
        pgrouper = BatchParamGrouper(params=model_parameters, shape=(n_seeds,))
        optim_params = pgrouper.param_groups
    else:
        pgrouper = None
        optim_params = model_parameters

    # Instantiating the optimizer
    if opt_type == 'adam':
        opt = torch.optim.Adam(optim_params, **opt_hparams)
    elif opt_type == 'sgd':
        opt = torch.optim.SGD(optim_params, **opt_hparams)
    else:
        raise NotImplementedError(f'opt/dstr="{opt_type}" not implmntd')

    # Instantiating the learning rate scheduler
    if schdlr_type is not None:
        scheduler = BatchLRScheduler(kind=schdlr_type, hparams=schdlr_hparams, 
            optimizer=opt, n_seeds=n_seeds, n_grps=pgrouper.n_grps)
    else:
        scheduler = None

    # Setting up the mixup beta distribution generator
    if has_mixup:
        alpha_mixuptch = torch.tensor([alpha_mixup], device=tch_device, dtype=tch_dtype)
        assert alpha_mixuptch.shape == (1,)

        beta_genr = torch.distributions.Beta(alpha_mixuptch, alpha_mixuptch)
    else:
        beta_genr, alpha_mixuptch = None, None

    # %%
    # Initializing the label network
    lblnn, y_dims = None, dict()
    if is_condvae and (lblnn_type == 'pretrained'):
        fpidx_lblnn = lblnn_hparams.pop('fpidx')
        resdir_lblnn = lblnn_hparams.pop('resdir', None)

        rio_lblnn = resio(fpidx=fpidx_lblnn, resdir=resdir_lblnn)
        lblnn_info = load_riomodel(rio_lblnn, 'enc', device_name, dtype_name, tch_device, tch_dtype)

        rng_seed_list_lblnn = lblnn_info.pop('rng_seed_list')
        assert rng_seed_list == rng_seed_list_lblnn, f'loaded label nn rng seeds do not match'

        assert lblnn_enctype == 'vae'
        encoder_lblnn = lblnn_info.pop('model')
        lblnn = VAELabelNN(encoder=encoder_lblnn, **lblnn_info, y_node=y_nodename)

        tmprng = BatchRNG(shape=(n_seeds,), lib='torch', device=tch_device, dtype=tch_dtype,
            unif_cache_cols=0, norm_cache_cols=0)
        tmprng.seed(np.broadcast_to(rng_seeds, rng.shape))
        lblnnkwsfk = dict(n_seeds=n_seeds, n_mb=n_mb, rng=tmprng, sigscale=y_sigscale)  
        y_dims = get_lbldims(lblnn, ul_dims, n_seeds, n_mb, tch_device, tch_dtype, lblnnkws=lblnnkwsfk)
        
        assert len(lblnn_hparams) == 0, f'unused options: {lblnn_hparams}'
    elif is_condvae:
        assert (lblnn_enctype, lblnn_hparams) == ('plain', None)
        lblnn_hparams = dict()
        for hparam, val in lblnn_hparamsraw.items():
            lblnn_hparams[hparam] = eval_formula(val, shapevars, catch=True)
            
        lblnn = make_nn(name=lblnn_type, hparams=lblnn_hparams, config=nn_mdlscfg,
            n_seeds=n_seeds, rng=rng, device=tch_device, dtype=tch_dtype)
        y_dims = get_lbldims(lblnn, ul_dims, n_seeds, n_mb, tch_device, tch_dtype)
        assert len(y_dims) > 0

    if w_indzy is not None:
        assert is_condvae, dedent(f'''
            An iid z and y loss weight was provided, but the 
            model is non-conditional. The iid z and y loss can 
            only make sense for conditional vae models:
                cri/indzy/w: {w_indzy}''')
        iid_lossfinder = IIDLossMaker(n_histindzy, cri_indzymtrccfg, rng, tch_device, tch_dtype)
        # If you're in absolute need of efficiency, you can set `has_indzyloss = (w_indzy != 0)`
        has_indzyloss = True
    else:
        assert n_histindzy is None, f'undefined cri/indzy/hist/n={n_histindzy}'
        assert len(cri_indzymtrccfg) == 0, dedent(f'''
            unused cri/indzy/mtrc options: {cri_indzymtrccfg}''')
        iid_lossfinder, has_indzyloss = None, False


    # %%
    if results_dir is not None:
        pathlib.Path(os.sep.join([results_dir, cfg_tree])
                    ).mkdir(parents=True, exist_ok=True)

    if storage_dir is not None:
        cfgstrgpnt_dir = os.sep.join([storage_dir, cfg_tree, cfg_name])
        pathlib.Path(cfgstrgpnt_dir).mkdir(parents=True, exist_ok=True)
        strgidx = sum(isdir(f'{cfgstrgpnt_dir}/{x}') for x in os.listdir(cfgstrgpnt_dir))
        dtnow_ = dtnow[2:].replace('-', '').replace(':', '').replace('.', '')
        cfgstrg_dir = f'{cfgstrgpnt_dir}/{strgidx:02d}_{dtnow_}'
        pathlib.Path(cfgstrg_dir).mkdir(parents=True, exist_ok=True)

        yaml.Dumper.ignore_aliases = lambda *args: True
        with open(f'{cfgstrg_dir}/config.yml', 'w') as fp:
            yaml.dump(cfg_dict_input, fp, sort_keys=False, default_flow_style=None)
    else:
        cfgstrg_dir = None

    if do_savefigs:
        assert cfgstrg_dir is not None
        fig_dir = f'{cfgstrg_dir}/figures'
        pathlib.Path(fig_dir).mkdir(parents=True, exist_ok=True)
    else:
        fig_dir = None

    if do_logtb:
        import logging
        if 'tbwriter' in locals():
            locals()['tbwriter'].close()
        tbwriter = tensorboardX.SummaryWriter(cfgstrg_dir)
        logging.getLogger("tensorboardX.x2num").setLevel(logging.CRITICAL)
    else:
        tbwriter = None
        
    if do_profile:
        profiler = Profiler()
        profiler.start()
    else:
        profiler = None

    # %%
    # Data writer construction
    hdfpth = None
    if results_dir is not None:
        hdfpth = f'{results_dir}/{cfg_tree}/{cfg_name}.h5'
    dwriter = DataWriter(flush_period=ioflsh_period*n_seeds, 
        compression_level=io_cmprssnlvl)

    # Data Prepper Construction
    dprepper = DataPrepper(n_seeds, rng_seed_list, io_avgfrq, chkpnt_period,
        iomon_period, device_name, dtype_name, dtnow, cfg_tree, cfg_name)

    # Identifying the hyper-parameter from etc config columns
    hppats = ['problem', 'opt/*', 'data/tree', 'split/*', 'nn/*', 'cri/*', 'eval/*']

    etcpats = ['desc', 'date', 'rng_seed/list', 'data/*', 'io/*', 'eval/*/viz/*/mplopts/*', 
        'nn/modules/*', 'pp/modules/*', 'mplopts/*', 'perf/*', 'metric/modules/*', 
        'asgn/modules/*', 'viz/*']

    # Making the hyper-param and etc dictionaries
    hp_dict, etc_dict = DataPrepper.make_hpetcopts(cfg_dict_input, 
        hppats, etcpats, n_seeds, device_name, dtnow)

    # Pushing the static output data once
    # TODO: store the label static nn parameters
    pp_statics = {'mdl/cri': cri_pp, 'mdl/enc': enc_pp, 'mdl/dec': dec_pp}
    pp_statics.update({f'eval/{eid}': eopts['pp'] for eid, eopts in eparams.items()})
    if is_condvae:
        pp_statics.update({'mdl/lbl': lbl_pp})
    dprepper.push_static(hp_dict, etc_dict, cfg_dict_input, shapevars, pp_statics)

    # Evaluation tools
    trn_sttime = time.time()
    dprepper.trn_sttime = trn_sttime
    last_perfdict = dict()
    ema = EMA(gamma=0.999, gamma_sq=0.998)
    timer = Timer()

# %% [markdown]
# # Training Loop

    # %% editable=true slideshow={"slide_type": ""}
    ###############################################################################
    ######################## Training and Evaluation Loop #########################
    ###############################################################################
    ii_trn = None
    for epoch in range(n_epochs + 1):
        timer.tic('train')

        encoder.train()
        decoder.train()

        opt.zero_grad()

        #######################################
        #  Sampling training data and indices #
        #######################################

        # First, we sample random indices to select from `trn_spltidxs`.
        iistrt, iiend = epoch * n_mb, (epoch + 1) * n_mb
        ii_mbidxs_ = torch.arange(iistrt, iiend, device=tch_device).remainder(n_trn).to(dtype=torch.long)
        assert ii_mbidxs_.shape == (n_mb,)

        renew_ii = (ii_mbidxs_ == 0).any().item()
        if renew_ii and (mb_rndmztn == 'none'):
            ii_trn_ = torch.arange(n_trn, device=tch_device, dtype=torch.long)
            assert ii_trn_.shape == (n_trn,)

            ii_trn = ii_trn_.reshape(1, n_trn).expand(n_seeds, n_trn)
            assert ii_trn.shape == (n_seeds, n_trn)
        if renew_ii and (mb_rndmztn == 'shuffle'):
            ii_trn_ = rng.rand(n_seeds, n_trn)
            assert ii_trn_.shape == (n_seeds, n_trn)

            ii_trn = ii_trn_.argsort(dim=-1).detach()
            assert ii_trn.shape == (n_seeds, n_trn)
        elif renew_ii and (mb_rndmztn == 'iid'):
            ii_trn = (rng.rand(n_seeds, n_trn) * n_trn).to(torch.long)
            assert ii_trn.shape == (n_seeds, n_trn)
        elif not renew_ii:
            assert ii_trn.shape == (n_seeds, n_trn)
        else:
            raise ValueError(f'undefined mini-batch randomization = {mb_rndmztn}')

        ii_mbidxs = ii_mbidxs_.reshape(1, n_mb).expand(n_seeds, n_mb)
        assert ii_mbidxs.shape == (n_seeds, n_mb)

        ii_mb = torch.take_along_dim(ii_trn, ii_mbidxs, dim=-1)
        assert ii_mb.shape == (n_seeds, n_mb)

        # First, we sample random indices to select from `trn_spltidxs`.
        ii_mb = (rng.rand(n_seeds, n_mb) * n_trn).to(torch.long)
        assert ii_mb.shape == (n_seeds, n_mb)

        # Next, we select those from the training split indices.
        i_mb = torch.take_along_dim(trn_spltidxs, ii_mb, dim=1)
        assert i_mb.shape == (n_seeds, n_mb)
        i_mbrvld = i_mb.ravel()
        assert i_mbrvld.shape == (n_seeds * n_mb,)
        i_mbunqs, i_mbinvrs, i_mbcnts, i_mbindx = torch_unique(i_mbrvld)
        assert i_mbinvrs.shape == (n_seeds * n_mb,)
        i_mbunqsnp = i_mbunqs.detach().cpu().numpy()
        i_mbinvrsnp = i_mbinvrs.detach().cpu().numpy()
        assert i_mbinvrsnp.shape == (n_seeds * n_mb,)

        # Now, we collect these indices from the full data.
        xedl_mbs = dict()
        for x_node, (n_xchnls, n_xlen) in xedl_dims.items():
            x_mb_ = datanp_dict[x_node][i_mbunqsnp][i_mbinvrsnp]
            assert x_mb_.shape == (n_seeds * n_mb, n_xchnls, n_xlen)
            x_mbth_ = (x_mb_ if torch.is_tensor(x_mb_) else 
                torch.tensor(x_mb_).to(device=tch_device, dtype=tch_dtype))
            assert x_mbth_.shape == (n_seeds * n_mb, n_xchnls, n_xlen)
            x_mb = x_mbth_.reshape(n_seeds, n_mb, n_xchnls, n_xlen)
            assert x_mb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)
            xedl_mbs[x_node] = x_mb

        # The encoder's pp input mini-batches
        xe_mbs = {x_node: xedl_mbs[x_node] for x_node in xe_dims}
        # The decoder's pp input mini-batches
        xd_mbs = {x_node: xedl_mbs[x_node] for x_node in xd_dims}
        # The labeler's pp input mini-batches
        xl_mbs = {x_node: xedl_mbs[x_node] for x_node in xl_dims}

        # Next, we apply the input pre-processing transformations
        # before feeding it to the encoder.
        ue_mbs, ue_shaper = enc_pp.forward(xe_mbs, full=False)
        for u_node, (n_uchnls, n_ulen) in ue_dims.items():
            u_mb = ue_mbs[u_node]
            assert u_mb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

        #######################################
        #         Applying the Labeler        #
        #######################################
        if is_condvae:
            with torch.no_grad():
                ul_mbs, ul_shaper = lbl_pp.forward(xl_mbs, full=False)
                for u_node, (n_uchnls, n_ulen) in ul_dims.items():
                    u_mb = ul_mbs[u_node]
                    assert u_mb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

                lblnnkws = dict()
                if lblnn_enctype == 'vae':
                    lblnnkws.update(n_seeds=n_seeds, n_mb=n_mb, rng=rng, sigscale=y_sigscale)
                assert lblnn_enctype in ('plain', 'vae')
            
                y_mbs = lblnn(**ul_mbs, **lblnnkws)
                assert isinstance(y_mbs, dict)
        else:
            y_mbs = dict()

        for y_node, (n_ychnls, n_ylen) in y_dims.items():
            y_mb = y_mbs[y_node]
            assert y_mb.shape == (n_seeds, n_mb, n_ychnls, n_ylen)

        #######################################
        #         Applying the Encoder        #
        #######################################
        # Encoding the pre-processed input
        musig_mbs = encoder(**ue_mbs, **y_mbs)
        assert isinstance(musig_mbs, dict), dedent(f'''
            The encoder should output a dictionary: {musig_mbs}''')

        # Getting the mu values out of the encoder
        mu_mbs = get_subdict(musig_mbs, 'mu', sep='.', pop=True)
        z_dims = {z_node: tuple(mu_mb.shape[2:]) for z_node, mu_mb 
            in mu_mbs.items()} if (z_dims is None) else z_dims
        assert set(mu_mbs) == set(z_dims), dedent(f'''
            mu nodes are different from the usual:
                extra nodes: {set(mu_mbs) - set(z_dims)}
                missing nodes: {set(z_dims) - set(mu_mbs)}''')
        for z_node, z_dim in z_dims.items():
            mu_mb = mu_mbs[z_node]
            assert mu_mb.shape == (n_seeds, n_mb, *z_dim)
        
        # Getting the log-sigma values out of the encoder
        logsigtld_mbs = get_subdict(musig_mbs, 'sig', sep='.', pop=True)
        assert set(logsigtld_mbs) == set(z_dims), dedent(f'''
            sigma nodes are different from the usual:
                extra nodes: {set(logsigtld_mbs) - set(z_dims)}
                missing nodes: {set(z_dims) - set(logsigtld_mbs)}''')
        for z_node, z_dim in z_dims.items():
            logsigtld_mb = logsigtld_mbs[z_node]
            assert logsigtld_mb.shape == (n_seeds, n_mb, *z_dim)

        # Making sure the encoder naming convention is followed
        assert len(musig_mbs) == 0, dedent(f'''
            The encoder output keys should be either `mu.*` or `sig.*`:
                offending items: {musig_mbs}''')
        
        # Sanity checking the total latent dimensionality
        assert sum(math.prod(z_dim) for z_dim in z_dims.values()) == ltnt_dim, dedent(f'''
            The total number of latent dimensions does not match:
                actual dims: {sum(math.prod(z_dim) for z_dim in z_dims.values())}
                expected dims: {ltnt_dim}''')

        # Applying the latent sigma transformation
        logsigma_mbs = dict()
        for z_node, z_dim in z_dims.items():
            if ltnt_sigtnsfm == 'exp':
                logsigma_mb = logsigtld_mbs[z_node]
                assert logsigma_mb.shape == (n_seeds, n_mb, *z_dim)
            elif ltnt_sigtnsfm == 'explin':
                logsigtlde_mb = logsigtld_mbs[z_node]
                assert logsigtlde_mb.shape == (n_seeds, n_mb, *z_dim)

                logsigma_mb = torch.where(logsigtlde_mb < 0, logsigtlde_mb, 
                    (logsigtlde_mb + 1).log())
                assert logsigma_mb.shape == (n_seeds, n_mb, *z_dim)
            else:
                raise ValueError(f'undefined ltnt_sigtnsfm={ltnt_sigtnsfm}')
            logsigma_mbs[z_node] = logsigma_mb

        #######################################
        ##### Computing the IID Z/Y Loss ######
        #######################################
        loss_indzy, loss_indzyunit = 0, 0
        if has_indzyloss:
            loss_indzyunit = iid_lossfinder(z_mbs=mu_mbs, 
                y_mbs=y_mbs, n_seeds=n_seeds, n_mb=n_mb)
            assert loss_indzyunit.shape == (n_seeds,)
            loss_indzy = w_indzy * loss_indzyunit
            assert loss_indzy.shape == (n_seeds,)

        #######################################
        #     Computing the KL-Divergences    #
        #######################################
        # Computing the KL-Divergences to the Normal prior in the latent space
        dimscl_kl = {'sum': 1, 'mean': ltnt_dim}[rdctn_kl]
        z_musqmbs, z_lossklmus = dict(), dict()
        for z_node, z_dim in z_dims.items():
            mu_mb = mu_mbs[z_node]
            assert mu_mb.shape == (n_seeds, n_mb, *z_dim)

            z_musqmb = mu_mb.square()
            assert z_musqmb.shape == (n_seeds, n_mb, *z_dim)

            kl_mumb = z_musqmb.sum(dim=-1) / (2.0 * dimscl_kl)
            assert kl_mumb.shape == (n_seeds, n_mb)

            z_lossklmu = kl_mumb.mean(dim=-1)
            assert z_lossklmu.shape == (n_seeds,)

            z_musqmbs[z_node] = z_musqmb
            z_lossklmus[z_node] = z_lossklmu

        z_sigmasqmbs, z_lossklsigs = dict(), dict()
        for z_node, z_dim in z_dims.items():
            logsigma_mb = logsigma_mbs[z_node]
            assert logsigma_mb.shape == (n_seeds, n_mb, *z_dim)
            
            logvar_mb = (2 * logsigma_mb)
            assert logvar_mb.shape == (n_seeds, n_mb, *z_dim)

            z_sigmasqmb = logvar_mb.exp()
            assert z_sigmasqmb.shape == (n_seeds, n_mb, *z_dim)

            kl_sigmb = (z_sigmasqmb - 1.0 - logvar_mb).sum(dim=-1) / (2.0 * dimscl_kl)
            assert kl_sigmb.shape == (n_seeds, n_mb)

            z_lossklsig = kl_sigmb.mean(dim=-1)
            assert z_lossklsig.shape == (n_seeds,)

            z_sigmasqmbs[z_node] = z_sigmasqmb
            z_lossklsigs[z_node] = z_lossklsig

        z_losskls = dict()
        for z_node, z_dim in z_dims.items():
            z_losskl = z_lossklmus[z_node] * w_klmu + z_lossklsigs[z_node] * w_klsig
            assert z_losskl.shape == (n_seeds,)
            z_losskls[z_node] = z_losskl

        #######################################
        #       Sampling the Latent Space     #
        #######################################

        z_mbs = dict()
        for z_node, z_dim in z_dims.items():
            mu_mb = mu_mbs[z_node]
            assert mu_mb.shape == (n_seeds, n_mb, *z_dim)

            logsigma_mb = logsigma_mbs[z_node]
            assert logsigma_mb.shape == (n_seeds, n_mb, *z_dim)

            eps_mb = rng.randn(n_seeds, n_mb, *z_dim)
            assert eps_mb.shape == (n_seeds, n_mb, *z_dim)

            z_mb = mu_mb + eps_mb * logsigma_mb.exp()
            assert z_mb.shape == (n_seeds, n_mb, *z_dim)

            z_mbs[z_node] = z_mb
        
        #######################################
        #     Applying Mixup Augmentation     #
        #######################################
        if has_mixup:
            j_mixup1 = rng.rand(n_seeds, n_mb).argsort(dim=-1)
            assert j_mixup1.shape == (n_seeds, n_mb)

            lam_mixup1 = rng(beta_genr, n_seeds, n_mb)
            assert lam_mixup1.shape == (n_seeds, n_mb)
            lam_mixup2 = 1.0 - lam_mixup1
            assert lam_mixup2.shape == (n_seeds, n_mb)
            lam_mixup3 = torch.maximum(lam_mixup1, lam_mixup2)
            assert lam_mixup3.shape == (n_seeds, n_mb)
            lam_mixup4 = torch.minimum(lam_mixup1, lam_mixup2)
            assert lam_mixup4.shape == (n_seeds, n_mb)

            z_mxdmbs = dict()
            for z_node, z_dim in z_dims.items():
                z_1dims = [1] * len(z_dim)

                j_mixup2 = j_mixup1.reshape(n_seeds, n_mb, *z_1dims)
                assert j_mixup2.shape == (n_seeds, n_mb, *z_1dims)

                lam_mixup5 = lam_mixup3.reshape(n_seeds, n_mb, *z_1dims)
                assert lam_mixup5.shape == (n_seeds, n_mb, *z_1dims)
                
                lam_mixup6 = lam_mixup4.reshape(n_seeds, n_mb, *z_1dims)
                assert lam_mixup6.shape == (n_seeds, n_mb, *z_1dims)

                zi_mb = z_mbs[z_node]
                assert zi_mb.shape == (n_seeds, n_mb, *z_dim)

                zj_mb = torch.take_along_dim(zi_mb, j_mixup2, dim=1)
                assert zj_mb.shape == (n_seeds, n_mb, *z_dim)

                z_mxdmb = zi_mb * lam_mixup5 + zj_mb * lam_mixup6
                assert z_mxdmb.shape == (n_seeds, n_mb, *z_dim)

                z_mxdmbs[z_node] = z_mxdmb

            y_mxdmbs = dict()
            for y_node, y_dim in y_dims.items():
                y_1dims = [1] * len(y_dim)

                j_mixup2 = j_mixup1.reshape(n_seeds, n_mb, *y_1dims)
                assert j_mixup2.shape == (n_seeds, n_mb, *y_1dims)

                lam_mixup5 = lam_mixup3.reshape(n_seeds, n_mb, *y_1dims)
                assert lam_mixup5.shape == (n_seeds, n_mb, *y_1dims)
                
                lam_mixup6 = lam_mixup4.reshape(n_seeds, n_mb, *y_1dims)
                assert lam_mixup6.shape == (n_seeds, n_mb, *y_1dims)

                yi_mb = y_mbs[y_node]
                assert yi_mb.shape == (n_seeds, n_mb, *y_dim)

                yj_mb = torch.take_along_dim(yi_mb, j_mixup2, dim=1)
                assert yj_mb.shape == (n_seeds, n_mb, *y_dim)

                y_mxdmb = yi_mb * lam_mixup5 + yj_mb * lam_mixup6
                assert y_mxdmb.shape == (n_seeds, n_mb, *y_dim)

                y_mxdmbs[y_node] = y_mxdmb
            
            xd_mxdmbs = dict()
            for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                j_mixup2 = j_mixup1.reshape(n_seeds, n_mb, 1, 1)
                assert j_mixup2.shape == (n_seeds, n_mb, 1, 1)

                lam_mixup5 = lam_mixup3.reshape(n_seeds, n_mb, 1, 1)
                assert lam_mixup5.shape == (n_seeds, n_mb, 1, 1)
                
                lam_mixup6 = lam_mixup4.reshape(n_seeds, n_mb, 1, 1)
                assert lam_mixup6.shape == (n_seeds, n_mb, 1, 1)

                xi_mb = xd_mbs[x_node]
                assert xi_mb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)

                xj_mb = torch.take_along_dim(xi_mb, j_mixup2, dim=1)
                assert xj_mb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)

                xd_mxdmb = xi_mb * lam_mixup5 + xj_mb * lam_mixup6
                assert xd_mxdmb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)

                xd_mxdmbs[x_node] = xd_mxdmb
        else:
            z_mxdmbs, y_mxdmbs, xd_mxdmbs = None, None, None

        #######################################
        #        The Decoding Pipeline        #
        #######################################
        u_lossrcs, u_mxuplossrcs = None, None
        for (z_tmbs, y_tmbs, xd_tmbs, stage_dec) in [
            (z_mbs, y_mbs, xd_mbs, 'main'), (z_mxdmbs, y_mxdmbs, xd_mxdmbs, 'mixup')]:
            
            if z_tmbs is None:
                continue

            #######################################
            #         Applying the Decoder        #
            #######################################
            udhat_tmbs = decoder(**z_tmbs, **y_tmbs)
            for u_node, (n_uchnls, n_ulen) in ud_dims.items():
                uhat_tmb = udhat_tmbs[u_node]
                assert uhat_tmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

            xdhat_tmbs = None
            if ppcri_inpspc == 'x':
                xdhat_tmbs = dec_pp.inverse(udhat_tmbs, ue_shaper, strict=True, full=False)
                for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                    x_tmb = xdhat_tmbs[x_node]
                    assert x_tmb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)

            #######################################
            #   Finding the Reconstruction Loss   #
            #######################################

            # Next, we apply the input pre-processing transformations
            # as if it was fed to a perfect decoder.
            need_udmbs1 = (ppcri_inpspc == 'x') and (ppcri_rndtrp == True)
            need_udmbs2 = (ppcri_inpspc == 'u')
            if need_udmbs1 or need_udmbs2:
                ud_tmbs, ud_shaper = dec_pp.forward(xd_tmbs, full=False)
                for u_node, (n_uchnls, n_ulen) in ud_dims.items():
                    u_tmb = ud_tmbs[u_node]
                    assert u_tmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)
            else:
                ud_tmbs, ud_shaper = None, None

            # Finding the right input to the criterion pre-processor
            if (ppcri_inpspc == 'x') and (ppcri_rndtrp == True):
                # In principle, `xdtld_tmbs` should be identical to `xe_tmbs`.
                # However, since some I/O transformations may not be truly
                # one-to-one, or due to numerical errors, they can be different.
                # Since this is not the VAE's fault, this should not exhibit
                # itself in the reconstruction loss, which is why we use the
                # `xdtld_tmbs` tensor instead of `x_pln` when computing the
                # reconstruction loss.
                xdtld_tmbs = dec_pp.inverse(ud_tmbs, ud_shaper, strict=True, full=False)
                for x_node, (n_xchnls, n_xlen) in xd_dims.items():
                    x_tmb = xdtld_tmbs[x_node]
                    assert x_tmb.shape == (n_seeds, n_mb, n_xchnls, n_xlen)
                cri_inpmbs, cri_inphatmbs = xdtld_tmbs, xdhat_tmbs
            elif (ppcri_inpspc == 'x') and (ppcri_rndtrp == False):
                cri_inpmbs, cri_inphatmbs, xdtld_tmbs = xd_tmbs, xdhat_tmbs, xd_tmbs
            elif (ppcri_inpspc == 'u'):
                cri_inpmbs, cri_inphatmbs, xdtld_tmbs = ud_tmbs, udhat_tmbs, None
            else:
                raise ValueError(f'undefined ppcri_inpspc={ppcri_inpspc}')

            # Applying the criterion pre-processor to the original
            uc_tmbs, uc_shaper = cri_pp.forward(cri_inpmbs, full=False)
            for u_node, (n_uchnls, n_ulen) in u_cridims.items():
                u_tmb = uc_tmbs[u_node]
                assert u_tmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

            # Applying the criterion pre-processor to the reconstruction
            uchat_tmbs, uchat_shaper = cri_pp.forward(cri_inphatmbs, full=False)
            for u_node, (n_uchnls, n_ulen) in u_cridims.items():
                uhat_tcmb = uchat_tmbs[u_node]
                assert uhat_tcmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

            # Finding the reconstruction loss
            u_tlossrcs = dict()
            for u_node, (n_uchnls, n_ulen) in u_cridims.items():
                u_tcmb = uc_tmbs[u_node]
                assert u_tcmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

                uhat_tcmb = uchat_tmbs[u_node]
                assert uhat_tcmb.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

                uhat_terr = u_tcmb - uhat_tcmb
                assert uhat_terr.shape == (n_seeds, n_mb, n_uchnls, n_ulen)

                if rcnst_cri == 'mse':
                    u_tlossrc = 0.5 * uhat_terr.square().mean(dim=[-1, -2, -3])
                    assert u_tlossrc.shape == (n_seeds,)
                elif rcnst_cri == 'mae':
                    u_tlossrc = uhat_terr.abs().mean(dim=[-1, -2, -3])
                    assert u_tlossrc.shape == (n_seeds,)
                else:
                    raise ValueError(f'undefined rcnst_cri={rcnst_cri}')

                u_tlossrcs[u_node] = u_tlossrc
            
            if stage_dec == 'main':
                u_lossrcs = u_tlossrcs
            elif stage_dec == 'mixup':
                u_mxuplossrcs = u_tlossrcs
            else:
                raise ValueError(f'undefined stage_dec={stage_dec}')

        if u_mxuplossrcs is None:
            u_mxuplossrcs = dict()
            for u_node, u_lossrc in u_lossrcs.items():
                u_mxuplossrc = torch.zeros_like(u_lossrc)
                assert u_mxuplossrc.shape == (n_seeds,)
                u_mxuplossrcs[u_node] = u_mxuplossrc

        loss_rc = torch.stack(list(u_lossrcs.values()), dim=1).sum(dim=1)
        assert loss_rc.shape == (n_seeds,)

        loss_mxuprcunit = torch.stack(list(u_mxuplossrcs.values()), dim=1).sum(dim=1)
        assert loss_mxuprcunit.shape == (n_seeds,)
        
        loss_mxuprc = w_mixup * loss_mxuprcunit
        assert loss_mxuprc.shape == (n_seeds,)

        loss_kl = torch.stack(list(z_losskls.values()), dim=1).sum(dim=1)
        assert loss_kl.shape == (n_seeds,)

        loss = loss_kl + loss_rc + loss_mxuprc + loss_indzy
        assert loss.shape == (n_seeds,)

        loss_sum = loss.sum()
        assert loss_sum.shape == tuple()
        assert not loss_sum.isnan()

        loss_sum.backward()

        # We will not update in the first epoch so that we will 
        # record the initialization statistics as well. Instead, 
        # we will update an extra epoch at the end.
        if epoch > 0:
            if pgrouper is not None:
                pgrouper.attach_grads()
            opt.step()

        if scheduler is not None:
            scheduler.step(metrics=loss, epoch=epoch)
        
        encoder.eval()
        decoder.eval()

        timer.toc()

        # Logging the loss statistics into a dictionary
        loss_dict = dict()
        with torch.no_grad():
            loss_dict['loss/net'] = loss
            loss_dict['loss/rcnst/net'] = loss_rc
            for u_node, u_lossrc in u_lossrcs.items():
                loss_dict[f'loss/rcnst/{u_node}'] = u_lossrc
            
            if has_mixup:
                loss_dict['loss/mixup/net'] = loss_mxuprc
                for u_node, u_mxuplossrc in u_mxuplossrcs.items():
                    loss_dict[f'loss/mixup/{u_node}'] = u_mxuplossrc * w_mixup

            loss_dict['loss/kl/net'] = loss_kl
            for z_node, z_lossklmu in z_lossklmus.items():
                loss_dict[f'loss/kl/mu/{z_node}'] = z_lossklmu
            for z_node, z_lossklsig in z_lossklsigs.items():
                loss_dict[f'loss/kl/sig/{z_node}'] = z_lossklsig
            for z_node, z_sigmasqmb in z_sigmasqmbs.items():
                loss_dict[f'ltnt/sig/l2sq/{z_node}'] = z_sigmasqmb.mean(dim=[-1, -2])
            z_musqmeanmbs = {z_node: z_musqmb.mean(dim=[-1, -2])
                for z_node, z_musqmb in z_musqmbs.items()}
            for z_node, z_musqmeanmb in z_musqmeanmbs.items():
                loss_dict[f'ltnt/mu/l2sq/{z_node}'] = z_musqmeanmb
            for z_node, z_sigmasqmb in z_sigmasqmbs.items():
                loss_dict[f'ltnt/sig/l2/{z_node}'] = z_sigmasqmb.mean(dim=-1).sqrt().mean(dim=-1)
            for z_node, z_musqmb in z_musqmbs.items():
                loss_dict[f'ltnt/mu/l2/{z_node}'] = z_musqmb.mean(dim=-1).sqrt().mean(dim=-1)
            if has_indzyloss:
                loss_dict['loss/indzy/net'] = loss_indzy

            # The unweighted set of losses
            loss_dict['ulss/rcnst/net'] = loss_rc
            for u_node, u_lossrc in u_lossrcs.items():
                loss_dict[f'ulss/rcnst/{u_node}'] = u_lossrc
            for z_node, z_lossklmu in z_lossklmus.items():
                loss_dict[f'ulss/kl/mu/{z_node}'] = z_lossklmu
            for z_node, z_lossklsig in z_lossklsigs.items():
                loss_dict[f'ulss/kl/sig/{z_node}'] = z_lossklsig
            if has_indzyloss:
                loss_dict['ulss/indzy/net'] = loss_indzyunit
            if has_mixup:
                for u_node, u_mxuplossrc in u_mxuplossrcs.items():
                    loss_dict[f'ulss/mixup/{u_node}'] = u_mxuplossrc
                
        for key, tnsr in loss_dict.items():
            assert tnsr.shape == (n_seeds,)

        #######################################
        #   Logging the training statistics   #
        #######################################
        lnet_ema, _ = ema('loss/net', loss)
        lklnet_ema, _ = ema('loss/kl/net', loss_kl)
        lrcnet_ema, _ = ema('loss/rcnst/net', loss_rc)
        if has_mixup:
            lmxuprc_ema, _ = ema('loss/mixup/net', loss_mxuprc)
        lmusqs_ema = {z_node: ema(f'ltnt/mu/l2sq/{z_node}', z_musqmeanmb)[0]
            for z_node, z_musqmeanmb in z_musqmeanmbs.items()}
        lkls_ema = {z_node: ema(f'loss/kl/{z_node}', z_losskl)[0]
            for z_node, z_losskl in z_losskls.items()}
        lrcs_ema = {u_node: ema(f'loss/rcnst/{u_node}', u_lossrc)[0]
            for u_node, u_lossrc in u_lossrcs.items()}
        if has_mixup:
            lmxuprcs_ema = {
                u_node: ema(f'loss/mixup/{u_node}', u_mxuplossrc * w_mixup)[0]
                for u_node, u_mxuplossrc in u_mxuplossrcs.items()}
        if has_indzyloss:
            lindzy_ema, _ = ema('loss/indzy/net', loss_indzy)
        if (epoch % 100 == 0) and (results_dir is not None):
            print_str = f'Epoch {epoch:06d}: EMA loss={lnet_ema:.4f}'
            for u_node, u_lrcema in lrcs_ema.items():
                print_str += f', rcnst/{u_node}: {u_lrcema:.4f}'
            for z_node, lkl_ema in lkls_ema.items():
                print_str += f', kl/{z_node}={lkl_ema:.4f}'
            for z_node, lmusq_ema in lmusqs_ema.items():
                print_str += f', musq/{z_node}: {lmusq_ema:.4f}'
            if has_mixup:
                for u_node, u_lmxuprcema in lmxuprcs_ema.items():
                    print_str += f', mixup/{u_node}: {u_lmxuprcema:.4f}'
            if has_indzyloss:
                print_str += f', indzy: {lindzy_ema:.4f}'
            if scheduler is not None:
                lr_median = np.median(scheduler.get_last_lr())
                print_str += f', median(lr)={lr_median:.4f}'
            print(print_str, flush=True)

        if do_logtb:
            with torch.no_grad():
                for loss_name, loss_tnsr in loss_dict.items():
                    tbwriter.add_scalar(loss_name, loss_tnsr.mean(), epoch)

        #######################################
        #      Performing the Evaluations     #
        #######################################
        e_vizinputs, e_mtrcinputs = dict(), dict()
        for eid, eopts in eparams.items():
            # The evaluation frequency
            e_frq = eopts['frq']
            if (epoch % e_frq) > 0:
                continue
            # The evaluation pre-processing
            e_pp = eopts['pp']
            # The evaluation pre-processor's input type (either 'x' or 'u')
            e_ppinpspc = eopts['inpspc']
            # The nodes and their dimensions after applying the `e_pp` pre-processor
            uv_dims = eopts['udims']
            # The evaluation split configs
            e_spltcfgs = eopts['splits']

            for vid, vcfg in vizcfgs.items():
                if vcfg['eval'] == eid:
                    assert epoch % io_avgfrq == 0, dedent(f'''
                        eval/{eid} requires storage, thus 
                        "eval/{eid}/frq" % "io/avg/frq" == 0 
                        should hold, but {e_frq} % {io_avgfrq} != 0.''')

            ######### Phase I: Applying the encoder to all splits ##########
            # Whether the original x inputs should go through the nn pre-prcessor's forward/inverse round trip.
            e_pprndtrp = eopts['rndtrp']
            # The randomization of evaluation indices ('none', 'shuffle', or 'iid')
            e_rndmztn = eopts['rndmztn']
            # The nodes and their dimensions before applying the `e_pp` pre-processor
            # This should practically be the same as `x_dimes` for VAEs.
            xv_dims = eopts['xdims']

            e_encpkgs = dict()
            for e_split, e_spltcfg in e_spltcfgs.items():
                # The split data indecis across all the data
                e_splitidxs = e_spltcfg['idxs']
                # The hyper-indices over `e_splitidxs` to keep track
                # of how much we made progress through them.
                eii = e_spltcfg['iidxs']
                # The number of samples in the split for evaluation
                n_evlpnts = e_spltcfg['n']
                # The vae label networks's noise injection sigma scale
                ey_sigscale = e_spltcfg['lbl']['sigscale']
                # The mixup configuration
                e_mixupcfg = e_spltcfg['mixup']
                # Whether we are applying mixup augmentation
                ehas_mixup = e_mixupcfg['enable']
                # The mixup alpha parameter of the beta distribution
                ealpha_mixup = e_mixupcfg['alpha']
                # The per-sample number of mixup interpolations
                n_emixper = e_mixupcfg['n']
                # Whether to max-flip the mixup lambda values.
                e_flipmax = e_mixupcfg['flipmax']
                # Whether the mixup sampling is done in a deterministic manner.
                e_detspace = e_mixupcfg['detspc']

                # Applying the Encoder
                timer.tic(f'sample/{eid}/{e_split}/encode')
                *pkg_enc, renew_eii, eii = Evaluation.encode(
                    eii, e_split, e_splitidxs, e_pp, e_frq, e_rndmztn,
                    ltnt_sigtnsfm, datanp_dict, enc_pp, encoder, 
                    e_ppinpspc, e_pprndtrp, xv_dims, uv_dims,
                    is_condvae, lblnn_enctype, ey_sigscale, 
                    lbl_pp, lblnn, xl_dims, ul_dims, z_dims, y_dims, 
                    dec_pp, ud_dims, xd_dims, xedl_dims, 
                    n_seeds, n_evlpnts, eval_bsnn, epoch,
                    xe_dims, ue_dims, erng, tch_device, tch_dtype)
                timer.toc()

                # Saving the same ordering for continuing in the next epochs
                if renew_eii:
                    e_spltcfg['iidxs'] = eii

                emu_mbs, esigma_mbs, exd_mbs, ey_mbs, ei_mb = pkg_enc

                # Applying mixup if needed
                if ehas_mixup:
                    pkg_mixup = Evaluation.mixup(emu_mbs, esigma_mbs, exd_mbs, ey_mbs, ei_mb, 
                        e_detspace, ealpha_mixup, e_flipmax, n_seeds, n_evlpnts, n_emixper,
                        z_dims, y_dims, xd_dims, erng, tch_dtype, tch_device)
                else:
                    ej_mxdmb = None
                    pkg_mixup = (emu_mbs, esigma_mbs, exd_mbs, ey_mbs, ei_mb, ej_mxdmb, n_evlpnts)

                (emu_mxdmbs, esigma_mxdmbs, exd_mxdmbs, ey_mxdmbs, ei_mxdmb, ej_mxdmb, n_mxdevlpnts) = pkg_mixup

                u_eppmxdmbs = Evaluation.decodeorig(exd_mxdmbs, e_split, e_pp, dec_pp, 
                    e_ppinpspc, e_pprndtrp, xv_dims, uv_dims, ud_dims, n_seeds, n_mxdevlpnts)

                pkg_encode = (emu_mxdmbs, esigma_mxdmbs, exd_mxdmbs, u_eppmxdmbs, 
                    ey_mxdmbs, ei_mxdmb, ej_mxdmb, n_mxdevlpnts)

                e_encpkgs[e_split] = pkg_encode

                # Making sure these variables would not leak elsewhere
                del n_evlpnts, eii, e_splitidxs

            for e_split, e_spltcfg in e_spltcfgs.items():
                # Getting the split encoded data
                (emu_mxdmbs, esigma_mxdmbs, exd_mxdmbs, u_eppmxdmbs, 
                    ey_mxdmbs, ei_mxdmb, ej_mxdmb, n_mxdevlpnts) = e_encpkgs[e_split]
                # The variant configs
                e_vrntcfgs = e_spltcfg['vrnts']

                eis_vrntdatas = dict()
                for e_vrnt, e_vrntcfg in e_vrntcfgs.items():
                    ######### Phase II: Compiling the latent variables of all variants #########
                    timer.tic(f'sample/{eid}/{e_split}/{e_vrnt}/zsample')
                    ltnt_pkg = Evaluation.sample_zy(
                        emu_mxdmbs, esigma_mxdmbs, ey_mxdmbs, ei_mxdmb, z_dims, y_dims,
                        n_seeds, n_mxdevlpnts, is_condvae, erng, e_encpkgs,
                        eid, e_split, e_vrnt, e_vrntcfg, tch_device, tch_dtype)

                    (emu_mxdmbvs, esigma_mxdmbvs, ez_mxdmbvs, ey_mxdmbvs, ei_mxdmbv, 
                     n_rcns, n_epsdim, sigma_multype, vrnt_type) = ltnt_pkg
                    timer.toc()

                    ############## Phase III: Applying the decoder to all splits ###############
                    # Applying the decoder to the evaluation samples
                    timer.tic(f'sample/{eid}/{e_split}/{e_vrnt}/decode')
                    if vrnt_type == 'orig':
                        assert n_rcns == 1
                        uhat_eppmxdmbvs = {u_node: u_tnsr.unsqueeze(2)
                            for u_node, u_tnsr in u_eppmxdmbs.items()}
                        for u_node, (n_uchnls, n_ulen) in uv_dims.items():
                            eu_mxdmbv = uhat_eppmxdmbvs[u_node]
                            assert eu_mxdmbv.shape == (n_seeds, n_mxdevlpnts, n_rcns, n_uchnls, n_ulen)
                        ey_mxdmbvs2 = ey_mxdmbvs
                    else:
                        assert ez_mxdmbvs is not None
                        uhat_eppmxdmbvs, eyhat_mxdmbvs = Evaluation.decode(
                            ez_mxdmbvs, ey_mxdmbvs, dec_pp, decoder, e_pp, e_ppinpspc, lbl_pp, lblnn,
                            is_condvae, lblnn_enctype, n_seeds, n_mxdevlpnts, n_rcns, eval_bsnn, 
                            xd_dims, ud_dims, uv_dims, xl_dims, ul_dims, z_dims, y_dims)
                        ey_mxdmbvs2 = eyhat_mxdmbvs
                    timer.toc()

                    eis_vrntdatas[e_vrnt] = (emu_mxdmbvs, esigma_mxdmbvs, ez_mxdmbvs, 
                        ey_mxdmbvs2, uhat_eppmxdmbvs, ei_mxdmbv, n_rcns, n_epsdim, sigma_multype)

                    if vrnt_type in ('genr', 'gaus'):
                        assert e_split == 'normal'
                        eis_vrntdatas[f'{e_vrnt}.inp'] = (None, None, None, ey_mxdmbvs, None, None, 
                            n_rcns, None, None)

                ######## Data Assembly for Metrics and Visualization #########
                eis_mtrcinputs, eis_vizinputs = Evaluation.assemble(
                    eid, e_split, eis_vrntdatas, n_seeds, n_mxdevlpnts,
                    uv_dims, z_dims, y_dims)
                e_mtrcinputs.update(eis_mtrcinputs)
                e_vizinputs.update(eis_vizinputs)

        #################### Metric Calculations ###################
        e_perfstats, e_perfhists = Evaluation.performance(perfcfgs, e_mtrcinputs, aero_cstinfo, 
            asgn_mdlscfg, metric_mdlscfg, eparams, split_vars, n_seeds, n_t, mrng, timer, 
            tch_device, tch_dtype, tch_cdtype)

        ############# Visualization: MDS Calculation ###############       
        e_vizoutputs = Visualization.mds(e_vizinputs, eparams, vizcfgs, vrng, n_seeds,
            z_dims, y_dims, epoch, timer, tch_device)

        with torch.no_grad():
            if do_logtb:
                for key, val in e_perfstats.items():
                    tbwriter.add_scalar(f'perf/{key}', val.mean(), epoch)
                
            ############ Visualizations: Part 3 (Plotting) ##############
            Visualization.draw(e_vizoutputs, vizcfgs, epoch, n_seeds,
                chem_species, d_histbins, fig_dir, timer, tbwriter)

            if do_logtb:
                tbwriter.flush()

        timer.tic('save/pump')
        if epoch == 0:
            # Just making sure there is an entry for 'save/pump' before 
            # writing the report in the `dprepper.prep` call.
            timer.toctic('save/pump')

        # `stat_dict` will become hierarchical in the `.prep()` call
        stat_dict = {'perf': e_perfstats, **loss_dict}
        nn_dict = {'mdl/enc': encoder, 'mdl/dec': decoder}
        if is_condvae and (lblnn_enctype == 'vae'): 
            nn_dict['mdl/lbl'] = lblnn.encoder
        elif is_condvae and (lblnn_enctype == 'plain'): 
            nn_dict['mdl/lbl'] = lblnn
        elif is_condvae:
            raise ValueError('undefined case')
        strg_dict = deep2hie({'var/eval/raw': e_vizinputs, 
            'var/eval/mds': e_vizoutputs})

        # Filtering out the data that should not be stored
        DataPrepper.filter_vizstrg(strg_dict, eparams, vizcfgs)
        strg_dict['var/perf/hists'] = e_perfhists

        # Prepping the data tuples to be stored to the disk
        dtups = dprepper.prep(strg_dict, stat_dict, nn_dict, timer, epoch)

        dwriter.add(data_tups=dtups, file_path=hdfpth)
        timer.toc()


# %% [markdown]
# # Storing the Results

    # %%
    if results_dir is not None:
        print(f'Training finished in {time.time() - trn_sttime:.1f} seconds.')

    # Closing up all the data/profile/log-writers
    timer.tic('save/close')
    dwriter.close()
    timer.toc()

    if do_logtb:
        timer.tic('save/tbflush')
        tbwriter.flush()
        timer.toc()

    if do_profile:
        timer.tic('save/profile')
        profiler.stop()
        html = profiler.output_html()
        htmlpath = f'{cfgstrg_dir}/profiler.html'
        with open(htmlpath, 'w') as fp:
            fp.write(html.encode('ascii', errors='ignore').decode('ascii'))
        timer.toc()

    if results_dir is not None:
        timer.print()

    # %%
    # Preparing the memory usage
    outdict = dict()
    tchmemusage = profmem()
    assert str(tch_device) in tchmemusage
    if 'cuda' in device_name:
        tch_dvcmem = torch.cuda.get_device_properties(tch_device).total_memory
    else:
        tch_dvcmem = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    outdict['dvc/mem/alloc'] = tchmemusage[str(tch_device)]
    outdict['dvc/mem/total'] = tch_dvcmem

    # %% tags=["active-py"] vscode={"languageId": "raw"}
    return outdict

if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('-c', '--configid', action='store', type=str, required=True)
        my_parser.add_argument('-d', '--device',   action='store', type=str, required=True)
        my_parser.add_argument('-s', '--nodesize', action='store', type=int, default=1)
        my_parser.add_argument('-r', '--noderank', action='store', type=int, default=0)
        my_parser.add_argument('-i', '--rsmindex', action='store', type=str, default="0.0")
        my_parser.add_argument('-n', '--nconfigs', action='store', type=int, default=-1)
        my_parser.add_argument('-g', '--nseedmax', action='store', type=int, default=0)
        my_parser.add_argument('--dry-run', action='store_true')
        args = my_parser.parse_args()
        args_configid = args.configid
        args_device_name = args.device
        args_nodesize = args.nodesize
        args_noderank = args.noderank
        arsg_rsmindex = args.rsmindex
        args_nconfigs = args.nconfigs
        args_dryrun = args.dry_run
        args_nseedmax = args.nseedmax
    else:
        args_configid = 'lvl1/lvl2/poiss2d'
        args_device_name = 'cuda:0'
        args_nodesize = 1
        args_noderank = 0
        arsg_rsmindex = 0
        args_nseedmax = 0
        args_dryrun = True

    assert args_noderank < args_nodesize
    cfgidsplit = args_configid.split('/')
    # Example: args_configid == 'lvl1/lvl2/poiss2d'
    config_id = cfgidsplit[-1]
    # Example: config_id == 'poiss2d'
    config_tree = '/'.join(cfgidsplit[:-1])
    # Example: config_tree == 'lvl1/lvl2'

    os.makedirs(configs_dir, exist_ok=True)
    # Example: configs_dir == '.../code_bspinn/config'
    os.makedirs(results_dir, exist_ok=True)
    # Example: results_dir == '.../code_bspinn/result'
    # os.makedirs(storage_dir, exist_ok=True)
    # Example: storage_dir == '.../code_bspinn/storage'
    
    if args_dryrun:
        print('>> Running in dry-run mode', flush=True)

    cfg_path_ = f'{configs_dir}/{config_tree}/{config_id}'
    cfg_exts = [cfg_ext for cfg_ext in ['json', 'yml', 'yaml'] if exists(f'{cfg_path_}.{cfg_ext}')]
    assert len(cfg_exts) < 2, f'found multiple {cfg_exts} extensions for {cfg_path_}'
    assert len(cfg_exts) > 0, f'found no json or yaml config at {cfg_path_}'
    cfg_ext = cfg_exts[0]
    cfg_path = f'{cfg_path_}.{cfg_ext}'
    print(f'>> Reading configuration from {cfg_path}', flush=True)
    
    if cfg_ext.lower() == 'json':
        with open(cfg_path, 'r') as fp:
            json_cfgdict = json.load(fp, object_pairs_hook=dict)
    elif cfg_ext.lower() in ('yml', 'yaml'):
        with open(cfg_path, 'r') as fp:
            json_cfgdict = dict(ruyaml.safe_load(fp))
    else:
        raise RuntimeError(f'unknown config extension: {cfg_ext}')
    
    if args_dryrun:
        import tempfile
        temp_resdir = tempfile.TemporaryDirectory(dir=f'{PROJPATH}/trash')
        temp_strdir = tempfile.TemporaryDirectory(dir=f'{PROJPATH}/trash')
        print(f'>> [dry-run] Temporary results dir placed at {temp_resdir.name}')
        print(f'>> [dry-run] Temporary storage dir placed at {temp_strdir.name}')
        results_dir = temp_resdir.name
        storage_dir = temp_strdir.name
        
        dr_maxfrq = 10
        dr_opts = {'opt/epoch': dr_maxfrq, 'io/cmprssn_lvl': 0, 'io/strg/logtb': False,
            'io/strg/profile': False, 'io/strg/savefigs': False, 'io/eval/enbl': False}
        for opt in dr_opts:
            assert opt in json_cfgdict
            json_cfgdict[opt] = dr_opts[opt]
        for opt in fnmatch.filter(json_cfgdict.keys(), '*/frq'):
            if json_cfgdict[opt] > dr_maxfrq:
                json_cfgdict[opt] = dr_opts[opt] = dr_maxfrq
        for opt in fnmatch.filter(json_cfgdict.keys(), '*/viz/*/store'):
            json_cfgdict[opt] = dr_opts[opt] = False

        print(f'>> [dry-run] The following options were made overriden:', flush=True)
        for opt, val in dr_opts.items():
            print(f'>>           {opt}: {val}', flush=True)
            
    nodepstfx = f'_{args_noderank:02d}'
    # Example: nodepstfx in ('_00', '_01', ...)
    json_cfgdict['io/config_id'] = f'{config_id}{nodepstfx}'
    # Example: ans in ('poiss2d', 'poiss2d_01')
    json_cfgdict['io/results_dir'] = f'{results_dir}/{config_tree}'
    # Example: ans == '.../code_bspinn/result/lvl1/lv2/poiss2d'
    json_cfgdict['io/storage_dir'] = f'{storage_dir}/{config_tree}'
    # Example: ans == '.../code_bspinn/storage/lvl1/lv2/poiss2d'
    json_cfgdict['io/tch/device'] = args_device_name
    # Example: args_device_name == 'cuda:0'

    # Pre-processing and applying the looping processes
    all_cfgdicts = preproc_cfgdict(json_cfgdict)
        
    # Selecting this node's config dict subset
    node_cfgdicts = [config_dict for cfgidx, config_dict in enumerate(all_cfgdicts) 
        if (cfgidx % args_nodesize == args_noderank)]
    n_nodecfgs = len(node_cfgdicts)
    
    # Going over the config dicts one-by-one
    rsmidx, rsmprt = tuple(int(x) for x in arsg_rsmindex.split('.'))
    for cfgidx, config_dict1 in enumerate(node_cfgdicts):
        # if a positive `args_nconfigs` is provided, we should only run 
        # `args_nconfigs` configs starting at `rsmidx`.
        is_toonew = ((args_nconfigs >= 0) and (cfgidx >= rsmidx + args_nconfigs))
        if (cfgidx < rsmidx) or is_toonew:
            continue
        
        # Parsing all the "/ref"-, "/def"-, and "/pop"-ending keys
        config_dict = parse_refs(config_dict1, trnsfrmtn='hie', pathsep=' -> ', 
            cfg_path=f'{config_tree}/{config_id}.{cfg_ext}')
        # Dropping the trailing references section
        ref_dict = get_subdict(config_dict, prefix='refs', pop=True)
        
        # Getting a single seed run to estimate the memory usage
        tempcfg = config_dict.copy()
        tempcfg['io/results_dir'] = None
        tempcfg['io/storage_dir'] = None
        tempcfg['opt/epoch'] = 0

        if args_nseedmax <= 0:
            # Taking a single-seed run
            args_device = torch.device(args_device_name)
            if 'cuda' in args_device_name:
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.reset_peak_memory_stats(device=args_device)
            tempcfg1 = dict(tempcfg)
            tempcfg1['rng_seed/list'] = [0]
            tod1 = main(tempcfg1)
            allocmem1, totmem = tod1['dvc/mem/alloc'], tod1['dvc/mem/total']

            # Taking a double-seed run
            if 'cuda' in args_device_name:
                torch.cuda.reset_peak_memory_stats(device=args_device)
            tempcfg2 = dict(tempcfg)
            tempcfg2['rng_seed/list'] = [0, 1000]
            tod2 = main(tempcfg2)
            allocmem2, totmem = tod2['dvc/mem/alloc'], tod2['dvc/mem/total']

            # Doing a linear interpolation to see how many seeds can fit
            allocmem2 = max(allocmem2, int(np.ceil(allocmem1 * 1.001)))
            nsd_max = 1 + int(0.75 * (totmem - allocmem1) / (allocmem2 - allocmem1))
        else:
            nsd_max = args_nseedmax
        
        # Computing how many parts we must split the original config into
        cfg_seeds = config_dict['rng_seed/list']
        nprts = int(np.ceil(len(cfg_seeds) / nsd_max))
        if args_nseedmax <= 0:
            mem_bias = 2 * allocmem1 - allocmem2
            mem_slope = allocmem2 - allocmem1
            print(f'>> Config index {cfgidx} takes {mem_bias/1e6:.1f} MB + {mem_slope/1e6:.1f} ' + 
                f'MB/seed (out of {totmem/1e9:.1f} GB)', flush=True)
            print(f'>> Config index {cfgidx} must be ' + 
                f'devided into {nprts} parts.', flush=True)
        
        # Looping over each part of the config
        for iprt in range(nprts):
            if (cfgidx == rsmidx) and (iprt < rsmprt):
                continue
            print(f'>>> Started Working on config index {cfgidx}.{iprt}' + 
                  f' (out of {nprts} parts and {n_nodecfgs} configs).', flush=True)
            iprtcfgseeds = cfg_seeds[(iprt*nsd_max):((iprt+1)*nsd_max)]
            iprtcfgdict = config_dict.copy()
            iprtcfgdict['rng_seed/list'] = iprtcfgseeds
            main(iprtcfgdict)
            print('-' * 40, flush=True)
        print(f'>> Finished working on config index {cfgidx} ' + 
              f'(out of {n_nodecfgs} configs).', flush=True)
        print('='*80, flush=True)
        
    if args_dryrun:
        print(f'>> [dry-run] Cleaning up {temp_resdir.name}')
        temp_resdir.cleanup()
        print(f'>> [dry-run] Cleaning up {temp_strdir.name}')
        temp_strdir.cleanup()

