import numpy as np
import pandas as pd
import h5py

import io
import time
import glob
import torch
import fnmatch
import warnings
from itertools import chain, cycle
from textwrap import dedent
from collections import defaultdict
from collections import OrderedDict as odict
from os.path import exists
from pandas.api.types import union_categoricals

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import EngFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bokeh.io import push_notebook, show, output_notebook, output_file, save
from bokeh.models import Band, Whisker, ColumnDataSource, FactorRange
from bokeh.models import TapTool, Select, Div, BoxZoomTool, ResetTool
from bokeh.models import HoverTool, CustomJS, TabPanel, Tabs
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot, layout, row, column, grid
from bokeh.transform import factor_cmap

import partnn
from partnn.io_utils import get_ovatgrps, drop_unqcols
from partnn.io_cfg import configs_dir, results_dir
from partnn.io_cfg import keyspecs, nullstr
from partnn.io_utils import deep2hie, hie2deep, get_subdict
from partnn.io_utils import save_h5data, load_h5data
from partnn.io_utils import get_h5du, resio, get_dfidxs
from partnn.summary import summarize

import yaml
from ruamel import yaml as ruyaml
from IPython import display as ICD

warnings.filterwarnings("ignore", message="found multiple competing values for " + 
    "'toolbar.active_drag' property; using the latest value")
warnings.filterwarnings("ignore", message="DataFrame.interpolate with object dtype " + 
    "is deprecated and will raise in a future version.")

dflt_dashcfg = dict(ncols=4, colsep=None, majoridcol='fpidxgrp', 
    minoridcol='fpidx', frame_width=350, frame_height=235, sharex=True, 
    sharey=True,  header='Ablation Studies', menu_width=250, colors='snsdark', 
    color_reset='figure')

dflt_bkfigcfg = dict(kind='line', xcol='epoch', ycol=None, huecol='fpidxgrp', 
    frame_width=350, frame_height=235, fig_title='Ablation: {ablname}', 
    tooltip=None, y_axis_type='auto', y_tick_fmt=None, y_tick_lbls=None, 
    x_axis_type='auto', x_tick_fmt=None, x_tick_lbls=None, marker=None, 
    label_color='black', label_font='10pt', header_color='black',
    theme=None, background_color='white')

dflt_datacfg = {'smry/idxcol': None, 'smry/selcol': None,
    'smry/selcri': None}

#################################################
########### Aggregation of RNG Seeds ############
#################################################
def get_aggdf(hpdf, stdf, xcol, huecol, rngcol, aggcfg):
    #################################################
    ############### Aggregator Setup ################
    #################################################
    aggtype = aggcfg.setdefault('type', 'mean')
    if aggtype == 'bootstrap':
        n_boot = aggcfg.setdefault('n_boot', 20)
        q = aggcfg.setdefault('q', [5, 95])
        stat = aggcfg.setdefault('stat', 'mean')
        device = aggcfg.setdefault('device', None)
        aggregator = BStrapAgg(n_boot=n_boot, q=q, stat=stat, 
            seed=12345, device=device)
        assert len(aggcfg) == 5
    elif aggtype in ('sem', 'mean'):
        aggregator = 'sem'
        assert len(aggcfg) == 1
    else:
        raise ValueError(f'aggregation type={aggtype} not defined')

    grpbycols = [huecol, xcol]

    # Making an initial copy
    grphpdf = hpdf.copy()
    grpstdf = stdf.copy()

    # Converting `category` columns to non-categorical types
    # Note: Due to a bug in pandas, the `.groupby` operation 
    # takes forever on categorical columns. For this, we have 
    # to first translate the categorical columns to their 
    # original type. Unfortunately, the existence of any categorical 
    # columns slows pandas down in `.groupby`, and it is not 
    # limited to the groupping columns.
    hpcols_ctgry = grphpdf.select_dtypes('category').columns.tolist()
    for col in hpcols_ctgry:
        grphpdf[col] = grphpdf[col].tolist()

    ########### Phase I: Merging `hpdf and `stdf` ###########
    grpstdf = grpstdf.infer_objects()
    stcols = [col for col in grpstdf.columns 
        if col not in (xcol, rngcol)]
    grpstdf = grpstdf.select_dtypes(include='number')
    assert all(col in grpstdf.columns for col in stcols), dedent(f'''
        Some of the `stdf` columns are non-numerical. I'm not 
        sure how I'm supposed to aggregate non-numerical statistics.
            Non-numerical columns: {set(stcols) - set(grpstdf.columns())}
    ''')

    hpcols = list(grphpdf.columns)
    grpdf = pd.concat([grphpdf, grpstdf], axis=1)
    ########### Phase II: Applying the Aggregator ###########
    grpdf_gb = grpdf.groupby(grpbycols, sort=False, observed=True)
    grpdf_gblst = list(grpdf_gb)
    all_agginplens = set(gdf.shape[0] for gvals, gdf in grpdf_gblst)

    if (len(all_agginplens) == 1) and hasattr(aggregator, 'batch'):
        n_samps = tuple(all_agginplens)[0]
        n_grps = len(grpdf_gblst)
        n_vars = len(stcols)
        samples_np = np.stack([gdf[stcols].values for gvals, gdf in grpdf_gblst], axis=0)
        assert samples_np.shape == (n_grps, n_samps, n_vars)

        samples_mean, samples_low, samples_high = aggregator.batch(samples_np)
        assert samples_mean.shape == (n_grps, n_vars)
        assert samples_low.shape == (n_grps, n_vars)
        assert samples_high.shape == (n_grps, n_vars)

        # The `grpbycols` will be added as an index after the `grpdf_gb.agg(aggpolicy)` 
        # call. Therefore, there is no reason to add them as columns for aggregation!
        aggpolicy1 = {c: 'first' for c in hpcols if c not in grpbycols}
        aggdf = grpdf_gb.agg(aggpolicy1)

        # resetting the `grpbycols` index columns
        for col in grpbycols:
            assert col not in aggdf.columns, f'"{col}" is in {aggdf.columns}'
        aggdf = aggdf.reset_index(names=grpbycols)
        assert all(col in aggdf.columns for col in grpbycols)

        for i_col, col in enumerate(stcols):
            aggdf[f'{col}/mean'] = samples_mean[:, i_col]
            aggdf[f'{col}/low'] = samples_low[:, i_col]
            aggdf[f'{col}/high'] = samples_high[:, i_col]
    elif callable(aggregator):
        # The `grpbycols` will be added as an index after the `grpdf_gb.agg(aggpolicy)` 
        # call. Therefore, there is no reason to add them as columns for aggregation!
        aggpolicy = {c: 'first' for c in hpcols if c not in grpbycols}
        aggpolicy.update({c: aggregator for c in stcols})
        aggdf = grpdf_gb.agg(aggpolicy)
        aggcolsdict = odict()
        for col in stcols:
            pkg = tuple(zip(*(aggdf[col].tolist())))
            aggcolsdict[f'{col}/mean'] = pkg[0]
            aggcolsdict[f'{col}/low'] = pkg[1]
            aggcolsdict[f'{col}/high'] = pkg[2]
        aggdf = pd.concat([aggdf.reset_index().drop(columns=stcols), pd.DataFrame(aggcolsdict)], axis=1)

        # resetting the `grpbycols` index columns
        assert all(col not in aggdf.columns for col in grpbycols)
        aggdf = aggdf.reset_index(names=grpbycols)
        assert all(col in aggdf.columns for col in grpbycols)
    elif aggregator in (None, 'mean', 'sem'):
        aggpolicy = {c: ['first'] for c in hpcols}
        aggpolicy.update({c: ['mean', 'sem'] for c in stcols})
        aggdf = grpdf_gb.agg(aggpolicy)
        for col in stcols:
            aggdf.loc[:, (col, 'low')] = aggdf[col]['mean'] - 1.96 * aggdf[col]['sem']
            aggdf.loc[:, (col, 'high')] = aggdf[col]['mean'] + 1.96 * aggdf[col]['sem']

        aggdf = aggdf.reindex(columns=hpcols+stcols, level=0)
        aggdf.columns = aggdf.columns.map('/'.join)
        aggdf = aggdf.reset_index().drop(columns=huecol)

        scc = [col for col in aggdf.columns if not col.endswith('/first')]
        aggdf = aggdf.rename(columns={f'{col}/first': col for col in hpcols})
        aggdf = aggdf[hpcols + scc]
    else:
        raise ValueError(f'aggregator={aggregator} not defined')
    
    ######### Phase III: Splitting `hpdf and `stdf` ##########
    nonhp_cols = [col for col in aggdf.columns if col not in hpcols]
    
    hpdf_agg = aggdf[hpcols]
    stdf_agg = aggdf[nonhp_cols]

    # Converting the category columns back
    hpdf_agg = hpdf_agg.copy(deep=True)
    for col in hpcols_ctgry:
        hpdf_agg[col] = hpdf_agg[col].astype('category')
        if nullstr not in hpdf_agg[col].cat.categories:
            hpdf_agg[col] = hpdf_agg[col].cat.add_categories(nullstr)

    outdict = dict(hpdf=hpdf_agg, stdf=stdf_agg, stcols=stcols)
    return outdict

class BStrapAgg:
    stat2func = {'mean': np.mean, 'median': np.median}

    def __init__(self, n_boot=20, q=(5, 95), 
        stat='mean', seed=12345, device=None):

        np_random = np.random.RandomState(seed)
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        tch_device = torch.device(device)
        tch_dtype = torch.float32
        th_random = torch.Generator(device=tch_device)
        th_random.manual_seed(np_random.randint(0, 2**31 - 1))

        self.n_boot = n_boot
        self.q = q
        self.stat = stat
        self.stat_f = self.stat2func[stat]
        self.np_random = np_random
        self.tch_device = tch_device
        self.tch_dtype = tch_dtype
        self.th_random = th_random

    def __call__(self, series):
        with np.errstate(invalid='ignore'):
            v = series.values
            n = v.size
            rsv = self.np_random.choice(v, self.n_boot*n, replace=True)
            x = rsv.reshape(self.n_boot, n)
            m = self.stat_f(x, axis=1)
            l = np.percentile(m, self.q[0])
            h = np.percentile(m, self.q[1])
            return (m.mean(), l, h)

    @torch.no_grad()
    def batch(self, samples_np):
        n_boot = self.n_boot
        q = self.q
        stat = self.stat
        tch_device = self.tch_device
        tch_dtype = self.tch_dtype
        th_random = self.th_random

        n_grps, n_samps, n_vars = samples_np.shape

        samp_thdata = torch.from_numpy(samples_np)
        samp_thdata = samp_thdata.to(device=tch_device, dtype=tch_dtype)
        assert samp_thdata.shape == (n_grps, n_samps, n_vars)

        randidxs = torch.empty(n_grps, n_boot * n_samps, 1, device=tch_device, dtype=torch.long)
        assert randidxs.shape == (n_grps, n_boot * n_samps, 1)

        randidxs.random_(0, n_samps, generator=th_random)
        assert randidxs.shape == (n_grps, n_boot * n_samps, 1)

        resamp_data_ = torch.take_along_dim(samp_thdata, randidxs, dim=-2)
        assert resamp_data_.shape == (n_grps, n_boot * n_samps, n_vars)

        resamp_data = resamp_data_.reshape(n_grps, n_boot, n_samps, n_vars)
        assert resamp_data.shape == (n_grps, n_boot, n_samps, n_vars)

        if stat == 'mean':
            resamp_stats = resamp_data.nanmean(dim=-2)
            assert resamp_stats.shape == (n_grps, n_boot, n_vars)
        elif stat == 'median':
            resamp_stats = resamp_data.nanmedian(dim=-2)
            assert resamp_stats.shape == (n_grps, n_boot, n_vars)
        else:
            raise ValueError(f'undefined stat={stat}')

        resamp_mean = resamp_stats.nanmean(dim=-2)
        assert resamp_mean.shape == (n_grps, n_vars)

        resamp_low = resamp_stats.nanquantile(q[0] / 100.0, dim=-2)
        assert resamp_low.shape == (n_grps, n_vars)

        resamp_high = resamp_stats.nanquantile(q[1] / 100.0, dim=-2)
        assert resamp_high.shape == (n_grps, n_vars)

        means = resamp_mean.detach().cpu().numpy()
        assert means.shape == (n_grps, n_vars)

        lows = resamp_low.detach().cpu().numpy()
        assert lows.shape == (n_grps, n_vars)

        highs = resamp_high.detach().cpu().numpy()
        assert highs.shape == (n_grps, n_vars)

        return (means, lows, highs)

#################################################
########### Dashboard Data & Building ###########
#################################################
def get_dflt_ablspec(hpdf, fpgrps, huecol='fpidxgrp', ablcol='ablgrp'):
    grpspec = []
    uhpdf = hpdf.drop_duplicates().copy()
    uhpdf = drop_unqcols(uhpdf, exclist=[huecol])
    uhpdf = uhpdf.reset_index(drop=True)
    uhpidf = uhpdf.copy().set_index(huecol)
    for ii, fpidxs in enumerate(fpgrps):
        grphpdf = drop_unqcols(uhpidf.loc[fpidxs].reset_index(huecol))
        grpid = set(grphpdf.columns).difference({ablcol, huecol})
        grpid = sorted(grpid)
        grpspec.append(dict(name=', '.join(grpid), columns=[grpid]))
    return grpspec

def set_dfltscfg(cfg, dflt_cfg, col_renamer, col_replacer, colsep='/'):
    outcfg, cfg = dict(), cfg.copy()
    for opt, dfltoptval in dflt_cfg.items():
        optval, n_pop = get_subdict(cfg, opt, pop=True, ret_npop=True) 
        if n_pop == 0:
            optval = dfltoptval                
        if opt.endswith('col'):
            optval = col_renamer.get(optval, optval)
            if isinstance(optval, str):
                optvalsplt = optval.split(colsep)
                if optvalsplt[-1] in ('mean', 'low', 'high'):
                    optvalsplt, later_appnd = optvalsplt[:-1], [optvalsplt[-1]]
                else:
                    optvalsplt, later_appnd = optvalsplt, []
                optvalsplt = [col_replacer.get(val, val) 
                    for val in optvalsplt] + later_appnd
                optval = colsep.join(optvalsplt)
        outcfg[opt] = optval
    
    assert len(outcfg) == len(dflt_cfg), outcfg
    assert len(cfg) == 0, f'unused options: {cfg}'
    return outcfg

def infer_ovatcol(df, hpcols, majoridcol, idcols):
    ovatcol = None
    agggrpby = df.groupby(majoridcol)
    
    hpcols_nontrvl = [col for col in hpcols 
        if (col not in idcols) and (col in df.columns)]
    for col in hpcols_nontrvl:
        unqdct = agggrpby[col].unique().to_dict()
        if all(len(val)==1 for val in unqdct.values()):
            ovatcol = col
            break
    
    assert ovatcol is not None, dedent('''
        Could not find a proper ovat-like ablation column.
        Here are the grouped hp columns:
        ''' + f'{agggrpby[hpcols_nontrvl]}')

    return ovatcol

def get_dashdata(data, ymlpath, write_yml=False,
    en_exclude=True, en_include=True):
    if isinstance(ymlpath, str) and exists(ymlpath):
        with open(ymlpath, 'r') as fp:
            dashconfig = ruyaml.load(fp, ruyaml.RoundTripLoader)
    elif isinstance(ymlpath, dict):
        dashconfig = ymlpath
    else:
        dashconfig = dict()

    figspec = dashconfig.setdefault('figures', dict())
    dropcfg = dashconfig.setdefault('drop', dict())
    renamecfg = dashconfig.setdefault('rename', dict())
    bkfigcfg_glbl = dashconfig.setdefault('bokeh/global', dict())
    datacfg_glbl = dashconfig.setdefault('data/global', dict())
    dashcfg = dashconfig.setdefault('dashboard', dict())

    #################################################
    ########### Column Definition Options ###########
    #################################################
    drop_colpats = dropcfg.setdefault('columns', [])
    assert len(dropcfg) == 1

    col_renamer = renamecfg.setdefault('columns', dict())
    col_replacer = renamecfg.setdefault('colrplc', dict())
    assert len(renamecfg) == 2

    dashcfg = set_dfltscfg(dashcfg, dflt_dashcfg, 
        col_renamer, col_replacer, colsep='/')
    
    bkfigcfg_glbl = set_dfltscfg(bkfigcfg_glbl, dflt_bkfigcfg, 
        col_renamer, col_replacer, colsep='/')

    datacfg_glbl = set_dfltscfg(datacfg_glbl, dflt_datacfg, 
        col_renamer, col_replacer, colsep='/')

    majoridcol = dashcfg['majoridcol']
    minoridcols = dashcfg['minoridcol']
    minoridcols = [] if minoridcols is None else minoridcols
    if isinstance(minoridcols, str):
        minoridcols = [minoridcols]
    idcols = [majoridcol] + minoridcols
    
    #################################################
    ################ Data Processing ################
    #################################################
    dashdata = dict()
    for tab_id, hpdf, stdf, stcols in data:
        tab_ablspec = figspec.setdefault(tab_id, dict())
        tabname = tab_ablspec.pop('name', tab_id)

        # Getting a copy of the input dataframes
        hpdf, stdf = hpdf.copy(), stdf.copy()

        # Making sure that `stdf` is already aggregated
        for col in stcols:
            for pstfx in ('mean', 'low', 'high'):
                assert f'{col}/{pstfx}' in stdf.columns, dedent(f'''
                    Looks like `stdf` is not properly aggregated. I 
                    cannot find a "{col}/{pstfx}" column in it.''')

        # Dropping some stat columns
        if drop_colpats is not None:
            drop_stcols = chain.from_iterable(fnmatch.filter(stcols, pat) 
                for pat in drop_colpats)
            drop_stcols = [f'{col}/{pstfx}' 
                for col in drop_stcols 
                for pstfx in ('mean', 'low', 'high')]
            stdf = stdf.drop(columns=drop_stcols)
            stcols = [col for col in stcols if col not in drop_stcols]
            
            hpcols_predrp = hpdf.columns.tolist()
            drop_hpcols = chain.from_iterable(fnmatch.filter(hpcols_predrp, pat) 
                for pat in drop_colpats)
            hpdf = hpdf.drop(columns=drop_hpcols)

        # Renaming the columns
        if col_renamer is not None:
            assert isinstance(col_renamer, dict)
            stdf = stdf.rename(columns=col_renamer)
            stcols = [col_renamer.get(col, col) for col in stcols]

            # Making sure that `stdf` and `stcols` are compatible
            for col in stcols:
                for pstfx in ('mean', 'low', 'high'):
                    assert f'{col}/{pstfx}' in stdf.columns, dedent(f'''
                        Looks like `stdf` and `stcols` are not properly synced.
                        I cannot find a "{col}/{pstfx}" column in it.''')
                        
        if col_replacer is not None:
            colsrplcd = dict()
            stcolsiter = set(stdf.columns)
            stcolsrplcd = []
            for col in stcols:
                colsplt = col.split('/')
                colsplt = [col_replacer.get(val, val) for val in colsplt]
                colrp = '/'.join(colsplt)
                stcolsrplcd.append(colrp)
                for pstfx in ('mean', 'low', 'high'):
                    assert f'{col}/{pstfx}' in stdf.columns, f'{col}/{pstfx}'
                    colsrplcd[f'{col}/{pstfx}'] = f'{colrp}/{pstfx}'
                    stcolsiter.remove(f'{col}/{pstfx}')
            stcols = stcolsrplcd

            for col in stcolsiter:
                colsplt = col.split('/')
                colsplt = [col_replacer.get(val, val) for val in colsplt]
                colrp = '/'.join(colsplt)
                colsrplcd[col] = colrp
                        
            stdf = stdf.rename(columns=colsrplcd)

            # Making sure that `stdf` and `stcols` are compatible
            for col in stcols:
                for pstfx in ('mean', 'low', 'high'):
                    assert f'{col}/{pstfx}' in stdf.columns, dedent(f'''
                        Looks like `stdf` and `stcols` are not properly synced.
                        I cannot find a "{col}/{pstfx}" column in it.''')

        # Shortening the fpidx and fpidxgrp columns to save space on html
        for col in idcols:
            hpdf[col] = hpdf[col].cat.codes

        # Grouping the hyper-parameters 
        fpgrps = get_ovatgrps(hpdf)
        dfltabldefs = get_dflt_ablspec(hpdf.drop(columns=minoridcols), 
            fpgrps, majoridcol, 'ablgrp')
        grphpcols = [tuple(x['columns'][0]) for x in dfltabldefs]
        hpcols2fpgrp = dict(zip(grphpcols, fpgrps))
        hpcols2fpgrp[('*',)] = hpdf[majoridcol].unique().tolist()
        
        # Adding existing ablations that do not exist in
        # the ablation config file
        mcols = [abldict['columns'] for abl_id, abldict in hie2deep(tab_ablspec, maxdepth=2).items()]
        mcols = [x if x != '*' else grphpcols for x in mcols]
        mohpc = chain.from_iterable(mcols)
        mohpc = [tuple(sorted(x)) for x in mohpc]
        used_fpgrpidxs = set(chain.from_iterable(hpcols2fpgrp[x] for x in mohpc))
        for i_abl, (fpgrp, abldict) in enumerate(zip(fpgrps, dfltabldefs)):
            if any(fpgrpidx not in used_fpgrpidxs for fpgrpidx in fpgrp):
                abl_id = f'abl{i_abl:02d}'
                assert abl_id not in tab_ablspec, f'please dont name any ablation "{abl_id}"'
                tab_ablspec[abl_id] = abldict

        # Adding the tab column
        hpdf.insert(0, 'tab_id', tab_id)

        # Adding the ablation group column
        hpdfgrp_list, stdfgrp_list = [], []
        abl2mpldict = dict()
        abl2bkfigcfg = dict()
        abl2datacfg = dict()
        abl2name = dict()
        for abl_id, abldict in tab_ablspec.items():
            ablname = abldict['name']
            abl2name[abl_id] = ablname
            exclist = abldict.get('exclude', [])
            exclist = exclist if en_exclude else []
            inclist = abldict.get('include', [])
            inclist = inclist if en_include else []
            abl2mpldict[abl_id] = get_subdict(abldict, prefix='matplotlib', pop=False)
            abl2bkfigcfg[abl_id] = get_subdict(abldict, prefix='bokeh', pop=False)
            abl2datacfg[abl_id] = get_subdict(abldict, prefix='data', pop=False)
            # allhpcols = abldict['columns'] if (abldict['columns'] != '*') else grphpcols
            allhpcols = abldict['columns']
            for hpcols_abl in allhpcols:
                fpgrp = hpcols2fpgrp[tuple(sorted(hpcols_abl))]
                ii = hpdf[majoridcol].isin(fpgrp)
                if len(exclist) > 0:
                    exc_il = [get_dfidxs(hpdf, excdict) for excdict in exclist]
                    jj = np.stack(exc_il, axis=1).any(axis=1)
                    ii = np.logical_and(ii, np.logical_not(jj))
                if len(inclist) > 0:
                    inc_il = [get_dfidxs(hpdf, incdict) for incdict in inclist]
                    jj = np.stack(inc_il, axis=1).any(axis=1)
                    ii = np.logical_and(ii, jj)
                hpdfgrp = hpdf[ii].copy()
                stdfgrp = stdf[ii]
                hpdfgrp.insert(1, 'ablgrp', abl_id)
                hpdfgrp_list.append(hpdfgrp)
                stdfgrp_list.append(stdfgrp)
        hpdf = pd.concat(hpdfgrp_list, axis=0, ignore_index=True)
        stdf = pd.concat(stdfgrp_list, axis=0, ignore_index=True)

        hpdf = hpdf.fillna(nullstr)
        stdf = stdf.fillna(nullstr)

        for fig_idx, (abl_id, ablhpdf) in enumerate(hpdf.groupby('ablgrp', sort=False)):
            ablname = abl2name[abl_id]
            ablstdf = stdf.loc[ablhpdf.index.values]
            assert ablstdf.drop_duplicates().shape[0] == ablstdf.shape[0]
            ablhpdf_nonunq = drop_unqcols(ablhpdf)
            ablhpcols = ablhpdf_nonunq.columns.tolist()
            ablhpcols = [col for col in ablhpcols if col not in idcols]
            abldf = pd.concat([ablhpdf_nonunq, ablstdf], axis=1)

            datacfg = abl2datacfg[abl_id]
            datacfg = set_dfltscfg(datacfg, datacfg_glbl, col_renamer, col_replacer, colsep='/')
            
            ovatcol = None
            if datacfg['smry/idxcol'] is not None:
                smryidxcollst_ = datacfg['smry/idxcol']
                smryidxcollst = [smryidxcollst_] if isinstance(smryidxcollst_, str) else smryidxcollst_
                assert isinstance(smryidxcollst, (list, tuple))

                # Finding the right ablation column
                if '__ovatcol__' in smryidxcollst:
                    ovatcol = infer_ovatcol(abldf, ablhpcols, majoridcol, idcols)
                    smryidxcollst = [ovatcol if col == '__ovatcol__' else col for col in smryidxcollst]

                selcol = datacfg['smry/selcol']
                assert selcol is not None, dedent(f'''
                    the selection column for summarizing the data for the 
                    "{ablname}" ablation figure is not specified. Make 
                    sure you specify a `smry/selcol` value in the `data` 
                    dictionary either globally or locally for this specific 
                    ablation figure.''')
                selcri = datacfg['smry/selcri']
                assert selcri is not None, dedent(f'''
                    the selection criterion for summarizing the data for the 
                    "{ablname}" ablation figure is not specified. Make 
                    sure you specify a `smry/selcri` value in the `data` 
                    dictionary either globally or locally for this specific 
                    ablation figure.''')
                
                smryidxcollst2 = []
                for col in smryidxcollst:
                    if (col in ablhpdf.columns) and (col not in ablhpdf_nonunq.columns):
                        continue
                    smryidxcollst2.append(col)

                abldf_gb = abldf.groupby(smryidxcollst2, observed=True, sort=True)

                for abl_grpid, abl_grpdf in abldf_gb:
                    assert len(abl_grpdf[majoridcol].unique()) == 1, dedent(f'''
                        When summarizing the data for the "{ablname}" ablation, 
                        there were multiple "{majoridcol}" identity values after 
                        grouping by the summary index column "{smryidxcollst}". 
                        This is a bad sign, since it means that we're summarizing 
                        data that do not belong to the same training.
                            summary index value: {abl_grpid}
                            unique {majoridcol} values: {abl_grpdf[majoridcol].unique()}
                            entire abldf groupby: {abldf_gb}
                    ''')

                if selcri == 'min':
                    abldf = abldf.loc[abldf_gb[selcol].idxmin()]
                elif selcri == 'max':
                    abldf = abldf.loc[abldf_gb[selcol].idxmax()]
                else:
                    raise ValueError(f'undefined selcri={selcri} for ablname={ablname}')

            mpldict = abl2mpldict[abl_id]
            bkfigcfg = abl2bkfigcfg[abl_id]
            bkfigcfg = set_dfltscfg(bkfigcfg, bkfigcfg_glbl, col_renamer, col_replacer, colsep='/')

            for key, val in bkfigcfg.items():
                if key.endswith('col') and (val == '__ovatcol__'):
                    bkfigcfg[key] = infer_ovatcol(abldf, ablhpcols, majoridcol, idcols)
            
            assert 'xcol' in bkfigcfg, f'ablname = {ablname} --> {bkfigcfg.keys()}'
            assert '/' not in tab_id, f'please use a different sep in the tab name {tab_id}'
            assert '/' not in abl_id, f'please use a different sep in the ablation name {abl_id}'
            dashdata[f'{tab_id}/{abl_id}'] = (tabname, ablname, abldf, ablhpcols, stcols, bkfigcfg, mpldict)

    if write_yml:
        assert isinstance(ymlpath, str), 'can only write to a file if the yaml path was provided'
        if exists(ymlpath):
            with open(ymlpath, 'w') as fp:
                ruyaml.dump(dashconfig, fp, ruyaml.RoundTripDumper)
        else:
            with open(ymlpath, 'w') as fp:
                yaml.dump(dashconfig, fp, sort_keys=False, default_flow_style=None)

    outdict = dict(dashcfg)
    outdict['data'] = dashdata
    outdict.pop('majoridcol', None)
    outdict.pop('minoridcol', None)
    return outdict

def build_dashboard(doc, data, header, ncols=4, sharex=True, sharey=True, 
    frame_width=350, frame_height=235, menu_width=250, colsep=None, 
    colors=None, color_reset='figure'):

    if colors in (None, 'snsdark'):
        colors = ['#001c7f', '#b1400d', '#12711c', '#8c0800', '#591e71', 
                  '#592f0d', '#a23582', '#3c3c3c', '#b8850a', '#006374']
    else:
        assert isinstance(colors, list)

    all_bkfigdata = []
    all_tabbkfigdata = []
    tabs_list = []

    tab2data, tab_id2name = odict(), dict()
    for tababl_id, (tabname, ablname, aggdf, hpcols, stcols, bokehdict, mpldict) in data.items():
        tab_id, abl_id = tababl_id.split('/')
        figsdata = tab2data.setdefault(tab_id, [])
        figsdata.append((ablname, aggdf, hpcols, stcols, bokehdict))
        if tab_id in tab_id2name:
            assert tab_id2name[tab_id] == tabname, dedent(f'''
                Multiple tab titles for the same tab id: 
                    title 1 = "{tabname}"
                    title 2 = "{tab_id2name[tab_id]}"
                ''')
        else:
            tab_id2name[tab_id] = tabname
        
    assert sharex in (True, False, 'tab', 'all', 'none')
    assert sharey in (True, False, 'tab', 'all', 'none')
    assert color_reset in ('figure', 'tab', 'all')
    if color_reset == 'all':
        colors_cycle = cycle(colors)
    
    header_color = 'black'
    doc_theme = None
    background_color = None
    max_scrnw, max_scrnh = None, None
    for tab_id, figsdata in tab2data.items():
        tabname = tab_id2name[tab_id]
        n_tabfig = len(figsdata)
        n_rows = int(np.ceil(n_tabfig / ncols))

        max_scrnw_ = int(frame_width*(ncols+0.1))
        max_scrnw = max_scrnw_ if max_scrnw is None else max(max_scrnw, max_scrnw_)

        max_scrnh_ = int(frame_height*(n_rows+0.1))
        max_scrnh = max_scrnh_ if max_scrnh is None else max(max_scrnh, max_scrnh_)

        tab_bkfigdata = []

        if color_reset == 'tab':
            colors_cycle = cycle(colors)
        for fig_idx, (ablname, aggdf, hpcols, stcols, bokehdict) in enumerate(figsdata):
            ax_row, ax_col = fig_idx // ncols, fig_idx % ncols

            assert bokehdict is not None
            bokehdict = bokehdict.copy()

            pltkind = bokehdict.pop('kind')
            xcol = bokehdict.pop('xcol')
            ycol = bokehdict.pop('ycol', None)
            huecol = bokehdict.pop('huecol', None)
            fig_title = bokehdict.pop('fig_title', None)
            tooltip = bokehdict.pop('tooltip', None)
            x_axis_type = bokehdict.pop('x_axis_type', 'linear')
            x_tick_fmt = bokehdict.pop('x_tick_fmt', None)
            x_tick_lbls = bokehdict.pop('x_tick_lbls', None)
            y_axis_type = bokehdict.pop('y_axis_type', 'linear')
            y_tick_fmt = bokehdict.pop('y_tick_fmt', None)
            y_tick_lbls = bokehdict.pop('y_tick_lbls', None)

            axis_label_text_color = bokehdict.pop('label_color', 'black')
            axis_label_fontsize = bokehdict.pop('label_font', '10pt')
            header_color = bokehdict.pop('header_color', header_color)
            doc_theme = bokehdict.pop('theme', doc_theme)
            background_color = bokehdict.pop('background_color', background_color)

            frame_width_fig = bokehdict.pop('frame_width')
            frame_height_fig = bokehdict.pop('frame_height')
            fig_title = 'Ablation: {ablname}' if fig_title is None else fig_title
            marker_opts = get_subdict(bokehdict, 'marker', pop=True)

            if ycol is None:
                ycol = [c[:-5] for c in aggdf.columns 
                    if c.endswith('/mean')][-1]
            assert ycol is not None
            assert xcol is not None

            idcols = [col for col in hpcols if col not in ('fpidx', huecol)]
            if tooltip is not None:
                idcols += tooltip

            show_xaxis = (sharex in (False, 'none')) or (ax_row == (n_rows-1))
            show_yaxis = (sharey in (False, 'none')) or (ax_col == 0)

            zoomtool = BoxZoomTool()
            figtools = [zoomtool, 'reset,pan,wheel_zoom,save']
            
            hover_opts = dict(show_arrow=False,
                line_policy='next', mode='mouse', visible=False)
            
            figure_opts = dict(x_axis_type=x_axis_type, y_axis_type=y_axis_type, 
                frame_height=frame_height_fig, frame_width=frame_width_fig, 
                sizing_mode='inherit')
            
            # Ordinary or stacked bar plots, where each `x` has a single bar which can 
            # be a stack of multiple ones.
            if pltkind == 'bar':
                assert x_axis_type == 'auto', dedent(f'''
                    The "{tabname}" tab and "{ablname}" ablation figure 
                    specified a bar plot but the `x_axis_type` was set to 
                    "{x_axis_type}". Make sure you specify this as `auto`.
                ''')
                figure_opts['x_range'] = aggdf[xcol].unique().tolist()
            
            # Grouped bar plots allow each x to have multiple bars side-by-side.
            if pltkind == 'grpbar':
                assert x_axis_type == 'auto', dedent(f'''
                    The "{tabname}" tab and "{ablname}" ablation figure 
                    specified a bar plot but the `x_axis_type` was set to 
                    "{x_axis_type}". Make sure you specify this as `auto`.
                ''')
                aggdf = aggdf.copy(deep=True)
                aggdf = aggdf.sort_values(by=[xcol, huecol])
                aggdf[huecol] = aggdf[huecol].astype(str)
                hueunqs = aggdf[huecol].drop_duplicates().tolist()
                aggdf[xcol] = [tuple(row) for row in aggdf[[xcol, huecol]].itertuples(index=False)]
                figure_opts['x_range'] = FactorRange(*aggdf[xcol].drop_duplicates().tolist())
                huecol = None
                # use the palette to colormap based on the x[1:2] values
                grpbarcolors = factor_cmap(xcol, palette=colors, factors=hueunqs, start=1, end=2)
            
            fig = figure(**figure_opts, tools=figtools)
            fig.toolbar.active_drag = zoomtool

            source_list = []
            fpidf_list = []
            if color_reset == 'figure':
                colors_cycle = cycle(colors)
            
            if huecol is not None:
                aggdf_gb = aggdf.groupby(huecol, sort=False, observed=True)
            else:
                aggdf_gb = (('All', aggdf),)

            for lineidx, (fpidx, fpidf) in enumerate(aggdf_gb):
                color = next(colors_cycle)
                if pltkind == 'grpbar':
                    color = grpbarcolors
                fpidf = fpidf.copy()

                fpidf = fpidf.set_index(xcol)
                with pd.option_context("future.no_silent_downcasting", True):
                    fpidf = fpidf.replace(np.inf, np.nan).infer_objects(copy=False)
                
                # Since `fpidf.interpolate()` cannot handle categorical columns, 
                # we have to temporarily convert them back.
                fpi_ctgrycols = fpidf.select_dtypes('category').columns.tolist()
                for col in fpi_ctgrycols:
                    fpidf[col] = fpidf[col].tolist()

                # When the x axis is numerical and the plot kind is a line plot, 
                # the x values must be ascending or descending.
                if (x_axis_type in ('linear', 'log')) and (pltkind in 'line'):
                    fpidf = fpidf.sort_values(by=xcol, kind='mergesort')

                # Applying the NaN missing data interpolations
                if not isinstance(fpidf.index.dtype, pd.CategoricalDtype):
                    fpidf = fpidf.interpolate(method='index', limit_direction='both')
                
                # Returning the categorical columns to 'category' dtype
                for col in fpi_ctgrycols:
                    fpidf[col] = fpidf[col].astype('category')
                    if nullstr not in fpidf[col].cat.categories:
                        fpidf[col] = fpidf[col].cat.add_categories(nullstr)

                fpidf = fpidf.reset_index()

                for col in fpidf.columns:
                    if fpidf[col].dtype == 'bool':
                        fpidf[col] = fpidf[col].astype(str)

                fpidf['y/low'] = fpidf[f'{ycol}/low']
                fpidf['y/mean'] = fpidf[f'{ycol}/mean']
                fpidf['y/high'] = fpidf[f'{ycol}/high']
                
                source = ColumnDataSource(fpidf)
                legend_label = str(fpidx)
                if pltkind == 'line':
                    line = fig.line(x=xcol, y='y/mean', source=source,  
                        legend_label=legend_label, color=color, 
                        line_width=5)
                    fig_renderes = [line]
                elif pltkind in ('bar', 'grpbar'):
                    bar = fig.vbar(x=xcol, top='y/mean', width=0.9, source=source, 
                        legend_label=legend_label, fill_color=color)
                    fig.xgrid.grid_line_color = None
                    # fig.axis.minor_tick_line_color = None
                    # fig.outline_line_color = None
                    fig_renderes = [bar]
                else:
                    raise ValueError(f'undefined plot kind = {pltkind}')

                if (marker_opts is not None):
                    marker_opts_ = marker_opts.copy()
                    marker_opts_.setdefault('color', color)
                    scat = fig.scatter(x=xcol, y='y/mean', **marker_opts_,
                        source=source)
                    fig_renderes = [scat]

                tooltip_tups = [(col, '@{'+col+'}') for col in idcols 
                    if not (fpidf[col] == nullstr).values.all()]
                if len(tooltip_tups) > 0:
                    fig.add_tools(HoverTool(renderers=fig_renderes, 
                        tooltips=tooltip_tups, **hover_opts))
                if pltkind == 'line':
                    band = Band(base=xcol, lower='y/low', 
                        upper='y/high', source=source,
                        fill_alpha=0.3, fill_color=color)
                    fig.add_layout(band)
                elif pltkind in ('bar', 'grpbar'):
                    band = Whisker(source=source, base=xcol, 
                        upper='y/high', lower='y/low', 
                        level='overlay', line_width=2.0, 
                        line_color='black')
                    fig.add_layout(band)
                else:
                    raise ValueError(f'undefined plot kind = {pltkind}')
                    
                source_list.append(source)
                fpidf_list.append(fpidf)

            assert len(bokehdict) == 0, f'unused bokeh options: {bokehdict.keys()}'

            if show_xaxis:
                fig.xaxis.axis_label = xcol
                fig.xaxis.axis_label_text_font_size = axis_label_fontsize
                fig.xaxis.major_label_text_font_size = axis_label_fontsize
                fig.xaxis.axis_label_text_color = axis_label_text_color
            else:
                fig.xaxis.visible = False

            if show_yaxis:
                ycol_ = ycol if len(ycol) <= 50 else ycol.split(colsep)[0]
                fig.yaxis.axis_label = ycol_
                fig.yaxis.axis_label_text_font_size = axis_label_fontsize
                fig.yaxis.major_label_text_font_size = axis_label_fontsize
                fig.yaxis.axis_label_text_color = axis_label_text_color
            else:
                fig.yaxis.visible = False

            fig.title.text = fig_title.format(ablname=ablname) 
            fig.legend.visible = False 
            fig.legend.destroy()

            for tick_fmt, tick_lbls, axis in [(x_tick_fmt, x_tick_lbls, fig.xaxis),
                                              (y_tick_fmt, y_tick_lbls, fig.yaxis)]:
                if tick_lbls is not None:
                    axis.ticker = tick_lbls
                if tick_fmt == 'eng':
                    msg_ = 'for eng fmt, tick labels must be specified'
                    assert tick_lbls is not None, msg_
                    mpleng = EngFormatter()
                    engfmtr = lambda n: mpleng.format_eng(n).replace(' ', '')
                    axis.major_label_overrides = {x: engfmtr(x) for x in tick_lbls}
                else:   
                    assert tick_fmt in ('eng', None)
            
            fig.outline_line_color = 'black'
            fig.min_border = 10
            # fig.min_border_left = 100
            # fig.min_border_bottom = 100

            figdict=dict(fig=fig, source_list=source_list, fpidf_list=fpidf_list)
            tab_bkfigdata.append(figdict)

        all_bkfigdata += tab_bkfigdata
        all_tabbkfigdata.append(tab_bkfigdata)
        tab_figs = [x['fig'] for x in tab_bkfigdata]
        gridfigs = gridplot(tab_figs, ncols=ncols)
        tabpanel = TabPanel(child=gridfigs, title=tabname, closable=True)
        tabs_list.append(tabpanel)
    
    tabgridfigs = Tabs(tabs=tabs_list, sizing_mode='inherit')
    all_figs = [x['fig'] for x in all_bkfigdata]
    all_tabfigs = [[x['fig'] for x in tab_bkfigdata] for tab_bkfigdata in all_tabbkfigdata]
    
    if sharey in (True, 'all'):
        for fig in all_figs:
            fig.y_range = all_figs[0].y_range
    elif sharey == 'tab':
        for tabfigs in all_tabfigs:
            for fig in tabfigs:
                fig.y_range = tabfigs[0].y_range
    elif sharey in (False, 'none'):
        pass
    else:
        raise ValueError(f'sharey={sharey} not defined')
    
    if sharex in (True, 'all'):
        for fig in all_figs:
            fig.x_range = all_figs[0].x_range
    elif sharex == 'tab':
        for tabfigs in all_tabfigs:
            for fig in tabfigs:
                fig.x_range = tabfigs[0].x_range
    elif sharex in (False, 'none'):
        pass
    else:
        raise ValueError(f'sharex={sharex} not defined')

    # Menu update callback in javascript
    code_js = """
        ////////////////////////////////////////////////////
        ///////////////// Menu Separation //////////////////
        ////////////////////////////////////////////////////
        var opts, menu, val;
        var ccd = cdeep;
        var sects = sectscds.data["sects"];
        for (let i=0; i < n_menu; i++) {
            if (i <= menu_idx) {
                if (i < menu_idx) {
                    val = sects[i];
                } else {
                    val = right_menus[0].value;
                    sects[i] = val;
                }
                ccd = ccd[val];
            } else {
                menu = right_menus[i-menu_idx];
                if (ccd == null) {
                    val = null;
                    menu.visible = false;
                    menu.options = [];
                } else {
                    opts = Object.keys(ccd);
                    if (opts.includes(sects[i])) {
                        val = sects[i];
                    } else {
                        val = opts[0];
                    }
                    ccd = ccd[val];
                    menu.visible = true;
                    menu.options = opts;
                    menu.value = val;
                }
                sects[i] = val;
            }

        }
        // The new axis
        var ycol_new = sects.filter(Boolean).join(colsep);
        
        ////////////////////////////////////////////////////
        ////////////////// Source Update ///////////////////
        ////////////////////////////////////////////////////
        // var ycol_new = cb_obj.value;
        const statnames = ["mean", "low", "high"];
        for (let j=0; j<all_srcs.length; j++) {
            var sdata = all_srcs[j].data;
            for (let k=0; k< statnames.length; k++) {
                var stn = statnames[k];
                var ycol_stn = ycol_new.concat(colsep, stn);
                var y_stn = "y".concat(colsep, stn);
                sdata[y_stn] = [];
                if (ycol_stn in sdata) {
                    for (let i=0;i<sdata[ycol_stn].length; i++) {
                        sdata[y_stn].push(sdata[ycol_stn][i]);
                    }
                }
            }
        }

        for (let j=0; j<all_srcs.length; j++) {
            all_srcs[j].change.emit();
        }

        var axlbl = ycol_new;
        if (axlbl.length > 50) {
            axlbl = axlbl.split(colsep)[0];
        }
        for (let j=0; j<all_yaxes.length; j++) {
            all_yaxes[j].axis_label = axlbl;
        }
    """
    # Creating Menus and seperating the columns
    if colsep is None:
        colsep = '######'
    cdeep = hie2deep({c: None for c in stcols}, sep=colsep)
    n_menu = max(c.count(colsep) for c in stcols) + 1
    sects = ycol.split(colsep)
    sects += (n_menu - len(sects)) * [None]
    all_menus = []
    dd = cdeep
    for i in range(n_menu):
        sect = sects[i]
        assert (dd is None) == (sect is None)
        opts = [] if sect is None else list(dd.keys())
        ddd = dict(title='Y-Axis') if i == 0 else dict(title="      ")
        menu = Select(options=opts, value=sect, **ddd,
            width=menu_width, height=50, sizing_mode='fixed')
        if sect is None:
            menu.visible = False
        menu.align = 'end'
        if dd is not None:
            assert sect in dd
            dd = dd[sect]
        all_menus.append(menu)

    sectscds = ColumnDataSource(data={'sects': sects})
    all_srcs = list(chain.from_iterable(figdata['source_list'] 
        for figdata in all_bkfigdata))
    all_yaxes = [fig.yaxis[0] for fig in all_figs]

    for menu_idx, menu in enumerate(all_menus):
        menu_args = dict(cdeep=cdeep, n_menu=n_menu, 
            menu_idx=menu_idx, right_menus=all_menus[menu_idx:], 
            sectscds=sectscds, colsep=colsep)
        data_args = dict(all_srcs=all_srcs, all_figs=all_figs, 
            all_yaxes=all_yaxes)
        jscb = CustomJS(args={**menu_args, **data_args}, code=code_js)
        menu.js_on_change('value', jscb)
    
    # heading fills available width
    # tab_screenwidth, tab_screenheight
    heading = Div(text=f'<h1 style="text-align: center">{header}</h1>', 
        width=max_scrnw-menu_width*n_menu, height=50, 
        styles={'color': header_color}, sizing_mode='stretch_width')
    
    # Add `background="black"` to the following for a dark style
    fulllayout = column(row(heading, *all_menus, sizing_mode='stretch_width', background=background_color), 
        tabgridfigs, sizing_mode='stretch_both', background=background_color)
    
    if doc_theme is not None:
        curdoc().theme = doc_theme
    
    if doc is not None:
        doc.add_root(fulllayout)
    else:
        return fulllayout

def draw_matplotlib(mpldict, aggdf):
    mpldict = hie2deep(deep2hie(mpldict))

    xcol = mpldict.pop('xcol')
    ycol = mpldict.pop('ycol')
    pltkind = mpldict.pop('kind')
    figsize = mpldict.pop('figsize', None)
    
    xlabel = mpldict.pop('xlabel', None)
    ylabel = mpldict.pop('ylabel', None)
    ylim = mpldict.pop('ylim', None)
    xlim = mpldict.pop('xlim', None)
    yscale = mpldict.pop('yscale', None)
    xscale = mpldict.pop('xscale', None)
    yticks = mpldict.pop('yticks', None)
    xticks = mpldict.pop('xticks', None)
    yticklabels = mpldict.pop('yticklabels', None)
    xticklabels = mpldict.pop('xticklabels', None)
    ylabelpad = mpldict.pop('ylabelpad', None)
    xlabelpad = mpldict.pop('xlabelpad', None)
    fig, ax = mpldict.pop('fig', None), mpldict.pop('ax', None)

    colorspec = get_subdict(mpldict, 'colorspec', pop=True)
    
    smrycfg = get_subdict(mpldict, 'smry', pop=True)
    selcol = smrycfg.pop('sel', None)
    smrycri = smrycfg.pop('cri', None)

    apkwg1 = get_subdict(mpldict, 'ax.plot', pop=True)
    apkwg2 = get_subdict(mpldict, 'ax.bar', pop=True)
    apkwg = apkwg1 if pltkind == 'line' else apkwg2

    barswidth = apkwg.pop('barswidth', 0.9)
    bargrpvar = apkwg.pop('grpby', 'x')

    akg = get_subdict(mpldict, 'ax.annotate', pop=True)
    apg = get_subdict(mpldict, 'arrowprops', pop=True)
    bbg = get_subdict(mpldict, 'bbox', pop=True)

    xtpg = get_subdict(mpldict, 'xtickparams', pop=True)
    ytpg = get_subdict(mpldict, 'ytickparams', pop=True)

    elmntsdict = get_subdict(mpldict, 'elements', pop=True)

    assert len(smrycfg) == 0, f'unused smry options: {smrycfg}'
    assert len(mpldict) == 0, f'unused mpldict options: {mpldict}'

    if ax is None:
        assert fig is None
        fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=100)
    else:
        assert fig is not None
    aggdf = aggdf.copy(deep=True)

    if selcol is not None:
        grpobj = aggdf.groupby('fpidxgrp', sort=False, observed=True)[selcol]
        assert smrycri in ('min', 'max')
        smrlocs = grpobj.idxmin() if smrycri == 'min' else grpobj.idxmax()
        smrydf = aggdf.loc[smrlocs].reset_index(drop=True)
    else:
        smrydf = aggdf
    n_smrrows = smrydf.shape[0]

    lbl2ap = dict()
    lbl2bb = dict()
    lbl2akw = dict()
    lbl2pkw = dict()
    for i_lbl, (lbl, props) in enumerate(elmntsdict.items()):
        props = dict(props)
        apl = get_subdict(props, 'arrowprops', pop=True)
        bbl = get_subdict(props, 'bbox', pop=True)
        bbl = bbl if len(bbl) > 0 else dict(facecolor='none', edgecolor= 'none')
        akl = get_subdict(props, 'ax.annotate', pop=True)
        apkwl1 = get_subdict(props, 'ax.plot', pop=True)
        apkwl2 = get_subdict(props, 'ax.bar', pop=True)
        apkwl = apkwl1 if pltkind == 'line' else apkwl2
        annotkws = {**akg, **akl}
        arrowprops = {**apg, **apl}
        bbox = {**bbg, **bbl}
        plotkws = {**apkwg, **apkwl}
        exclist = props.pop('exclude', [])
        inclist = props.pop('include', [])
        ii = np.array([True] * n_smrrows)
        if len(exclist) > 0:
            exc_il = [get_dfidxs(smrydf, excdict) for excdict in exclist]
            jj = np.stack(exc_il, axis=1).any(axis=1)
            ii = np.logical_and(ii, np.logical_not(jj))
        if len(inclist) > 0:
            inc_il = [get_dfidxs(smrydf, incdict) for incdict in inclist]
            jj = np.stack(inc_il, axis=1).any(axis=1)
            ii = np.logical_and(ii, jj)
        smrydf.loc[ii, 'label'] = lbl
        smrydf.loc[ii, 'ilbl'] = i_lbl
        lbl2pkw[lbl] = plotkws
        lbl2akw[lbl] = annotkws
        lbl2ap[lbl] = arrowprops
        lbl2bb[lbl] = bbox

    smrydf = smrydf[~smrydf['label'].isnull()].copy(deep=True)

    if pltkind == 'line':
        for (i_lbl, label), lbldf in smrydf.groupby(['ilbl', 'label'], sort=True, observed=True):
            lbldf2 = lbldf.sort_values(by=[xcol])
            plotkws = lbl2pkw[label].copy()
            color = plotkws.pop('color', None)
            color = colorspec.get(color, color)
            ax.plot(lbldf2[xcol], lbldf2[f'{ycol}/mean'], 
                color=color, label=label, **plotkws)
            ax.fill_between(lbldf2[xcol], lbldf2[f'{ycol}/low'], 
                lbldf2[f'{ycol}/high'], color=color, alpha=0.2)
    elif pltkind in ('grpbar', 'grpbarh'):
        # Computing the error bar sizes: err = [mean - low, high - mean]
        smrydf[f'{ycol}/ehigh'] = smrydf[f'{ycol}/high'].values - smrydf[f'{ycol}/mean'].values
        smrydf[f'{ycol}/elow'] = smrydf[f'{ycol}/mean'].values - smrydf[f'{ycol}/low'].values

        # Making sure all xcol and label pairs are unique
        smryxlbldf = smrydf[[xcol, 'label']]
        assert not smryxlbldf.duplicated().any(), dedent(f'''
            ّThe ("{xcol}", "label") columns did not define unique rows;
            the following combinations defined multiple rows:
                {smryxlbldf[smryxlbldf.duplicated(keep=False)]}
        ''')

        assert bargrpvar in ('hue', 'x')

        # The unique `xcol` values in the same order they appear in `smrydf`.
        xnames = smrydf[xcol].unique().tolist()
        xname2i = {xname: i_xnm for i_xnm, xname in enumerate(xnames)}
        smrydf['ixcol'] = [xname2i[xname] for xname in smrydf[xcol]]

        # Sorting smrydf according to `x` and `label` index values.
        sort_ordr = {'x': ['ixcol', 'ilbl'], 'hue': ['ilbl', 'ixcol']}[bargrpvar]
        smrydf2 = smrydf.sort_values(by=sort_ordr)

        # Finding the maximum number of labels for a single x
        grpcol = {'x': xcol, 'hue': 'label'}[bargrpvar]
        n_maxgrpdbars = max(grpdf.shape[0] for _, grpdf in smrydf2.groupby(grpcol))
        # Finding the bar widths so that they utilize 90% of the space between two groups
        wbar = barswidth / n_maxgrpdbars

        # Figuring out the numerical x location of each bar
        grpbyvars = {'x': ['ixcol', xcol], 'hue': ['ilbl', 'label']}[bargrpvar]
        for (i_xnm, xname), xdf in smrydf2.groupby(grpbyvars, sort=False, observed=True):
            n_grpdbars = xdf.shape[0]
            # Each bar takes a fixed size of `wbar`.
            barxs = np.arange(n_grpdbars) * wbar
            # Recentering all bars of this group so that they have a zero mean on the x-axis. 
            barxs = barxs - barxs.mean()
            # Shifting all bars to the right
            barxs = barxs + i_xnm
            # Registerring these bar x values
            smrydf2.loc[xdf.index, 'barx'] = barxs
        
        # The plotting of bars should happen one label at a time
        for label, lbldf in smrydf2.groupby('label', sort=False, observed=True):
            # Getting the ax.bar keyword arguments 
            plotkws = lbl2pkw[label].copy()
            color = plotkws.pop('color', None)
            color = colorspec.get(color, color)

            # The number of x values for this particular label
            n_xlbl = lbldf.shape[0]

            # The numerical x location of each bar
            barxlbl = lbldf['barx'].values
            assert barxlbl.shape == (n_xlbl,)

            # The yerr values for this particular label
            yerrlbl = np.array([lbldf[f'{ycol}/elow'], lbldf[f'{ycol}/ehigh']])
            assert yerrlbl.shape == (2, n_xlbl)

            # The bar plot rectangles
            if pltkind == 'grpbar':
                ax.bar(x=barxlbl, height=lbldf[f'{ycol}/mean'], width=wbar, 
                    label=label, color=color, yerr=yerrlbl, **plotkws)
            elif pltkind  == 'grpbarh':
                ax.barh(y=barxlbl, width=lbldf[f'{ycol}/mean'], height=wbar, 
                    label=label, color=color, xerr=yerrlbl, **plotkws)
            else:
                raise ValueError(f'undefined pltkind={pltkind}')

        # Setting the x ticks and labels
        if bargrpvar == 'x':
            if pltkind == 'grpbar':
                ax.set_xticks(np.arange(len(xnames)), xnames)
            elif pltkind  == 'grpbarh':
                ax.set_yticks(np.arange(len(xnames)), xnames)
            else:
                raise ValueError(f'undefined pltkind={pltkind}')
        elif bargrpvar == 'hue':
            x_ticks = smrydf2['barx']
            x_ticklabels = smrydf2[xcol]
            if pltkind == 'grpbar':
                ax.set_xticks(x_ticks, x_ticklabels)
            elif pltkind  == 'grpbarh':
                ax.set_yticks(x_ticks, x_ticklabels)
            else:
                raise ValueError(f'undefined pltkind={pltkind}')
        else:
            raise ValueError(f'undefined bargrpvar={bargrpvar}')

        smrydf = smrydf2    
    elif pltkind == ('bar', 'barh'):
        assert len(smrydf[xcol].unique()) == 1, dedent(f'''
            you have multiple xcol values. You may want to consider 
            using `kind: grpbar`:
                xcol: {xcol}
                xcol unique values: {smrydf[xcol].unique()}''')
        # Computing the error bar sizes: err = [mean - low, high - mean]
        yerr = smrydf[[f'{ycol}/low', f'{ycol}/high']].values.T
        yerr = yerr - smrydf[f'{ycol}/mean'].values.reshape(1, -1)
        yerr = yerr * np.array([[-1], [1]])

        x_ticks_, x_ticklabels_ = [], []
        for i_row, row in smrydf.reset_index(drop=True).iterrows():
            label = row['label']
            plotkws = lbl2pkw[label].copy()
            color = plotkws.pop('color', None)
            color = colorspec.get(color, color)
            x_tick, x_ticklabel = row['ilbl'], row['label']

            # The bar plot rectangles
            if pltkind == 'bar':
                ax.bar(x=x_tick, height=row[f'{ycol}/mean'], color=color,
                    tick_label=x_ticklabel, yerr=yerr[:, (i_row):(i_row+1)], 
                    **plotkws)
            elif pltkind == 'barh':
                ax.barh(y=x_tick, width=row[f'{ycol}/mean'], color=color,
                    tick_label=x_ticklabel, xerr=yerr[:, (i_row):(i_row+1)], 
                    **plotkws)
            else:
                raise ValueError(f'undefined pltkind={pltkind}')
            x_ticks_.append(x_tick)
            x_ticklabels_.append(x_ticklabel)
        
        if pltkind == 'bar':
            ax.set_xticks(x_ticks_)
            ax.set_xticklabels(x_ticklabels_)
        elif pltkind  == 'barh':
            ax.set_yticks(x_ticks_)
            ax.set_yticklabels(x_ticklabels_)
        else:
            raise ValueError(f'undefined pltkind={pltkind}')
    else:
        raise ValueError(f'undefined pltkind={pltkind}')

    if xlabel is not None: ax.set_xlabel(xlabel, labelpad=xlabelpad)
    if ylabel is not None: ax.set_ylabel(ylabel, labelpad=ylabelpad)
    if ylim is not None: ax.set_ylim(*ylim)
    if yscale is not None: ax.set_yscale(yscale)
    if xlim is not None: ax.set_xlim(*xlim)
    if xscale is not None: ax.set_xscale(xscale)
    if yticks is not None: ax.set_yticks(yticks)
    if yticklabels is not None: ax.set_yticklabels(yticklabels)
    if xticks is not None: ax.set_xticks(xticks)
    if xticklabels is not None: ax.set_xticklabels(xticklabels)
    if len(xtpg) > 0: ax.tick_params(axis='x', **xtpg)
    if len(ytpg) > 0: ax.tick_params(axis='y', **ytpg)

    ax2data = ax.transAxes + ax.transData.inverted()
    data2ax = ax2data.inverted()
    for label, lbldf in smrydf.groupby('label', sort=False, observed=True):
        row = lbldf.iloc[0]
        annotkws = lbl2akw[label].copy()
        text = annotkws.pop('text', None)
        xytext = annotkws.pop('xytext', None)
        label = row['label']
        if text is None: continue
        
        # The top of the bar in data coordinates
        arrowpos_x, arrowpos_y = annotkws.pop('xy', (None, None))
        xycoords = annotkws.get('xycoords', None)
        if pltkind in ('bar', 'barh'):
            assert (xycoords == 'axes fraction'), 'not implemented yet'
            assert arrowpos_x is None
            assert arrowpos_y is None
            if pltkind == 'bar':
                xy_posdata = row['x'], row[f'{ycol}/mean']
            elif pltkind == 'barh':
                xy_posdata = row[f'{ycol}/mean'], row['x']
            else:
                raise ValueError(f'undefined pltkind={pltkind}')
            xy_posfrac = data2ax.transform(xy_posdata)
        elif pltkind in ('grpbar', 'grpbarh'):
            assert arrowpos_x is None
            assert arrowpos_y is None
            if pltkind == 'grpbar':
                xy_posdata = row['barx'], row[f'{ycol}/mean']
            elif pltkind == 'grpbarh':
                xy_posdata = row[f'{ycol}/mean'], row['barx']
            else:
                raise ValueError(f'undefined pltkind={pltkind}')
            xy_posfrac = data2ax.transform(xy_posdata)
            annotkws['xycoords'] = xycoords = 'data'
        elif pltkind == 'line':
            assert (xycoords == 'axes fraction'), 'not implemented yet'
            xy_posxdata = ax2data.transform([arrowpos_x, 0.0])[0]
            xy_posydata = arrowpos_y
            if xy_posydata is None:
                lbldf2 = lbldf.sort_values(by=[xcol])
                from scipy.interpolate import interp1d
                fi = interp1d(lbldf2[xcol], lbldf2[f'{ycol}/mean'], 
                    fill_value='extrapolate')
                assert arrowpos_x is not None
                xy_posydata = fi(xy_posxdata).item()
                assert xy_posydata is not None
            xy_posdata = xy_posxdata, xy_posydata
            xy_posfrac = data2ax.transform(xy_posdata)
        else: 
            raise ValueError(f'pltkind={pltkind} undefined')
        
        if (xycoords == 'axes fraction'): xy = xy_posfrac 
        elif (xycoords == 'data'): xy = xy_posdata 
        else: raise ValueError(f'xycoords={xycoords} undefined')

        ax.annotate(xy=xy, xytext=xytext, text=text,
            arrowprops=lbl2ap[label], bbox=lbl2bb[label], **annotkws)
    
    if pltkind in ('barh', 'grpbarh'):
        ax.yaxis.set_inverted(True)

    return fig, ax

#################################################
############### Utility Functions ###############
#################################################

def squeeze_deepdict(names_deep: dict):
    """
    This is a helper function to be called inside the `squeeze_colnames` function.
    It basically takes a deep dictionary of of column names, and it will remove 
    depth levels with a single key according to a set of rules (see the implementation 
    if/else cases for details on these rules). 

    Parameters
    ----------
        names_deep: (dict) a deep dictionary of the column names. The 
            leaf values must always be `None`. 
        
        Example:
            names_deep = 
                {'perf': {'reconst': {'a': {'b': None}}}, 
                'eval': None}

    Returns
    -------
        out2: (dict) a deep dictionary of column names with the unique 
            levels removed.
    """
    

    # Leaf-level closure of recursion
    if names_deep is None:
        return None

    # First, we will recursively reduce all the children
    out1 = {key: squeeze_deepdict(subdict) for key, subdict in names_deep.items()}
    # Example:
    #   out1 = 
    #       {'perf': {'reconst': {'b': None}}, 
    #        'eval': None}

    # Next, we "carefully "check if this level can be squeezed; see the example, 
    # and how the `recons` level survived the squeezing but `a` did not. The 
    # priority in these scenarios is for the leftmost level to survive.
    out2 = out1
    if len(out1) == 1:
        # Let's name the only existing key `key1`
        key1 = tuple(out1.keys())[0]
        # The sub-dictionary of `key1` is `val1`
        val1 = out1[key1]
        if val1 is not None:
            assert isinstance(val1, dict)
            if len(val1) == 1:
                key2 = tuple(val1.keys())[0]
                val2 = val1[key2]
                # Since we have
                #   `out1 == {key1: {key2: val2}}`, 
                # then we should squeeze the `key2` level and return 
                #   `out2 == {key1: val2}`.
                out2 = {key1: val2}
            else:
                # Since we have
                #   `out1 == {key1: {key2: val2, key3: val3, ...}}`, 
                # then we should squeeze the `key1` level and return 
                #   `out2 == val1 == {key2: val2, key3: val3, ...}`.
                assert len(val1) > 1
                out2 = val1
        else:
            # Since we have
            #   `out1 == {key1: None}`, 
            # we should not make any further reductions and we should return 
            #   `out2 = out1 = {key1: None}`.
            pass

    # Example:
    #   out2 = 
    #       {'perf': {'reconst': None}, 
    #        'eval': None}
    return out2

def squeeze_colnames(col_names: list, mindepth: int = 1):
    """
    This function squeezes the unique levels given a set of hierarchical 
    (i.e., '/'-separated) column names

    Parameters
    ----------
        col_names: (list) The list of `/`-separated strings.

            Example:

                col_names = ['perf/reconst/a', 'perf/reconst/b', 'perf/eval']
        
        mindepth: (int) Having `mindepth > 1` causes the leftmost `mindepth`-levels 
            to remain intact throughout this process. The default value of 
            `mindepth=1` has a neutral effect.

    Outputs
    -------
        sqzd_names (list) The corresponding list of columns with the 
            unique levels squeezed out.

            Example:

                sqzd_names = ['perf/a', 'perf/b', 'perf/eval']
    """


    # First, we create a dummy dictionary out of this to be able to use the he
    names_dict = {name: None for name in col_names}
    # Example:
    #    names_dict = {
    #       'perf/reconst/a': None, 
    #       'perf/reconst/b': None, 
    #       'perf/eval': None}
    
    # Next, we convert this dictionary
    names_deep = hie2deep(names_dict, mindepth=mindepth)
    # Example:
    #   names_deep = 
    #       {'perf': {'reconst: {'a': None, 'b': None}, 
    #        'eval': None}

    sqzd_namesdict = deep2hie(squeeze_deepdict(names_deep))
    # Example:
    #   sqzd_namesdict = 
    #       {'perf/a': None, 'perf/b': None, 'perf/eval': None} 

    sqzd_names = list(sqzd_namesdict)
    # Example:
    #   sqzd_names = ['perf/a', 'perf/b', 'perf/eval']

    assert len(sqzd_names) == len(col_names)
    return sqzd_names

def adjust_mtrcnames(statdf):
    """
    This function adjusts a few metric names in the statdf columns.

    Parameters
    ----------
        statdf: (pd.DataFrame) The training stat dataframe.
    
    Output
    ----------
        statdf: (pd.DataFrame) The same as input's dataframe, but 
            with the columns being renamed.
    """
    colrplcmnts = {
        'mahal:1': 'mahal/abserr', 
        'mahal:2': 'mahal/sqerr',
        'uslcwass:1': 'uslcwass/abserr',
        'uslcwass:2': 'uslcwass/sqerr',
        'wslcwass1:1': 'wslcwass1/abserr',
        'wslcwass1:2': 'wslcwass1/sqerr',
        'wslcwass2:1,-0.5': 'wslcwass2:-0.5/abserr',
        'wslcwass2:1,-1.0': 'wslcwass2:-1.0/abserr',
        'wslcwass2:1,0.0': 'wslcwass2:0.0/abserr',
        'wslcwass2:1,0.5': 'wslcwass2:0.5/abserr',
        'wslcwass2:1,1.0': 'wslcwass2:1.0/abserr',
        'wslcwass2:2,-0.5': 'wslcwass2:-0.5/sqerr',
        'wslcwass2:2,-1.0': 'wslcwass2:-1.0/sqerr',
        'wslcwass2:2,0.0': 'wslcwass2:0.0/sqerr',
        'wslcwass2:2,0.5': 'wslcwass2:0.5/sqerr',
        'wslcwass2:2,1.0': 'wslcwass2:1.0/sqerr'}

    col_rnms = dict()
    for old_col in statdf.columns.tolist():
        new_col = old_col
        for old_lvl, new_lvl in colrplcmnts.items():
            new_col = new_col.replace(old_lvl, new_lvl)
        col_rnms[old_col] = new_col
    statdf = statdf.rename(columns=col_rnms)

    return statdf
