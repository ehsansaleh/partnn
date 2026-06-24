from pathlib import Path

PROJPATH = str(Path(__file__).resolve().parents[1])

cache_dir = f'{PROJPATH}/.cache'
configs_dir = f'{PROJPATH}/configs'
results_dir = f'{PROJPATH}/results'
storage_dir = f'{PROJPATH}/storage'
data_dir = f'{PROJPATH}/data'
datalfs_dir = f'{PROJPATH}/../data_partnn'
summary_dir = f'{PROJPATH}/summary'
nullstr = '-'

keyspecs = [('hp',         'last',    'pd.cat'),
            ('stat',       'mean',        None),
            ('etc',        'last',    'pd.cat'),
            ('mon',        'mean',        None),
            ('mdl',        'last',        None),
            ('trg',        'last',        None),
            ('var/eval/*', 'last',        None)]
