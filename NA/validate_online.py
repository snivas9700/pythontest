from datetime import datetime as dt
from pandas import DataFrame
from os import path as os_path

from online_main import process_quote
from shared.io_utils import load_models, load_data

try:
    wd = os_path.dirname(__file__)
except NameError:  # local dev/ipython; replace with user's path
    wd = '/Users/jbrubaker/projects/copra_project/copra_hw'


DATA_PATH = os_path.join(wd, 'data')
src_path = os_path.join(DATA_PATH, 'NA')
data = load_data(src_path, 'testing')

t = dt.now()

qids = data['quoteid'].unique()[:5]

mods = {'NA': load_models(region='NA', compressed=False)}

q_track = DataFrame()
for qid in qids:
    print(qid)
    dat = data.loc[data['quoteid'].eq(qid)].reset_index(drop=True)
    q = process_quote(dat, mods)
    q_track = q_track.append(q, ignore_index=True)

print('Running online flow across testing data took {}'.format(dt.now()-t))
