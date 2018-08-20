from os import mkdir, listdir
from os.path import join as os_join, abspath, dirname
from pandas import concat, read_csv

from write_data import write_model_dict, write_pickle

# try:
#     _ = abspath(__file__)
# except NameError:
#     MODEL_PATH = '/Users/jbrubaker/projects/copra_project/copra_hw/models'
# else:
#     MODEL_PATH = os_join(dirname(dirname(abspath(__file__))), 'models')  # move one level up


def write_models(comp_model, c_quant, folder, region, compressed):
    # store model outputs to has folder
    write_model_dict(comp_model,
                     fname=os_join(folder, region+'comp_model.json'),
                     convert_keys=True)

    fnames = {'c_quant': 'comp_quant.pkl'}
    if compressed:
        fnames['c_quant'] += '.gz'

    p_file = os_join(folder, region+fnames['c_quant'])
    write_pickle(p_file, c_quant)

    # store model outputs to local models/ folder
    # NOTE: because the structure of the models/ folder hasn't been confirmed, logic here will make children
    #       directories, if missing
    # print(('Attempting to save models to models directory at path {}'.format(MODEL_PATH)))
    # try:
    #     listdir(os_join(MODEL_PATH, region))
    # except OSError:
    #     print(('Folder for output region {r} does not exist. Creating'.format(r=region)))
    #     mkdir(os_join(MODEL_PATH, region))
    #
    # m_file = os_join(MODEL_PATH, region, 'comp_model.json')
    # write_model_dict(comp_model,
    #                  fname=m_file,
    #                  convert_keys=True)
    #
    # p_file = os_join(MODEL_PATH, region, fnames['c_quant'])
    # write_pickle(p_file, c_quant)



# def load_models(region, compressed=False):
#     seg_dict = load_segmentation(region)
#     seg_mod = load_model(region, 'comp')
#
#     c_mod = load_model(region, 'c_quant', compressed)
#     q_mod = load_model(region, 'q_quant', compressed)
#
#     quant_mod = {'pn': c_mod, 'qt': q_mod}
#
#     return seg_mod, seg_dict, quant_mod
#
#
# def load_data(src_path, label='training'):
#     csvs = [x for x in listdir(src_path) if label in x]
#
#     data = concat([read_csv(os_join(src_path, csv)) for csv in csvs])
#     data = data.drop('Unnamed: 0', axis=1) if 'Unnamed: 0' in data.columns else data
#     data.columns = [x.lower() for x in data.columns]
#
#     return data