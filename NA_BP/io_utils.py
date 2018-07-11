from os.path import join as os_join, abspath, dirname

from NA_BP.api_io.load_data import load_segmentation, load_model

try:
    _ = abspath(__file__)
except NameError:
    MODEL_PATH = '/Users/jbrubaker/projects/copra_project/copra_hw/models'
else:
    MODEL_PATH = os_join(dirname(abspath(__file__)), 'models')


def load_models(region, compressed=False):
    seg_dict = load_segmentation(region)
    seg_mod = load_model(region, 'comp')

    c_mod = load_model(region, 'c_quant', compressed)
    q_mod = load_model(region, 'q_quant', compressed)

    quant_mod = {'pn': c_mod, 'qt': q_mod}

    return seg_mod, seg_dict, quant_mod
