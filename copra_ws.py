# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:55:41 2017
@author: Sanket Maheshwari
"""

import warnings
import pandas as pd
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

import sys
#sys.path.insert(0, '.')
#sys.path.insert(0, './engine')

from flask import Flask, request
from system import System
from NA_Direct.shared.io_utils import load_models as load_direct_models
import time
import xml.etree.ElementTree as etree

# TEMP: Currently we have to separate the NA BP and NA Direct model code
from NA_BP.io_utils import load_models as load_bp_models

app = Flask(__name__)
sys = System()

print("Please wait, model is loading")

loaded_models = {
    "Direct": {
        "NA": load_direct_models("NA_direct_nonSAPHANA_apprved_quant_intercept", compressed=False),
        "JP": load_direct_models("JP_direct", compressed=False),
    },
    
    "BP": {
        "NA": load_bp_models("NA", compressed=False),
        "JP": load_direct_models("JP_BP", compressed=False),
    }
}

print("Model load complete")


@app.route('/optimalPricingEngine/<modelId>', methods=['POST'])
def call_system(modelId):
    quote_xml = request.data

    root = etree.fromstring(quote_xml)
    header = root.find('RequestHeader')
    quote_id = header.find("QuoteID").text + "-" + header.find("Countrycode").text

    print("Started processing quote " + quote_id)

    start_time = time.time()
    res_xml = sys.run_logic(modelId, quote_xml, loaded_models)

    processing_time = time.time() - start_time
    print("Finished processing quote %s (Processing Time: %f)" % (quote_id, processing_time))

    return res_xml


if __name__ == "__main__":
    app.run(debug=False, host='0.0.0.0')
