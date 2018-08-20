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
from NA.shared.io_utils import load_models as load_models_NA
from JP.shared.io_utils import load_models as load_models_JP
import time
import xml.etree.ElementTree as etree

app = Flask(__name__)
sys = System()

print("Please wait, model is loading")

# TODO: Load the EMEA LME models here
loaded_models = {
    "Direct": {
        "NA": load_models_NA("NA_Direct_July20", compressed=False),
        "JP": load_models_JP("JP_direct", compressed=False),
    },
    
    "BP": {
        "NA": load_models_NA("NA_BP_July20", compressed=False),
        "JP": load_models_JP("JP_BP", compressed=False),
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
    app.run(debug=True, host='0.0.0.0', port = 5001)
