from lme import lmeMod
from pandas import Series, DataFrame
from time import time
from copy import deepcopy
from numpy import unique, absolute, mean, inf, sum as np_sum
from statsmodels.regression.quantile_regression import QuantReg



def lme_train(train_data, config):
    # # fit model and issue prediction
    cols = list(config['FIELD_TYPES'].keys())
    mod = lmeMod(train_data[cols], formula_fe=config['model']['formula_fe'], formula_re=config['model']['formula_re'], field_types=config['FIELD_TYPES'])
    lme_start_time = time()
    mod = mod.fit()

    print("Fitting lme model takes", round(time()-lme_start_time, 2), "seconds.")
    return mod


def lme_predict(test_data, mod, ind, label='TESTING'):
    test_ = test_data.loc[:, mod.field_types.keys()]
    if ind == 'hw':
        test_['won'] = 1
    y_pred = mod.predict(test_).fillna(0.)
    test_data['y_pred'] = y_pred.values

    # enforce hard bounds on the 1-discount prediction
    test_data.loc[test_data['y_pred'] > 1.0, 'y_pred'] = 1.0
    test_data.loc[test_data['y_pred'] < 0.05, 'y_pred'] = 0.05

    if ind == 'hw':
        target = 'p_pct_list_hw'
    else:
        target = 'p_pct_list_hwma'

    test_data['APE'] = absolute(test_data['y_pred'] - test_data[target]) / test_data[target]
    test_data['APE2'] = absolute(test_data['y_pred'] - test_data[target]) / test_data['y_pred']

    # if isinstance(y_pred, Series):
    #     y_pred.index = y_pred.index.astype(int)
    # clean up distances
    # rel_dist = rel_dist.fillna(0.).replace([-inf, inf], 0.0)  # any discount = 0.0 generates distance of +/-inf
    mape = mean(test_data['APE'])
    print(('MAPE on {l} predictions: {m}'.format(l=label, m=mape)))

    return test_data


# Convert from R-like output to fully explicit model for use in the real-time flow
def process_params(mod, data_in, name_dict, KNOWN_CATEGORICALS, KNOWN_BINARIES):
    # print('Expanding model dataframe...')

    # don't alter the stored model parameters
    fe = deepcopy(mod.fe)
    re = deepcopy(mod.re)

    # step 1: track index+column fields and their mappings back to original names
    idx_dict = {}
    col_dict = {}
    for k in list(re.keys()):
        for j in list(re[k].keys()):
            idx_dict.update({x: (name_dict[x] if x in list(name_dict.keys()) else x) for x in re[k][j].index.names})
            col_dict.update({x: (name_dict[x] if x in list(name_dict.keys()) else x) for x in re[k][j].columns})

    # p.reset_index(inplace=True)

    # step 2: consolidate the mixed effects weights

    # find unique mixed effects columns
    # me_cols = unique([x.split('_grp')[0] for x in p.columns if '_grp' in x])
    #
    # for col in me_cols:
    #     # find all mixed effects for given column
    #     # NOTE: any column with ":" in it is an interaction term and should not be included here
    #     cols = [x for x in p.columns if col in x and ':' not in x]
    #     # overwrite column with sum of all mixed effects
    #     p[col] = p[cols].sum(axis=1)
    #
    #     # remove column from list of mixed effects cols and drop those mixed effects cols
    #     cols.remove(col)
    #     p.drop(cols, axis=1, inplace=True)

    # step 3: interpolate missing categoricals (i.e. per cat, add missing value back in and set weigh to zero)
    # NOTE: binaries can be conformed to the same process, but want binaries to be processed AFTER categoricals
    #           so that the interaction terms will be handled properly
    for cat in KNOWN_CATEGORICALS + KNOWN_BINARIES:

        if cat in KNOWN_BINARIES:
            # Binaries are implicitly 1 if present, and 0 if missing. Make this explicit by adding the 1 to
            #   each present binary category. From here, binary can be treated same as categorical
            # p.columns = [x.replace(cat, cat+'1') if cat in x else x for x in p.columns]
            fe.index = [x.replace(cat, cat+'1') if cat in x else x for x in fe.index]

        fe = interpolate_missing(fe, cat, data_in)

    # step 4 A: convert model column names back to original data names FOR FIXED EFFECTS
    # NOTE: This is based on a partial match; cannot do in a one-liner
    col_list = []
    for col in fe.index:
        for k, v in name_dict.items():
            if k in col:
                col = col.replace(k, v)
        col_list.append(col)
    fe.index = col_list

    # establish return object
    ret = {}
    ret.update({'fixed_effects': fe})
    ret['mixed_effects'] = {}

    # step 4 B: convert model column names back to original data names FOR MIXED EFFECTS
    for k in list(re.keys()):
        store = {}
        # random effects has 3 dataframes per hierarchy; one df per level. iterate across all dataframes
        for j in list(re[k].keys()):
            df = re[k][j]
            df.index.names = [idx_dict[x] if x in list(idx_dict.keys()) else x for x in df.index.names]
            df.columns = [col_dict[x] if x in list(col_dict.keys()) else x for x in df.columns]
            store.update({j: df})

        ret['mixed_effects'].update({k: store})

    return ret


def interpolate_missing(p, cat, data_in):
    # find all indices that are interaction terms with category "cat" (if any)
    intrxn = [[y for y in x.split(':') if cat not in y] for x in p.index if cat in x and ':' in x]
    # reduce down to unique entries
    intrxns = list(unique([i for s in intrxn for i in s]))

    # all possible values for this cat from training data
    # NOTE: binary categoricals will have numeric values; convert to strings
    data_vals = [str(x) for x in data_in[cat].unique()]
    # all values for this cat output by model
    model_vals = [x.split(cat)[-1] for x in p.index if cat in x and ':' not in x]

    # Find all missing values in the model output for this cat
    missing = list(set(data_vals) - set(model_vals))
    for entry in missing:  # should only be one missing value
        add_col = cat + entry
        # add missing categorical back in with zero weight
        p[add_col] = 0.

        # Add in any interaction terms that should also include this missing category:
        for intr in intrxns:
            intr_col = intr + ':' + add_col
            p[intr_col] = 0.

    # add key character '-' to identify categorical
    for val in data_vals:
        col = cat + val
        repl_col = cat + '-' + val
        p.index = [x.replace(col, repl_col) if col in x else x for x in p.index]

    return p