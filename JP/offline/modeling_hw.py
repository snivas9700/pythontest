from numpy import tile
from pandas import Series, DataFrame, concat, MultiIndex
from time import time

from offline.lme import lmeMod
from shared.modeling_hw import quant_reg_pred, quant_reg_train


def build_lme_model(train_data, formula_fe, formula_re, field_types):
    mod = lmeMod(train_data, formula_fe, formula_re, field_types)
    return mod


def train_lme_model(mod):
    lme_start_time = time()
    mod = mod.fit()

    print("Fitting lme model takes", round(time()-lme_start_time, 2), "seconds.")
    return mod


def run_quants_offline(data, idx_cols, in_feats, out_feat, grp_cols):
    dat = data.copy()

    # store trained model objects
    mod_out = DataFrame()

    skipped = []
    if len(grp_cols) > 0:
        # raw quantile calculations
        qs = concat([dat.groupby(grp_cols)[out_feat].apply(lambda s: s.quantile(q=q)).to_frame(str(q))
                        for q in [0.05, 0.5, 0.95]], axis=1)
        qs = qs.rename(columns={'0.05': 'L', '0.5': 'M', '0.95': 'H'})

        q_reg = DataFrame()
        for t3, grp in data.groupby(grp_cols):
            # train = grp.loc[grp['winloss'].eq(1)].reset_index(drop=True)
            train = grp.reset_index(drop=True)
            if train.shape[0] > len(in_feats):
                full_rank = True #added by BB on May 31st, 2018
                # establish filter to find testing data relevant to this groupby index
                # allow for arbitrary number of groupby columns
                mask = [True] * data.shape[0]
                masks = [data[c].eq(t3[i]) if len(grp_cols) > 1 else data[c].eq(t3) for i, c in enumerate(grp_cols)]
                for m in masks:
                    mask = mask & m

                # extract testing data + set all values to win = 1; i.e. predict price that would WIN the quote
                test = data.loc[mask].copy().reset_index(drop=True)
                test.loc[:, 'winloss'] = 1
#                q_mods = quant_reg_train(train.set_index(idx_cols), [0.05, 0.5, 0.95], in_feats, out_feat)
#                q_res = quant_reg_pred(test.set_index(idx_cols), in_feats, q_mods, qs.columns)
#
#                q_reg = q_reg.append(q_res)
#
#                q_mods.index = MultiIndex.from_tuples(list(zip(tile(t3, len(q_mods.index)), q_mods.index)))
#                q_mods.index.names = grp_cols + ['quantile']
#                mod_out = mod_out.append(q_mods)  # store multi-indexed quantile model parameters
                train.to_csv('./debug/quants_training_grp.csv') # for debug #Commented out for testing purpose by BB on May 31st, 2018
                try:
                   q_mods = quant_reg_train(train.set_index(idx_cols), [0.05, 0.5, 0.95], in_feats, out_feat)# if design matrix is not full rank, quantile regression will throw a ValueError 
                except ValueError:
                   full_rank = False
                if full_rank:
                   q_res = quant_reg_pred(test.set_index(idx_cols), in_feats, q_mods, qs.columns)
                   q_reg = q_reg.append(q_res)
                   q_mods.index = MultiIndex.from_tuples(list(zip(tile(t3, len(q_mods.index)), q_mods.index)))
                   q_mods.index.names = grp_cols + ['quantile']
                   mod_out = mod_out.append(q_mods)  # store multi-indexed quantile model parameters
                else:
                   print('design matrix is not full rank! Using ALL...')
                    # remove observed quantiles for skipped entries
                   qs = qs.drop(t3)
                   skipped.append(t3)
            else:
                print(('Not enough samples in {g} level {v} ({n} samples)'.format(g=grp_cols, v=t3, n=train.shape[0])))
                # remove observed quantiles for skipped entries
                qs = qs.drop(t3)
                skipped.append(t3)

    # calculate population-level to fill in for segment_id values that were skipped
    qs_ = Series(data={str(q): dat[out_feat].quantile(q=q) for q in [0.05, 0.5, 0.95]})
    qs_ = qs_.rename(index={'0.05': 'L', '0.5': 'M', '0.95': 'H'})
    qs_.name = 'ALL'

    qs = qs.append(qs_)

    train = dat
    test = dat.copy()
    test.loc[:, 'winloss'] = 1
    q_mods = quant_reg_train(train.set_index(idx_cols), [0.05, 0.5, 0.95], in_feats, out_feat)

    q_mods.index = MultiIndex.from_tuples(list(zip(tile('ALL', len(q_mods.index)), q_mods.index)))
    q_mods.index.names = grp_cols + ['quantile']
    mod_out = mod_out.append(q_mods)
    
    q_mods.to_csv('./debug/q_mods.csv')#updated by B.B. on May 30th, 2018
    
    if len(skipped) > 0:
        # ID the entries that were skipped above
        mask = [True] * test.shape[0]
        masks = [test[c].isin(skipped[i]) if len(grp_cols) > 1 else test[c].isin(skipped) for i, c in enumerate(grp_cols)]
        for m in masks:
            mask = mask & m
        # run predictions on subset of data that was skipped
        test_ = test.loc[mask, :].set_index(idx_cols)
        # NOTE: Because q_mods is multiindexed, and quant_reg_pred() can't handle that, have to
        #       extract out the relevant q_mods entry before passing in
        #q_res = quant_reg_pred(test_, in_feats, q_mods.xs('ALL', level='segment_id'), qs_.index) #commented out on May 30th, 2018 by BB
        q_res = quant_reg_pred(test_, in_feats, q_mods.xs('ALL', level=grp_cols[0]), qs_.index)
        
        # update outputs predictions
        q_reg = q_reg.append(q_res)

    pred_cols = [c for c in q_reg.columns if 'pred_' in c]  # get prediction labels

    return q_reg, qs, pred_cols, mod_out
