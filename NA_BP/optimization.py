from numpy import isnan, exp, nan, array, ceil, array_split, reshape, tile, expand_dims, transpose, where, argmax
from pandas import merge, DataFrame
from itertools import compress
from time import time

from utils import timeit
from method_defs import opt_method


def w_prob_logit_2d(alpha, xdat, coeff):
    # inputs will be matrices

    tmp = alpha * xdat * coeff
    t = tmp.sum(axis=1)

    y = 1 / (1 + exp(-t))

    # apply rule: if not abs(coeff).sum() > 0, then y = 0.
    #   otherwise, wherever abs(coeff).sum() = 0, y = 0.5
    y[(coeff.sum(axis=1) == 0.)] = 0.

    return y


def w_prob_logit_np(search, b0, b1, op_fail):
    t = b0 + b1 * search  # N x K

    y = 1. / (1. + exp(-1. * t))  # N x K

    y[op_fail] = 0.

    return y


def merge_data_coef(data, coef):
    # Merge coef(_cv) to data. coef(_cv) is DataFrame, index=tier_0,...,tier_layer,(fold). Columns have suffix=layer
    if 'fold' in coef.index.names:  # If 'fold' is found in coef, search it in data, for the purpose of matching
        if 'fold_cv' in data.columns:
            print('Error: data has been unfolded as a training data set.')  # Current data is the training set
        if 'fold' not in data.columns:  # coef had fold ID, but the data are not ready for cross validation
            print('Error: data has not included any fold.')
            data['fold'] = 0  # put a default value in the column of'fold'
    index_list = coef.index.names
    data_coef = merge(data, coef.reset_index(), how='left', on=index_list)  # left merge coef to data

    return data_coef


@timeit
def optprice_mtier(data, coef):
    opt_key = opt_method.keys()[0]
    dec_vars = opt_method[opt_key]['decision_vars']
    bounds = opt_method[opt_key]['var_bounds']

    coef_layer_list = [l for l in coef.index.names if 'tier_' in l]  # coef_layer_list=[0] for root node
    layer = len(coef_layer_list) - 1
    # Use data_coef to prepare for data_x_var & data_x_opt
    data_coef = merge_data_coef(data, coef)  # Merge coef to data
    #: some columns have '_' in their names (before suffix). have to be careful what you split on
    split_on = '_' + str(layer)
    x_var = [c.split(split_on)[0] for c in coef.columns]
    if 'const' in x_var:
        data_coef['const'] = 1

    data_x_var = data_coef[x_var]

    mask = [not (isinstance(x, float) and isnan(x)) for x in dec_vars]
    x_dec = list(compress(dec_vars, mask))

    dec_vars = ['NA_'+str(i) if not mask[i] else x for i, x in enumerate(dec_vars)]

    data_x_opt = DataFrame({}, columns=dec_vars, index=data_coef.index)  # initialize data_x_opt
    data_x_opt[x_dec] = data_coef[x_dec]

    if len(x_dec) == 1:  # a single decision variable
        lbd, ubd = bounds['lower'], bounds['upper']
        x_min = -data_coef[lbd] if type(lbd) is str else -lbd
        x_max = data_coef[ubd].values if type(ubd) is str else ubd

        # are these necessary??
        data_coef['LW'] = x_min  # cost(or zero) as lower bound of x
        data_coef['UP'] = x_max  # list(or entitled) price as upper bound of x

    data_coef = numpy_approach(data_coef, data_x_opt, data_x_var, coef, x_var, x_dec, dec_vars)

    data_coef['discount_act'] = (data_coef['UP'] - data_coef['price_act'])/data_coef['UP']
    data_coef['discount_opt'] = (data_coef['UP'] - data_coef['price_opt'])/data_coef['UP']

    return data_coef


def update_scale_df(scale, x_dec, data_coef, price_col):
    for col in x_dec:
        scale[col] = data_coef[price_col]/data_coef[col]

    return scale


def numpy_approach(data_coef, data_x_opt, data_x_var, coef, x_var, x_dec, dec_vars):
    print('\nStarting numpy approach..')
    t_tot = time()

    rev_data = data_x_opt.join(data_x_var).join(data_coef[['LW', 'UP', 'quoteid', 'list_qt', 'tmc_qt'] + list(coef.columns)])
    rev_data['bounds'] = data_coef[['LW', 'UP']].apply(lambda row: (row['LW'], row['UP']), axis=1)
    rev_data['price_act'] = data_coef[x_dec]

    coeff_tot = rev_data[coef.columns].fillna(0.)

    # Establish target variable(s)
    xvar_b1 = ['q_v']  # to extract x data
    coef_b1 = [x for x in coef.columns if any([y in x for y in xvar_b1])]  # to extract coefficients
    # Establish input features
    xvar_b0 = [x for x in x_var if x not in xvar_b1]
    coef_b0 = [x for x in coef.columns if any([y in x for y in xvar_b0])]

    # beta0 = sum(x*c, axis=1)
    rev_data['beta0'] = (rev_data[xvar_b0].values * rev_data[coef_b0].fillna(0.).values).sum(axis=1)  # N
    rev_data['beta1'] = rev_data[coef_b1].fillna(0.)
    # NOTE: convert Q_V value back to 1/VS, so that multiplying x will result in normalized price values
    rev_data['VS'] = rev_data[x_dec].values / rev_data[xvar_b1].values   # QP / (QP/VS) = VS

    # y = 1/(1+exp(-t)); t = beta0 + (beta1 * (x/VS) )

    # NOTE: for a check after running the OP calcs, anywhere ALL coefficients are zero are defaulted to
    #       upper limit values
    rev_data['op_fail'] = rev_data[coef.columns].fillna(0.).sum(axis=1) == 0.

    rev_data['search_bound'] = rev_data['list_qt']/rev_data['VS']

    t = time()

    rev_data = run_np_wrapper(rev_data)

    print('  -> Numpy runtime: {}'.format(round(time() - t, 2)))

    t2 = time()

    # Using ALL columns would cause a lot of duplicated column names. Subset here
    use_cols = ['price_opt', 'price_act', 'premask_opt', 'op_fail', 'lb', 'ub', 'quoteid']

    data_coef = merge(data_coef, rev_data[use_cols], on='quoteid', how='left')

    # everything not modified by decision variable is set to 1.
    scale = DataFrame(data=1., index=data_coef.index, columns=dec_vars)
    # calculate the WP at the known quoted price, i.e. the "actual" price
    data_coef['wp_act'] = w_prob_logit_2d(scale.values, data_coef[x_var].values, coeff_tot.values)

    scale = update_scale_df(scale, x_dec, data_coef, 'price_opt')
    data_coef['wp_opt'] = w_prob_logit_2d(scale.values, data_coef[x_var].values, coeff_tot.values)

    print('  -> array WP calcs took {} seconds'.format(round(time() - t2, 2)))
    print('Numpy approach took {} seconds\n'.format(round(time() - t_tot, 2)))

    return data_coef


def run_np_wrapper(rev_data):
    store = DataFrame()
    n_splits = ceil(rev_data.shape[0]/1000.)
    for i, df in enumerate(array_split(rev_data, n_splits)):
        store = store.append(run_np(df))

    return store


def run_np(rev_data):
    b0 = rev_data['beta0'].values
    b1 = rev_data['beta1'].values
    bounds = rev_data[['LW', 'UP']].values
    vs = rev_data['VS'].values
    op_fail = rev_data['op_fail'].values
    rev_adj = (rev_data['tmc_qt']/rev_data['VS']).values  # normalized material cost; non-zero for hardware

    # establish search criteria

    # want to allow the predictions to run all the way up to entitled value. for this, we need to allow the search
    # range to go all the way up to ENTITLED_SW/Value. the previous setting of 3.0 was arbitrary and capped
    # certain entries. this was to keep the process efficient, but it makes testing near impossible. because
    # this logic is based on matrix math, all entries must contain as many points as the max. we rely heavily
    # on the bounding logic to bring values back to realistic numbers
    m = min(max(int(max(rev_data['list_qt']/(rev_data['VS']+1.))), 3), 100)  # max of search range for entire

    # resolution of the search space (in VS * 1/res )
    res = 1E1

    # range to search over, per quote:
    # NOTE: search space is NORMALIZED BY VS. Needs to be projected back
    # search = array(range(int(0.), int(m*res + 1)))/res

    # establish constants
    N = b0.shape[0]     # number observations
    K = int(m*res + 1)  # number points to search

    # search = reshape(tile(search, N), (N, K))  # N x K

    # Offset for revenue line (adjust HW estimates for material costs); needs to be 2D
    rev_adj = expand_dims(rev_adj, 1)  # N x 1

    # Constant offset for logistic curve
    b0 = transpose(reshape(tile(b0, K), (K, N)), (1, 0))  # N x K

    # Weight for normalized price vector
    b1 = transpose(reshape(tile(b1, K), (K, N)), (1, 0))  # N x K

    opt_prices, vs_opt, op_fail, lb, ub = rev_logit_np(m, res, b0, b1, bounds, vs, op_fail, rev_adj)

    rev_data['price_opt'] = opt_prices
    rev_data['premask_opt'] = vs_opt
    rev_data['op_fail'] = op_fail
    rev_data['lb'] = lb
    rev_data['ub'] = ub

    return rev_data


def rev_logit_np(m, res, b0, b1, bounds, vs, op_fail, rev_adj):

    max_search = run_search(m, res, b0, b1, op_fail, rev_adj, level=0, max_levels=4)

    vs_opt = max_search * vs  # N

    lb = bounds[:, 0]  # N
    ub = bounds[:, 1]  # N

    # consider the optimal price search a failure anywhere the coefficients were all zero
    # at these locations, set "optimal" to the upper bound of the search space
    # NOTE: values = tf.where(boolean_mask, values_where_TRUE, values_where_FALSE)
    fail_opt = where(op_fail, ub, vs_opt)  # N

    f_opt = where(fail_opt >= ub, ub, where(fail_opt <= lb, lb, fail_opt))  # N

    return f_opt, vs_opt, op_fail, lb, ub


def run_search(m, res, b0, b1, op_fail, rev_adj, level, max_levels, max_search=None):

    if level < max_levels:  # check that the number of searches run (level) is less than max allowed
        if level == 0:
            # first search runs from 0 to K
            search = array(range(int(0.), int(m*res + 1)))/(res**(level+1))
            search = reshape(tile(search, b0.shape[0]), (b0.shape[0], len(search)))  # N x K
        else:
            # center the search around max_search and search [-K/2, K/2]
            search = array(range(int(-1*m*res/2), int(m*res/2 + 1)))/(res**(level+1))
            search = max_search + reshape(tile(search, b0.shape[0]), (b0.shape[0], len(search)))  # N x K

        # search over normalized space
        win_prob = w_prob_logit_np(search, b0, b1, op_fail)  # N x K

        rev_2d = (search - rev_adj) * win_prob  # N x K

        #: Find location of each max revenue value - start by finding the index of the max values
        # Actual revenue values aren't important. we care about the VS-normalized QP at max revenue
        max_idx = argmax(rev_2d, axis=1)  # max(N x K, across K) = N

        N = max_idx.shape[0]

        # use the index to find the search value at max revenue. The search value corresponds to the normalized price.
        max_search = search[range(N), max_idx].reshape(-1, 1)  # [N x 1]
        return run_search(m, res, b0, b1, op_fail, rev_adj, level+1, max_levels, max_search)
    # if level == max_level, return the max_search
    else:
        return max_search.ravel()  # N
