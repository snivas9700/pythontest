from collections import OrderedDict
from numpy import inf

segmethod = OrderedDict([('categorical_1', [['leading_brand'], None]),
                         ('binary_2', [['indirect'], None]),
                         ('tree_3', [['ln_vs', 'discount_qt'], [0, 10000, 1, 50, 20]])])

regmethod = {
    # list of modeling methodologies - logit, linear, etc.
    'logit':
    {
        'feats': {  # list of input features
            'const': {
                'hierarchy': 0,         # designations of KEY (1) or ALT (0) variables. ALTs can potentially be dropped
                'pval_lim': 0.15,       # max pval allowed per feature
                'lower_bound': -inf,    # lower bound on parameter weight (coefficient)
                'upper_bound': inf      # upper bound on parameter weight (coefficient)
            },
            'q_v':          {'hierarchy': 1, 'pval_lim': 0.05, 'lower_bound': -inf, 'upper_bound': 0},
            'gap_ev':       {'hierarchy': 0, 'pval_lim': 0.15, 'lower_bound': -inf, 'upper_bound': inf},
            'ln_ds':        {'hierarchy': 0, 'pval_lim': 0.15, 'lower_bound': -inf, 'upper_bound': 0},
            'tc':           {'hierarchy': 0, 'pval_lim': 0.15, 'lower_bound': -inf, 'upper_bound': inf}
        },
        'targets': ['winloss'],  # list of target (output) variables
    }
}

# Define optimization problem in optmethod
opt_method = {'revenue_logit': {
    'decision_vars': ['quoted_price'],  # list of prediction variables
    'var_bounds': {
        'lower': 0,  # lower-bound of dec var
        'upper': 'list_price'}  # upper-bound of dec var
}
}
