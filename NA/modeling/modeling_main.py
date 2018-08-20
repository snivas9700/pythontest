from numpy import array, where

from NA.modeling.modeling_utils import find_model, apply_model, calc_win_prob, calc_bounds, apply_bounds, calc_profit
from NA.modeling.segmentation_utils import navigate_tree


def apply_lme(row, m_dict):
    """
    Takes a single component and attempts to calculate a value score. First the appropriate model is found via 
     helper function find_model(). If a model is found, helper function apply_model() is called, which performs 
     some logic (to avoid breaking and to allow for transforms) as it applies the model to the data. 
    
    :param row: A component-level entry
    :type row: pandas.Series
    :param m_dict: Component-level models, in a tree structure
    :type m_dict: dict
    :return vs: Calculated value score of the component
    :rtype: float or NoneType
    """

    if 'mixed_effects' in list(m_dict.keys()):
        vs = 0.
        for k in list(m_dict['mixed_effects'].keys()):
            model = navigate_tree(row, m_dict['mixed_effects'][k])
            if model is not None:
                vs += apply_model(row, model)
            else:
                print('Couldnt find mixed-effects model for component. Will not adjust value score calculation')
                vs = 0.

        if vs is not None:
            vs += apply_model(row, {'inputs': m_dict['fixed_effects']})  # run through fixed effects

    return vs


def calc_quote_constants(quote, model, price_field='Q_V'):
    """
    Similar logic to modeling.calc_value_score() - reads in quote object and appropriate quote model and then 
     applies that model. In this case, though, it is a little more complicated than a simple weighted sum - most model 
     features contribute to beta0, but at least one will apply to beta1, which will become a weight to the price 
     variable during optimization. So, separate out those features and split the quote model into two separate models
     that can each be handled by apply_model()

    :param quote: Prepped quote-level data object with all pre-calculated values
    :type quote: pandas.Series
    :param model: Quote-level model to apply
    :type model: dict
    :param price_field: The coefficient to apply to the optimal price values during optimization search
    :type price_field: str
    :return beta0, beta1: Calculated constants to use for profit optimization
    :rtype: float, float
    """

    feats = model['inputs']

    # extract out weight for price
    beta1 = [x for x in feats if x['feat_name'] == price_field][0]['weight']

    subset_model = {'inputs': [x for x in feats if x['feat_name'] != price_field]}
    beta0 = apply_model(quote, subset_model)

    return beta0, beta1


def search_profit(b0, b1, min_norm_price=0.0, max_norm_price=3.0, res=1E3, plot=False):
    """
    Runs search on profit using normalized (1/total_vs) prices in range (min_norm_price, 3) with 3*res step points 
     each spaced 1/res apart. Calculates profit at each normalized point, then finds max() of profit points, 
     which should be the normalized price value that maximizes profit, and therefore the optimal price.
    
    :param b0: beta0; offset used in calculating parameter for logit curve
    :type b0: float
    :param b1: beta1; weighting on price as part of calculating parameter for logit function
    :type b1: float
    :param min_norm_price: minimum allowed price. Determines where the search is allowed to start at
    :type min_norm_price: float
    :param max_norm_price: maximum allowed price. Determines where the search must end
    :type max_norm_price: float
    :param res: Dictates the number of points to calculate derivative at. Also limits the step-size between each 
                calculated point.
    :type res: numeric; int or float
    :param plot: Whether or not to plot optimal price search values. Useful for debugging
    :type plot: bool
    :return optimal, win_prob: Tuple of optimal normalized price and the win probability of that optimal price
    :rtype: float, float
    """

    # range() doesn't support float values
    # NOTE: add res + 1 to include the bound
    vals = array(list(range(int(min_norm_price*res), int(max_norm_price*(res+1.)))))/res

    profit = calc_profit(b0, b1, vals)

    idx = where(profit == max(profit))[0][0]

    optimal = vals[idx]

    if plot:
        import matplotlib.pyplot as plt
        pr = calc_win_prob(b0, b1, vals)

        plt.figure()
        plt.plot(profit, 'black')
        plt.plot(vals, 'blue')
        plt.plot(pr, 'red')

    win_prob = calc_win_prob(b0, b1, optimal)

    return optimal, win_prob


def run_quote_model(quote, m_dict, ep_bound=True, vs_bound=True):
    """
    Uses prepped quote object to calculate the optimal price - the price at which the profit function is maximized.
     The flow is simple, so it probably shouldn't change:
        * Find appropriate model for the quote (via find_model() )
        * Use model to calculate beta0 and beta1, weights used in logistic function
        * Search the profit function using calculated beta0, beta1. Find (normalized) optimal price
        * Project normalized price into original space
        * Trim optimal price, if necessary (dictated by trim_price flag)
        
    NOTE: Current bounds are defined as follows
        * EP-based bounds: 0.15*EP <= x <= EP
        * VS-based bounds: 0.9*VS <= x <= 1.1*VS
    
    :param quote: Prepped quote object with all fields pre-calculated
    :type quote: pandas.Series
    :param m_dict: All quote models, separated by segment_id
    :type m_dict: dict
    :param ep_bound: Whether or not to apply Entitled Price (EP) bounds to calculated optimal price
    :type ep_bound: bool
    :param vs_bound: Whether or not to apply Value Score (VS) bounds to the calculated optimal price
    :type vs_bound: bool
    :return opt_price, win_prob: Calculated (and possible trimmed) optimal price, along with win
                                  probability evaluated at the (normalized) optimal price, and any error code
    :rtype: float, float, int or NoneType
    """
    err_code = None

    model = find_model(quote, m_dict)

    if model is not None:
        beta0, beta1 = calc_quote_constants(quote, model)

        max_range = 3. * quote['ENTITLED_SW'] / float(quote['value'])
        opt_price_norm, win_prob = search_profit(beta0, beta1, min_norm_price=0.0, max_norm_price=max_range, res=1E4)
        opt_price = opt_price_norm * quote['value']  # rescale back to original range

        quote['beta0'] = beta0
        quote['beta1'] = beta1

        if ep_bound or vs_bound:
            lb, ub = calc_bounds(quote, ep=ep_bound, vs=vs_bound)

            opt_price, delta = apply_bounds(opt_price, lb, ub)
            win_prob = calc_win_prob(beta0, beta1, opt_price/quote['value'])
        else:
            delta = 0.

        disc = 1. - (opt_price/quote['ENTITLED_SW'])

        quote['optimal_price'] = opt_price
        quote['win_prob'] = win_prob
        quote['discount'] = disc
        quote['delta'] = delta
        return quote, err_code
    else:
        SCRIBE.scribe('Couldnt generate quote optimal price, no model found.', "error")
        quote['optimal_price'] = None
        err_code = 3
        return quote, err_code
