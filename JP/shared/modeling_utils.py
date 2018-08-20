from numpy import maximum, minimum, arange
from pandas import Series, DataFrame
from copy import copy


def price_adj(low, med, high):
    """low=Low Price(% of Ref), med=Median Price(% of Ref), high=High Price(% of Ref)"""
    """
     Written by Glenn Melzer on:  22 Feb 2016
      Last update:                1 Apl 2018 by Yuxi Chen

    This function ensures that the Low, Median, and High prices (as percents
    of reference) are in the correct order.  It then calculates the skew of
    the curve and if it is excessively low (defined by min_skew), then
    high price is increased to compensate for what appears to be an entitled
    discount that prevents the high price points to be captured in the
    historical data.  (The smaller the skew, the more the high price is
    increased.)  The is also an adjustment to the median price if the skew
    is excessively high (defined by max_skew).  In this case, the median price
    is increased to reduce the skew in an attept to make the median price
    more reasonable.

    INPUT:
      low   (Low price(% of Ref) - the price that yeilds a 95% chance of winning)
      med   (Median price(% of Ref) - the price that yeilds a 50% chance of winning)
      high  (High price(% of Ref) - the price that yeilds a 5% chance of winning)

    OUTPUT:
      low   (Low price(% of Ref) - the price that yeilds a 95% chance of winning)
      med   (Median price(% of Ref) - the price that yeilds a 50% chance of winning)
      high  (High price(% of Ref) - the price that yeilds a 5% chance of winning)
    """
    min_val = .005  # this is the minimum value of low
    bound = .051  # the Median may not be closer than this to either Low or High
    min_skew = .95
    max_skew = 5
    #max_val = 1.2  # this is the maximum value of High

    '''
    # this makes any needed adjustments to the low, med, and high price points to ensure proper order
    low = maximum(minimum(low, med), min_val)  # ensures low is not too close to zero
    high = minimum(maximum(med, high), max_val)  # ensures high isn't too large
    '''
    # make sure the L/M/H are all lower bounded
    low = maximum(low, min_val)
    med = maximum(med, min_val)
    high = maximum(high, min_val)
    # then simply sort them
    [low, med, high] = sorted([low, med, high])

    if round(high, 5) == round(low, 5):  # ensures high <> low
        high = low * (1 + bound)
        med = (high + low) / 2

    #med = maximum(low, minimum(high, med))  # ensures med is between high and low


    # this adjusts the high price if the skew is too small
    if (med - low) > 0:
        skew = (high - med) / (med - low)
    else:
        skew = 999

    if skew < min_skew:
        high = maximum(high, ((minimum((2 * med) - low, 0.95)) + high) / 2)

    # this adjusts the median price if the skew is too large
    if (skew > max_skew) | (skew == 0.):
        med = (low + med + high) / 3

    return low, med, high


def calc_quant_op(l, m, h, cf, cv, fcw, fcl):
    """L=low price, M=median price, H=high price, Cf=fixed cost, Cv=variable cost(%ofnet), FCw=fin contrib if win,
    FCl=fin contrib if loss

    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                11 Jan 2016

    This function calculates the optimal price by maximizing the expected financial
    contribution of the deal.  The probability of winning is a fuction of the Low,
    Median, and High prices.  The financial contribution is based on price minus
    fixed and variable costs.  There is also an adjustment for follow-on
    incremental expected financial contribution of winning and loosing.

    INPUT:
      L   (Low price - the price that yeilds a 95% chance of winning)
      M   (Median price - the price that yeilds a 50% chance of winning)
      H   (High price - the price that yeilds a 5% chance of winning)
      Cf  (the fixed cost)
      Cv  (the variable cost, i.e. it is multiplied by the price P)
      FCw (the incremental financial contribution if the deal WINS)
      FCl (the incremental financial contribution if the deal LOSES)

    OUTPUT:
      OptPrice (the financial contribution maximizing price given the above inputs)
    """

    iterations = 8
    slices = 10
    start = l - (m - l) / 5.0 #edited by Bonnie
    finish = h + (h - m) / 5.0 #edited by Bonnie
    #start = 0.
    #finish = 3
    for i in range(iterations):
        gp_max = -99999999.0
        p_opt = -99999999.0
        d = (finish - start) / slices
        if d<1e-10: #added by Bonnie
            start = start/1.1 #added by Bonnie
            finish = finish*1.1 #added by Bonnie
            d = (finish - start)/slices #added by Bonnie
        for p in arange(start, finish, d):
            egp = exp_fc(p, l, m, h, cf, cv, fcw, fcl)
            if egp > gp_max:
                gp_max = egp
                p_opt = p
        start = p_opt - d
        finish = p_opt + d
    if cf > (p_opt * (1 - cv)):
        p_opt = cf / (1 - cv)
    optprice = p_opt
    return optprice


def exp_fc(p, l, m, h, cf, cv, fcw, fcl):
    """P=price input, L=low price, M=median price, H=high price, Cf=fixed cost, Cv=variable cost(%ofnet), FCw=fin contrib if win, FCl=fin contrib if loss"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                10 Apr 2014

    This function calculates the expected financial contribution of a particular
    price given the fixed cost, the variable cost,the probability of winning
    (from Low, Median, and High price data), and the follow-on incremental expected
    financial contribution of winning and losing.

    INPUT:
      P   (the price to be evaluated)
      L   (Low price - the price that yeilds a 95% chance of winning)
      M   (Median price - the price that yeilds a 50% chance of winning)
      H   (High price - the price that yeilds a 5% chance of winning)
      Cf  (the fixed cost)
      Cv  (the variable cost, i.e. it is multiplied by the price P)
      FCw (the incremental financial contribution if the deal WINS)
      FCl (the incremental financial contribution if the deal LOSES)

    OUTPUT:
      ExpFC (the expected financial contribution)
    """
    q = calc_win_prob(p, l, m, h)
    e = q * (p * (1 - cv) - cf + fcw) + (1 - q) * fcl
    return e


def calc_win_prob(p, l, m, h):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                07 May 2015

    This function finds the probability of winning a deal using price P given a
    marketplace defined with a market probability curve of Low, Median, and High.

    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)

    OUTPUT:
      Q (the win probability)
    """
    s = best_trans(p, l, m, h)
    wp = cum_prob(s)
    return wp


def best_trans(p, l, m, h):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                07 May 2015

    This function performs either a linear or hyperbolic transformation depending on
    whether the H, M, and L data is linear or not.

    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)

    OUTPUT:
      S (the number of Standard Deviations from the mean (or median))
    """
    l1 = (1.0 * (h - m) / (m - l)) - 1
    if abs(l1) <= 0.0005:
        t = lin_trans(p, l, m, h)
    else:
        t = hyper_trans(p, l, m, h)
    return t


def cum_prob(S):
    """ S=number of standard deviations """
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                06 May 2015

    This function calculates the cumulative probablility under a normal
    bell curve as a function of the distance from zero measured in
    standard deviations.

    INPUT:
      S (The number of standard deviations)

    OUTPUT:
      Q (The cumulative probability)
          [NOTE: in limit, large S->0%, large negative S->100%]

    FORMULAS USED:
      Z(S) = (e^(-(S^2)/2))/sqrt(2Pi)
      t(S) = 1/(1 + (p * abs(S)))
      q(S) = Z(S) * (t(b1 +t(b2+t(b3+t(b4+t*b5))))
      Q(S) = If S<0 then Q(S)= 1 - Q(S)
    This function takes S (the number of standard deviations from zero
    and calculates the area under a normal bell curve to the right of S.
    Example: If S=0, then CumProb=50%; if S=1.644853, then CumProb=5%,
    if S=-1.644853, then CumProb=95%.

    CONSTANTS REQUIRED:
      sqrt(2Pi) =  2.506628275
      P         =  0.231641900
      b1        =  0.319381530
      b2        = -0.356563782
      b3        =  1.781477937
      b4        = -1.821255978
      b5        =  1.330274429
    """
    zs = (2.7182818285 ** (-(S * S) / 2)) / 2.506628275
    ts = 1 / (1 + 0.2316419 * abs(S))
    qps = zs * (ts * (0.31938153 + ts * (-0.356563782 + ts * (1.781477937 + ts * (-1.821255978 + ts * 1.330274429)))))
    if S < 0:
        p = 1 - qps
    else:
        p = qps
    return p


def lin_trans(p, l, m, h):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                07 May 2015

    This function transforms a point P into the number of standard
    deviations it is from the mean.  It is based on a normal probability
    curve that uses Low price (95% probability of winning),
    Median (50% prob), and High (5% prob).
    NOTE:  This transformation is linear and assumes that (H - M) = (M - L)

    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)

    OUTPUT:
      S (the number of Standard Deviations from the mean (or median))

    FORMULAS USED:
      m1 = C1 / (H - M)
      b1 = m1 * M
      S = m1 * P + b1

    CONSTANTS REQUIRED:
    c1 = 1.644853475  'The std dev for a 5% cumulative probability.

    NOTES:
    Positive S values are to the right of the mean; negative to the left
    If P = L, then LinTrans = -1.644853475; if P = M, then LinTrans = 0;
    if P = H then LinTrans = 1.644853475.  All other values of P are
    translated linearly.
    """
    c1 = 1.644853475
    m1 = c1 / (h - m)
    b1 = -m1 * m
    t = (m1 * p) + b1
    return t


def hyper_trans(p, l, m, h):
    """P=price input, L=low price, M=median price, H=high price"""
    """
    Written by Glenn Melzer on:   25 Aug 2006
      Last update:                09 Apr 2014

    This function transforms a point P into the number of standard
    deviations it is from the mean.  It is based on a normal probability
    curve that uses Low price (95% probability of winning),
    Median price (50% prob), and High price(5% prob).

    Notes:  This transformation is non-linear and assumes that (H - M) <> (M - L)
    It assumes that the data best fit a hyperbolically skewed normal bell
    curve.  This transformation removes the skew from the data and outputs
    the standard deviation from the mean so normal bell curve cumulative
    probability analysis is possible.
    If P = L, then HyperTrans = -1.644853475; if P = M, then HyperTrans = 0; if P = H
    then HyperTrans = 1.644853475.  All other translations of P fall on a hyperbolic
    curve defined by the points (L,-1.644853475), (M,0), and (H,1.644853475).

    The function may not stable when:
         (P < L)   or   (P > H)
    In these cases the S value may flip to the other half of the hyperbola.
    This usually doesn't happen unless the curve is highly skewed - when M
    is relatively close to L or M.  Logic has been added after the hyperbolic
    skew to ensure that this potential anomoly is prevented.

    INPUT:
      P (the price to be probability tested)
      L (Low price - the price that yeilds a 95% chance of winning)
      M (Median price - the price that yeilds a 50% chance of winning)
      H (High price - the price that yeilds a 5% chance of winning)

    OUTPUT:
      S (the number of Standard Deviations from the mean (or median))

    FORMULAS USED:
      B = (2HL - M(H + L))/(2M - H - L)
      A = c1((M + B)(H + B))/(M - H)
      C = -A / (M + B)
      S = (A / (P + B)) + C

    CONSTANTS REQUIRED:
    c1 = 1.644853475  'The std dev for a 5% cumulative probability.
    """
    c1 = 1.644853475
    b = (2. * h * l - m * (h + l)) / (2. * m - h - l)
    a = c1 * ((m + b) * (h + b)) / (m - h)
    c = -a / (m + b)
    temp = (a / (p + b)) + c
    t = temp

    # Logic added to prevent unstable results:
    if (p < l) and (temp > c1):
        t = -c1 * ((m - p) / (m - l))
    elif (p > h) and (temp < c1):
        t = c1 * ((p - m) / (h - m))
    return t


# TODO - adjust input variables to match lowercase naming scheme
def opt_price_ci(OptPrc, L, M, H, Cf, Cv=0, FCw=0, FCl=0, Pct=.95):
    """OptPrc=Optimal Price, L=low price, M=median price, H=high price, Cf=fixed cost, Cv=variable cost(%ofnet), FCw=fin contrib if win, FCl=fin contrib if loss, Pct=% of expected GP at Optimal Price"""
    """
    Written by:   Glenn Melzer on:   18 Mar 2016
    Last update:  Glenn Melzer on:   30 Jun 2016              

    This function calculates the price range confidence interval around the
    optimal price.  The approach is based on the idea that the optimal price
    maximizes the expected profit contribution given the price uncertainty 
    defined by the L, M, and H price points.  This function assumes that the 
    user may be willing to accept a different price than the optimal price as long
    as the different price doesn't imply a significantly lower expected profit
    contribution.  The Pct value indicates how much lower the expected profit
    contribution can be.  Pct has a default value of .95, meaning that the 
    confidence interval price points (one below the optimal price and one above
    the optimal price) will have an expected profit contribution of 95% of the
    optimal price's profit contribution.  The purpose of these confidence 
    interval price points is to communicate to the user how much flexibility
    the user has in straying off the optimal price before it significantly 
    affects the deal.  Some deals will have broad confidence intervals and
    some will be narrow.

    INPUT:
      OptPrc (Optimal Price - around which is calculated the interval)
      L   (Low price - the price that yeilds a 95% chance of winning)
      M   (Median price - the price that yeilds a 50% chance of winning)
      H   (High price - the price that yeilds a 5% chance of winning)
      Cf  (the fixed cost)
      Cv  (the variable cost) [default=0]
      FCw (the incremental financial contribution if the deal WINS) [default=0]
      FCl (the incremental financial contribution if the deal LOSES) [default=0]
      Pct (the percent of the Optimal Price's expected GP that defines the interval) [default=.95]

    OUTPUT:
      ConfPriceLow  (the confidence interval price below the optimal price)
      ConfPriceHigh (the confidence interval price above the optimal price)
    """
    # this define the iterations and number of slices to find the confidence price point
    iterations = 8
    slices = 10

    # this section is for finding the lower confidence interval price point
    #  this defines the range in which to find the confidence interval price point
    Finish = OptPrc
    Start = min(OptPrc, L) - (M - L) / 5.0
    OptPrcEGP = exp_fc(OptPrc, L, M, H, Cf, Cv, FCw, FCl)
    #  this searches through the slices within the interations to find the interval price point
    for i in range(iterations):
        GPMax = -99999999.0
        PConf = -99999999.0
        d = (Finish - Start) / slices
        # print Start, Finish, d
        if d < 1e-10:
            Start = Start/1.1
            Finish = Finish * 1.1
            d = (Finish - Start) / slices #added by BB on May 30th, 2018
            
        for P in arange(Start, Finish, d):
            ConfEGP = -abs(exp_fc(P, L, M, H, Cf, Cv, FCw, FCl) - (Pct * OptPrcEGP))
            if ConfEGP > GPMax:
                GPMax = ConfEGP
                PConf = P
        Start = PConf - d
        Finish = min(PConf + d, OptPrc)
    ConfPriceLow = PConf

    # this section is for finding the higher confidence interval price point
    #  this defines the range in which to find the confidence interval price point
    Start = OptPrc
    Finish = max(OptPrc, H) + (H - M) / 5.0
    OptPrcEGP = exp_fc(OptPrc, L, M, H, Cf, Cv, FCw, FCl)
    #  this searches through the slices within the interations to find the interval price point
    for i in range(iterations):
        GPMax = -99999999.0
        PConf = -99999999.0
        if (Start == Finish):
            if (Finish != 0):
                Start = Finish / 1.01
            else:
                Finish = Start + 0.01
        d = (Finish - Start) / slices
        # print Start, Finish, d
        for P in arange(Start, Finish, d):
            ConfEGP = -abs(exp_fc(P, L, M, H, Cf, Cv, FCw, FCl) - (Pct * OptPrcEGP))
            if ConfEGP > GPMax:
                GPMax = ConfEGP
                PConf = P
        Start = max(PConf - d, OptPrc)
        Finish = PConf + d
    ConfPriceHigh = PConf

    # this adjusts the bounds to be > 1% different than the optimal price
    # print ConfPriceLow, OptPrc, ConfPriceHigh
    if ConfPriceLow > (OptPrc * .99):  # this forces the lower bound to be at least 1% below the optimal price
        ConfPriceLow = OptPrc * .99
        # print 'Lower bound adjusted to be just below the optimal price for this component'
    if ConfPriceHigh < (OptPrc * 1.01):  # this forces the higher bound to be at least 1% above the optimal price
        ConfPriceHigh = OptPrc * 1.01

    return ConfPriceLow, ConfPriceHigh


def bound_quant(entry, q_ranks, grp_cols):
    """
    Bound 5th quantile estimate by max(5th by quant reg, 5th by rank) and
          95th quantile by min(95th by quant reg, 95th by rank)
    """

    if len(grp_cols) > 1:
        t = zip(entry[grp_cols])
        if t in q_ranks.index:    ### if t present in raw_qs index (segment_id)
            q_rank = q_ranks.xs(t)
        else:
            q_rank = q_ranks.xs('ALL')
    elif len(grp_cols) == 1:
        t = entry[grp_cols[0]]
        if t in q_ranks.index:
            q_rank = q_ranks.xs(t)
        else:
            q_rank = q_ranks.xs('ALL')
    else:  # implies component-level processing
        q_rank = q_ranks

    low = max(entry['pred_L'], q_rank['L'])
    high = min(entry['pred_H'], q_rank['H'])

    s = Series(data={'pred_L': low, 'pred_M': entry['pred_M'], 'pred_H': high}, name=entry.name)
    return s


def bound_op(data_obj, alpha):
    # anywhere TMC > LP, replace OP with LP
    #   requires adjustment of norm_op too, as downstream calcs are impacted

    if isinstance(data_obj, DataFrame):
        data_obj['orig_OP'] = data_obj['price_opt'].copy()
        mask = data_obj['tmc'] > data_obj['list_price']
        if sum(mask) > 0:
            print(('Found TMC > LP in {} samples'.format(sum(mask))))
            data_obj.loc[mask, 'price_opt'] = data_obj.loc[mask, 'list_price']
            data_obj.loc[mask, 'norm_op'] = data_obj.loc[mask, 'price_opt']/data_obj.loc[mask, 'value_score']

        lb = data_obj['price_opt'] < (alpha * data_obj['value_score'])
        if sum(lb) > 0:
            print(('Bounding {} samples at lower bound'.format(sum(lb))))
            data_obj.loc[lb, 'price_opt'] = alpha * data_obj.loc[lb, 'value_score']

        ub = data_obj['price_opt'] > data_obj['list_price']
        if sum(ub) > 0:
            print(('Bounding {} samples at upper bound'.format(sum(ub))))
            data_obj.loc[ub, 'price_opt'] = data_obj.loc[ub, 'list_price']
    else:
        data_obj['orig_OP'] = copy(data_obj['price_opt'])
        if data_obj['tmc'] > data_obj['list_price']:
            data_obj.loc['price_opt'] = data_obj.loc['list_price']
            data_obj.loc['norm_op'] = data_obj.loc['price_opt']/data_obj.loc['value_score']

        if data_obj['price_opt'] < (alpha * data_obj['value_score']):
            data_obj.loc['price_opt'] = alpha * data_obj.loc['value_score']

        if data_obj['price_opt'] > data_obj['list_price']:
            data_obj.loc['price_opt'] = data_obj.loc['list_price']

    return data_obj
