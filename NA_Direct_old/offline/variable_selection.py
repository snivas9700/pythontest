# COPRA Code (ZX 04/03/2017)

from pandas import DataFrame
from numpy import nan, ones, argmax, unique, mean, ceil
import statsmodels.api as sm
from itertools import combinations
from copy import deepcopy

################################################################################################################
# Variable Selection
################################################################################################################


def delete_odds(X, Y, residual):
    # Delete the data points that result in the largest deviance residual
    idx = abs(residual).argmax()
    # idx = abs(residual).sort_values(ascending=False).iloc[:N].index

    Y = Y.drop(idx)
    X = X.drop(idx)
    return X, Y


def glm_lm_del_odd(X, Y, reg_vals, data_vol, min_score):
    # GLM_LM: (generalized) linear model with delete_odds()
    # Y: DataFrame of response variables
    # X: DataFrame of selected independent variables, corresponding to Xvar_sub
    # x_var_sub = X.columns.tolist() #an array of strings indicates the variables to be selected except for intercept
    # x_var: an array of strings includes all the possible variables and intercept if required
    # data_vol: maximum fraction of data need to be retained
    # min_score: minimum score of data quality to be assured of regression
    # param: [[lower bound], # lower bound on regression coefficients
    #         [upper bound], # upper bound on regression coefficients
    #         [pval threshold] # p-value thresholds used to determine variable selection
    #         [var hierachy]] # indicator of the hierachy (importance) for each variable

    reg_key, x_var, y_var, x_param = reg_vals[0], reg_vals[1], reg_vals[2], reg_vals[3]

    num_data = Y.shape[0]  # num_data: A integer indicating the sample size
    # Define result, a DataFrame containing the output information
    resdf = DataFrame({}, index=x_var)  # old: ['const'] + x_var
    resdf['param'] = nan
    resdf['pvalue'] = ones(len(x_var))  # initial pval = ones to assure (resdf.pvalue >= pval).any() is True
    # Run the regression, if
    # 1. outlier deletion is not beyond the percentage allowed by data_vol
    # 2. data quality is good enough for regression checked by score()>min_score

    # number of points that are in 1% of input data
    # will drop this many points on each loop when dropping outliers
    # N = int(ceil(0.01 * Y.shape[0]))

    while (len(Y) / num_data >= data_vol) and (score(X, Y, reg_vals) > min_score):
        try:
            # Binomial_Logit Regression
            if reg_key == 'logit':
                result = sm.GLM(Y, X, family=sm.families.Binomial()).fit()
                residual = result.resid_deviance
            if reg_key == 'linear':
                result = sm.OLS(Y, X).fit()
                residual = result.resid

            resdf['param'] = result.params
            resdf['pvalue'] = result.pvalues

            # Delete outliers if pval is insignificant
            # NOTE: NaN can never be greater than the pval_lim, therefore any missing feature will return False
            #       on comparison. This check will only apply to those features present
            if any(resdf['pvalue'].sort_index() > x_param.xs('pval_lim').sort_index()):
                X, Y = delete_odds(X, Y, residual)
            else:
                break
        except:  # TODO figure out what kind of exception this is and catch it specifically
            print('hit exception')
            resdf = DataFrame({}, index=x_var)  # old: ['const'] + x_var
            resdf['param'] = nan
            resdf['pvalue'] = ones(len(x_var))  # initial pval = ones to assure (resdf.pvalue >= pval).any() is True
            break

    return resdf


def regress_select_const(data, reg_vals, x_var_sub, data_vol, min_score):
    # data: a DataFrame obj includes all the columns listed by x_var and y_var
    # regmethod will define the x_var, y_var and x_param corresponding to the reg_key
    # y_var: a list of dependent variables, data[y_var] is DataFrame of Y variables
    # x_var: a list of all possible intercept and independent variables,
    # x_var_sub: a sub list of x_var (intercept is included if and only if it is a key variable)
    # data_vol: maximum fraction of data need to be retained
    # min_score: minimum score of data quality to be assured of regression
    # x_param: [[var hierarchy] # indicator of the hierarchy (importance) for each variable
    #          [pval threshold] # p-value thresholds used to determine variable selection
    #          [lower bound], # lower bound on regression coefficients
    #          [upper bound]] # upper bound on regression coefficients
    # reg_names = data.columns.tolist()
    # get and check all control paramters
    x_var, y_var, x_param = reg_vals[1], reg_vals[2], reg_vals[3]

    # Define result, a DataFrame containing the output information
    resdf = DataFrame({}, index=x_var)
    resdf['param'] = nan
    resdf['pvalue'] = ones(len(x_var))  # initial pval = ones to assure (resdf.pvalue >= pval).any() is True
    # get column by
    Y = data[y_var].copy().reset_index(drop=True)
    # get data columns listed by x_var_sub, where data doesn't have a column of ones named by 'const'
    x_var_sub = [x for x in x_var_sub if x != 'const']
    X = data[x_var_sub].copy().reset_index(drop=True)
    # if intercept is included in original list of input features, try adding it and running model
    if 'const' in x_param.columns:
        X = sm.add_constant(X, prepend=True)  # adds column of 1s
        resdf = glm_lm_del_odd(X, Y, reg_vals, data_vol, min_score)

        x_param_sub = x_param.loc[:, resdf.index.values]

        # check if const p-value is significant (to within limit)
        const_insig = resdf['pvalue']['const'] > x_param_sub.xs('pval_lim')['const']

        # check that there are ANY coefficients below their respective lower bounds
        # lb_violate = resdf['param'].lt(x_param_sub.xs('lower_bound')).any()
        lb_violate = (resdf['param'].sort_index() < x_param_sub.xs('lower_bound').sort_index()).any()

        # check that there are ANY coefficients above their respective upper bounds
        # ub_violate = resdf['param'].gt(x_param_sub.xs('upper_bound')).any()
        ub_violate = (resdf['param'].sort_index() > x_param_sub.xs('upper_bound').sort_index()).any()

        # if const is KEY, it cannot be dropped, regardless of violations
        const_alt = x_param_sub.xs('hierarchy')['const'] == 0

        #: if any violations (insig, lb, ub) AND const_alt, then drop constant variable and try again
        if (const_insig or lb_violate or ub_violate) and const_alt:
            Y = data[y_var].copy()
            X = data[x_var_sub].copy()  # try fitting again, without the column of ones
            resdf = glm_lm_del_odd(X, Y, reg_vals, data_vol, min_score)
    else:
        resdf = glm_lm_del_odd(X, Y, reg_vals, data_vol, min_score)

    return resdf.fillna(0.)


def check_var_pval(resdf, x_param):
    xpar = x_param[list(resdf.index)]

    #: Establish penalties for violating the bounds:
    #:    - any violation of KEY variable (e.g. hier = 1) penalized by 1
    #:    - any violation of ALT variable (e.g. hier = 0) penalized by 0.5

    # param = 0 and pvalue = 0 will allow missing entries to pass all checks
    dat = resdf.fillna(0.).join(xpar.T)
    # do any violate upper bound?
    # key_ub_violate = resdf['param'].gt(xpar.xs('upper_bound')).multiply(xpar.xs('hierarchy'))
    # alt_ub_violate = resdf['param'].gt(xpar.xs('upper_bound')).multiply((1 - xpar.xs('hierarchy')) * 0.5)
    key_ub_violate = dat['param'].gt(dat['upper_bound']).multiply(dat['hierarchy'])
    alt_ub_violate = dat['param'].gt(dat['upper_bound']).multiply((1 - dat['hierarchy']) * 0.5)
    coef_ubd = key_ub_violate + alt_ub_violate  # upper bound violation penalty

    # do any violate lower bound?
    # key_lb_violate = resdf['param'].lt(xpar.xs('lower_bound')).multiply(xpar.xs('hierarchy'))
    # alt_lb_violate = resdf['param'].lt(xpar.xs('lower_bound')).multiply((1 - xpar.xs('hierarchy')) * 0.5)
    key_lb_violate = dat['param'].lt(dat['lower_bound']).multiply(dat['hierarchy'])
    alt_lb_violate = dat['param'].lt(dat['lower_bound']).multiply((1 - dat['hierarchy']) * 0.5)
    coef_lbd = key_lb_violate.add(alt_lb_violate)  # lower bound violation penalty

    # Check that all coefficients are significant and within bounds
    # coef_sig = resdf['pvalue'].le(xpar.xs('pval_lim'))  # check that coefficient is significant to within pval_lim
    # ub_pass = resdf['param'].le(xpar.xs('upper_bound'))  # check coefficient is <= upper bound
    # lb_pass = resdf['param'].ge(xpar.xs('lower_bound'))  # check coefficient is >= lower bound
    coef_sig = dat['pvalue'].le(dat['pval_lim'])
    ub_pass = dat['param'].le(dat['upper_bound'])
    lb_pass = dat['param'].ge(dat['lower_bound'])

    pval_ind = (coef_sig & ub_pass & lb_pass).all()

    # vector shows pvalue and bound-check scores for all variables
    # pval_vec = resdf['pvalue'].add(coef_ubd).add(coef_lbd)
    pval_vec = dat['pvalue'].add(coef_ubd).add(coef_lbd)

    # scalar shows overall degree of pvalue and bound violation for all variables
    pval_tot = pval_vec.sum()

    # shows pvalue and bound violation for ALT variables
    # pval_alt = pval_vec.multiply(~coef_sig).multiply(1 - xpar.xs('hierarchy'))
    pval_alt = pval_vec.multiply(~coef_sig).multiply(1 - dat['hierarchy'])

    check = (pval_ind, pval_tot, argmax(pval_alt))
    return check


def var_drop(data, reg_vals, x_var_sub, x_var_alt, data_vol, min_score, limit):
    # If the current subset of Xvar is insignificant: drop a variable from this subset after a myopic check
    # Repeat the above process, until:
    # 1. find some subset of Xvar significant
    # 2. the number of alternative variables in the current subset is below limit

    x_var, x_param = reg_vals[1], reg_vals[3]

    x_var_key = x_param.loc[:, x_param.xs('hierarchy').eq(1)].columns.values

    # if const is in Xvar but optional, then it won't be included in Xvar_alt and Xvar_sub
    # else const is either excluded in Xvar or included in Xvar_key
    # Xvar_sub: a sub list of Xvar (const is included if and only if it is a key variable)
    num_max = len(x_var_alt)  # maximum number of variables to be deleted

    # note: 1.const is never in Xvar_alt; 2. either Xvar_key or Xvar_alt is possibly empty
    # initial run of regression
    resdf = regress_select_const(data, reg_vals, x_var_sub, data_vol, min_score)
    check = check_var_pval(resdf, x_param)  # check = (pval_ind, pval_tot, pval_loc)
    # Drop one variable in each iterative, while:
    # 1. number of remaining alternatives is more than limit
    # 2. no model is qualified so far
    var0_out_list = []  # list of variables eventually removed during iterations
    num = len(var0_out_list)  # number of variables already been removed
    while (num < num_max - limit) and not check[0]:
        # check[2] is the name of the variable with the worst pval rating, per check_var_pval()
        var0_out_list.append(check[2])
        x_var_alt_temp = list(set(x_var_alt) - set(var0_out_list))
        x_var_sub_temp = x_var_key + x_var_alt_temp

        resdf = regress_select_const(data, reg_vals, x_var_sub_temp, data_vol, min_score)
        check = check_var_pval(resdf, x_param)
        # Break, if either model is qualified or variable deletion reaches the limit
        if check[0] or (len(var0_out_list) == num_max - limit):
            # get the remaining variables after drop var0
            x_var_sub = list(set(x_var_sub) - set(var0_out_list))
            break
        else:
            num = len(var0_out_list)

    return resdf, x_var_sub


def var_trim(data, reg_vals, x_var_sub, x_var_alt, data_vol, min_score, limit):
    # If the current subset of Xvar is insignificant: drop a variable from this subset by 1-step forward looking check
    # Repeat the above process, until:
    # 1. find some subset of Xvar significant
    # 2. the number of alternative variables in the current subset is below limit
    num_max = len(x_var_alt)  # maximum number of variables to be deleted

    # note: 1.const is never in Xvar_alt; 2. either Xvar_key or Xvar_alt is possibly empty
    # initial run of regression
    resdf = regress_select_const(data, reg_vals, x_var_sub, data_vol, min_score)
    check = check_var_pval(resdf, reg_vals)  # check = (pval_ind, pval_tot, pval_loc)

    x_param = reg_vals[3]
    x_var_key = x_param.loc[:, x_param.xs('hierarchy').eq(1)].columns.values

    # Trim one variable in each iterative, while:
    # 1. number of remaining alternatives is more than limit
    # 2. no model is qualified so far
    var0_out_list = []  # list of variables eventually removed during iterations
    num = len(var0_out_list)  # number of variables already been removed
    while (num < num_max - limit) and not check[0]:

        var_out_list = []  # list of variable to be removed from Xvar_alt_temp
        resdf_list = []  # list of all resdf by removing 1 more variable with 1-step forward looking
        check_list = []  # list of quality check corresponding to each DataFrame obj in resdf_list

        x_var_alt_temp = list(set(x_var_alt) - set(var0_out_list))
        for var in x_var_alt_temp:  # any single variable to pop out of Xvar_alt_temp
            x_var_sub_temp = list(set(x_var_key) + (set(x_var_alt_temp) - set(var)) )

            resdf_temp = regress_select_const(data, reg_vals, x_var_sub_temp, data_vol, min_score)
            check_temp = check_var_pval(resdf_temp, reg_vals)

            var_out_list.append(var)  # Use it as the index of checkdf
            resdf_list.append(resdf_temp)  # append() always add the new resdf in sequence
            check_list.append(check_temp)

        # Select the best model in the resdf_list, based on the check_list
        pval_ind, pval_tot, pval_loc = list(zip(*check_list))  # checkdf has three columns: 'ind', 'tot', 'loc'
        # checkdf has 3 columns: ['ind', 'tot', resdf'], index is the list of var
        checkdf = DataFrame({'ind': pval_ind, 'tot': pval_tot, 'resdf': resdf_list},
                            index=var_out_list)
        # checkdf_sorted = checkdf.sort_index(by=['ind', 'tot'], ascending=[False, True])
        checkdf_sorted = checkdf.sort_values(by=['ind', 'tot'], ascending=[False, True], axis=0)

        var0 = checkdf_sorted.index[0]  # get the mostly preferred variable to be removed
        var0_out_list.append(var0)

        resdf = checkdf.resdf[var0]  # get the mostly preferred result from the resdf_list
        check = (checkdf.ind[var0], checkdf.tot[var0])  # get the mostly preferred check scores

        # Break, if either some model is qualified or variable deletion reaches the limit
        if True in pval_ind or len(var0_out_list) == num_max - limit:
            # get the best combination of variables after trim
            x_var_sub = list(set(x_var_sub) - set(var0_out_list))
            break
        else:
            num = len(var0_out_list)

    return resdf, x_var_sub


def var_comb(data, reg_vals, x_var_sub, x_var_alt, data_vol, min_score, limit):
    x_param = reg_vals[3]

    # x_var_sub will be updated in the loop if conditions pass and some variables are removed, or the condition in
    # the loop will never pass and it will be returned as-is

    # While the current subset of x_var is insignificant by removing any combination of m variables from x_var_alt:
    # Exhaustively search any subset of x_var by removing any combination of m+1 variables from x_var_alt:

    num_max = len(x_var_alt)  # maximum number of variables to be deleted

    # note: 1.const is never in x_var_alt; 2. either x_var_key or x_var_alt is possibly empty
    # initial run of regression without any variable deletion
    num = 0  # count the current size of variable combinations to be removed, start at 0
    resdf = regress_select_const(data, reg_vals, x_var_sub, data_vol, min_score)

    check = check_var_pval(resdf, x_param)  # check = (pval_ind, pval_tot, pval_loc)

    # try to remove any combination of num+1 variables in the coming iterative, while:
    # 1. current size of variable combinations is less than num_max - limit
    # 2. no result is satisfied by removing any combination of num variables from x_var_alt
    while num < num_max - limit and not check[0]:
        var_list = []  # list of combinations of num+1 variables to be deleted in this iteration
        resdf_list = []  # list of all resdf (DataFrame obj) by removing num+1 variables from x_var_sub
        check_list = []  # list of results from p-value check for each resdf in the resdf_list

        # add one more variable into the combinations, and get all combinations of num+1 variables in x_var_alt
        # NOTE: Each iteration of the while loop adds 1 additional variable to the combinations generated.
        #       For example, loop 0 runs through all combinations of 1 variable, loop 1 runs through all
        #       combinations of 2 variables, etc
        x_var_combo = set(combinations(x_var_alt, num + 1))
        for var_combo in x_var_combo:  # for any combo of num variables from x_var_alt
            # remove var_combo combination of variables from input x_var_alt
            # NOTE: Can never remove a KEY feature (i.e. a KEY feature is never in var_combo)
            x_var_temp = list(set(x_var_sub) - set(var_combo))

            # run outlier removal and calculate pvalue + coefficient weights
            resdf_temp = regress_select_const(data, reg_vals, x_var_temp, data_vol, min_score)
            check_temp = check_var_pval(resdf_temp, x_param)

            var_list.append(var_combo)  # use it as the index of checkdf
            resdf_list.append(resdf_temp)  # append() add the resdf in the same sequence as in the 'for' loop
            check_list.append(check_temp)  # append() add the check in the same sequence as in the 'for' loop

        # Select the best model in the resdf_list, based on the check_list
        pval_ind, pval_tot, pval_loc = list(zip(*check_list))

        checkdf = DataFrame({'ind': pval_ind, 'tot': pval_tot, 'resdf': resdf_list}, index=var_list)

        # sort checkdf on 'ind' DESCENDING and 'tot' ASCENDING - so pick the index of the combo where
        #   all features passed the pvalue check and/or had the lowest total pval violation value
        # get the varcombo mostly preferred to be removed
        # worst_combo = checkdf.sort_index(by=['ind', 'tot'], ascending=[False, True]).iloc[0, :]
        best_combo = checkdf.sort_values(by=['ind', 'tot'], ascending=[False, True], axis=0).iloc[0, :]
        # default to returning the BEST resdf values, not the WORST !!
        resdf = best_combo['resdf']
        # result of check[0] is used in while loop check. If BEST combo DOES NOT pass check[0], then remove
        #   WORST combo and move forward
        check = (best_combo['ind'], best_combo['tot'])

        # Use this sorting to find the WORST parameter combination and potentially drop it from consideration
        worst_combo = checkdf.sort_values(by=['ind', 'tot'], ascending=[True, False]).iloc[0, :]

        # break the while iteration if
        # 1. some model is qualified after removing varcombo in this iteration
        # 2. the num of vars in varcombo reaches the limit
        # if True in pval_ind or (len(worst_combo.name) == num_max - limit):
        #     x_var_sub = list(set(x_var_sub) - set(worst_combo.name))
        if True in pval_ind or (len(best_combo.name) == num_max - limit):
            x_var_sub = list(set(x_var_sub) - set(best_combo.name))
            break
        else:
            # num = len(worst_combo.name)
            num = len(best_combo.name)

    return resdf, x_var_sub


def regress_select_var(data, reg_vals, data_vol, min_score, limit=[6, 4, 0]):
    # Regression allows multiple methods to select variables, with limit = [drop_lim, trim_lim, comb_lim]
    # use var_drop() when len(x_var_alt) > drop_lim
    # use var_trim() when drop_lim >= len(x_var_alt) > trim_lim
    # use var_comb() when trim_lim >= len(x_var_alt) > comb_lim
    # data_vol: maximum fraction of data need to be retained
    # min_score: minimum score of data quality to be assured of regression
    ##################################################################################
    # data: a DataFrame obj includes all the columnes listed by Xvar and Yvar
    # y_var: a list of dependent variables, data[y_var] is DataFrame of Y variables
    # x_var: a list of all possible intercept and independent variables,
    # param: [[lower bound], # lower bound on regression coefficients
    #         [upper bound], # upper bound on regression coefficients
    #         [pval threshold] # p-value thresholds used to determine variable selection
    #         [var hierarchy]] # indicator of the hierarchy (importance) for each variable
    # reg_names = data.columns.tolist()
    x_var, y_var, x_param = reg_vals[1], reg_vals[2], reg_vals[3]

    drop_lim = limit[0]  # max variables allowed in model
    trim_lim = limit[1]
    comb_lim = limit[2]

    x_var_key = x_param.loc[:, x_param.xs('hierarchy').eq(1)].columns.values

    # all input features, minus 'const' if 'const' is ALT
    x_var_sub = x_param.xs('hierarchy')
    if ('const' in x_var_sub) and (x_var_sub['const'] == 0.):
        x_var_sub = x_var_sub.drop('const').index.values
    else:
        x_var_sub = x_var_sub.index.values

    # array of alternatives (ALTs) still alive
    x_var_alt = x_param.loc[:, x_param.xs('hierarchy').eq(0)].columns
    if 'const' in x_var_alt:  # if 'const' is an ALT variable, DROP IT (per old methodology)
        x_var_alt = x_var_alt.drop('const').values
    else:
        x_var_alt = x_var_alt.values

    # TODO - confirm we want to preserve these. With the current model definitions, we will only ever use var_comb()
    if len(x_var_alt) > drop_lim:
        resdf, x_var_sub = var_drop(data, reg_vals, x_var_sub, x_var_alt, data_vol, min_score, drop_lim)
        x_var_alt = list(set(x_var_sub) - set(x_var_key))

    if (len(x_var_alt) <= drop_lim) and (len(x_var_alt) > trim_lim):
        resdf, x_var_sub = var_trim(data, reg_vals, x_var_sub, x_var_alt, x_param, data_vol, min_score, trim_lim)
        x_var_alt = list(set(x_var_sub) - set(x_var_key))

    # if the number of ALT variables passes thresholds, go into var_comb() and see about dropping some
    if (len(x_var_alt) <= trim_lim) and (len(x_var_alt) > comb_lim):
        resdf, x_var_sub = var_comb(data, reg_vals, x_var_sub, x_var_alt, data_vol, min_score, comb_lim)

    return resdf, x_var_sub


def main_regression(data, reg_vals, data_vol, min_score):
    # Main regression function combines variable selection and outlier deletion to produce regression results
    # that are significant and correct
    # data: DataFrame includes all possible independent variables and response variables
    # regmethod: dictionary of regression method: name of regression variables, response variables, and
    # regulations on regression parameters
    # data_vol: maximum fraction of data points need to be retained
    reg_vals_temp = deepcopy(reg_vals)

    x_param = reg_vals_temp[3]

    resdf, x_var_comb = regress_select_var(data, reg_vals_temp, data_vol, min_score, limit=[4, 4, 1])
    check = check_var_pval(resdf, x_param)
    if not check[0]:
        x_param.T['pval_lim'] = x_param.xs('pval_lim').add(0.025 * x_param.xs('hierarchy'))
        reg_vals_temp = (reg_vals_temp[0], reg_vals_temp[1], reg_vals_temp[2], x_param)

        resdf, x_var_comb = regress_select_var(data, reg_vals_temp, data_vol, min_score, limit=[4, 4, 1])
        check = check_var_pval(resdf, x_param)

        if not check[0]:
            x_param.T['pval_lim'] = x_param.xs('pval_lim').add(0.025 * x_param.xs('hierarchy'))
            reg_vals_temp = (reg_vals_temp[0], reg_vals_temp[1], reg_vals_temp[2], x_param)

            resdf, x_var_comb = regress_select_var(data, reg_vals_temp, data_vol, min_score, limit=[10, 4, 0])

    # If the result is still insignificant, there is nothing we can do, return whatever the last result is.
    return resdf['param']


def score(X, Y, reg_vals):
    # Helper function to calculate the scores for the quality of data input for regression.
    # Response: Response variable for the regression model
    # Regressor: Parameters for the score calculation
    # Return: The score value for the data inputs
    reg_key = reg_vals[0]

    # score_size = 0  # 0~4
    # score_win = 0  # 0~2
    num_data = Y.shape[0]
    num_var = X.shape[1]

    # Green (1991) indicates that N>50+8m (where m is the number of independent variables) is needed for
    # testing multiple correlation N>104+m for testing individual predictors.
    # Harris (1985) says that the number of participants should exceed the number of predictors by at least 50 (50+m).
    # Van Voorhis & Morgan (2007) using 6 or more predictors the absolute minimum of participants should be 10 (10m).
    # Though it is better to go for 30 participants per variable (30m).
    # ZX 2017.06.16: add a threshold of data volume '20+num_var'
    score_size = (num_data >= 20 + num_var) + \
                 (num_data >= 50 + num_var) + \
                 (num_data >= min(50 + 8 * num_var, 104 + num_var)) + \
                 (num_data >= max(50 + 8 * num_var, 104 + num_var, 30 * num_var))

    if reg_key == 'logit':
        num_win = sum(Y.values[:, 0])
        win_rate = (sum(Y.values[:, 0]) / num_data).item()
        score_win = ((win_rate > 0.05) and (win_rate < 0.55)) + ((win_rate > 0.15) and (win_rate < 0.45))
        # ZX 2017.06.16: data is not qualified unless score_size > 0
        score_val = 0 if (score_win == 0 or score_size == 0 or num_win < 4) else (score_size + score_win)
    elif reg_key == 'linear':
        score_val = score_size

    return score_val


def data_score(data, reg_vals):
    reg_key, x_var, y_var = reg_vals[0], reg_vals[1], reg_vals[2]

    num_data = data.shape[0]  # number of records in data
    num_var = len(x_var)  # number of regression variables
    num_fold = len(unique(data.fold)) if 'fold' in data.columns else 1  # count unique fold IDs in each data set

    # number of data in the smallest training set = num_data - number of data in the largest testing set (fold)
    min_train = num_data - max(data.groupby('fold')['fold'].count()) if 'fold' in data.columns else 0  # smallest training data set

    # score_size = 0 # 0~3; score_fold = 0 # 0~2; score_win = 0 # 0~2; score_mape = 0 # 0~3
    # Green (1991) indicates that N>50+8m (where m is the number of independent variables) is needed for
    # testing multiple correlation N>104+m for testing individual predictors.
    # Harris (1985) says that the number of participants should exceed the number of predictors by at least 50 (50+m).
    # Van Voorhis & Morgan (2007) using 6 or more predictors the absolute minimum of participants should be 10 (10m).
    # Though it is better to go for 30 participants per variable (30m).
    # ZX 2017.06.16: add a threshold of data volume '20+num_var'
    score_size = (num_data >= 20 + num_var) + \
                 (num_data >= 50 + num_var) + \
                 (num_data >= min(50 + 8 * num_var, 104 + num_var)) + \
                 (num_data >= max(50 + 8 * num_var, 104 + num_var, 30 * num_var))

    score_fold = (num_fold >= 2 and min_train >= num_var) + (min_train >= min(50 + num_var, 10 * num_var))

    # no training set is ready if there is a single fold (if num_fold < 2)
    if reg_key == 'logit':
        win_rate = sum(data[y_var[0]]) / num_data
        score_win = ((win_rate > 0.05) and (win_rate < 0.55)) + ((win_rate > 0.15) and (win_rate < 0.45))
        # Assume 'Value' and 'SUM_QUOTED_PN_PRICE' are ready in data
        mape = mean(abs(1 - data['value_qt'] / data['quoted_qt']))
        score_mape = (mape < 0.30) + (mape < 0.20) + (mape < 0.10)
        # Computing the overall score ZX 2017.06.16
        score_val = 0 if (score_win == 0 or score_fold == 0 or score_size == 0) else \
            score_size + score_fold + score_win + score_mape
    else:
        score_val = (score_size + score_fold) * 2

    return score_val
