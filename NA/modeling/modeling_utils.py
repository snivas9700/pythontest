

from numpy import log10, log, sqrt, exp


KNOWN_OPERATIONS = ['*', '/', '+', '-']


def apply_transform(x, transform):
    """
    Helper function containing all known transforms
    
    :param x: Any value (or list of values) to apply transform to
    :type x: float
    :param transform: Specifies the transform function
    :type transform: str
    :return: If match is found for transform, returns transformed version of x, else returns x
    :rtype: float
    """

    if transform is not None:
        if transform == 'ln':
            return log(x)
        elif transform == 'log10':
            return log10(x)
        elif transform == 'sqrt':
            return sqrt(x)
        elif transform == 'squared':
            return x**2
        else:
            SCRIBE.scribe('Unhandled transform {}. Will return unmodified values/series'.format(transform), "warn")
            return x
    else:  # Null/None transform - return original value/series
        return x


def find_model(obj, m_dict):
    """
    Simple function to search model dictionary m_dict for the segment_id of the quote object passed in.
     If no match is found, default to returning None.
    
    :param obj: Quote-level data object
    :type obj: dict
    :param m_dict: Model definitions, by segment_id
    :type m_dict: dict
    :return: Model definition, if matching segment_id is found, else None
    :rtype: dict or NoneType
    """

    s_id = obj['segment_id']
    if s_id is not None:
        try:
            model = m_dict[s_id]
        except KeyError:
            SCRIBE.scribe('No model found matching segment_id {}'.format(s_id), "warn")
            return None
        else:
            return model
    else:
        SCRIBE.scribe('No segment ID found', "warn")
        return None


def calc_simple(obj, f_info):
    """
    Contains all the logic necessary to compute a "simple" feature that is just a (potentially modified) version
     of the original, input feature. Accepts a component object and a model feature definition, and based on the 
     contents of f_info, it extracts out the appropriate field from obj, preps it if a transform is required, and 
     runs apply_operation on it to calculate the new value(s). This calculated value is then weighted according to 
     feature weight, and that weighted sum is returned.

    :param obj: Component- or quote-level entry containing all input fields
    :type obj: pandas.Series
    :param f_info: Single feature within a model definition
    :type f_info: dict
    :return: Calculated value weighted by feature weight
    :rtype: float
    """
    try:
        if ('const' in f_info['field_name'].lower()) | ('const' in f_info['match_type'].lower()):  # only special case
            val = 1.  # to be multiplied by feat['weight'] -> produces constant
        elif f_info['match_type'] == 'continuous':
            val = obj[f_info['field_name']]  # extract feature from component
        elif f_info['match_type'] == 'discrete':
            val = (str(obj[f_info['field_name']]) == f_info['match_val']) * 1.
    except KeyError:
        SCRIBE.scribe('Entry doesnt contain feature {}. Will not contribute to calculation'.format(f_info['field_name']), "warn")
        val = 0.

    return val


def apply_operation(x, y, op):
    """
    Accepts two values (x, y), either individual numerics or array-like, and evaluates an expression with x on
     the left hand side of that expression, the operation op in the middle, and y on the right hand side. A check
     is run that the operation contained within op is valid, as there are a large number of ways eval() can fail,
     and it is not reasonable to try to catch all those potential errors.
    
    :param x: numeric variable to be evaluated on the LHS of the equation
    :type x: float or list-like
    :param y: numeric variable to be evaluated on the RHS of the equation
    :type y: float or list-like
    :param op: Operation to perform between x and y
    :type op: str
    :return val: calculated value of x op y
    :rtype: float or list-like
    """

    # check that all operations can be handled
    if op in KNOWN_OPERATIONS:
        # Evaluate arbitrary expression
        # Convert variables to floats, then to strings
        val = eval('x' + op + 'y')
    else:
        SCRIBE.scribe('Cannot process operation: {}'.format(op), "error")
        raise NotImplementedError

    return val


def calc_complex(obj, feat):
    """
    Contains all the logic necessary to compute a "complex" feature that is a combination of at least two input
     features. Accepts a component object (obj, typically a Series) and a model feature definition (feat), and
     based on the contents of feat, it extracts out the relevant fields from the component object, preps them
     if a transform is required, and runs apply_operation on them to calculate the new values. This entails iterating
     across all input operations, extracting out the LHS and RHS values, and then using the operation to combine them.
     These calculated values (val) are then weighted according to feature weight, and that weighted sum is returned.
    
    :param obj: Component-level entry containing all input fields
    :type obj: pandas.Series
    :param feat: Single feature within a model definition
    :type feat: dict
    :return: calculated value weighted by feature weight
    :rtype: numeric
    """

    val = 0.
    for i, op in enumerate(feat['operation']):
        # feat['feat'][i] = LHS of operation
        # feat['feat'][i+1] = RHS of operation
        # NOTE: If a key error occurs here, meaning one of the features is not present in the component entry,
        #       allow this function to raise the KeyError to be caught in the calling function apply_model().
        #       Don't want to catch it here.
        # Complex features may involve chaining together a series of operations. If so, start with the initial features,
        # then carry through the value calculated on the previous step
        if i == 0:
            l = apply_transform(obj[feat['feat_name'][i]], feat['transform'][i])
        else:
            l = apply_transform(val, feat['transform'][i])

        r = apply_transform(obj[feat['feat_name'][i + 1]], feat['transform'][i])

        try:
            val += apply_operation(l, r, op)
        except NotImplementedError:
            SCRIBE.scribe('Feature {} unable to be properly calculated. Returning 0'.format(feat), "error")
            return 0

    return val


def calc_interaction(obj, f_info):
    """
    Helper function to calculate interaction terms, present in the linear mixed effects model. As an interaction
     is a convolution of two features, the model definition must contain full definitions for each sub-feature of
     the interaction term. Assuming this is present, this function multiplies the results of calc_simple() run on
     each interaction term, thereby generating the interaction.
    
    :param obj: Component- or quote-level data object
    :type obj: pandas.Series
    :param f_info: Feature definition for the interaction term
    :type f_info: dict (nested)
    :return: interaction term value
    :rtype: float
    """

    # For interaction terms, feat_info contains a list of both interaction features, plus info on how to handle them
    val = 1.
    for inner_feat in f_info:
        val *= calc_simple(obj, inner_feat)

    return val


def apply_model(obj, model):
    """
    Logic to read in a component- or quote-level object and apply the appropriate model to it. 
     This function sums up each calculated feature, which means this is implicitly a hard-coded
     linear model, with the weights being accounted for in the calc_simple/complex() functions. 
    
    :param obj: Component- or quote-level object containing all relevant data
    :type obj: pandas.Series or dict
    :param model: Model definition to apply to the component/quote object passed in
    :type model: dict
    :return calc: The predicted value (weighted sum of inputs)
    :rtype: float
    """

    feats = model['inputs']

    # Iterate across all features specified in model dict. If any are not found, set to 0. For each found,
    # check if a transform needs to be applied or not, and if so, attempt to apply it. Then multiply the
    # (potentially transformed) value by the model weight and add into value score
    calc = 0.
    for feat in feats:
        try:
            if feat['feat_type'] == 'simple':
                val = calc_simple(obj, feat['feat_info'])
            elif feat['feat_type'] == 'complex':
                val = calc_complex(obj, feat)  # TODO - conform calc_complex()
            elif feat['feat_type'] == 'interaction':
                val = calc_interaction(obj, feat['feat_info'])
        except KeyError:
            SCRIBE.scribe('Feature {} not found in component. Will not contribute to value score calculation'.format(feat), "warn")
            # What to do here? Return None, set this feature = 0, etc?
            calc += 0.
        else:
            calc += val * feat['weight']

    return calc


def apply_model_old(obj, model):
    """
    Given component/quote object and model passed in, apply the model (as a cumulative sum) by the following:
        * Start with an offset of model['CONSTANT'] if present, else 0
        * Per feature in the model, 
            check that feature is present in the data (else, set to 0)
            apply any transform specified by the model
            apply feature weight to (potentially transformed) feature
            add to cumulative sum
    
    
    :param obj: Component- or quote-level object containing all relevant data
    :type obj: pandas.Series or dict
    :param model: Specific model to apply to component/quote data
    :type model: dict
    :return calc: Calculated value
    :rtype: float
    """
    # value score starts at whatever the constant offset is
    # Retrieve constant from model, if exists
    if 'CONSTANT' in list(model.keys()):
        calc = model['CONSTANT']
        # model.pop('CONSTANT')
        # Can't pop because it will permanently modify model dict
        subset_model = {k: v for k, v in model.items() if k != 'CONSTANT'}
    elif 'const' in list(model.keys()):
        calc = model['const']
        # model.pop('CONSTANT')
        # Can't pop because it will permanently modify model dict
        subset_model = {k: v for k, v in model.items() if k != 'const'}
    else:
        calc = 0.
        subset_model = model

    # Iterate across all feature specified in model dict. If any are not found, set to 0. For each found,
    # check if a transform needs to be applied or not, and if so, attempt to apply it. Then multiply the
    # (potentially transformed) value by the model weight and add into value score
    for feat in list(subset_model.keys()):
        try:
            val = obj[feat]
        except KeyError:
            SCRIBE.scribe('Feature {} not found in component. Will not contribute to value score calculation'.format(feat), "warn")
            # What to do here? Return None, set this feature = 0, etc?
            val = 0.
        else:
            # If a transform is specified, attempt to apply it. Else use unmodified series
            val = apply_transform(val, model[feat]['transform'])

        calc += val * model[feat]['weight']

    return calc


def calc_win_prob(b0, b1, norm_price):
    """
    Helper function to calculate win probability (logistic function) given the two weights beta0 and beta1, 
     and the normalized price vector to calculate on. Normalized price vector can be interpreted as quote price
     divided by value score.
    
    :param b0: Constant offset in exponential term
    :type b0: float
    :param b1: Feature weight in exponential term
    :type b1: float
    :param norm_price: Range of prices over which to calculate the win probability
    :type norm_price: numpy.array
    :return: Win probability calculated at each normalized price
    :rtype: numpy.array
    """

    t = b0 + b1*norm_price
    wp = 1./(1. + exp(-1.*t))  # win probability is logistic function
    return wp


def calc_profit(b0, b1, norm_price):
    """
    Estimation of profit by convolution of the normalized price and the win probability at that price. Evaluates
     profit over a range of prices and returns all potential profits. 
    
    :param b0: Constant offset in exponential term
    :type b0: float
    :param b1: Feature weight in exponential term
    :type b1: float
    :param norm_price: Range of prices over which to calculate the win probability
    :type norm_price: numpy.array
    :return: Profit estimates across the range of prices provided
    :rtype: numpy.array
    """

    return norm_price * calc_win_prob(b0, b1, norm_price)


def calc_bounds(quote, ep, vs):
    """
    Two-phase boundary calculation. First, calculate bounds based on entitled price:
            lower = 0.15*entitled, upper = entitled
    
    Second, calculate bounds based on value_score:
            lower = 0.8*value_score, upper = 1.2*value_score
    
    The optimal price is then bounded by
            max(0.15*entitled, 0.8*value_score) <= OP <= min(entitled, 1.2*value_score)
    
    NOTE: These values are in terms of dollars (i.e. are not normalized)
    
    :param quote: Prepped quote-level data object
    :type quote: pandas.Series
    :return lower_bound, upper_bound: Calculated lower and upper bounds for given quote
    :rtype: float, float
    """

    if ep:
        ep_lb = 0.15 * quote['ENTITLED_SW']  # 85% discount, max
        ep_ub = quote['ENTITLED_SW']  # 0% discount, min

    if vs:
        # bind tightly around value score
        vs_lb = quote['value'] * 0.8
        vs_ub = quote['value'] * 1.20

    if ep and vs:
        SCRIBE.scribe('EP and VS bounds active', "info")
        lb = max(ep_lb, vs_lb)
        ub = min(ep_ub, vs_ub)
    elif ep:
        SCRIBE.scribe('EP bounds active ONLY', "info")
        lb = ep_lb
        ub = ep_ub
    elif vs:
        SCRIBE.scribe('VS bounds active ONLY', "info")
        lb = vs_lb
        ub = vs_ub

    return lb, ub


def apply_bounds(price, lower_bound, upper_bound):
    """
    Helper function to trim input price between lower and upper bounds. If bounds are applied, the distance
     between the input OP and the bounded price is captured in delta and returned. Else, delta = 0.

    :param price: Calculated optimal price
    :type price: float
    :param lower_bound: Calculated lower bound
    :type lower_bound: float
    :param upper_bound: Calculated upper bound
    :type upper_bound: float
    :return price: Trimmed optimal price
    :rtype price: float
    """

    if price < lower_bound:
        SCRIBE.scribe('Binding price at LOWER bound', "info")
        delta = lower_bound - price
        price = lower_bound
    elif price > upper_bound:
        SCRIBE.scribe('Binding price at UPPER bound', "info")
        delta = upper_bound - price
        price = upper_bound
    else:
        delta = 0.

    return price, delta
