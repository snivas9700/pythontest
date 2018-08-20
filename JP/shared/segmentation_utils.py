# List of implemented matching category
IMPLEMENTED_MATCH_CATS = ['categorical', 'discrete', 'range', 'tree', 'binary']
RANGE_MATCHES = ['tree', 'range', 'continuous']         # expect a nested tuple to specify upper and lower bounds
EXACT_MATCHES = ['categorical', 'discrete', 'binary']   # expect exactly 3 values in key


def replace_key(d, key, to_string):
    """
    The key is converted to/from string/tuple - either from string to tuple or from tuple to string. 
     Which direction the conversion operates is dictated by to_string.
    
    This function expects keys from the segmentation definition dictionary. Conversion from string to tuple expects
     first string value to be the feature (e.g. brand), second string value to be the feature category 
     (e.g. discrete, range), all subsequent items are packed into a tuple
    
    NOTE: Replacement of key done in-place
    
    :param d: Model or segmentation definitions, at arbitrary depth
    :type d: dict
    :param key: Key to be converted
    :type key: dict or str
    :param to_string: Switch to decide which direction to convert key
    :type to_string: bool
    :return: N/A
    """
    if to_string:

        # NOTE: When converting to JSON string for push to database, the keys can't contain single quotes, or those
        #       will interfere with SQL's ability to interpret the JSON string as a variable. Need to remove
        #       single quotes here. Inversion (i.e. from string to list) handles this fine.
        if isinstance(key, tuple) | isinstance(key, list):
            # Convert from tuple to string
            # TODO - more elegant solution here?
            # make sure tuple entries are strings and not unicode - the leading 'u' will throw everything off
            key = tuple([str(k) if isinstance(k, str) else k for k in key])

            if key[1] in RANGE_MATCHES:
                vals = key[2]  # unpack nested tuple/list
                tmp_key = (key[0], key[1], vals[0], vals[1])
            else:
                tmp_key = key
            # str_key = str(key).replace("'", "").replace('"', '')
            str_key = '\t'.join([str(k).replace('.', ',') for k in tmp_key])
        else:
            str_key = str(key)  # should be conversion from int to str

        d[str_key] = d.pop(key)
    else:
        # Convert from string to tuple
        if '\t' in key:  # implies new key structure
            l = [x.strip().replace(',', '.') for x in key.split('\t')]
        else:  # old key structure
            # TEMPORARY HACK
            tmp_key = key[1:] if key[0] == '(' else key
            tmp_key = tmp_key[:-1] if tmp_key[-1] == ')' else tmp_key
            l = [str(x.strip()) for x in tmp_key.replace("'", "").replace('"', '').split(',')]
        if len(l) > 1:  # meant to be a tuple/list, apply logic
            if l[1] in EXACT_MATCHES:
                tup_key = tuple(l)  # known 3 entries
            elif l[1] == 'binary':
                try:
                    l[2] = int(l[2])
                except ValueError:
                    print(('Cannot convert binary value {} to integer. (WHY??)'.format(l[3])))
                tup_key = tuple(l)
            elif l[1] in RANGE_MATCHES:
                if '\t' in key:
                    # known 4 entries, 2 of which are actually numbers
                    tup_key = (l[0], l[1], (float(l[2]), float(l[3])))
                else:
                    tup_key = (l[0], l[1], (float(l[2].replace('(', '')), float(l[3].replace(')', ''))))
            else:
                # Should raise error, but for now will fill
                # TODO - raise NotImplementedError
                tup_key = ('MISSING', None, None)

            d[tup_key] = d.pop(key)


def convert_dict_keys(d, to_string):
    """
    Recursively navigates dictionary tree and replaces all keys at child nodes below parent node, then replaces key
     at parent node. If child node has further children nodes, logic steps through to repeat the process at the child
     node. Logic stops and starts replacing keys once there are no more child nodes to step through. 
    
    NOTE: Key replacement performed in-place
    
    :param d: Model or segmentation definitions, at arbitrary depth
    :type d: dict
    :param to_string: Whether to convert from tuple to string, or vice versa
    :type to_string: bool
    :return: 
    """
    try:
        # if 'inputs' and 'target' are not excluded, the check for child nodes will fail even if
        #  child nodes are present, because 'inputs' or 'target' is usually the first key listed when .keys()
        #  is called
        keys = [k for k in list(d.keys()) if k not in ['inputs', 'target']]
    except AttributeError:
        # New model definition format allows for nested dicts at this level to fully capture the interaction terms
        # Because of this allowance, the check in this inner try: statement (d[keys[0]].keys()) can pass
        #   if keys[0] is an interaction term, whereas keys[1] may not pass because it's a "simple" term.
        # To accommodate this new definition format, whenever the dict d at this level has no keys, skip over it
        pass
    else:
        try:  # check if there is another level
            list(d[keys[0]].keys())
        except (AttributeError, IndexError):  # At the lowest level of the tree, replace keys with string versions
            for k in keys:
                replace_key(d, k, to_string)
        else:  # there is another level, keep going
            for key in keys:
                convert_dict_keys(d[key], to_string)
                replace_key(d, key, to_string)


def navigate_tree(obj, d):
    """
    Recursively navigates segmentation dictionary tree by attempting to match the rules found at each node (key). 
     Ideally, a match is found at each node and the process proceeds until a final value (segment id) is found.
     If a match is NOT found at a given node, the code will attempt to fall back to any default action(s) specified
        If no default is specified, the program will print out a generic error statement and return None
     Logic outside of this function will handle any Nones returned

    :param obj: Data object to find model for (component- or quote-level)
    :type obj: pandas.Series
    :param d: Model or segmentation definitions at arbitrary depth
    :type d: dict
    :return: The final matching dictionary, segment value, or None if no match
    :rtype: dict, int, or NoneType
    """
    # if a dict was passed in, try to find a key match and dig into next layer
    # with new component model format, need to confirm the key is a tuple, else we've found the model definition
    if isinstance(d, dict):
        keys = [k for k in list(d.keys()) if k not in ['inputs', 'target']]
    else:
        keys = []

    if isinstance(d, dict) and len(keys) > 0 and isinstance(keys[0], (tuple, list)):
        # doesn't matter which key is pulled out for this purpose, so long as it's not an unintentional dead end
        feature = keys[0][0]  # constant across tier
        match_cat = keys[0][1]  # non-default match category for this tier
        match_vals = [k[2] for k in keys]  # all match_vals for this tier

        if match_cat in IMPLEMENTED_MATCH_CATS:
            try:
                if match_cat in EXACT_MATCHES:
                    # In this case, match_vals is a list of strings (or maybe numbers?) ['analytics', 'watson',...]
                    # Try for exact match
                    idx = match_vals.index(str(obj[feature]))
                elif match_cat in RANGE_MATCHES:
                    # In this case, match_vals is a list of tuples [(0, 10), (10, 19), (19, 30)...]
                    # Need to find where the value for this component falls within these ranges, if any
                    # Return enumerate index i of all (e.g. the only) match_vals entries where obj[feature]
                    # value falls between (min, max) values defined in match_vals tuples. LEFT INCLUDE
                    idx = [i for (i, v) in enumerate(match_vals) if
                           (obj[feature] >= v[0]) & (obj[feature] < v[1])][0]
                else:
                    # Should never trigger because of "if match_cat in IMPLEMENTED_MATCH_CATS" requirement
                    print(('Match category {cat} not in list of handled categories, yet somehow made it past the check'.format(cat=match_cat)))

            # Can't find match
            except (ValueError, IndexError):
                if 'inputs' in list(d.keys()):
                    return {'inputs': d['inputs'], 'target': d['target']}
                else:
                    print(('Cant find value "{v}" of {c} feature {f} for component, '
                          'and no default handling in place'.format(v=obj[feature], c=match_cat, f=feature)))
                    return None
            else:
                # Found match - dig deeper into tree
                # Use index of key match (idx) to call appropriate sub-dictionary to then try to navigate
                key = (feature, match_cat, match_vals[idx])
                return navigate_tree(obj, d[key])  # YOU HAVE TO RETURN THE RESULT OF THE RECURSION

        else:
            raise NotImplementedError

    else:  # There are no more layers. Return value we have arrived at
        return d


def find_seg_id(obj, seg_dict):
    """
    Simple wrapper function for segment_id finder navigate_tree()

    Any special logic for handling when None is returned is handled here

    :param obj: Data object
    :type obj: pandas.Series
    :param seg_dict: Segmentation definitions
    :type seg_dict: dict
    :return: segmentation ID if found, None if no match is found
    :rtype: int or NoneType
    """
    value_match = navigate_tree(obj, seg_dict)
    if value_match is None:
        # TODO: Match not found. What should happen?
        print(('Couldnt find segment ID. Returning None'.format(obj.name)))
    else:
        print(('Found segment ID: {}'.format(value_match)))

    return value_match
