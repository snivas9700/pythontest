import __builtin__

from copy import deepcopy
from collections import OrderedDict
from pandas import MultiIndex, Index, Series, DataFrame, merge
from numpy import unique
from utils import timeit
from numpy import array_split
from multiprocessing import Process, Queue, cpu_count

from modeling.segmentation_utils import find_seg_id

# Column present in the quote outdf model dataframe that aren't model features
NON_FEAT_COLS = ['Discount', 'Count', 'MAPE', 'Level']


def process_model_entry(m_def):
    parsed = []
    for param, weight in m_def.iteritems():
        feat = {}
        f_list = []
        if ':' in param:  # interaction term
            # print 'Found interaction term {}'.format(param)
            f_type = 'interaction'
            for field in param.split(':'):
                f_dict = process_field(field)
                if '-' in field:
                    field = field.split('-')[0]
                f_dict.update({'field_name': field})
                f_list.append(f_dict)
            # interaction terms will contain 2 features, each to be handled separately and multiplied together
            feat.update({'feat_info': f_list})

        else:
            # print 'Found single term {}'.format(param)
            f_type = 'simple'
            f_dict = process_field(param)
            if '-' in param:
                field = param.split('-')[0]
            else:
                field = param
            f_dict.update({'field_name': field})
            feat.update({'feat_info': f_dict})

        feat.update({'feat_type': f_type,
                     'weight': weight,
                     'feat_name': param})

        parsed.append(feat)

    return parsed


def process_field(field):
    if 'Intercept' in field:
        # print('Found constant field {}'.format(field))
        m_type = 'CONSTANT'
        val = None
    elif '-' in field:
        # print('Found categorical field {}'.format(field))
        m_type = 'discrete'
        val = field.split('-')[-1]
    else:
        # print('Found continuous field {}'.format(field))
        m_type = 'continuous'
        val = None

    d = {'match_type': m_type,
         'match_val': val}

    return d


# Recursively build tree out of dataframe indices
#   each node is a match on brand, set, group, IOT, country, or iot_region
#   each leaf is a fully defined model
def build_model_tree(df, key=None):
    df_ = df.copy()

    if isinstance(df_.index, MultiIndex):
        hold = {}
        all_vals = df_.index.levels[0]
        idx = df_.index.labels[0]
        field = df_.index.names[0]

        # all_vals contains all unique values in this level across entire dataframe,
        #   not all of which are applicable to this particular level of the tree
        # Find all entries (labels) in the level that are applicable
        vals = unique([all_vals[i] for i in idx])
        for val in vals:
            # key for this node
            tup_key = (field, 'discrete', val)
            d = build_model_tree(df_.xs(val, level=field), tup_key)
            # dig deeper
            hold.update({tup_key: d})

        return deepcopy(hold)

    elif isinstance(df_.index, Index):
        # can't dig deeper, will convert each row (model) into a dict
        hold = {}
        vals = df_.index.values
        field = df_.index.name

        for val in vals:
            # key for this leaf
            tup_key = (field, 'discrete', val)
            d = process_model_entry(df_.ix[val])

            # match format of quote-level model
            hold.update({tup_key: {'inputs': deepcopy(d),
                                   'target': None}
                         })

        return deepcopy(hold)

    else:
        print('Cannot handle index of type {}'.format(type(df_.index)))
        raise NotImplementedError


# parent function to build model tree out of prepped dataframe
@timeit
def component_model_to_dict(df):
    # per row in df, make keys out of
    m_dict = build_model_tree(df)
    # res_queue.put(m_dict)
    print('Finished parsing {} rows.'.format(df.shape[0]))
    return m_dict


@timeit
def component_parse_wrapper(params):

    p_ = deepcopy(params)

    print('Processing fixed effects into dict...')
    fe = p_['fixed_effects']
    fe_list = process_model_entry(fe)  # parses series into model dict entries
    p_.update({'fixed_effects': fe_list})  # overwrite entry with parsed values

    print('Converting mixed effects dataframe(s) to dict...')

    # The old tree build logic assumed that models were only necessary at the leaf node of each hierarchy
    #  as there could be no partial matches for a given input quote. We have changed that; we want to allow for
    #  partial matches, and allow the engine to bring in as much adjustment to the coefficients as possible, as
    #  provided for in the hierarchy. To do this, we have to provide models at ALL nodes in the tree, and the tree
    #  search will have to traverse the tree as far as it can, then return whatever it finds, whether that is a true
    #  leaf node or not.
    # This requires incorporating the coefficients at each node, which is what the following code does.
    me = p_['mixed_effects']
    me_dict = {}
    for k in me.keys():
        print('Processing mixed effects hierarchy {}...'.format(k))
        hier_dict = {}
        for j in me[k].keys():
            df_ = me[k][j]

            # max (minus 1) the parallelization
            n_procs = cpu_count()-1 or 1
            # print('Spreading tree building across {} cores...'.format(n_procs))

            result_dict = {}

            # chunk the dataframe, but in a stratified manner (preserve the top-level keys)
            idx_vals = list(set(df_.index.get_level_values(0)))
            for i, vals in enumerate(array_split(idx_vals, n_procs)):
                m = df_.index.get_level_values(0).isin(vals)
                d = df_.loc[m, :]
                # print('Processing slice {i} ({n1} unique top-level keys; {n2} rows)...'.format(
                #   i=i, n1=len(vals), n2=d.shape[0]))
                partial = component_model_to_dict(d)
                result_dict.update(partial)

            # update parsed tree for this hierarchy level
            hier_dict.update({j: result_dict})

        # after all hierarchy levels have been processed, merge them into a single tree
        tree = coalesce_trees(hier_dict)

        # after processing all trees in the hierarchy, return the final prepped tree
        me_dict.update({k: tree})

    p_.update({'mixed_effects': me_dict})

    return p_


def coalesce_trees(tree_dict):
    # sorting on length allows the merge to start at the top-level
    # start at the top of the hierarchy, and add the keys of the next level into the current structure
    for i, k in enumerate(sorted(tree_dict.keys(), key=len)):
        tree = tree_dict[k]
        if i == 0:
            out_dict = tree
        else:
            coalesce(out_dict, tree)

    return out_dict


def coalesce(parent, child):
    # NOTE: The child key will always have one more level than the parent
    # NOTE: All updates are done IN-PLACE
    # check current level of parent node
    valid_keys = [z for z in parent.keys() if z not in ['inputs', 'target']]
    if len(valid_keys) > 0:  # the parent node has at least one more level to go
        for z in valid_keys:
            coalesce(parent[z], child[z])
    else:
        # insert contents of child node into parent node; i.e. add the level in child into parent
        parent.update(child)


# Convert from R-like output to fully explicit model for use in the real-time flow
def process_params(mod, data_in, name_dict, KNOWN_CATEGORICALS, KNOWN_BINARIES):
    # print('Expanding model dataframe...')

    # don't alter the stored model parameters
    fe = deepcopy(mod.fe)
    re = deepcopy(mod.re)

    # step 1: track index+column fields and their mappings back to original names
    idx_dict = {}
    col_dict = {}
    for k in re.keys():
        for j in re[k].keys():
            idx_dict.update({x: (name_dict[x] if x in name_dict.keys() else x) for x in re[k][j].index.names})
            col_dict.update({x: (name_dict[x] if x in name_dict.keys() else x) for x in re[k][j].columns})

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
        for k, v in name_dict.iteritems():
            if k in col:
                col = col.replace(k, v)
        col_list.append(col)
    fe.index = col_list

    # establish return object
    ret = {}
    ret.update({'fixed_effects': fe})
    ret['mixed_effects'] = {}

    # step 4 B: convert model column names back to original data names FOR MIXED EFFECTS
    for k in re.keys():
        store = {}
        # random effects has 3 dataframes per hierarchy; one df per level. iterate across all dataframes
        for j in re[k].keys():
            df = re[k][j]
            df.index.names = [idx_dict[x] if x in idx_dict.keys() else x for x in df.index.names]
            df.columns = [col_dict[x] if x in col_dict.keys() else x for x in df.columns]
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


def model_entry_to_dict(model_entry, seg_dict, tier_map, m_dict, tier_dict):
    # storage
    feat_list = []
    # take dataframe entry index, which contains value to match per tier level, and tie back to
    # the feature that defines that tier by using the metadata
    # NOTE: Skip tier_0 ('All')
    if 'All' in model_entry.name:
        model_entry.name = tuple([x for x in model_entry.name if x != 'All'])

    s_dict = {}  # to track match values for finding segment id
    t_dict = {}  # to recover range on each tier entry
    for i, v in enumerate(model_entry.name):
        tier = 'tier_' + str(i + 1)
        if tier_map[tier]['type'] in ['categorical', 'binary']:
            tier_val = {tier_map[tier]['feature']: v}
            feat_range = None
        else:
            feat_range = tier_map[tier]['match_vals'][v]
            # segmentation tree search operates on LEFT INCLUDE, so fill in missing value with MIN of range
            tier_val = {tier_map[tier]['feature']: min(feat_range)}
        # track
        s_dict.update(tier_val)
        t_dict.update({tier: feat_range})

    # build series object to be passed in to tree search function
    row = Series(s_dict, name=model_entry.name)

    try:
        # find segment id for given set of tier values
        seg_id = find_seg_id(row, seg_dict)
    except NotImplementedError:
        print 'Model definition could not match key values: {}'.format(row.name)
        print 'Model metadata: {}'.format(tier_map)
        print 'Skipping'
        return None

    # ID the columns in the dataframe that represent features used in the model
    feat_cols = [x for x in model_entry.index if not any([y in x for y in NON_FEAT_COLS])]
    # removes the trailing '_'+str(level) from the end of each feature, e.g. 'TC_3' -> 'TC'
    clean_feat_cols = ['_'.join(x.split('_')[:-1]) for x in feat_cols]

    # map the feature names present in the dataframe to cleaned names for use by the API/Engine
    feat_names = dict(zip(feat_cols, clean_feat_cols))
    for feat in feat_cols:
        # Fill out default feature dictionary
        feat_dict = {'feat_name': feat_names[feat],
                     'feat_info':
                         {'match_type': 'continuous',
                          'match_val': None,
                          # not useful here, but needed to conform component and quote levels:
                          'field_name': feat_names[feat]},
                     'feat_type': 'simple',
                     'weight': model_entry[feat]}
        feat_list.append(feat_dict)

    # inplace update of model definition dict
    # TODO: Figure out how to include target information
    m_dict.update({seg_id: {'inputs': feat_list,
                            'target': None}
                   })
    tier_dict.update({seg_id: t_dict})

    return seg_id


def quote_model_to_dict(df, seg_dict, tier_map, level):
    m_dict = {}  # will update in place and return after all loops have run
    tier_dict = {}  # to track ranges on each segment_id
    model_df = deepcopy(df)

    model_df['segment_id'] = model_df.apply(
        lambda row: model_entry_to_dict(row, seg_dict, tier_map, m_dict, tier_dict), axis=1)

    # code to include tier match ranges for tiers that have ranges
    range_tiers = [k for k in tier_map.keys() if tier_map[k]['type'] not in ['categorical', 'discrete', 'binary']]
    tier_names = {k: tier_map[k]['feature'] for k in range_tiers}
    tier_df = DataFrame().from_dict(tier_dict, orient='index')[range_tiers].rename(columns=tier_names)
    for col in tier_df.columns:
        tier_df[[col+'_min', col+'_max']] = tier_df[col].apply(Series)
    tier_df = tier_df.drop(tier_names.values(), axis=1).reset_index().rename(columns={'index': 'segment_id'})

    model_df.reset_index(inplace=True)
    all_tiers = [x for x in model_df.columns if 'tier_' in x]
    model_df = merge(model_df, tier_df, on='segment_id', how='left').set_index(all_tiers)

    print('Finished processing {l}-level model dataframe to dict'.format(l=level))
    return deepcopy(m_dict), model_df


def apply_tier_map(data, train_qt, tier_map):
    # This function takes the segment ids generated in the training data and applies them to the test data

    # Don't need/care about duplicate entries; just want to map unique tier values to segment ids
    tiers = train_qt.drop_duplicates(['tier_0', 'tier_1', 'tier_2', 'tier_3']).reset_index(drop=True)

    # Enforce an order on tier_map
    tier_map = OrderedDict([
        ('tier_1', tier_map['tier_1']), ('tier_2', tier_map['tier_2']), ('tier_3', tier_map['tier_3'])
    ])

    # tier_0 is a default; always equals "All"
    data['tier_0'] = 'All'

    #: Loop logic covers the following:
    # data['tier_1'] = data[tier_map['tier_1']['feature']]
    # data['tier_2'] = data[tier_map['tier_2']['feature']]
    # # have to find any range tier matches
    # data['tier_3'] = data.apply(lambda row: find_tier(row, tiers, feats), axis=1)

    # build tier values
    for i, (k, v) in enumerate(tier_map.iteritems()):
        if v['type'] in ['continuous', 'tree', 'range']:  # continuous match categories require careful logic
            data[k] = data.apply(lambda row: find_tier(row, k, tiers, tier_map, i), axis=1)
        else:  # exact match categories are easy
            data[k] = data[v['feature']]

    return data


def find_tier(row, curr_tier, tiers, tier_map, i):
    # segment_id for continuous tier is dependent on the previous tiers' hierarchy

    # based on tier number iter, use all PREVIOUS tiers to filter out what should apply to this tier
    prev_tiers = ['tier_{}'.format(t) for t in range(1, i+1)]

    # the loop logic matches the following:
    # m1 = tiers['tier_1'] == row[tier_map['tier_1']['feature']]
    # m2 = tiers['tier_2'] == row[tier_map['tier_2']['feature']]
    # m3 = tiers['LN_Value_MIN'] <= row['LN_Value']
    # m4 = tiers['LN_Value_MAX'] > row['LN_Value']
    #
    # mask = m1 & m2 & m3 & m4

    # Initialize mask as True for all entries of tiers
    # This mask will be used to find the tiers entry that applies to this row of the test data
    mask = [True] * tiers.shape[0]
    for tier in prev_tiers:
        tmp = tiers[tier] == row[tier_map[tier]['feature']]
        mask &= tmp

    feat = tier_map[curr_tier]['feature']  # pull out feature name for easy access
    lbd = tiers[feat + '_MIN']  # find all minimum values for this feature
    ubd = tiers[feat + '_MAX']  # find all maximum values for this feature

    # find the values within the range established by lbd and ubd
    mask &= ((lbd <= row[feat]) & (ubd > row[feat]))

    match = tiers.loc[mask, curr_tier].reset_index(drop=True)
    if match.shape[0] > 1:
        print('Found multiple matches')
        match = match.iloc[0]
    elif match.shape[0] == 0:
        print('Found NO matches')
        match = None
    else:
        match = match.iloc[0]

    return match
