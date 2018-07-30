from numpy import append, zeros, arange, int32, ravel
from pandas import DataFrame, merge
from sklearn import tree

from shared.utils import timeit


################################################################################################################
# Segmentation ZX 2017.06.12 revise based on Krishnan's version
################################################################################################################


def tree_seg(X, Y, seg_min=1.0, seg_max=10.0, maxm_depth=2, min_split=50, min_leaves=20):
    clf = tree.DecisionTreeRegressor(criterion='mse', max_depth=maxm_depth, min_samples_split=min_split,
                                     min_samples_leaf=min_leaves)
    clf = clf.fit(X, Y)
    seg_array = clf.tree_.threshold[(clf.tree_.children_left + clf.tree_.children_right) != -2]
    seg_array = append(seg_array, [seg_min, seg_max])
    seg_array.sort()

    n_segments = len(seg_array) - 1
    segments = zeros((n_segments, 2))
    for i in arange(len(seg_array) - 1):
        segments[i, 0] = seg_array[i]
        segments[i, 1] = seg_array[i + 1]

    leaf_array = (clf.tree_.children_left + clf.tree_.children_right) == -2
    num_leaves = sum(leaf_array)
    i = 0
    leaves = zeros((num_leaves, 3), dtype=int32)
    if clf.tree_.node_count > 1:

        node_index = arange(clf.tree_.node_count)
        for k in arange(len(leaf_array)):
            if leaf_array[k] == True:
                leaves[i, 0] = k
                leaves[i, 1] = node_index[(clf.tree_.children_left == k) | (clf.tree_.children_right == k)]
                if sum(clf.tree_.children_left == k) == 1:
                    leaves[i, 2] = 1
                i = i + 1

        leaf_values = zeros((num_leaves, 1))
        leaf_sample_count = zeros((num_leaves, 1), dtype=int32)
        leaf_threshold = clf.tree_.threshold[leaves[:, 1]]
        for j in arange(num_leaves):
            leaf_sample_count[segments[:, leaves[j, 2]] == leaf_threshold[j]] = clf.tree_.n_node_samples[leaves[j, 0]]
            leaf_values[segments[:, leaves[j, 2]] == leaf_threshold[j]] = clf.tree_.value[leaves[j, 0]]
    else:
        leaf_sample_count = len(X)
        leaf_values = (Y.mean())[0]
        num_leaves - 0

    segment_full = DataFrame(segments, columns=['MIN', 'MAX'])
    segment_full['COUNT'] = leaf_sample_count
    segment_full['AVG_VAL'] = leaf_values
    segment_full['SEG_ID'] = arange(1, (num_leaves + 1))  # ZX 2017.06.12: seg ID by 1,...,num_leaves
    segment_full = DataFrame(segment_full, columns=['SEG_ID', 'MIN', 'MAX', 'COUNT', 'AVG_VAL'])

    return segments, leaf_values, leaf_sample_count, segment_full


@timeit
def byTree(data, x):
    cols = x[0]
    tier = x[2]
    previousTiers = ["tier_" + str(t) for t in range(0, tier)]
    for c in cols: assert c in data.columns
    for p in previousTiers: assert p in data.columns

    dset = data.copy()

    if len(previousTiers) > 0:

        temp = dset.groupby(previousTiers)
        out = DataFrame()

        for key, g in temp:
            gset = DataFrame(g).copy()
            gset['SEG_ID'] = 1

            seg, values, counts, tree_out = tree_seg(
                gset[[cols[0]]], gset[[cols[1]]], x[1][0], x[1][1], x[1][2], x[1][3], x[1][4])

            tree_out.columns = ['REV_SEG_ID', 'VAL_MIN', 'VAL_MAX', 'VAL_COUNT', 'AVG_VAL']
            tree_out['SEG_ID'] = 1

            tree_out = DataFrame(tree_out,
                                 columns=['SEG_ID', 'REV_SEG_ID', 'VAL_MIN', 'VAL_MAX', 'VAL_COUNT', 'AVG_VAL'])
            t = gset[['SEG_ID'] + previousTiers]

            out = out.append(merge(t, tree_out, on='SEG_ID', how='inner'))

        out = out[previousTiers + ['VAL_MIN', 'VAL_MAX']]
        out = out.drop_duplicates().reset_index(drop=True)
        out['tier_' + str(tier)] = out.index
        dset = merge(dset, out, on=previousTiers, how='inner')
        dset = dset[(dset[cols[0]] >= dset['VAL_MIN']) & (dset[cols[0]] < dset['VAL_MAX'])]
        dset.rename(columns={'VAL_MIN': cols[0] + '_MIN', 'VAL_MAX': cols[0] + '_MAX'},
                    inplace=True)

    else:

        gset = dset.copy()

        gset['SEG_ID'] = 1
        seg, values, counts, tree_out = tree_seg(gset[[cols[0]]], gset[[cols[1]]], \
                                                 x[1][0], x[1][1], x[1][2], x[1][3], x[1][4])

        tree_out.columns = ['REV_SEG_ID', 'VAL_MIN', 'VAL_MAX', 'VAL_COUNT', 'AVG_VAL']
        tree_out['SEG_ID'] = 1

        tree_out = DataFrame(tree_out,
                                columns=['SEG_ID', 'REV_SEG_ID', 'VAL_MIN', 'VAL_MAX', 'VAL_COUNT', 'AVG_VAL'])
        t = gset[['SEG_ID'] + previousTiers]

        out = merge(t, tree_out, on='SEG_ID', how='inner')

        out = out[['VAL_MIN', 'VAL_MAX']]
        out = out.drop_duplicates().reset_index(drop=True)
        out['tier_' + str(tier)] = out.index
        dset['dummy'] = 'dummy'
        out['dummy'] = 'dummy'

        dset = merge(dset, out, on='dummy', how='inner')
        dset = dset[(dset[cols[0]] >= dset['VAL_MIN']) & (dset[cols[0]] < dset['VAL_MAX'])]
        dset.rename(columns={'VAL_MIN': cols[0] + '_MIN', 'VAL_MAX': cols[0] + '_MAX'},
                    inplace=True)

    return dset, [[cols[0] + '_MIN', cols[0] + '_MAX'], 'tree', cols[0]]


def preprocess(dset):
    assert (isinstance(dset, DataFrame))
    columns = dset.columns
    assert ('leading_brand' in columns)
    assert ('quoteid' in columns)
    newColNames = {'quoteid': 'SampleSize_Quote'}
    temp = dset[['leading_brand', 'quoteid']].groupby('leading_brand').count().reset_index().rename(
        columns=newColNames)
    return merge(dset, temp, on='leading_brand', how='inner')


@timeit
def byRange(data, x):
    cols = x[0]
    tier = x[2]
    previousTiers = ["tier_" + str(t) for t in range(0, tier)]
    for c in cols: assert c in data.columns
    for p in previousTiers: assert p in data.columns

    dset = data.copy()

    dset['REV_MIN'] = 0
    dset['REV_MAX'] = 0

    values = x[1]
    tier = x[2]

    for i in range(len(values)):
        rlow = values[i][0]
        rhigh = values[i][1]
        temp = dset.copy()
        for j in range(len(cols)):
            assert cols[j] in dset.columns
            low = values[i][2 * j]
            high = values[i][2 * j + 1]
            if (low is not None) and (high is not None):
                temp = temp[(temp[cols[j]] > low) & (temp[cols[j]] <= high)]
        dset.loc[temp.index, 'REV_MIN'] = rlow
        dset.loc[temp.index, 'REV_MAX'] = rhigh

    if len(previousTiers) > 0:
        lead = dset[previousTiers + ['REV_MIN', 'REV_MAX']].drop_duplicates().reset_index(drop=True)
        lead['tier_' + str(tier)] = lead.index
        return merge(dset, lead, on=previousTiers + ['REV_MIN', 'REV_MAX']), [['REV_MIN', 'REV_MAX'], 'range',
                                                                                 cols[0]]
    else:
        lead = dset[['REV_MIN', 'REV_MAX']].drop_duplicates().reset_index(drop=True)
        # lead['tier' + str(tier)] = lead.index
        lead['tier_' + str(tier)] = lead.index
        return merge(dset, lead, on=['REV_MIN', 'REV_MAX']), [['REV_MIN', 'REV_MAX'], 'range', cols[0]]


@timeit
def byCategorical(data, x, label):
    cols = x[0]
    tier = x[2]

    previousTiers = ["tier_" + str(t) for t in range(0, tier)]
    for c in cols: assert c in data.columns
    for p in previousTiers: assert p in data.columns

    dset = data.copy()

    brands = dset[previousTiers + cols].drop_duplicates().reset_index(drop=True)
    # brands['tier' + str(tier)] = brands.index
    brands['tier_' + str(tier)] = brands[cols[0]]
    return merge(dset, brands, on=previousTiers + cols), [cols, label, cols[0]]


@timeit
def segment(data, segmethod):
    dset = data.copy()
    dset['tier_0'] = 'all'
    meta = []

    for seg_key, seg_param in list(segmethod.items()):
        layer = int(seg_key.split('_')[-1])  # take last char after '_' and convert to int
        key = seg_key.split('_')[0]  # take first word before '_'
        seg_param.append(layer)
        if key in ['categorical', 'binary']:
            dset, temp = byCategorical(dset, seg_param, key)
            meta.append(temp)
        if key == 'range':
            dset, temp = byRange(dset, seg_param)
            meta.append(temp)
        if key == 'tree':
            dset, temp = byTree(dset, seg_param)
            meta.append(temp)

    # Quick hack to get a mapping of tier IDs to feature min, max (if range or tree)
    t = {}
    for i, val in enumerate(meta):
        tier = 'tier_'+str(i+1)
        t_type = val[1]
        feature = val[2]

        if t_type in ['binary', 'categorical']:
            match_vals = None  # categorical entries don't get IDs associated with them
        else:
            fields = val[0]
            match_vals = dict(list(zip(dset[tier], list(zip(dset[fields[0]], dset[fields[1]])))))

        t[tier] = {'feature': feature,
                   'type': t_type,
                   'match_vals': match_vals}

    return dset, meta, t


def toDict(df, meta, seg_dict):
    if len(meta) > 1:

        group, category, meta = meta[0][2], meta[0][1], meta[1:]
        temp = df.groupby([group])
        for k, g in temp:
            key = (group, category, k)
            seg_dict[key] = {}
            toDict(g, meta, seg_dict[key])

    elif len(meta) == 1:
        group, category = meta[0][2], meta[0][1]
        temp = df.groupby([group])
        for k, g in temp:
            key = (group, category, k)
            seg_dict[key] = ravel(g['tier'])[0]

    return seg_dict


@timeit
def seg_to_dict(dset, meta):
    temp_cols = []
    cols = []
    for x in meta:
        temp_cols += x[0]
        cols.append(x[2])

    t = dset[temp_cols].drop_duplicates().reset_index(drop=True)
#    t['tier'] = t.index.astype(int)

    for x in meta:
        # NOTE: All exact matches are expected to be string type, so forcefully convert to string
        t[x[2]] = list(zip(*[t[y] for y in x[0]])) if x[1] not in ['categorical', 'binary'] else t[x[0]].astype(str)

    t = t[cols]
    t['tier'] = t.index.astype(str)

    seg_dict = {}

    return toDict(t, meta, seg_dict)
