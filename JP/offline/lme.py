from copy import copy

import rpy2.robjects as ro
from numpy import intersect1d, union1d
from pandas import Series, merge
from rpy2.robjects import pandas2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.lib.dplyr import DataFrame as r_dataframe
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector, BoolVector, IntVector, FloatVector

from shared.utils import timeit

lme4 = importr('lme4')
base = importr('base')
stats = importr('stats')
r_ranef = ro.r['ranef']
r_fixef = ro.r['fixef']
r_formula = ro.r['formula']
print('Loading lme script')


class lmeMod(object):

    def __init__(self, train_data, formula_fe, formula_re, field_types):
        print('Initializing model object...')
        self.train_data = copy(train_data)
        self.formula_fe = formula_fe
        self.formula_re = formula_re
        self.field_types = field_types
        self.fe = None  # store fixed effects Series
        self.re = None  # store random effects as DataFrame, per hierarchy
        self.model = None  # store R model object
        self.key = None  # store key info, per hierarchy

    def fit(self):
        print('Transforming input data...')
        self.train_data = self.map_df_fields(self.train_data)
        r_train = convert_pandas_df(self.train_data)
        formula = self.formula_fe + '+' + self.formula_re
        kwargs = {'allow.new.levels': True
                  , 'formula': formula
                  , 'data': r_train
                  }
        
        print('Fitting the model...')
        self.model = self.run_fit(kwargs)
        
        # store fixed effects in model object for later access
        self.fe = self.extract_fixed_effects()

        # store random effects in model object
        ranef = self.extract_random_effects()
        self.key, self.re = process_ranef(ranef)

        return self

    def predict(self, test):
        print('Converting data fields...')
        test = self.map_df_fields(test)
        r_test = convert_pandas_df(test)
        
        print('Making predictions...')
        y_pred = stats.predict(self.model, newdata=r_test)

        preds = convert_r_obj(y_pred, 'series')
        return preds
    
    @timeit
    def run_fit(self, kwargs):
        return lme4.lmer(**kwargs)

    @timeit
    def run_predict(self, data):
        return stats.predict(self.model, newdata=data)

    def extract_fixed_effects(self):
        # extract fixed effects, keep in R
        fe_ = r_fixef(self.model)
        # convert R array to pandas Series
        s = convert_r_obj(fe_, 'series')

        return s
        
    def extract_random_effects(self):
        # etract random effects, keep in R
        ranefs = r_ranef(self.model)
        # map ranefs name to dataframe contents
        dfs = {ranefs.names[i]: convert_r_obj(x, 'df') for i, x in enumerate(ranefs)}
        
        return dfs
    
    # rpy2 exhibits strange behavior when NaN/None values are present in pandas
    # source data. Explicitly set the contents of the pandas dataframe so
    # rpy2 won't guess (and get it wrong)
    # See https://bitbucket.org/rpy2/rpy2/issues/421/pandas2ripy2ri-use-way-too-much-ram
    def map_df_fields(self, data):
        print('Mapping train_data fields to rpy2 R vectors...')
        mapper = {'str': StrVector, 'int': IntVector, 'float': FloatVector, 'bool': BoolVector}
        for col in data.columns:
            if col in self.field_types.keys():
                col_type = self.field_types[col]
            else:
                print('Column {c} doesnt have field type'.format(c=col))
                col_type = 'str'

            if col_type in mapper.keys():
                vec = mapper[col_type]  # extract appropriate vector function
            else:
                print('Unable to map col {c} (type {t}). Defaulting to StrVector'.format(c=col, t=col_type))
                vec = mapper['str']

            # replace train_data field with vector
            data[col] = vec(data[col])
        
        return data


@timeit
def convert_pandas_df(df):
    # local initialization of pandas2ri, since we don't want it global
    # specifying the dplyr DataFrame() converter gives _significant_
    # improvement over generic pandas2ri.py2ri()
    with localconverter(default_converter + pandas2ri.converter):
        r_df = r_dataframe(df)
    
    return r_df


@timeit
def convert_r_obj(r_obj, type='obj'):
    with localconverter(default_converter + pandas2ri.converter):
        if type == 'df':
            py_obj = pandas2ri.ri2py_dataframe(r_obj)
        elif type == 'series':
            # retrieve the weights
            obj_ = pandas2ri.ri2py(r_obj)
            # construct pandas.Series out of numpy vector and name list
            py_obj = Series(data=obj_, index=r_obj.names)
        else:
            py_obj = pandas2ri.ri2py(r_obj)
    
    return py_obj


def process_ranef(random_effects):
    # break out brand, set, and group for each df
    # replace original dict entry with nested entry that will allow for more maneuverability
    key_track = {}
    for i, k in enumerate(sorted(list(random_effects.keys()), key=len, reverse=True)):
        print(k)
        keys = k.replace('(', '').replace(')', '').split(':')
        df = random_effects[k].reset_index()
        df_cols = list(df.columns.drop('index'))  # if sort worked properly, first loop will be the base column names
        tmp = df['index'].apply(lambda x: Series(x.split(':'), index=keys))  # unpack index into (potentially) separate columns
        df = df.join(tmp).drop('index', axis=1)
        random_effects[k] = {'data': df, 'keys': keys, 'cols': df_cols}
        if i == 0:  # start the key tracking dict
            key_track.update({'hier_0': {'top': [k], 'keys': keys}})
        else:
            # iterate over all key groupings in key_track and see if current set of keys matches any key sets
            # already contained in key_track groupings
            overlap = {y: any([x in keys for x in key_track[y]['keys']]) for y in list(key_track.keys())}
            if not any([overlap[key] for key in list(overlap.keys())]):  # if no overlap found, do this
                key = 'hier_' + str(len(list(overlap.keys())))
                key_track.update({key: {'top': [k], 'keys': keys}})  # create new grouping in key_track
            else:  # if overlap found, do this
                # find key grouping with overlap; there must be one
                key = [key for key, value in overlap.items() if value][0]
                # extend list of top-level keys
                key_track[key]['top'] += [k]
                # extend list of hierarchy keys contained in this grouping
                key_track[key]['keys'] += [z for z in keys if z not in key_track[key]['keys']]

    # key_track contains all key groupings, i.e. hierarchies, separated out by arbitrary top-level keys "grp_"
    # use this to sum together all of the random effects in each hierarchy so that there is only one set
    # of model coefficients per hierarchy
    hier_track = {}
    for key in list(key_track.keys()):
        print(('Processing key grouping {}..'.format(key)))
        # pull out the grouped random effects keys for this hierarchy
        # sort by increasing length, which corresponds to the levels of the hierarchy (longer = deeper)
        ranef_keys = sorted(key_track[key]['top'], key=len)
        # store each layer of the hierarchy, with coefficients summed up to that layer
        store = {}
        for i in range(len(ranef_keys)):
            print(('Adding in data from ranef_key "{}"'.format(ranef_keys[i])))
            if i == 0:
                # on first loop, store top-level hierarchy info
                df = random_effects[ranef_keys[i]]['data']
                cols = random_effects[ranef_keys[i]]['cols']
            else:
                # on all subsequent loops, bring in next hierarchy level (r_dat) and add in the effects of the
                # previous level (l_dat) to the new level
                l_dat = df.copy()
                l_cols = copy(cols)
                l_keys = copy(list(df.columns.drop(cols)))

                # pull out all right-hand values
                r_dat = random_effects[ranef_keys[i]]['data']
                r_cols = random_effects[ranef_keys[i]]['cols']
                r_keys = random_effects[ranef_keys[i]]['keys']

                cols = list(intersect1d(l_cols, r_cols))

                # check that all coefficient columns overlap on left and right side
                if any([x not in cols for x in union1d(l_cols, r_cols)]):
                    print('!! WARNING !! process_ranef() has failed\n'
                          'There are coefficient columns in the union not present in the intersection')
                    print(('Examine the following mixed effects hierarchy: {}'.format(ranef_keys)))
                    break

                keys = list(intersect1d(l_keys, r_keys))

                tmp = merge(l_dat, r_dat, on=keys, how='outer')
                for col in cols:
                    tmp[col] = tmp[[col+'_x', col+'_y']].sum(axis=1)  # sum the coefficient columns together
                    tmp.drop([col+'_x', col+'_y'], axis=1, inplace=True)  # drop the source columns

                df = tmp

            # establish hierarchy key order according to the formula passed in to the model
            # NOTE: (0|brand/set/group) states order is brand -> set -> group, where brand is top-level key
            #       The R model outputs its keys in reverse order, as group:(set:brand)
            #       Knowing this, we can split on ':' and reverse the order to retrieve the hierarchy we want
            vals = (ranef_keys[i].replace('(', '').replace(')', '').split(':')[::-1])

            # per loop across hierarchy, store coefficients UP TO that level
            store.update({ranef_keys[i]: df.set_index(vals).copy(deep=True)})

        # hier_track.update({key: df.set_index(vals)})
        hier_track.update({key: store})

    return key_track, hier_track
