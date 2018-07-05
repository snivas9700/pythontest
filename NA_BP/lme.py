# NOTE: If rpy2 import fails, it is LIKELY due to incorrect build of rpy2 (e.g. with other version of R than
#       what is present on your system. To fix, download the rpy2 source .tar.gz file, unpackage, cd into folder
#       and run
#       export LDFLAGS="-Wl,-rpath,<r_home>/lib"
#           r_home = /Library/Frameworks/R.framework/Resources or whatever "R RHOME" returns
#       `python setup.py build` - confirm the R listed at the end is the proper version, then run
#       `python setup.py install`
#       SHOULD fix
from rpy2.robjects.packages import importr
import rpy2.robjects as ro
#import pandas.rpy.common as com
from rpy2.robjects import pandas2ri
pandas2ri.activate()
from numpy import intersect1d, union1d
from pandas import Series, merge
from copy import deepcopy, copy

# `conda install -c r r-lme4` SHOULD have fixed this
# ALSO went into R console, ran `install.packages('lme4')` within R
lme4 = importr('lme4')
base = importr('base')  # `conda install -c r r`  (will install r-base, and others)
stats = importr('stats')
r_ranef = ro.r['ranef']
r_fixef = ro.r['fixef']
r_formula = ro.r['formula']
print ('Loading lme script')


class lmeMod(object):

    def __init__(self, train_data, formula_fe, formula_re):
        print('Initializing model object...')
        self.train_data = train_data
        self.formula_fe = formula_fe
        self.formula_re = formula_re
        self.fe = None  # store fixed effects Series
        self.re = None  # store random effects as DataFrame, per hierarchy
        self.model = None  # store R model object
        self.key = None  # store key info, per hierarchy

    def fit(self):
        print('Fitting the model...')
        rtrain_data = pandas2ri.py2ri(self.train_data)
        formula = self.formula_fe + '+' + self.formula_re
        kwargs = {'allow.new.levels': True}
        self.model = lme4.lmer(formula, data=rtrain_data, **kwargs)
        self.fe = pandas2ri.ri2py(r_fixef(self.model))
        ranef = deepcopy(pandas2ri.ri2py(r_ranef(self.model)))

        # store random effects in model object for later access
        self.key, self.re = process_ranef(ranef)

        return self

    def predict(self, newdata):
        print('Making predictions...')
        rvalid_data = pandas2ri.py2ri(newdata)
        y_pred = stats.predict(self.model, newdata=rvalid_data)

        return pandas2ri.ri2py(y_pred)


def process_ranef(random_effects):
    # break out brand, set, and group for each df
    # replace original dict entry with nested entry that will allow for more maneuverability
    key_track = {}
    for i, k in enumerate(sorted(random_effects.keys(), key=len, reverse=True)):
        print (k)
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
            overlap = {y: any([x in keys for x in key_track[y]['keys']]) for y in key_track.keys()}
            if not any([overlap[key] for key in overlap.keys()]):  # if no overlap found, do this
                key = 'hier_' + str(len(overlap.keys()))
                key_track.update({key: {'top': [k], 'keys': keys}})  # create new grouping in key_track
            else:  # if overlap found, do this
                # find key grouping with overlap; there must be one
                key = [key for key, value in overlap.iteritems() if value][0]
                # extend list of top-level keys
                key_track[key]['top'] += [k]
                # extend list of hierarchy keys contained in this grouping
                key_track[key]['keys'] += [z for z in keys if z not in key_track[key]['keys']]

    # key_track contains all key groupings, i.e. hierarchies, separated out by arbitrary top-level keys "grp_"
    # use this to sum together all of the random effects in each hierarchy so that there is only one set
    # of model coefficients per hierarchy
    hier_track = {}
    for key in key_track.keys():
        print('Processing key grouping {}..'.format(key))
        # pull out the grouped random effects keys for this hierarchy
        # sort by increasing length, which corresponds to the levels of the hierarchy (longer = deeper)
        ranef_keys = sorted(key_track[key]['top'], key=len)
        # store each layer of the hierarchy, with coefficients summed up to that layer
        store = {}
        for i in range(len(ranef_keys)):
            print('Adding in data from ranef_key "{}"'.format(ranef_keys[i]))
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
                    print('Examine the following mixed effects hierarchy: {}'.format(ranef_keys))
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
