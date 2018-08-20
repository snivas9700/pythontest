# Descriptive analytics

import pandas as pd
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from cleansing import to_category
import myplotlibrary4_3 as myplt
from PlotColors import *

def distributions(df, attribs, indices=[], xsections=[], cumul=False):
    """
    Plots the frequency distribution of one or more variables, across the
    entire data set, or one or more(?) cross sections (slices) thereof

    Args:
        df: input DataFrame
        attribs (list): variables for which to compute and plot
        frequency
            distributions
        indices: variables used to (multi)Index the DataFrame, which enables
            flexible slicing
        xsections (list of tuples): Values of each level of the multiIindex
            slice

    Returns:

    """
    # Need another for loop here to cover multiple cross sections
    if len(indices) > 0:
        dfi = df.set_index(indices)
        dfi.sort_index(inplace=True)

    for feature in attribs:
        if len(xsections) > 0:
            for xsection in xsections:
                idx = pd.IndexSlice[xsection]
                # Derive a chart title from the cross section values
                title = str(xsection).strip('(').strip(')')

                # Define chart data
                cdata = dfi.loc[idx][feature]
                # Plot histogram
                n1, bins1 = iVis.histogram(cdata, neutralcolor, nbins=20,
                                           title1=feature + '\n' + title,
                                           ylabel1='Count', y_rot=90,
                                           xlabel1=feature, cumul=cumul,
                                           filename=feature + '_' + title)
        # Define chart data
        cdata = df[feature]
        # Plot histogram
        n1, bins1 = iVis.histogram(cdata, neutralcolor, nbins=20,
                                   title1=feature,
                                   ylabel1='Count', y_rot=90,
                                   xlabel1=feature,
                                   filename=feature)
    pass


def plot_dendrogram_external(children, path, levels=10, title1='', xlabel1='',
                             ylabel1='', filename='temp'):
        """
        Uses scipy to visualize an array of distances between clusters as a
        tree; the distances are produced by a separate clustering algorithm

        Args:
            children:

        Returns:

        """
        # Encode the hierarchical clustering output such that it can be
        # rendered as a dendrogram
        Z = hierarchy.linkage(children)

        # Set the color(s) of the dendrogram branches
        hierarchy.set_link_color_palette(['0.7'])

        # Create a Matplotlib figure and axes
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        # Construct the dendrogram and add it to the axes; show the last
        # number of the highest levels and label the termini of the branches
        # with the number of unseen lower level clusters
        dendrogram(Z, truncate_mode='lastp', p=levels,
                        count_sort='ascending',
                        show_leaf_counts=True, orientation='left',
                        above_threshold_color=tableau20[0],
                        ax=ax1)  # distance_sort

        # Add title, label and format axes, draw a grid, etc.
        myplt.psup.axAccessories(ax1, gridax='x', title=title1,
                                 xlabel=xlabel1,
                                 rot=-90, ylab_pos='right', ylabel=ylabel1,
                                 ylab_coord_x=1.1, ylab_coord_y=1.0,
                                 ylab_halign='left')

        # Ensure plot layout does not cut off elements
        plt.tight_layout()

        # Save the dendrogram as a graphics file
        fig.savefig(path + filename + '_dendrogram.png', dpi=200,
                    transparent=True)


def cluster(df, attribs, levels=10, clusters=5, draw_dendrogram=True,
            title1='', xlabel1='', ylabel1='', path_out='',
            filename='temp'):
    """
    Clusters observations from the bottom up, creating an unsupervised
    hierarchy. The distance between newly formed clusters is computed using
    Ward minimum variance. You may cluster observations on the basis of
    categorical attribs, but they must be encoded numerically.

    The number of clusters determines the height at which the segmentation
    tree is cut, which determines which underlying branches comprise a
    segment designated by the labels_ attribute. This parameter is akin to k
    in K-means clustering and so can be used to compare the two methods.

    Useful overview: http://bit.ly/2BnUaiO
    Technical reference: Introduction to Statistical Learning

    Args:
        df: Data set from which the feature matrix will be derived
        attribs (list): list of attribs by which to cluster
        the instances
        levels: pertains to the truncation of the dendrogram (p)
        clusters: determines where the clustering hierarchy is cut to group
            instances into clusters
        draw_dendrogram: determines whether a dendrogram is drawn, or just the
            clustering performed
        title1: optional title for the dendrogram graph
        xlabel1: horizontal axis label of the dendrogram
        ylabel1: horizontal axis label of the dendrogram
        path_out: folder in which to store the dendrogram graphics file
        filename: dendrogram graphics file name

    Returns:
        Input DataFrame with cluster labels appended and the number of leaves
        & components generated by the clustering algorithm
    """

    def plot_dendrogram(children):
        """
        Uses scipy to visualize an array of distances between clusters as a
        tree; the distances are produced by a separate clustering algorithm

        Args:
            children:

        Returns:

        """
        # Encode the hierarchical clustering output such that it can be
        # rendered as a dendrogram
        Z = hierarchy.linkage(children)

        # Set the color(s) of the dendrogram branches
        hierarchy.set_link_color_palette(['0.7'])

        # Create a Matplotlib figure and axes
        fig = plt.figure(figsize=(8, 6))
        ax1 = fig.add_subplot(1, 1, 1)

        # Construct the dendrogram and add it to the axes; show the last
        # number of the highest levels and label the termini of the branches
        # with the number of unseen lower level clusters
        dendrogram(Z, truncate_mode='lastp', p=levels,
                        count_sort='ascending',
                        show_leaf_counts=True, orientation='left',
                        above_threshold_color=tableau20[0],
                        ax=ax1)  # distance_sort

        # Add title, label and format axes, draw a grid, etc.
        myplt.psup.axAccessories(ax1, gridax='x', title=title1,
                                 xlabel=xlabel1,
                                 rot=-90, ylab_pos='right', ylabel=ylabel1,
                                 ylab_coord_x=1.1, ylab_coord_y=1.0,
                                 ylab_halign='left')

        # Ensure plot layout does not cut off elements
        plt.tight_layout()

        # Save the dendrogram as a graphics file
        fig.savefig(path_out + filename + '_dendrogram.png', dpi=200,
                    transparent=True)

    cluster_ward = AgglomerativeClustering(n_clusters=clusters,
        affinity='euclidean', connectivity=None,
                                           compute_full_tree=False,
        linkage='ward')

    # cluster_kmeans = KMeans(n_clusters=8)

    cluster_dbscan = cluster_dbscan(eps=0.5, min_samples=5,
        metric='euclidean', metric_params=None, algorithm='auto',
        leaf_size=30, p=None)  # metric='precomputed'

    # Convert object (strings) to category dtype utilizing the external
    # function to_category
    df.dropna(subset=attribs, inplace=True)

    X = to_category(df[attribs])  # .reset_index(drop=True)

    # Standardize numeric (float) attributes to zero mean and a standard
    # deviation of 1
    X[X.select_dtypes(include=['float64'], exclude=['object', 'int64']).
        columns] = RobustScaler().fit_transform(
            X[X.select_dtypes(include=['float64'],
                              exclude=['object', 'int64']).columns])

    # Unstack categorical variables per unique value & encode as (0, 1)
    X = pd.get_dummies(X)

    # Cluster the observations and store results in an object
    cluster_ward_1 = cluster_ward.fit(X)
    # kmeans_1 = cluster_kmeans.fit(X)

    # Retrieve clustering output
    cluster_labels = cluster_ward_1.labels_
    children = cluster_ward_1.children_
    leaves = cluster_ward_1.n_leaves_
    components = cluster_ward_1.n_components_

    # Add the cluster labels to the (null value-trimmed) data
    df.loc[:, 'cluster'] = cluster_labels

    if draw_dendrogram:
        plot_dendrogram(children)

    return df, X, leaves, components


def analyze_cluster(dfc, cluster_label, attribs, piecemeal=True):
    """
    Once instances in a data set are clustered by a select group of
    attributes, investigate how the instances are similar with respect to
    other attributes, such as business unit, offering type, sales channel,
    etc.  This function aims to provide a measure of heterogeneity (called
    entropy in the domain of decision trees):

    entropy = -sum[p_i * log(p_i)], where p_i is the incidence of
    instances of property p_i relative to all other values of that property p.

    p can be one attribute, or some unique combination of attributes

    In this way, the utility of a predefined hierarchy or product mapping in
    a predictive modeling context can be assessed.

    Args:
        dfc: data set with cluster labels
        cluster_label: name of the column containing cluster labels
        attribs: one or more attribs by which to evaluate
        instance
            heterogeneity
        piecemeal: eavluate heterogeneity per each attribute listed in the
            list attribs or by unique combination of all attributes
            listed in
            attribs

    Returns:

    """
    # Currently testing in script mode
    # Define attribs
    attribs = ['platform', 'prod_div', 'machine_type_x']

    # Create a groupby object per attribs and extract its keys
    gbo = dfc.groupby(attribs)
    attrib_tuples = gbo.groups.keys()

    # To use these keys to access portions of the data, index the DataFrame
    dfc.set_index(attribs, inplace=True)

    # Heterogeneity of
    # dfc.to_csv(path_out + 'dfc.csv')
    entropy_table = pd.DataFrame({'cluster0': np.nan, 'cluster0': np.nan,
                                  'cluster0': np.nan})
    # Iterate over the cluster labels
    for c in dfc[cluster_label].unique():

        entropy_table_col = pd.DataFrame({'cluster' + str(c): np.nan})

        # Iterate over unique combinations of attributes
        for attrib_comb in attrib_tuples:

            entropy = 0

            dfcs = dfc.loc[list(feature_sets)[0]]
            n_ = len(dfcs[(dfcs[cluster_label] == c)])

            p = n_ / len(dfc[dfc[cluster_label] == c])

        # Iterate over individual attributes
        for f in attribs:
            # List the unique values of the current feature
            feat_vals = dfc[attribs[f]].unique()

            entropy = 0

            # Iterate over the values of the current feature
            for v in feat_vals:
                # Simplest (binary) case
                n_v = len(dfc[(dfc[cluster_label] == c) &
                              (dfc[feat_vals[v]] == v)])

                p = n_v / len(dfc[dfc[cluster_label] == c])

                entropy -= p * np.log10(p)


# def attrib_rank(df):
