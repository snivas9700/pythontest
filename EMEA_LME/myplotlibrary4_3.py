'''DATA VISUALIZATION FUNCTIONS
(c) 2016-2018 Aaron Slowey, IBM Chief Analytics Office

Utilizes matplotlib (in object-oriented mode*) and pandas (Python 3.x)
*Highly recommended reading:
1. Tosi, S. Matplotlib for Python Developers
2. matplotlib Artist tutorial: http://matplotlib.org/users/artists.html

Note: these functions are not designed to fulfill all edge cases. They
balance simplicity with the ability to customize charts to a decent degree
out-of-the-box.  That said, many specific cases will require further
customization or simplification of the code.
'''
# ----------------------------------------------------------------------------
# LIBRARIES
# ----------------------------------------------------------------------------
import numpy as np
# import xarray
import pandas as pd
# import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
# from matplotlib.ticker import MultipleLocator, AutoMinorLocator
# from matplotlib.dates import date2num
from statsmodels.graphics.mosaicplot import mosaic
from PlotColors import *
from plot_regional_settings import *  # unclear if this happens
import plot_support as psup


# ----------------------------------------------------------------------------
# BAR CHART -- VERTICAL
# ----------------------------------------------------------------------------
# Width is not constant, but rather denotes some aspect of the data
# l locates the left/top side of the rectangle(s)
# h is the height of the rectangle(s)
# w is the width of the rectangle(s), from <<1 to 1, where 1 means bars touch
# each other.
# b is the height at which the bottom of the rectangle is plotted, useful for
# stacked bar & waterfall charts.
# Typical order of arguments: l, h, w, b; switching ? and h might enable
# a tree diagram (Mekko chart)
def barv(cdata, v1, scale, path_out, filename='temp.png', figx='fig1', axx='ax1',
    startpoint=0, title1='', xlabel1='', ylabel1='', ticklabels=''):
    '''Arguments:
    -figx & axx are labels for matplotlib's figure and axes objects; when
    calling the function, choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    -startpoint accommodates rare use cases to plot a subsample; pass 0 to
    include all values
    -title1, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    '''
    l = np.arange(startpoint, len(cdata))
    w = 0.8
    b = 0
    figx = plt.figure(figsize = (figwidth, fig_ht))
    axx = figx.add_subplot(p, q, 1)
    # ax2 = figx.add_subplot(p, q, 1)
    for i in np.arange(startpoint, len(cdata)):
        h1 = cdata[v1][i]
        # Boolean for conditional formatting
        positive = pd.Series(h1 > 0)
        rects1 = axx.bar(l[i - startpoint], h1, w, b, align='edge',
                        color=positive.map({True: ibm_tealGreen[4], False: ibm_RedOrange[4]}),
                        linewidth=0, edgecolor=edgecol, alpha=transp+0.2)
        
        # h2 = cdata[v2][i]
        # rects2 = ax2.bar(l[i - startpoint], h2, w, b, align='edge',
        #                 color=positive.map({True: negcolor, False: negcolor}),
        #                 linewidth=0, edgecolor=edgecol, alpha=transp+0.2)
        psup.autolabel_v(rects1, axx, h1)
    psup.axAccessories(axx, title1, xlabel1, ylabel1, ticklabels)
    plt.tight_layout()
    figx.savefig(path_out + filename+'_barv.png', dpi=200, transparent = True)
    # plt.show()

# ----------------------------------------------------------------------------
# BAR CHART -- VERTICAL + WIDTH-SENSITIVITY
# ----------------------------------------------------------------------------
# Width is not constant, but rather denotes some aspect of the data
def barv_width(figx, axx, l, cdata, startpoint, path_out, filename, title1='', 
    xlabel1='', ylabel1='', ticklabels=''):
    '''Arguments:
    -figx & axx are labels for matplotlib's figure and axes objects; when
    calling the function, choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    -startpoint accommodates rare use cases to plot a subsample; pass 0 to
    include all values
    -title1, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    '''
    b = 0
    figx = plt.figure(figsize = (figwidth, fig_ht))
    axx = figx.add_subplot(p, q, 1)
    for i in np.arange(startpoint, len(cdata)):
        w = l[i+1] - l[i]
        h = cdata.instances[i]
        # Boolean for conditional formatting
        positive = pd.Series(h > 0)
        rects = axx.bar(l[i - startpoint], h, w, b, align='edge',
                        color=positive.map({True: barcolor, False: negcolor}),
                        linewidth=2, edgecolor='w', alpha=transp+0.2)
        # psup.autolabel_v(rects, axx, cdata.prob_pos[i])
        h2 = cdata.positives[i]
        axx.bar(l[i - startpoint], h2, w, b, align='edge',
                        color=positive.map({True: posbarcolor2, False: negcolor}),
                        linewidth=2, edgecolor='w', alpha=transp)
    psup.axAccessories(axx, title1, xlabel1, ylabel1, ticklabels)
    plt.tight_layout()
    figx.savefig(path_out + filename+'_barv.png', dpi=200, transparent = True)
    # plt.show()


# ----------------------------------------------------------------------------
# BAR & STEP
# ----------------------------------------------------------------------------
def barv_step(cdata, v1, scale, path_out, filename, number_form='{:.1f}',
    figx='fig1', title='', xlabel='', ylabel1='', xticklabels=''):
    '''Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata is the DataFrame containing values to visualize
    - accommodates rare use cases to plot a subsample; 0 includes
    all values
    -v1, etc. are series of values to visualize, usually columns in a DataFrame
    -scale is a scalar divided into v1; e.g., 1 (million) instead of 1000000
    -title, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    '''
    l = np.arange(len(cdata))
    w = 1
    b = 0
    figx = plt.figure(figsize = (figwidth, fig_ht))
    ax1 = figx.add_subplot(p, q, 1)
    for i in range(len(cdata)):
        h = cdata[v1][i]/scale
        # Boolean for conditional formatting
        positive = pd.Series(h > 0)
        rects = ax1.bar(l[i], h, w, b, align='center',
                        color=positive.map({True: posbarcolor2, False: negcolor}),
                        linewidth=0.5, edgecolor='w', alpha=transp+0.2)
        psup.autolabel_v(rects, ax1, h, number_form)

    # ! Average line -- is it just a simple average of the % changes,
    # or does it need to be weighted by revenue?
    ax1.axhline(cdata[v1].mean()/scale, linestyle='dotted', linewidth=1,
        color = barcolor, xmin = 0.05, xmax = 0.95)

    ax2 = figx.add_subplot(p, q, 1)
    # ax2 = ax1.twinx()
    y1 = cdata[v1]/scale
    # x =
    ax2.plot(l, y1, color = barcolor, linewidth=2, ls = 'steps-mid')

    psup.axAccessories(ax1, title, ylabel1, xticklabels=xticklabels)

    ax1.set_xticks(range(len(l) + 1))
    ax1.set_xticklabels(['{:.0f}'.format(x) for x in xticklabels])

    psup.axAccessories(ax2, title, xlabel=xlabel)
    ax2.tick_params(axis="y", which="major", bottom="off", top="off",
                labelbottom="off", left="off", right="off", labelleft="off",
                labelright="off", labelsize=ticklabelsize, labelcolor = subtitlecolor)

    plt.tight_layout()
    figx.savefig(path_out + filename+'_barv_step.png', dpi=200, transparent = True)


# %%--------------------------------------------------------------------------
# BAR CHART -- HORIZONTAL
# ----------------------------------------------------------------------------
# l locates the left/top side of the rectangle(s)
# h is the height of the rectangle(s)
# w is the width of the rectangle(s), from <<1 to 1, where 1 means bars touch
# each other.
# b is the height at which the bottom of the rectangle is plotted, useful for
# stacked bar & waterfall charts.
# Typical order of arguments: l, h, w, b; switching ? and h might enable
# a tree diagram (Mekko chart)
def barh(cdata, v1, v2, path_out, filename='temp.png', scale=1, 
    figx='fig1', startpoint=0, axx='ax1', title1='', xlabel1='', ylabel1='',
    xticklabels=''):
    '''Arguments:
    -figx & axx are labels for matplotlib's figure and axes objects; when
    calling the function, choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    -startpoint accommodates rare use cases to plot a subsample; pass 0 to
    include all values
    -title1, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    '''
    l = np.arange(startpoint, len(cdata))
    w = 0.8
    b = 0
    figx = plt.figure(figsize = (3, 4))
    axx = figx.add_subplot(p, q, 1)
    ax2 = figx.add_subplot(p, q, 1)
    for i in np.arange(startpoint, len(cdata)):
        h1 = cdata[v1][i] / scale
        # Boolean for conditional formatting
        positive1 = pd.Series(h1 > 0)
        rects1 = axx.barh(l[i - startpoint], h1, w, b, align='center',
            color=positive1.map({True: ibm_bluehues[0], False: ibm_RedOrange[4]}),
            linewidth=0, edgecolor=edgecol, alpha=transp+0.2)
        h2 = cdata[v2][i] / scale
        # Boolean for conditional formatting
        positive2 = pd.Series(h2 > 0)
        rects2 = ax2.barh(l[i - startpoint], h2, w, b, align='center',
            color=positive2.map({True:negcolor, False:negcolor}),
            linewidth=0, edgecolor=edgecol, alpha=transp+0.2)
        psup.autolabel_h(rects1, axx, h1)
    # axx.axis(xmin = -0.06, xmax= 0.16)
    psup.axAccessories(axx, title1, xlabel1, ylabel1, xticklabels)
    psup.axAccessories(ax2, title1, xlabel1, ylabel1, xticklabels)
    # To customize a different set of tick parameters
    # ax2.tick_params(axis="y", which="major", bottom="off", top="off",
    #         labelbottom="off", left="off", right="off", labelleft="off",
    #         labelright="off", labelsize=ticklabelsize, labelcolor = subtitlecolor)
    # plt.tight_layout()
    figx.savefig(path_out + filename+'_barh.png', dpi=200, transparent = True)


# ----------------------------------------------------------------------------
# BOX PLOT (pandas wrapped)
# ----------------------------------------------------------------------------
def box(figx, cdata, path_out, filename, vert=False):
    '''Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    -vert boolean determines whether boxes are vertical or horizontal
    '''
    # Create the chart space (figure and axes)
    figx = plt.figure(figsize=(figwidth, fig_ht))
    cdata.plot.box()  # positions=[1, 4, 5, 6, 8] custom positions
    figx.savefig(path_out + filename+'_box.png', dpi=200,
        transparent = True)


# ----------------------------------------------------------------------------
# BOX & WHISKER CHART -- VERTICAL
# ----------------------------------------------------------------------------
# l locates the left/top side of the rectangle(s)
# h is the height of the rectangle(s)
# w is the width of the rectangle(s), from <<1 to 1, where 1 means bars touch
# each other.
# b is the height at which the bottom of the rectangle is plotted, useful for
# stacked bar & waterfall charts.
# Typical order of arguments: l, h, w, b; switching ? and h might enable
# a tree diagram (Mekko chart)
def boxv(figx, axx, cdata, v1, scale, startpoint, title1, xlabel1, 
         ylabel1, xticklabels, path_out, filename):
    '''Arguments:
    -figx & axx are labels for matplotlib's figure and axes objects; when
    calling the function, choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    -scale is a scalar divided into v1; e.g., 1 (million) instead of 1000000
    -startpoint accommodates rare use cases to plot a subsample; pass 0 to
    include all values
    -title1, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    '''
    # 10 arguments
    l = np.arange(startpoint, len(cdata))
    w = 0.8
    b = 0
    figx = plt.figure(figsize = (figwidth, fig_ht))
    axx = figx.add_subplot(p, q, 1)
    
    for i in np.arange(len(cdata)):
        datei = yq[1]
        y1 = cdata[yq[i]].v1
        axx.boxplot(cdata[v1]/scale, showmeans=True)

    psup.axAccessories(axx, title1, xlabel1, ylabel1, xticklabels)
    figx.savefig(path_out + filename+'_box.png', dpi=200, transparent = True)


# ----------------------------------------------------------------------------
# RADIAL BAR
# ----------------------------------------------------------------------------
# Uses pie chart function
# http://bit.ly/2r3SC8V  |  http://bit.ly/2q5TNHw
def pie(ax, values, **kwargs):
    total = sum(values)
    def formatter(pct):
        return '{:0.0f} ({:0.0f}%)'.format(pct * total / 100, pct.round(0))
    wedges, _, autotexts = ax.pie(values, autopct=formatter, **kwargs)  # http://bit.ly/2pZ3YwV
    # autotexts is a list of Text instances for the numeric labels
    for label in autotexts:
        label.set_color('0.4')
        label.set_fontsize(14)
    return wedges

def donuts(cdata1, v1, scale, categories, met1, met2, met3, filename, path_out,
    cdata2='', figx='fig1', title1='', xlabel1='', ylabel1='', xticklabels=''):
    figx, ax = plt.subplots()
    ax.axis('equal')

    grapefruit_grays = [(1.0, 0.39215686274509803, 0.40784313725490196),
        (0.8627450980392157, 0.8627450980392157, 0.8627450980392157),
        (0.8274509803921568, 0.8274509803921568, 0.8274509803921568),
        (0.7529411764705882, 0.7529411764705882, 0.7529411764705882),
        (0.6627450980392157, 0.6627450980392157, 0.6627450980392157),
        (0.5019607843137255, 0.5019607843137255, 0.5019607843137255),
        (0.4117647058823529, 0.4117647058823529, 0.4117647058823529),
        (0.4666666666666667, 0.5333333333333333, 0.6),
        (0.1843137254901961, 0.30980392156862746, 0.30980392156862746)]

    # Dictionary of properties for the pie function
    # Determine optimal angle
    # angle = 180 + float(sum(small[::2])) / sum(reordered) * 360
    kwargs = dict(colors=grapefruit_grays, startangle=10)
    width = 0.35

    # Construct the wedges of the pie with aid of functions above;
    # autopct labels wedges
    # pctdistance is the ratio between the center text generated by autopct;
    # labeldistance seems to be an alternative to pctdistance
    outside = pie(ax, cdata1[v1]/scale, radius=1,
        pctdistance=1+width/2,
        wedgeprops = dict(alpha=0.8),
        **kwargs)
   
    # Add a second donut inside
    # inside = pie(ax, cdata2[v2]/scale, radius=1-width, 
    #             pctdistance=1 - (width/2) / (1-width), **kwargs)

    # Set the width for all wedges generated by ax.pie;
    # The inside radius for each donut will be Radius - Width)
    plt.setp(outside, width=width, edgecolor='white', linewidth=0.5)

    # Legends are complicated; http://bit.ly/2qOGo7s
    ax.legend(categories, frameon=False, bbox_to_anchor=(1.15, 0.75),
        loc='center left')  # outside

    kwargs = dict(va='center', fontweight='normal')
    ax.text(0, 0.17, '2017 Q1 YoY', ha='center', size=14, color=ibm_bluehues[1], **kwargs)
    ax.text(0, 0.35, '{:+.1%}'.format(met1), ha='center', size=24, color=ibm_bluehues[1], **kwargs)
    ax.text(0, -0.1, u'\u00B1'+ 'Prev 5Q '+'{:+.0%}'.format(met2), ha='center', size=14, color=ibm_YellowOrange[3], **kwargs)
    ax.text(0, -0.3, u'\u00B1' + 'IBM '+'{:+.0%}'.format(met3), ha='center', size=14, color=ibm_YellowOrange[3], **kwargs)  # '\u00B1
    ax.text(-1.4, 1.4, 'millions USD', ha='left', size=14, color='0.4', **kwargs)
    # ax.annotate(title2, (0, 0), xytext=(np.radians(-45), 1.1), 
    #             bbox=dict(boxstyle='round', facecolor='0.8', edgecolor='none'),
    #             textcoords='polar', ha='left', **kwargs)

    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.6)

    # plt.tight_layout()
    figx.savefig(path_out + filename+'_donut.png', dpi=200, transparent = True)  # bbox_inches="tight"
    # plt.show()


# ----------------------------------------------------------------------------
# HEXAGONAL BIN (pandas wrapped)
# ----------------------------------------------------------------------------
# Non-overlapping scatterplot colored to bin count
def hexbin(x, y, path_out, grid=20, filename='temp', figx='fig1', axx='ax1',
           title1='', xlabel1='', ylabel1='', xticklabels1='',
           cblabel='', colormap='Greys'):
    """Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata is the DataFrame containing x and y
    -x and y are the data you want to plot: enclose DataFrame column names in
    quotes
    -grid is a scalar that defines the resolution of the 2-D bins
    """
    # Create the chart space (figure and axes)
    figx = plt.figure(figsize = (6, 4))
    axx = figx.add_subplot(p, q, 1)
    hb = axx.hexbin(x, y, gridsize=grid, cmap = colormap)

    medianx = x.quantile(q=0.5)
    mediany = y.quantile(q=0.5)
    plt.axvline(x=medianx, color='0.7', linewidth = 1, linestyle='--')
    plt.axhline(y=mediany, color='0.7', linewidth = 1, linestyle='--')

    psup.axAccessories(axx, title=title1, xlabel=xlabel1, ylabel=ylabel1,
                       xticklabels=xticklabels1)

    cb = figx.colorbar(hb, ax=axx)
    cb.set_label(cblabel)

    plt.tight_layout()
    figx.savefig(path_out + filename+'_hexbin.png', dpi=200)


def histogram(v1, path_out, v2='', scale=1, barcolor1=posbarcolor,
              barcolor2=neutralcolor,
              nbins=10, draw_quantiles=False,
              norm=False, logged=False, title1='', xlabel1='',
              ylabel1='', xticklabels='', y_rot=0, cumul=False,
              filename='temp'):
    """
    Plots a histogram, or multiple superimposed histograms

    Args:
        v1: series of values to visualize, usually one column in a DataFrame
        barcolor1: usually a reference to an RGB tuple from the module PlotColors
        barcolor2: could also be a single value, such as '0.5'
        nbins (int): number of equally spaced bins with which to divide instances
        norm (Boolean): whether counts are normalized to integrate to 1
        title1 (str): Figure title
        xlabel1 (str): horizontal axis label
        ylabel1 (str): vertical axis label
        xticklabels: tick labels, typically on the horizontal axis
        filename (str): graphics file name

    Returns:
        Count in each bin (first placeholder (n) in call to hist) and the
        coordinate of the edge of each bin (second placeholder (bins))

    """
    # Create the chart space (figure and axes)
    figx = plt.figure(figsize=(6, 4))
    ax = figx.add_subplot(p, q, 1)

    # Scale the values
    z1 = v1 / scale

    # Plot the histogram on the chosen axes, and retrieve
    # the histogram elements
    n1, bins1, patches1 = ax.hist(z1, bins=nbins, orientation='vertical',
                                  log=logged, normed=norm, color=barcolor1,
                                  edgecolor=edgecol,
                                  linewidth=1, cumulative=cumul,
                                  stacked=False, alpha=transp)
    # When normed = True, the ordinal values correspond to a probability
    # density whose integral is 1.0

    # Create labels for the bars
    # binwidth1 = (max(v1) - min(v1)) / nbins
    # nreal1 = n1 * len(v1) * binwidth1
    # nrealf1 = nreal1 / len(v1)
    # Call the external labeling function
    # psup.autolabel_v(patches1, ax, nrealf1)
    # for i in range(6): #np.arange(len(n1)):
    #     ax.text(bins1[i] + binwidth1/2., n1[i], '%4.0f%%' % (100*nrealf1[i]),
    #         ha='center', va='bottom', fontsize=ticklabelsize, color=barcolor1,
    #         weight = 'semibold')

    # (Optional) add a second histogram to the axes
    if len(v2) > 0:
        z2 = v2 / scale
        n2, bins2, patches2 = ax.hist(z2, bins=nbins, orientation='vertical',
                                      log=logged, normed=norm, color=barcolor2,
                                      edgecolor=edgecol,
                                      linewidth=1, cumulative=cumul,
                                      stacked=False, alpha=transp)
    else:
        n2 = None
        bins2 = None
        # binwidth2 = (max(v2) - min(v2))/nbins
    # nreal2 = n2*len(v2)*binwidth2
    # nrealf2 = nreal2/len(v2)
    # for i in range(6): #np.arange(len(n2)):
    #     ax.text(bins2[i] + binwidth2/2., n2[i], '%4.0f%%' % (100*nrealf2[i]),
    #         ha='center', va='bottom', fontsize=ticklabelsize, color=barcolor2,
    #         weight = 'semibold')

    # SET COUNT AXIS LIMIT
    # need to take max of count variable (n)
    # ymax =  np.array([r1i.min(), r2i.min(), r1e.min(), r2e.min(), r1c.min(), r2c.min(), r1cs.min(), r2cs.min()]).min() * 100
    # plt.axis(xmax=xmax)
    # ax.set_xticks(np.arange(0, ymax+1, ymax/4))

    # Add vertical lines denoting quantiles
    if draw_quantiles:
        # Compute quantiles
        quan1 = z1.quantile(q=0.05)
        quan2 = z1.quantile(q=0.5)
        # quan3 = z1.quantile(q=0.8)
        quan4 = z1.quantile(q=0.95)

        plt.axvline(x=quan1, color='white', linewidth=1, linestyle='--')
        plt.axvline(x=quan2, color='white', linewidth=1, linestyle='--')
        # plt.axvline(x=quan3, color=ibm_YellowOrange[2], linewidth=1,
        #             linestyle='--')
        plt.axvline(x=quan4, color='white', linewidth=1, linestyle='--')

    psup.axAccessories(ax, title=title1, xlabel=xlabel1, ylabel=ylabel1,
                       xticklabels=xticklabels, rot=y_rot)
    plt.tight_layout()
    figx.savefig(path_out + filename + '_hist.png', dpi=200,
                      transparent=True)
    return n1, bins1, n2, bins2

# ----------------------------------------------------------------------------
# BAR & LINE SUPERIMPOSED
# ----------------------------------------------------------------------------
def barvline(figx, cdata, startpoint, v1, scale, title1, xlabel1,
            ylabel1, xticklabels, path_out, filename):
    '''Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata is the DataFrame containing values to visualize
    -startpoint accommodates rare use cases to plot a subsample; pass 0 to
    include all values
    -v1, etc. are series of values to visualize, usually columns in a DataFrame
    -scale is a scalar divided into v1; e.g., 1 (million) instead of 1000000
    -title1, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    '''
    l = np.arange(startpoint+1, len(cdata)+1)
    w = 0.8
    b = 0
    figx = plt.figure(figsize = (figwidth, fig_ht))
    ax1 = figx.add_subplot(p, q, 1)
    for i in np.arange(startpoint, len(cdata)):
        h = cdata[i]
        # Boolean for conditional formatting
        positive = pd.Series(h > 0)
        rects = ax1.bar(l[i - startpoint], h, w, b, align='center',
                        color=positive.map({True: posbarcolor2, False: negcolor}),
                        linewidth=0, edgecolor=edgecol, alpha=transp+0.2)
        psup.autolabel_v(rects, ax1, h)

    ax1.set_xlim(1,11)
    # ax2 = figx.add_subplot(p, q, 1)
    ax2 = ax1.twinx()
    y1 = cdata[v1]/scale  # v1 = volume of opportunities
    # x = np.arange(len(cdata))
    ax2.plot(l, y1, color = ibm_bluehues[3], linewidth=4)
    # ax2.plot(l, y1, 'o', markerfacecolor = ibm_bluehues[3], markersize=3, markeredgecolor = 'w', markeredgewidth=4)
    psup.axAccessories(ax1, title1, xlabel1, ylabel1, xticklabels)
    psup.axAccessories(ax2, title1, xlabel1, ylabel1, xticklabels)
    ax2.tick_params(axis="y", which="major", bottom="off", top="off",
                labelbottom="off", left="off", right="on", labelleft="off",
                labelright="on", labelsize=ticklabelsize, labelcolor = subtitlecolor)
    plt.tight_layout()
    figx.savefig(path_out + filename+'_barvline.png', dpi=200, transparent = True)


# ----------------------------------------------------------------------------
# MOSAIC
# ----------------------------------------------------------------------------
# data = np.array([32, 53, 10, 3, 11, 50, 10, 30, 10, 25, 7, 5, 3, 15, 7, 8,
#                  36, 66, 16, 4,  9, 34,  7, 64,  5, 29, 7, 5, 2, 14, 7, 8])

# _dim = (4, 4, 2)
# data = data.reshape(_dim[::-1])

# _dims = ['Hair', 'Eye', 'Sex']
# _coords = [['Black', 'Brown', 'Red', 'Blond'], 
#            ['Brown', 'Blue', 'Hazel', 'Green'],
#            ['Male', 'Female']]

# data = xarray.DataArray(
#     data, dims=_dims[::-1],
#     coords=_coords[::-1], name='Number'
# )

# assert int(data.loc['Female', 'Green', 'Black']) == 2


# ----------------------------------------------------------------------------
# SCATTER PLOT
# ----------------------------------------------------------------------------
def scat(x, y, path_out, figx='fig1', axx='ax1', x2='', y2='',
         plot_two=False, scale_x=1, scale_y=1, title1='', cmap_='Greys',
         xlabel1='', ylabel1='', xticklabels1='', filename='temp',
         symbolsize=6, symbolcolor='0.5', transp_=0.5, medians=False,
         diag=False, legend_=False, symbol_label='', figwidth_=figwidth,
         fig_ht_=fig_ht):
    """Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata is the DataFrame containing values to visualize
    -startpoint accommodates rare use cases to plot a subsample; pass 0 to
    include all values
    -v1, etc. are series of values to visualize, usually columns in a DataFrame
    -scale is a scalar divided into v1; e.g., 1 (million) instead of 1000000
    -title1, xlabel1, ylabel1 are all strings to label the figure and axes;
    pass '' to not label
    -xticklabels would be a custom list in lieu of the default integer label
    """

    # Create a figure (blank canvas) and chart placeholder
    figx = plt.figure(figsize=(figwidth_, fig_ht_))
    axx = figx.add_subplot(p, q, 1)  # time series of pipeline value

    # Specify x and y coordinates
    x_ = x / scale_x
    y_ = y / scale_y

    # Plot x,y series in axes instances
    axx.scatter(x_, y_, c=symbolcolor, s=symbolsize,
                edgecolor='none', cmap=cmap_,
                label=symbol_label, alpha=transp_)

    if diag:
        xd = [0, 1]; yd = [0, 1]
        axx.plot(xd, yd, c='w', linewidth=1)

    if medians:
        axx.axhline(y.median() / scale_y, linestyle='dotted',
                    linewidth=1, color='0.2', xmin=0.05, xmax=0.95)

        axx.axvline(x.median() / scale_x, linestyle='dotted',
                    linewidth=1, color='0.2', ymin=0.05, ymax=0.95)

    if plot_two:
        axx.plot(x2, y2)

    psup.axAccessories(axx, title=title1, xlabel=xlabel1, ylabel=ylabel1,
                       xticklabels=xticklabels1, legend_on=legend_)

    # Determine which axis has a larger limit & set axis limits
    if y.max() > x.max():
        ax_line_ub = round(y_.max(), 2)
    else:
        ax_line_ub = round(x_.max(), 2)

    if y.min() < x.min():
        ax_line_lb = round(y_.min(), 2)
    else:
        ax_line_lb = round(x_.min(), 2)

    # Set asymmetrical line chart limits, starting from zero
    axx.set_xlim(ax_line_lb, ax_line_ub * 1.)
    axx.set_ylim(ax_line_lb, ax_line_ub * 1.)

    plt.tight_layout()
    plt.show()
    figx.savefig(path_out + filename+'_scatter.png', dpi=200, transparent=True)


# ----------------------------------------------------------------------------
# SCATTER MATRIX (CROSS-CORRELATION)
# ----------------------------------------------------------------------------
# Non-overlapping scatterplot colored to bin count
def scatmat(figx, cdata, x, y, path_out, filename):
    '''Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    -x and y are the data you want to plot: enclose DataFrame column names in
    quotes
    '''
    # Create the chart space (figure and axes)
    figx = plt.figure(figsize = (figwidth, fig_ht))
    scatter_matrix(cdata, alpha=transp, diagonal='kde')  # figsize=(figwidth, fig_ht)
    figx.savefig(path_out + filename+'_scatterMatrix.png', dpi=200,
        transparent=True)


# ----------------------------------------------------------------------------
# PARALLEL COORDINATE
# ----------------------------------------------------------------------------
# USING PANDAS BUILT-IN WRAPPER
def parcoord(figx, cdata, names, path_out, filename):
    '''Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    '''
    figx = plt.figure(figsize = (figwidth, fig_ht))
    parallel_coordinates(cdata, names)
    figx.savefig(path_out + filename+'_parcoord.png', dpi=200, transparent = True)

# HOMEMADE
# as of now, rescaling each feature so that they spread out over their 'posts'
# to a similar degree is a work in progress
# def parcoord(figx, axx, cdata, title1, xlabel1, ylabel1,
#          xticklabels, path_out, filename):
#     figx = plt.figure(figsize = (figwidth, fig_ht))
#     axx = figx.add_subplot(p, q, 1)  # time series of pipeline value
#     n = len(cdata.columns)
#     x1 = np.arange(n) + 1

#     for i in np.arange(len(cdata)):
#         y1 = cdata.iloc[i]
#         axx.plot(x1, y1, color = ibm_bluehues[0])

# Draw horizontal lines denoting quantiles
# xmin and xmax in axhline command attempts to
#     for i in np.arange(n + 1):
#         axx.axhline(y=quan1, xmin = i / (n + 1) * 0.8,
#                     xmax = i / (n + 1) * 1.2, color='0.5', linewidth = 2,
#                     linestyle='--')

# Draw vertical lines denoting the feature locations
# Left and rightmost lines are cut off by figure dimension limits?
    # axx.axvline(x=x1[0], color='0.5', linewidth = 3, linestyle='-')
    # axx.axvline(x=x1[1], color='0.5', linewidth = 3, linestyle='-')
    # axx.axvline(x=x1[2], color='0.5', linewidth = 3, linestyle='-')
    
    # psup.axAccessories(axx, title1, xlabel1, ylabel1, cdata.columns)
    # figx.savefig(path_out + filename, dpi=200, transparent = True)


# ----------------------------------------------------------------------------
# RADVIZ with a spring tension minimization algorithm
# ----------------------------------------------------------------------------
# Draws each variable at the outer rim of a unit 1-diameter circle
# Each sample in the data set is attached to each of these points by a spring,
# the stiffness of which is proportional to the numerical value of that
# attribute (normalized to unit interval)

def radviz(figx, cdata, names, path_out, filename):
    '''Arguments:
    -figx is a label for matplotlib's figure object; when calling the function,
    choose any text and place it in quotes
    -cdata are the values to visualize, usually one column in a DataFrame
    '''
    figx = plt.figure(figsize = (figwidth, fig_ht))
    radviz(cdata, names)
    figx.savefig(path_out + filename+'_radviz.png', dpi=200, transparent = True)


def waterfall(cdata, baselines, path, v1='', cdata2='', scale=1,
              filename='temp', title='', frmt='{:3.0f}', transp_=transp):
        """
        Compares one year-to-year transition of two segments on the basis of some
        metric, in three stacked bars: previous state of each segment, the
        changes in each segment, and the final state of each segment

        Args:
            cdata: list of DataFrames; one per segment, potentially containing
                several periods of data
            path (str): path to the graphics archive
            filename (str): name for the archived graphic file

        Returns:
            Nothing directly; saves graphics files

        """
        # Define list of colors
        colors = [ibm_bluehues[1], ibm_YellowOrange[3], '0.5']

        # Define the width of the rectangles of the bar chart
        w = 0.8

        # Create a figure and four axes, aligning the x-axis
        figx = plt.figure(figsize=(6, 4))  # width, height
        figx.suptitle(title, x=0.02, horizontalalignment='left',
            fontsize=ticklabelsize, color=titlecolor)

        ax1 = figx.add_subplot(1, 1, 1)

        l = np.arange(1, len(cdata) + 1)  # [0, 1, 2, 3, 4]

        ax1.bar(l, cdata, w, baselines, align='center',
                color=tableau20[0], linewidth=0, alpha=transp_)

        # h1 = cdata[0] / scale
        # h2 = cdata[1] / scale

        # b1 = 0
        # b2 = h1
        # b3 = h1 + h2
        # b4 = b3 + h3  # b3 + h3
        # b5 = 0

        # for i in np.arange(len(cdata)):
        #     l = (i+1, i+2)
        #     h = (cdata[i] / scale, abs(cdata[delta][i+1])
        #
        #     ax1.bar(j, cdata[v1][i], w, 0, align='center',
        #             color=tableau20[0], linewidth=0, alpha=transp)
        #
        #     if i + 1 < len(cdata):
        #         bi = cdata[v1][i + 1] / scale
        #         ax1.bar(i + 2, abs(cdata[delta][i+1]), w, bi, align='center',
        #                 color=tableau20[0], linewidth=0, alpha=transp)

        # Label segment in the middle of each rectangle
        # ax1.text(l[0], h1 / 2, 'FS', color='w', ha='center',
        #          fontsize=ticklabelsize)
        # ax1.text(l[0], b2 + h2 / 2, 'Alt', color='w', ha='center',
        #          fontsize=ticklabelsize)
        # ax1.text(l[3], h5 / 2, 'FS', color='w', ha='center',
        #          fontsize=ticklabelsize)
        # ax1.text(l[3], b6 + h6 / 2, 'Alt', color='w', ha='center',
        #          fontsize=ticklabelsize)

        # Label value of metric changes in the delta portions of the chart
        # if h3[0] < 0:
        #     vertalign='top'
        # else:
        #     vertalign='bottom'
        # ax1.text(l[1], b3 + h3, '${:3.0f}'.format(h3[0]), color=subtitlecolor,
        #          ha='center', va=vertalign, fontsize=ticklabelsize)
        # if h4[0] < 0:
        #     vertalign='top'
        # else:
        #     vertalign='bottom'
        # ax1.text(l[2], b4 + h4, '${:3.0f}'.format(h4[0]), color=subtitlecolor,
        #          ha='center', va=vertalign, fontsize=ticklabelsize)

        psup.axAccessories(ax1)

        vals = ax1.get_yticks()
        ax1.set_yticklabels([frmt.format(x) for x in vals])

        # Customize each axes by redoing portions of axAccessories
        ax1.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off',
                        labelbottom='on', labelleft='on', labelright='off',
                        labelsize=ticklabelsize, labelcolor=subtitlecolor)

        # Override tick coordinate labels
        # Ensure there are the right number of ticks
        ax1.set_xticks(range(len(l)))
        xticklabels = [''] + [str(x) for x in l]
        ax1.set_xticklabels(xticklabels, color=subtitlecolor, rotation=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        figx.savefig(path + filename + '.png', dpi=200,
                          transparent=True)
        # figx.savefig(archive + filename + '_pareto.png', dpi=200,
        #                   transparent=True)
