"""
Autonomous Analytics

Aaron Slowey, lead developer
IBM Chief Analytics Office
2017-2018

Data visualization support functions for commonly used settings
"""
import numpy as np

# LIBRARIES
from plot_regional_settings import *


# %%--------------------------------------------------------------------------
# AXES OUTLINE VISIBILITY
def axspines(axx):
    """
    Sets the visibility of each of four sides of a rectangle surrounding an
    axes

    Args:
        axx: Matplotlib axes instance

    Returns:
        Nothing; simply sets the spine properties

    """
    axx.spines["top"].set_visible(False)
    axx.spines["bottom"].set_visible(False)
    axx.spines["right"].set_visible(False)
    axx.spines["left"].set_visible(False)


# ----------------------------------------------------------------------------
# AXES SETTINGS
# ----------------------------------------------------------------------------
def axAccessories(axx, l='', rot=90, title='', xlabel='', ylabel='',
                  xticklabels='', gridax='y', ylab_pos='left',
                  legend_on=False, ylab_coord_x=-0.15, ylab_coord_y=1.0,
                  ylab_halign='right'):
    """
    Facilitates the setting of multiple element properties (content & format),
    including grid lines, a title for the axes as a whole, labels of the x and
    y axes, tick labels, and a legend. It is often necessary, after applying
    this function to an axes, to reapply portions to further get the figure
    to appear as you would like.

    Args:
        axx: Matplotlib axes instance
        l: vector of positions; e.g., where rectangles are placed on a bar chart
        rot: rotation angle in degrees, with positive angles going counterclockwise
        title (str): axes title
        xlabel (str): horizontal axis label
        ylabel (str): vertical axis label
        xticklabels (list): list of tick labels
        legend_on: Include a legend

    Returns:
        Nothing explicitly; formats an axes during its production
    """
    # Add a grid to an axes
    axx.grid(which='major', axis=gridax, color='0.9', linestyle='dotted')

    # Add titles to an axes
    axx.set_title(title, loc='left', fontname='Sans', fontsize=titlefontsize,
        fontstyle=titlefontstyle, weight=subtitlewt, color=subtitlecolor)

    # Label each axis and format the labels
    axx.set_xlabel(xlabel, rotation=0, horizontalalignment='right',
                   verticalalignment='top')
    axx.set_ylabel(ylabel, rotation=rot, horizontalalignment=ylab_halign,
                   verticalalignment='bottom')
    axx.xaxis.set_label_coords(1.0, -0.15)
    axx.yaxis.set_label_coords(ylab_coord_x, ylab_coord_y)

    # Format the axis labels
    axx.xaxis.get_label().set(color=titlecolor, fontsize=subtitlefontsize)
    axx.yaxis.get_label().set(color=titlecolor, fontsize=subtitlefontsize)

    # Override tick coordinate labels
    # Ensure there are the right number of ticks
    # axx.set_xticks(range(len(l)))
    # axx.set_xticklabels(xticklabels, color=subtitlecolor, rotation=30)
    # # axx.set_yticklabels(ticklabels, color=subtitlecolor)

    # Tick and label visibility
    # axx.tick_params(axis='both', which='major', top='off', bottom='off',
    #                 left='on', right='off', labeltop='off', labelbottom='on',
    #                 labelleft='on', labelright='off', labelsize=ticklabelsize,
    #                 labelcolor=subtitlecolor)

    # Use scientific notation if value < 0.01 or >100,000, for example
    # plt.ticklabel_format(axis='y', style='sci', scilimits=(-2,5))

    # Add a legend
    if legend_on:
        axx.legend(loc='best', frameon=False)

    # Hide the outline ('spines') of an axes
    axspines(axx)


# %% -------------------------------------------------------------------------
# VALUE LABELING FUNCTIONS
# ----------------------------------------------------------------------------
# Text labels of the height (or length) of rectangles, placed near their ends
# Two arguments: the bar rectangle(s) and the axes in which it/they
# reside (among the multiple possible axes contained within a plot)

def autolabel_v(rects, axx, h, number_form, boost=1):
    """
    Labels a vertical bar chart with the values of the underlying data

    Args:
        rects: Matplotlib Artist object representing the rectangle(s) used to
            plot the bar chart
        axx: Matplotlib axes instance
        h: value(s) to label the bars with
        boost: multiplier to modify the position of the label along the direction
            of the bar to avoid overlapping labels
        number_form: format of the label; e.g., percentage or certain number
            of decimal places

    Returns:
        Nothing explicitly; modifies a chart during its construction
    """
    # (y_bottom, y_top) = axx.get_ylim()
    # y_height = y_top - y_bottom
    i = 0
    for rect in rects:
        if h < 0:
            # height = -rect.get_height()
            label_position = boost * h  # - (y_height * 0.1)
            axx.text(rect.get_x() + rect.get_width()/2., label_position,
                     number_form.format(h), ha='center', va='top',
                     fontsize=10, color='0.5', weight='normal')  # semibold
        elif h > 0:
            # height = rect.get_height()
            label_position = boost * h # - (y_height * 0.1)
            axx.text(rect.get_x() + rect.get_width()/2., label_position,
                     number_form.format(h), ha='center', va='bottom',
                     fontsize=10, color='0.5', weight='normal')


def autolabel_h(rects, axx, h, number_form, b=0, in_or_out=['left', 'right'],
                color='0.3'):
    """
    Labels a horizontal bar chart with the values of the underlying data

    Args:
        b: base of label, for stacked bars
        in_or_out: to position labels, set horizontal justification depending
         on whether value is positive or negative
        color: font color
        rects: Matplotlib Artist object representing the rectangle(s) used to
            plot the bar chart
        axx: Matplotlib axes instance
        h: value(s) to label the bars with
        number_form: format of the label; e.g., percentage or certain number
            of decimal places

    Returns:
        Nothing explicitly; modifies a chart during its construction
    """
    # (x_bottom, x_top) = axx.get_xlim()
    # x_height = x_top - x_bottom
    for rect in rects:
        # width = rect.get_width()
        if h < 0:
            label_position = h + b  #* in_or_out
            axx.text(label_position, rect.get_y() + rect.get_height() / 2.,
                     number_form.format(h), ha=in_or_out[1], va='center',
                     fontsize=ticklabelsize, color=color)
        else:
            label_position = h + b #* in_or_out
            axx.text(label_position, rect.get_y() + rect.get_height() / 2.,
                     number_form.format(h), ha=in_or_out[0], va='center',
                     fontsize=ticklabelsize, color=color)


def annotate_points(axx, x, y, metric_totals, number_form, vertical_position):
    """
    Annotates points in a scatter plot

    Args:
        vertical_position:
        axx: Matplotlib axes instance
        x: horizontal cartesian coordinate at which to place the annotation
        y: vertical cartesian coordinate at which to place the annotation
            metric_totals: auxilliary (list or Series of) value(s) with which
            to derive an annotation value; e.g., the whole on which a proportion
            is based
        metric_totals: divisor
        number_form: format of the label; e.g., percentage or certain number
            of decimal places

    Returns:
        Nothing directly; labels points in the process of generating the
        chart
    """
    # Label the first point
    if len(vertical_position) > 1:
        axx.annotate(number_form.format(y[0]/metric_totals[0]), xy=(x[0],
                                                                    y[0]),
                     xycoords='data', horizontalalignment='left',
                     verticalalignment='bottomcenter', fontsize=footnotefs,
                     color=subtitlecolor, xytext=(0, 6),
                     textcoords='offset points')
    else:
        axx.annotate('Share ' + number_form.format(y[0] / metric_totals[0]),
                     xy=(x[0], vertical_position[0]), xycoords='data',
                     horizontalalignment='right', verticalalignment='bottom',
                     fontsize=footnotefs, color=subtitlecolor)

    # Label the rest of the points, placing label above the point if it is
    # higher than the previous point & below if it is lower
    for i in np.arange(1, len(x)):
        if len(vertical_position) > 1:
            label_position = (x[i], vertical_position[i])

            if y[i] > y[i-1]:
                axx.annotate(number_form.format(y[i]/metric_totals[i]),
                    xy=label_position, xycoords='data', xytext=(0, 6),
                    textcoords='offset points', horizontalalignment='center',
                    verticalalignment='bottom', fontsize=footnotefs,
                    color=subtitlecolor)
            else:
                axx.annotate(number_form.format(y[i]/metric_totals[i]),
                    xy=label_position, xycoords='data', xytext=(0, 6),
                    textcoords='offset points', horizontalalignment='center',
                    verticalalignment='bottom', fontsize=footnotefs,
                    color=subtitlecolor)
        else:
            label_position = (x[i], vertical_position[0])
            axx.annotate('{:3.0f}'.format(100 * y[i] / metric_totals[i]),
                xy=label_position, xycoords='data',
                horizontalalignment='center', verticalalignment='bottom',
                fontsize=footnotefs, color=subtitlecolor)