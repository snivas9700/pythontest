"""
Autonomous Analytics

Aaron Slowey, lead developer
IBM Chief Analytics Office
2017-2018

Data visualization utilizing Matplotlib (in object-oriented mode) and
pandas (Python 3.x)

These functions are not designed to fulfill all edge cases. They
balance simplicity with the ability to customize charts to a decent degree.
Specialized cases will require further customization of the code.

Recommended reading:
1. Tosi, S. Matplotlib for Python Developers
2. matplotlib Artist tutorial: http://matplotlib.org/users/artists.html
"""
# LIBRARIES
import numpy as np
import pandas as pd
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['IBM Plex Sans']
import matplotlib.pyplot as plt
# import matplotlib.path as mpath
# import matplotlib.patches as mpatches
# from matplotlib.colors import ListedColormap
# from matplotlib.ticker import MultipleLocator, AutoMinorLocator
# from matplotlib.dates import date2num
# from statsmodels.graphics.mosaicplot import mosaic
from plot_regional_settings import *  # unclear if this happens
import plot_support as psup


class AAvisualization:
    def __init__(self, path, fig, metric, period, lag, scale, symbol,
                 palette):
        """

        Args:
            path: local folder to store graphics files
            fig (str): Matplotlib figure object in which to place axes
            metric (str): field of interest to assess performance
            period: usually current fiscal period (e.g., quarter)
            lag: number of periods to compute period-to-period change
            scale: number to divide results for clarity
            symbol (str): scatter plot symbol
            palette: list of tuples of normalized RGB values or objects
                thereof

        """
        self.path_out = path
        self.figx = fig
        self.metric = metric
        self.period = period
        self.lag = lag
        self.scale = scale
        self.symbol = symbol
        self.colors = palette
        pass

    @staticmethod
    def pie(ax, values, **kwargs):
        """
        Called by method donuts to create a pie or radial bar chart
        http://bit.ly/2r3SC8V  |  http://bit.ly/2q5TNHw

        Args:
            ax: Matplotlib axes in which to draw the pie or radial bar chart
            values: quantities to represent by pie slices or radial bars
            ``**kwargs``: formatting parameters

        Returns:
            Pie or radial bar chart pieces & numeric labels

        """
        # To compute fractions, calculate the sum
        total = sum(values)

        # Format the numerical labels
        def formatter(pct):
            return '{:0.0f}\n({:0.0f}%)'.format(pct * total / 100, round(pct, 0))
        # Create the pie chart: wedges are the 'patches' or pie wedges, _ is a
        # placeholder of label text instances, and autotexts is a placeholder
        # for numeric labels; http://bit.ly/2pZ3YwV
        # _ is a throwaway label; it means, ignore that element
        wedges, _, autotexts = ax.pie(values, autopct=formatter, **kwargs) #

        # Format numeric labels
        for label in autotexts:
            label.set_color('0.3')
            label.set_fontsize(ticklabelsize)
        return wedges, autotexts

    def radbar(self, cdata1, cdata2, v1, segments1, segments2, met1,
               met2, met3, filename, archive, recyc_no,
               function_verb='owned', title1=''):
        """
        Generates a labeled radial bar chart with contextual metrics

        Args:
            cdata1: DataFrame
            cdata2: Second DataFrame; optional, to add a concentric layer
            v1: field within the DataFrame containing the data to be visualized
            segments1: cata1 v1's labels, typically for legend annotation
            segments2: cdata2 v1's labels, typically for legend annotation
            met1: contextual datum placed inside the radial bar chart
            met2: contextual datum placed inside the radial bar chart
            met3: contextual datum placed inside the radial bar chart
            filename: prefix of graphics file to be archived
            archive: path to the graphics archive
            title1: optional chart title
            recyc_no: tba
            function_verb: tba

        Returns:
            Nothing directly, but saves two copies of the chart, one to be
            placed in the report (and subsequently overwritten) and the other
            archived.

        """
        # self.figx, ax = plt.subplots()
        self.figx = plt.figure(figsize=(3.9, 2.5))  # width, height
        ax1 = self.figx.add_subplot(p, q, 1)
        ax1.axis('equal')
        # ax2 = self.figx.add_subplot(p, q, 1)
        # ax2.axis('equal')

        # Dictionary of properties for the pie function
        # Determine optimal angle
        # angle = 180 + float(sum(small[::2])) / sum(reordered) * 360
        kwargs = dict(colors=tableau20, startangle=0, wedgeprops=dict(alpha=
                                                                      transp))
        width1 = 0.35
        # width2 = 0.15

        # Construct the wedges of the pie using the functions above;
        # autopct labels wedges
        # pctdistance is the ratio between the center text generated by autopct;
        # labeldistance seems to be an alternative to pctdistance
        inside, inside_autotexts = self.pie(ax1, cdata1[v1]/self.scale,
            radius=1, pctdistance=1+width1, **kwargs)

        # Add a second donut inside
        # outside = pie(ax2, cdata2[v1]/self.scale, radius=1+width2,
        #             pctdistance=1 - (width2/2) / (1-width2), **kwargs)

        # Set the width for all wedges generated by ax.pie;
        # The inside radius for each donut will be Radius - Width)
        plt.setp(inside, width=width1, edgecolor='white', linewidth=0)
        # plt.setp(outside, width=width2, edgecolor='white', linewidth=0)

        # Legends are complicated; http://bit.ly/2qOGo7s
        ax1.legend(segments1, title='', frameon=False, markerfirst=False,
                   loc='upper left', bbox_to_anchor=(1.1, 0.99))
        # ax2.legend(title='identified', frameon=False,
        #    markerfirst=False, loc='upper left', bbox_to_anchor=(1, 0.4))

        kwargs = dict(va='center', fontweight='semibold')
        # ax1.text(0, 0.17, '2017 Q2 YoY', ha='center', size=12, color=ibm_bluehues[1], **kwargs)
        ax1.text(0, 0.17, '$M\n' + function_verb, ha='center',
                 size=ticklabelsize, color='0.3', **kwargs)
        # ax1.text(0, 0.35, '{:+.1%}'.format(met1), ha = 'center', size = 18, color = ibm_bluehues[1], ** kwargs)
        # ax1.text(0, -0.1, u'\u00B1'+ 'Prev 5Q '+'{:+.0%}'.format(met2), ha='center', size=12, color=ibm_YellowOrange[3], **kwargs)
        # ax1.text(0, -0.3, u'\u00B1' + 'IBM '+'{:+.0%}'.format(met3), ha='center', size=12, color=ibm_YellowOrange[3], **kwargs)

        # plt.subplots_adjust(left=0.05, right=0.7, top = 0.99, bottom=0.01)
        plt.tight_layout()

        # left, bottom, width, height in relative 0,1 coordinates
        # get_position may come in handy:
        # pos1 = ax1.get_position()
        # where then you can access pos1.x0, .y0, .width, .height
        # subplot2grid is potentially useful; see
        # http://matplotlib.org/users/gridspec.html#basic-example-of-using-subplot2grid
        ax1.set_position([-0.1, 0.1, 0.75, 0.75])

        self.figx.savefig(self.path_out + str(recyc_no) + '.png', dpi=200,
                     transparent=True)  # bbox_inches="tight"
        self.figx.savefig(archive + filename + '_radial.png', dpi=200,
                     transparent=True)  # bbox_inches="tight"

    def barv(self, axx, cdata, vals, idx_cdata, idx_vals, val_labels=False,
             w=0.8, b=0, startpoint=0):
        """

        Args:
            axx (str): Matplotlib axes object
            cdata: DataFrame containing data to plot
            vals: name(s) of field(s) to plot, numeric reference to same, or
                list thereof
            idx_cdata (int): integer reference to list of DataFrames
            idx_vals (int): integer reference to list of field names
            val_labels: custom strings with which to label ticks
            w: bar widths
            b: baseline height of bars (often used for stacked bars or
                waterfall-type visualizations)
            startpoint (int): specifies which row to begin plotting

        Returns:
            Nothing; generates a graphics file

        """

        # Define the horizontal locations of the lower left corners, width &
        # vertical position of the rectangles of the bar chart
        l = np.arange(startpoint + 1, len(cdata[idx_cdata]) + 1)

        for i in np.arange(startpoint, len(cdata[idx_cdata])):
            h = cdata[idx_cdata][vals[idx_vals]][i]
            # Boolean for conditional formatting
            positive = pd.Series(h > 0)
            rects = axx.bar(l[i - startpoint] + 0.5, h, w, b, align='center',
                            color=positive.map({True: self.colors[0],
                                                False: self.colors[2]}),
                            linewidth=0, alpha=transp)

            # Position bar value labels such that they do not overlap
            if i > 0 and h * cdata[idx_cdata][vals[idx_vals]][
                        i - 1] > 0 and h / \
                    cdata[idx_cdata][vals[idx_vals]][i - 1] - 1 < 0.6:
                boost = 1.8
            else:
                boost = 1

            if val_labels:
                psup.autolabel_v(rects, axx, h, boost, '{:3.0%}')

    def line_symbol(self, axx, cdata, vals, idx_cdata, idx_vals, scale,
                    line_or_symbol='line', startpoint=0, symbolsize=6,
                    symbolcolor='0.5', transp_=0.7):
        """

        Args:
            axx (str): Matplotlib axes object
            cdata: DataFrame containing data to plot
            vals: name(s) of field(s) to plot, numeric reference to same, or
                list thereof
            idx_cdata (int): integer reference to list of DataFrames
            idx_vals (int): integer reference to list of field names
            scale: number that large metric values are divided by for clarity
            line_or_symbol (str): specifies which type of graph to draw
            startpoint(int): specifies which row to begin plotting
            symbolsize:
            symbolcolor:
            transp_:


        Returns:
            Nothing; generates a graphics file

        """

        # Define the horizontal locations of the points
        x = np.arange(startpoint + 1, len(cdata[idx_cdata]) + 1)

        # Define vertical coordinates of line points; cdata is a list of two
        y = cdata[idx_cdata][vals[idx_vals]][startpoint:] / scale

        if line_or_symbol == 'symbol':
            axx.plot(x, y, self.symbol, markerfacecolor=symbolcolor,
                     markersize=symbolsize, markeredgecolor=self.colors[0],
                     markeredgewidth=2, alpha=transp_)
        else:
            axx.plot(x, y, color=self.colors[0], linewidth=1)

    def barvline(self, cdata, v1, v2, filename, archive, startpoint=0,
                 title1='', xlabel1='Fiscal quarter',
                 ylabel1='YtY revenue growth', ylabel2='Revenue ($M)',
                 ticklabels=''):
        """
        Veritical bars with superimposed line on a secondary y axis and
        shared x axis

        Args:
            cdata: chart data to be plotted (Pandas DataFrame)
            v1: name of field (column) in cdata
            v2: another field (column) in cdata
            filename: name of graphic file to be saved
            archive: local folder to permanently store graphics output
            startpoint: first row to plot
            title1: Matplotlib figure title
            xlabel1: x axis label
            ylabel1: y axis label
            ylabel2: secondary y axis label
            ticklabels: tick labels (could be x or y axis)

        Returns:
            May display charts (if iteractive plotting mode is on); additionally,
            graphics files.

        """
        l = np.arange(startpoint+1, len(cdata)+1)
        w = 0.8
        b = 0
        self.figx = plt.figure(figsize=(4, 2.6))  # width, height
        ax1 = self.figx.add_subplot(p, q, 1)
        for i in np.arange(startpoint, len(cdata)):
            h = cdata[v1][i]
            # Boolean for conditional formatting
            positive = pd.Series(h > 0)
            rects = ax1.bar(l[i - startpoint], h, w, b, align='center',
                            color=positive.map(
                                {True: ibm_bluehues[3], False: '0.4'}),
                            linewidth=0, edgecolor=edgecol,
                            alpha=transpnsp + 0.2)
            # psup.autolabel_v(rects, ax1, h, '{:3.0%}')

        # Add a zero-vertical reference line for the bar chart
        ax1.axhline(y=0, linestyle='dashed', color = '0.5')

        # Determine which bar chart has the larger vertical limits
        if abs(cdata[v1].min()) > abs(cdata[v1].max()):
            ax_ub = abs(cdata[v1].min())
            ax_lb = cdata[v1].min()
        else:
            ax_ub = cdata[v1].max()
            ax_lb = -ax_ub

        # Set symmetrical bar chart limits
        ax1.set_ylim(round(ax_lb, 2), round(ax_ub, 2))

        ax2 = ax1.twinx()
        y1 = cdata[v2]/self.scale  # v1 = volume of opportunities
        # x = np.arange(len(cdata))
        sec_color = ibm_YellowOrange[3]
        ax2.plot(l, y1, color = sec_color, linewidth=1)

        # ax2.axhline(linestyle = 'dashed', color = sec_color)
        ax2.plot(l, y1, 'o', markerfacecolor = 'w', markersize=6,
            markeredgecolor = sec_color, markeredgewidth=2)

        ax2.set_ylim(0, 1.01*y1.max())

        # Format left vertical axis tick labels as percentages
        vals = ax1.get_yticks()
        ax1.set_yticklabels(['{:3.0f}%'.format(x * 100) for x in vals])

        psup.axAccessories(ax1, xlabel=xlabel1, ylabel=ylabel1, rot = 90,
                           xticklabels = ticklabels)
        psup.axAccessories(ax2, ylabel=ylabel2, rot=-90)
        ax2.tick_params(axis="y", which="major", bottom="off", top="off",
            labelbottom="off", left="off", right="on", labelleft="off",
            labelright="on", labelsize=ticklabelsize-2, labelcolor=sec_color)

        # Label both vertical axis's
        # self.figx.suptitle('Revenue YtY growth (left) & absolute value (right)',
        #     x=0.025, horizontalalignment='left', fontsize=ticklabelsize - 4, color='0.5')
        # # ax1.set_title('Revenue growth YtY', loc = 'left', fontsize=12, color='0.5')
        # ax2.set_title('$M', loc = 'right', fontsize=12, color=sec_color)

        # ax1.text(1, ax_lb-0.15, '2015\nQ1', horizontalalignment = 'left', fontsize=12, color=subtitlecolor)
        # ax1.text(len(cdata), ax_lb-0.15, '2017\nQ1', horizontalalignment = 'left', fontsize=12, color=subtitlecolor)
        # plt.subplots_adjust(bottom=0.2)
        plt.tight_layout()
        self.figx.savefig(self.path_out + '0.png', dpi=200,
                     transparent=True)
        self.figx.savefig(archive + filename+'_barvline.png', dpi=200,
                     transparent = True)

    def twobarstwolines(self, cdata, vals, revenue_totals, scale, filename,
                        archive, recyc_no, idx_cdata=0, idx_vals=0,
                        startpoint=0, title1='', xlabel1='Fiscal quarter',
                        ylabel1='YtY revenue growth', ylabel2='Revenue ($M)',
                        ticklabels=''):
        """
        Up to two bars and two lines on a secondary y axis and shared x axis.
        This function visualizes two metrics across two cross sections.  It does
        so in four axes: (1, 2) bar charts of YtY metric change and (3, 4) lines
        showing absolute revenue over time (on a second axis)

        Args:
            scale:
            idx_cdata: index selecting a DataFrame in the list cdata
            idx_vals: index selecting the field in the list of field names in
                vals
            cdata: Pandas DataFrame containing the chart data
            vals: list of strings specifying field (column) names in cdata
            revenue_totals: total revenue for a segment at a point in time;
                used to calculate percentage of revenue
            filename (str): name of archived file
            archive (str): local folder and permament repository of graphics
                files
            recyc_no (int): used as a filename for an impermanent graphics
                file
            startpoint (int): first row to plot
            title1: Matplotlib figure title
            xlabel1 (str): x axis label
            ylabel1 (str): primary y axis label
            ylabel2 (str): secondary y axis label
            ticklabels (str): tick labels (x or y axis)

        Returns:
            Nothing; saves graphics files

        """
        # Create a figure and four axes, aligning the x-axis
        self.figx = plt.figure(figsize=(9.3, 6.5))
        ax1 = self.figx.add_subplot(2, 2, 3)  # FS bar chart (revenue change)
        ax2 = self.figx.add_subplot(2, 2, 4)  # Alt bar chart (revenue change)
        ax3 = self.figx.add_subplot(2, 2, 1)  # FS line (revenue)
        ax4 = self.figx.add_subplot(2, 2, 2)  # Alt line (revenue)

        # ------------------------------------------------------------------------
        # Create the rectangles of the bar charts, where positive & negative
        # values can be colored differently (with a for loop); rects stores
        # the rectangles to access in other functions; eg, to add value labels
        self.barv(ax1, cdata, vals, 0, 0)
        self.barv(ax2, cdata, vals, 1, 0)

        self.line_symbol(ax3, cdata, vals, 0, 2, scale, startpoint=3)
        self.line_symbol(ax3, cdata, vals, 0, 1, scale, line_or_symbol='symbol')
        self.line_symbol(ax4, cdata, vals, 1, 2, scale, startpoint=3)
        self.line_symbol(ax4, cdata, vals, 1, 1, scale, line_or_symbol='symbol')

        # ------------------------------------------------------------------------
        # Add a zero-vertical reference line to the bar chart
        ax1.axhline(y=0, linestyle='solid', linewidth=1, color=self.colors[2])
        ax2.axhline(y=0, linestyle='solid', linewidth=1, color=self.colors[2])

        # Add average growth rate reference lines to the bar chart
        # To set start point for average reference line, sample non-zero part
        # of the chart data
        # non_zero = cdata[0][vals[0]]
        # non_zero = non_zero[non_zero != 0]
        ax1.axhline(xmin=1.05*(1-len(cdata[0][cdata[0][vals[0]] != 0])/len(cdata[0][vals[0]])),
                    xmax=0.95, y=cdata[0][vals[0]].mean(), linestyle='dashed',
                    linewidth=0.9, color='0.5')
        ax2.axhline(xmin=1.05*(1-len(cdata[1][cdata[1][vals[0]] != 0])/len(cdata[1][vals[0]])),
                    xmax=0.95, y=cdata[1][vals[0]].mean(), linestyle='dashed',
                    linewidth=0.9, color='0.5')

        ax1.annotate('Trailing\naverage', xy=(5, cdata[0][vals[0]].mean()),
                     xycoords='data', horizontalalignment='right',
                     verticalalignment='top', fontsize=10, color='0.5')

        # ------------------------------------------------------------------------
        # Bound the vertical axis of the bar chart such that it is as tall as
        # the largest range & symmetrical about zero
        # Determine which bar chart has larger vertical limits & tentatively
        # define upper and lower boundaries
        if cdata[idx_cdata][vals[idx_vals]].min() < cdata[idx_cdata+1][
            vals[idx_vals]].min():
            ax_lb = round(cdata[idx_cdata][vals[idx_vals]].min(), 2)
        else:
            ax_lb = round(cdata[idx_cdata+1][vals[idx_vals]].min(), 2)

        if cdata[idx_cdata][vals[idx_vals]].max() > cdata[idx_cdata+1][
            vals[idx_vals]].max():
            ax_ub = round(cdata[idx_cdata][vals[idx_vals]].max(), 2)
        else:
            ax_ub = round(cdata[idx_cdata+1][vals[idx_vals]].max(), 2)

        # Set symmetrical bar chart limits, considering three possibilities:
        if ax_lb > 0 and ax_ub > 0:  # 1. Both lb and ub are positive
            ax_lb = -ax_ub
        elif ax_lb < 0 and ax_ub < 0:  # 2. Both lb and ub are negative
            ax_ub = abs(ax_lb)
        elif abs(ax_lb) > abs(ax_ub):  # 3. lb is negative and ub is positive
            ax_ub = abs(ax_lb)  # Change upper boundary
        else:
            ax_lb = -ax_ub  # Change lower boundary

        ax1.set_ylim(ax_lb, ax_ub)
        ax2.set_ylim(ax_lb, ax_ub)

        # Determine which line chart has the larger vertical limits
        if cdata[idx_cdata][vals[idx_vals+1]].max() > cdata[idx_cdata+1][vals[idx_vals+1]].max():
            ax_line_ub = round(cdata[idx_cdata][vals[idx_vals+1]].max(), 2)
        else:
            ax_line_ub = round(cdata[idx_cdata+1][vals[idx_vals+1]].max(), 2)

        # Set asymmetrical line chart limits, starting from zero
        ax3.set_ylim(0, ax_line_ub/self.scale*1.1)
        ax4.set_ylim(0, ax_line_ub/self.scale*1.1)

        # ! NEED TO MOVE THIS UP TO line_symbol
        # Label each point in the line plot with the corresponding metric share
        # ax3.annotate('REVENUE\nSHARE', xy=(l[1], y1[1]), xycoords='data',
        #              xytext=(8, 16), textcoords='offset points',
        #              horizontalalignment='right', verticalalignment='bottom',
        #              fontsize=10, color='0.5')
        x = np.arange(startpoint + 1, len(cdata[0]) + 1)
        y = cdata[0][vals[1]] / scale
        psup.annotate_points(ax3, x, y, revenue_totals / self.scale, '{:3.0%}',
                             list([ax_line_ub / self.scale * 1.1]))

        x_ = np.arange(startpoint + 1, len(cdata[1]) + 1)
        y_ = cdata[1][vals[1]] / scale
        psup.annotate_points(ax4, x_, y_, revenue_totals / self.scale,
                             '{:3.0%}', list([ax_line_ub / self.scale * 1.1]))

        # Draw vertical lines at specific points along the horizontal axis
        for i in np.arange(1, max(len(y), len(y_)), 2):
            ax3.axvline(x=i, ymin=0, ymax=y[i - 1] / (ax_line_ub / self.scale
                * 1.05), linestyle='dotted', linewidth=1, color='0.8')
            ax4.axvline(x=i, ymin=0, ymax=y_[i - 1] / (ax_line_ub / self.
                                                        scale * 1.05),
                        linestyle='dotted', linewidth=1, color='0.8')
        # ------------------------------------------------------------------------

        # Format left vertical axis tick labels as percentages
        vals = ax1.get_yticks()
        ax1.set_yticklabels(['{:.0%}'.format(x) for x in vals])
        vals2 = ax2.get_yticks()
        ax2.set_yticklabels(['{:.0%}'.format(x) for x in vals2])
        # vals3 = ax3.get_yticks()
        # ax3.set_yticklabels(['${:.0f}'.format(x) for x in vals3])
        # vals4 = ax4.get_yticks()
        # ax4.set_yticklabels(['${:.0f}'.format(x) for x in vals4])

        # Override tick coordinate labels
        # Ensure there are the right number of ticks
        ax3.set_xticks(range(len(y) + 1))
        ax3.set_xticklabels(ticklabels, color=subtitlecolor, rotation=0)
        ax4.set_xticks(range(len(y_) + 1))
        ax4.set_xticklabels(ticklabels, color=subtitlecolor, rotation=0)

        l = np.arange(startpoint + 1, len(cdata[0]) + 1)
        ax1.set_xticks(range(len(l) + 1))
        ax1.set_xticklabels(['  {}'.format(elem) for elem in ticklabels], color=subtitlecolor, rotation=0)
        ax2.set_xticks(range(len(l) + 1))
        ax2.set_xticklabels(['  {}'.format(elem) for elem in ticklabels], color=subtitlecolor, rotation=0)

        # ------------------------------------------------------------------------
        # Tick and label visibility
        # Set common properties
        psup.axAccessories(ax1, ylabel='REVENUE CHANGE\nYEAR-TO-YEAR', rot=90)
        psup.axAccessories(ax2)
        psup.axAccessories(ax3, ylabel='REVENUE ($M)', rot=90, xlabel='QUARTER')
        psup.axAccessories(ax4, xlabel='QUARTER')

        ticklabelsize = 10

        # Customize each axes by redoing portions of axAccessories
        ax1.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off', labelbottom='on',
                        labelleft='on', labelright='off', labelsize=ticklabelsize,
                        labelcolor=subtitlecolor)
        ax2.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off', labelbottom='on',
                        labelleft='on', labelright='off', labelsize=ticklabelsize,
                        labelcolor=subtitlecolor)
        ax3.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off',
                        labelbottom='on', labelleft='on', labelright='off',
                        labelsize=ticklabelsize, labelcolor=subtitlecolor)
        ax4.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off', labelbottom='on',
                        labelleft='on', labelright='off', labelsize=ticklabelsize,
                        labelcolor=subtitlecolor)
        # ------------------------------------------------------------------------

        # Label both vertical axis's
        kwargs = dict(fontsize=ticklabelsize, color=titlecolor, weight='semibold')
        # ax1.set_title('YtY change in revenue', loc='left', **kwargs)
        # ax2.set_title('YtY change in revenue', loc='left', **kwargs)
        ax1.set_title('Field Sales-owned\n ', loc='left', **kwargs)
        ax2.set_title('Alternate Routes\n ', loc='left', **kwargs)
        ax3.set_title('Field Sales-owned\n', loc='left', **kwargs)
        ax4.set_title('Alternate Routes\n ', loc='left', **kwargs)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        # self.figx.subplots_adjust(hspace=0.5)

        self.figx.savefig(self.path_out + str(recyc_no) + '.png', dpi=200, transparent = True)
        self.figx.savefig(archive + filename + '_barvline.png', dpi=200,
                          transparent=True)

    def pareto(self, cdata, archive, filename, recyc_no,
               segment='geo', title1='', xlabel1='Fiscal quarter',
               ylabel1='YtY revenue growth', ylabel2='Revenue ($M)',
               ticklabels='',
               sec_colors=[ibm_YellowOrange[2], ibm_RedOrange[4], '0.7']):
        """
        Horizontal bar chart with three colors: positive, negative, or neither
        label with values on bars and, optionally, on y axis tick labels

        Args:
            sec_colors: list of colors: bar1 positive=line1, bar2 positive=
                line2, bar negative
            cdata: DataFrame containing the data to be plotted
            archive (str): local folder for permanent storage of graphics
                files
            filename (str): name of permanent graphics file
            recyc_no (int): used as a filename for an impermanent graphics
                file
            segment (str): level in DataFrame multiIndex containing category
                labels
            title1: figure title
            xlabel1: x axis label
            ylabel1: primary y axis label
            ylabel2: secondary y axis label
            ticklabels: tick labels (x or y axis)
            self.figx (str): Matplotlib figure object name

        Returns:
            Nothing directly; saves graphics files

        """
        # Define the horizontal locations of the lower left corners, width &
        # vertical position of the rectangles of the bar chart
        l = np.arange(0, len(cdata))
        w = 0.8
        b = 0

        # Create a figure and four axes, aligning the x-axis
        self.figx = plt.figure(figsize=(4.7, 2.66))  # width, height
        ax1 = self.figx.add_subplot(1, 1, 1)
        ax1.axvline(x=0, linestyle='dotted', color='0.7')

        # ------------------------------------------------------------------------
        # Create the rectangles of the bar charts, where certain values can be
        # colored differently (requires a for loop); rects stores
        # the rectangles to access their properties; e.g., to add value labels
        for i in np.arange(len(cdata)):
            h1 = cdata.iloc[i, 0]/self.scale  # fixed indexing with cdata[v1][i]/self.scale
            # Boolean for conditional formatting
            # positive = pd.Series(h1 > 0)

            if cdata.iloc[i, -1] == 1:  # had to replace cdata[flag][i] to fix indexing
                rects = ax1.barh(l[i], h1, w, b, align='center',
                    color= sec_colors[0], linewidth=0, alpha=transp)
            elif cdata.iloc[i, -1] == -1:
                rects = ax1.barh(l[i], h1, w, b, align='center',
                    color=sec_colors[1], linewidth=0, alpha=transp)
            else:
                rects = ax1.barh(l[i], h1, w, b, align='center',
                                  color=sec_colors[2], linewidth=0, alpha=transp)

            # if i > 0 and h1 * cdata[v1][i-1] > 0 and h1 / cdata[v1][i-1] - 1 < 0.6:
            #     boost = 1.8
            # else:
            #     boost = 1
            psup.autolabel_h(rects, ax1, h1, '{:3.1f}')  # , boost

        # ------------------------------------------------------------------------
        # Set number of vertical axis tick labels
        ax1.set_yticks(range(len(l) + 1))

        # Specify categorical tick label
        lab1 = cdata.index.get_level_values(segment)

        # Specify supplemental tick annotations
        lab2 = cdata.iloc[:, 1]
        lab1_2 = ["{}  {:2.0%} YtY".format(lab1_, lab2_) for lab1_, lab2_ in
                  zip(lab1, lab2)]
        ax1.set_yticklabels(lab1_2)

        # Shift tick labels
        # ax1.yaxis.get_majorticklabels().set_x(-0.1)

        # Tick and label visibility
        ticklabelsize = 10

        # Customize each axes by redoing portions of axAccessories
        ax1.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off', labelbottom='off',
                        labelleft='on', labelright='off', labelsize=ticklabelsize,
                        labelcolor=subtitlecolor)

        psup.axspines(ax1)

        # ------------------------------------------------------------------------
        # Label both vertical axis's
        kwargs = dict(fontsize=12, color=subtitlecolor)
        ax1.set_title(title1, loc='left', **kwargs)

        ax1.set_xlabel(xlabel1, rotation=0, horizontalalignment='right',
                       verticalalignment='top', color = '0.5')
        ax1.xaxis.set_label_coords(1, -0.02)
        ax1.set_xlim(left=1.5 * cdata.iloc[:, 0].min()/self.scale)

        plt.tight_layout()

        self.figx.savefig(self.path_out + str(recyc_no) + '.png', dpi=200, transparent = True)
        self.figx.savefig(archive + filename + '_pareto.png', dpi=200,
                     transparent=True)

    def pareto_stack(self, cdata, archive, filename, recyc_no, v1=1,
                     v2=2, v3='offering_success', segment='geo',
                     title1='', xlabel1='', ylabel1='YtY revenue growth',
                     ylabel2='Revenue ($M)', ticklabels='',
                     sec_colors=[ibm_YellowOrange[2], ibm_RedOrange[4], '0.7',
                                 ibm_tealGreen[4]]):
        """
        Horizontal bar chart with three colors: positive, negative, and
        neither label with values on bars and, optionally, on y axis tick
        labels

        Args:
            sec_colors: list of colors: bar1 positive=line1, bar2 positive=
                line2, bar negative
            cdata: DataFrame containing the data to be plotted
            v1: name of the field (column) of the DataFrame containing the
                particular
            data to plot
            v2: name of the field (column) of the DataFrame containing other
            data, either to plot or annotation
            archive (str): local folder for permanent storage of graphics
                files
            filename (str): name of permanent graphics file
            recyc_no (int): used as a filename for an impermanent graphics
                file
            segment (str): level in DataFrame multiIndex containing category
                labels
            title1: figure title
            xlabel1: x axis label
            ylabel1: primary y axis label
            ylabel2: secondary y axis label
            ticklabels: tick labels (x or y axis)

        Returns:
            Nothing directly; saves graphics files

        """
        # Create a figure and four axes, aligning the x-axis
        self.figx = plt.figure(figsize=(7, 5.7))  # width, height
        ax1 = self.figx.add_subplot(1, 1, 1)
        ax1.axvline(x=0, linestyle='dotted', color='0.7')

        # Define the horizontal locations of the lower left corners, width, and
        # vertical position of the rectangles of the bar chart
        l = np.arange(0, len(cdata))
        w = 0.8
        b1 = 0
        # ------------------------------------------------------------------------
        # Create the rectangles of the bar charts, where certain values can be
        # colored differently (requires a for loop); rects stores
        # the rectangles to access their properties; e.g., to add value labels
        shover = 1.4
        for i in np.arange(len(cdata)):
            h1 = cdata.iloc[i, v1]/self.scale
            h2 = cdata.iloc[i, v2]/self.scale

            # Set proper base for each rectangle
            if h1*h2 > 0:
                b2 = h1
            else:
                b2 = 0

            rects1 = ax1.barh(l[i], h1, w, b1, align='center',
                color='0.3', linewidth=0, alpha=transp)
            rects2 = ax1.barh(l[i], h2, w, b2, align='center',
                color=ibm_YellowOrange[3], linewidth=0, alpha=transp)

            # Label one of the route contributions, inside the bar
            psup.autolabel_h(rects1, ax1, h1, '{:3.1f}',
                             in_or_out=['right', 'left'])
            # Label another of the route contributions, inside the bar
            psup.autolabel_h(rects2, ax1, h2, '{:3.1f}', b=b2,
                             in_or_out=['left', 'right'])

            # Plot and label the overall change
            # ax.plot(cdata.iloc[i, 0]/self.scale, l[i], '|', markersize=18,
            #         markeredgecolor=sec_colors[1], markeredgewidth=1)
            # Label the overall change
            # psup.autolabel_h(rects2, ax, cdata.iloc[i, 0]/self.scale, '{:3.1f}',
            #                  in_or_out=['left', 'right'])

            # Plot labels: successes, then problems
            if cdata.iloc[i, -1] == 1:
                ax1.plot(min(cdata.iloc[:, 1].min(), cdata.iloc[:, 2].min()) /
                         self.scale*shover, rects1[0].get_y() +
                         rects1[0].get_height() / 2., 'o', markersize=12,
                         color=sec_colors[3], markeredgewidth=0)  #
            elif cdata.iloc[i, -1] == -1:
                ax1.plot(min(cdata.iloc[:, 1].min(), cdata.iloc[:, 2].min()) /
                         self.scale*shover, rects1[0].get_y() +
                         rects1[0].get_height() / 2., 'o', markersize=12,
                         color=sec_colors[1], markeredgewidth=0)

        # ------------------------------------------------------------------------
        # Set number of vertical axis tick labels
        ax1.set_yticks(range(len(l) + 1))

        # Specify categorical tick label
        lab1 = cdata.index.get_level_values(segment)

        # Specify supplemental tick annotations
        lab2 = cdata.iloc[:, 0]
        lab3 = cdata.iloc[:, 3]
        lab4 = cdata.iloc[:, 5]

        lab1_3 = ['{}  \n${:+2.1f}M ({:2.0%}) YtY  \nto ${:2.1f}M  '.format(
            lab1_, lab2_/self.scale, lab3_, lab4_/self.scale)
                  for lab1_, lab2_, lab3_, lab4_ in zip(lab1, lab2, lab3, lab4)]
        ax1.set_yticklabels(lab1_3)
        plt.setp(ax1.get_yticklabels(), rotation=0, horizontalalignment='right')

        # Shift tick labels
        # ax.yaxis.get_majorticklabels().set_x(-0.1)

        # Tick and label visibility
        # Customize each axes by redoing portions of axAccessories
        ax1.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off', labelbottom='off',
                        labelleft='on', labelright='off', labelsize=ticklabelsize,
                        labelcolor=subtitlecolor)
        psup.axspines(ax1)

        # ------------------------------------------------------------------------
        # Label both vertical axis's
        kwargs = dict(fontsize=12, color=subtitlecolor)
        ax1.set_title(title1, loc='left', **kwargs)

        ax1.set_xlabel(xlabel1, rotation=0, horizontalalignment='right',
                      verticalalignment='top', color='0.5')
        ax1.xaxis.set_label_coords(1, -0.02)
        ax1.set_xlim(left=(shover + 0.5) * min(cdata.iloc[:, 1].min(), cdata.iloc[:, 2].min()) / self.scale)

        plt.tight_layout()

        self.figx.savefig(self.path_out + str(recyc_no) + '.png', dpi=200, transparent = True)
        self.figx.savefig(archive + filename + '_pareto.png', dpi=200,
                     transparent=True)

    def waterfall(self, cdata, filename, archive, recyc_no, qlevel=1):
        """
        Compares one year-to-year transition of two segments on the basis of some
        metric, in three stacked bars: previous state of each segment, the
        changes in each segment, and the final state of each segment

        Args:
            cdata: list of DataFrames; one per segment, potentially containing
                several periods of data
            qlevel: multiIndex level where the period can be found in the
                DataFrame
            recyc_no (int): name for the temporary graphic file
            archive (str): path to the graphics archive
            filename (str): name for the archived graphic file

        Returns:
            Nothing directly; saves graphics files

        """
        # Define list of colors
        colors = [ibm_bluehues[1], ibm_YellowOrange[3], '0.5']

        # Define the width of the rectangles of the bar chart
        w = 0.8

        # Create a figure and four axes, aligning the x-axis
        self.figx = plt.figure(figsize=(3.5, 4.8))  # width, height
        self.figx.suptitle('REVENUE ($M)', x=0.02, horizontalalignment='left',
            fontsize=ticklabelsize, color=titlecolor)

        ax1 = self.figx.add_subplot(1, 1, 1)
        l = [0, 1, 2, 3]

        h1 = cdata[0].xs(self.period - self.lag, level=qlevel)[self.metric] / self.scale
        h2 = cdata[1].xs(self.period - self.lag, level=qlevel)[self.metric] / self.scale
        h3 = cdata[1].xs(self.period, level=qlevel)[self.metric + '_d' +
                                                    str(self.lag)] / self.scale
        h4 = cdata[0].xs(self.period, level=qlevel)[self.metric + '_d' +
                                                    str(self.lag)] / self.scale
        h5 = cdata[0].xs(self.period, level=qlevel)[self.metric] / self.scale
        h6 = cdata[1].xs(self.period, level=qlevel)[self.metric] / self.scale

        b1 = 0
        b2 = h1
        b3 = h1 + h2
        b4 = b3 + h3  # b3 + h3
        b5 = 0
        b6 = h5

        ax1.bar(l[0], h1, w, b1, align='center', color=tableau20[0],
                linewidth=0, alpha=transp)
        ax1.bar(l[0], h2, w, b2, align='center', color=colors[1],
                linewidth=0, alpha=transp)
        ax1.bar(l[1], h3, w, b3, align='center', color=colors[1],
                linewidth=0, alpha=transp)
        ax1.bar(l[2], h4, w, b4, align='center', color=tableau20[0],
                linewidth=0, alpha=transp)
        ax1.bar(l[3], h5, w, b5, align='center', color=tableau20[0],
                linewidth=0, alpha=transp)
        ax1.bar(l[3], h6, w, b6, align='center', color=colors[1],
                linewidth=0, alpha=transp)

        # Label segment in the middle of each rectangle
        ax1.text(l[0], h1 / 2, 'FS', color='w', ha='center',
                 fontsize=ticklabelsize)
        ax1.text(l[0], b2 + h2 / 2, 'Alt', color='w', ha='center',
                 fontsize=ticklabelsize)
        ax1.text(l[3], h5 / 2, 'FS', color='w', ha='center',
                 fontsize=ticklabelsize)
        ax1.text(l[3], b6 + h6 / 2, 'Alt', color='w', ha='center',
                 fontsize=ticklabelsize)

        # Label value of metric changes in the delta portions of the chart
        if h3[0] < 0:
            vertalign='top'
        else:
            vertalign='bottom'
        ax1.text(l[1], b3 + h3, '${:3.0f}'.format(h3[0]), color=subtitlecolor,
                 ha='center', va=vertalign, fontsize=ticklabelsize)
        if h4[0] < 0:
            vertalign='top'
        else:
            vertalign='bottom'
        ax1.text(l[2], b4 + h4, '${:3.0f}'.format(h4[0]), color=subtitlecolor,
                 ha='center', va=vertalign, fontsize=ticklabelsize)

        psup.axAccessories(ax1)

        vals = ax1.get_yticks()
        ax1.set_yticklabels(['${:3.0f}'.format(x) for x in vals])

        # Customize each axes by redoing portions of axAccessories
        ax1.tick_params(axis='both', which='major', top='off', bottom='off',
                        left='off', right='off', labeltop='off',
                        labelbottom='on', labelleft='on', labelright='off',
                        labelsize=ticklabelsize, labelcolor=subtitlecolor)

        # Override tick coordinate labels
        # Ensure there are the right number of ticks
        ax1.set_xticks(range(len(l)))
        xticklabels = [str(self.period - self.lag), r'$\Delta$'+ 'Alt',
                       r'$\Delta$'+ 'FS', str(self.period)]
        ax1.set_xticklabels(xticklabels, color=subtitlecolor, rotation=0)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        self.figx.savefig(self.path_out + str(recyc_no) + '.png', dpi=200,
                          transparent=True)
        self.figx.savefig(archive + filename + '_pareto.png', dpi=200,
                          transparent=True)

    def histogram(self, v1, barcolor1, barcolor2='', nbins=10,
                  norm=False, title1='', xlabel1='',
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
        self.figx = plt.figure(figsize = (6, 4))
        ax = self.figx.add_subplot(p, q, 1)

        # Scale the values
        z1 = v1/self.scale

        # Plot the histogram on the chosen axes, and retrieve
        # the histogram elements
        n1, bins1, patches1 = ax.hist(z1, bins=nbins, orientation='vertical',
            log=False, normed=norm, color=barcolor1, edgecolor=edgecol,
            linewidth=1, cumulative=cumul, stacked=False, alpha=transp)
        # When normed = True, the ordinal values correspond to a probability
        # density whose integral is 1.0

        # Create labels for the bars
        binwidth1 = (max(v1) - min(v1))/nbins
        nreal1 = n1*len(v1)*binwidth1
        nrealf1 = nreal1/len(v1)
        # Call the external labeling function
        # psup.autolabel_v(patches1, ax, nrealf1)
        # for i in range(6): #np.arange(len(n1)):
        #     ax.text(bins1[i] + binwidth1/2., n1[i], '%4.0f%%' % (100*nrealf1[i]),
        #         ha='center', va='bottom', fontsize=ticklabelsize, color=barcolor1,
        #         weight = 'semibold')

        # (Optional) add a second histogram to the axes
        # z2 = v2/self.scale
        # n2, bins2, patches2 = ax.hist(z2, orientation='vertical', log=False,
        #     normed=True, color=barcolor2, edgecolor=edgecol,
        #     linewidth=1, stacked=False, alpha=transpnsp)
        # binwidth2 = (max(v2) - min(v2))/nbins
        # nreal2 = n2*len(v2)*binwidth2
        # nrealf2 = nreal2/len(v2)
        # for i in range(6): #np.arange(len(n2)):
        #     ax.text(bins2[i] + binwidth2/2., n2[i], '%4.0f%%' % (100*nrealf2[i]),
        #         ha='center', va='bottom', fontsize=ticklabelsize, color=barcolor2,
        #         weight = 'semibold')

        # SET COUNT AXIS LIMIT
        # need to take max of count variable (n)
        #ymax =  np.array([r1i.min(), r2i.min(), r1e.min(), r2e.min(), r1c.min(), r2c.min(), r1cs.min(), r2cs.min()]).min() * 100
        # plt.axis(xmax=xmax)
        #ax.set_xticks(np.arange(0, ymax+1, ymax/4))

        # Add vertical lines denoting quantiles
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
        self.figx.savefig(self.path_out + filename + '_hist.png', dpi=200,
                          transparent=True)
        return n1, bins1
