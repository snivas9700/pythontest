"""
Set of standard element properties to consistently format data visualizations.
'Axes' refers to a visualization of a single series of data. Multiple axes can
be arranged in a 'figure'
"""
from PlotColors import *

# Size the figure in which all axes will be drawn
figwidth = 6
fig_ht = 4

# Specify the number of rows and columns of axes that will be drawn
p = 1  # rows of axes
q = 1  # columns of axes

# Transparency of fill colors
transp = 0.6  # used to set alpha; 0 transparent, 1 opaque

# BAR CHART SETTINGS
# Rectangle width (<<1 to 1; 1 = bars touching)
w = 0.8
# Lateral shift of grouped bars
sh = 0.4
neutralcolor = '0.5'
posbarcolor = ibm_bluehues[1]  # ibm_tealGreen[3]
posbarcolor2 = ibm_PinkPurple[0]  # ibm_oliveYellow[5]
negbarcolor1 = ibm_YellowOrange[3]
negcolor = '0.5'
edgecol = 'w'

# TITLE SETTINGS
titlefontsize = 18
titlecolor = '0.3'
titlefontstyle = 'normal'
titlewt = 'bold'  # Choices include light, normal, medium, bold, extrabold,
# black, ultralight’ | ‘light’ | ‘normal’ | ‘regular’ | ‘book’ | ‘medium’ |
# ‘roman’ | ‘semibold’ | ‘demibold’ | ‘demi’ | ‘bold’ | ‘heavy’ |
# ‘extra bold’ | ‘black’
font_norm = {'fontname': 'Segoe UI'}
font_light = {'fontname': 'Segoe UI Light'}

subtitlefontsize = titlefontsize - 4
subtitlecolor = '0.4'
subtitlewt = 'normal'

ticklabelsize = subtitlefontsize - 4
ticklabelcolor = '0.4'

footnotefs = subtitlefontsize - 4
footnotecolor = subtitlecolor

valuelabelcolor = ticklabelcolor
