"""
COLOR PALETTES
Created July 2016
@author: Aaron Slowey, PhD, MBA
Senior Strategy Consultant, IBM Chief Analytics Office
aaron.slowey@ibm.com | +1 914 765 6732

References
http://www.randalolson.com/2014/06/28/how-to-make-beautiful-data-visualizations-in-python-with-matplotlib/
"""

from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------------
# Gray colors can be specified by color = 0.x, where the closer the value
# is to 1.0, the LIGHTER the gray will be
# Note the order -- dark to light, then light to dark
graysdl = ('0.3', '0.4', '0.5', '0.6', '0.7', '0.8')
graysld = ('0.8', '0.7', '0.6', '0.5', '0.4', '0.3')

# Light to dark, in RGB
grays = [(220,220,220), (211,211,211), (192,192,192), (169,169,169),
    (128,128,128), (105,105,105), (119,136,153), (47,79,79)]

for i in range(len(grays)):
    r, g, b = grays[i]
    grays[i] = (r / 255., g / 255., b / 255.)

# ----------------------------------------------------------------------------
# RGB from https://www.ibm.com/innovate/brand/color-palette/
ibm_yellows = [(255, 255, 79), (255, 207, 1), (253, 184, 19), (197, 147, 45)]
ibm_oranges = [(241, 144, 39), (221, 115, 28), (184, 71, 27), (129, 49, 23)]
ibm_pinks = [(243, 137, 175), (238, 62, 150), (186, 0, 110), (140, 0, 82)]
ibm_reds = [(240, 78, 55),  (217, 24, 45), (168, 16, 36), (126, 35, 34)]
ibm_blueEvens = [(0, 176, 218), (0, 138, 191), (0, 63, 105), (0, 25, 52)]
ibm_blueOdds = [(131, 209, 245), (0, 178, 239), (0, 100, 157), (0, 66, 102)]
ibm_teals = [(0, 166, 160), (0, 118, 112), (0, 96, 88), (0, 68, 61)]
ibm_greens = [(140, 198, 63), (23, 175, 75), (0, 138, 82), (0, 81, 43)]
ibm_olives = [(165, 162, 21), (131, 131, 41), (89, 79, 19), (60, 57, 0)]
ibm_purples = [(171, 26, 134), (127, 28, 125), (59, 2, 86), (40, 0, 62)]

# Monochromatic palettes
ibm_YellowOrange = [(255, 255, 79), (255, 207, 1), (253, 184, 19), (241, 144, 39), (221, 115, 28), (184, 71, 27)]
ibm_RedOrange = [(241, 144, 39), (221, 115, 28), (184, 71, 27), (240, 78, 55),  (217, 24, 45), (168, 16, 36)]
ibm_PinkPurple = [(243, 137, 175), (238, 62, 150), (186, 0, 110), (171, 26, 134), (127, 28, 125), (59, 2, 86)]
ibm_bluehues = [(131, 209, 245), (0, 178, 239), (0, 100, 157), (0, 176, 218), (0, 138, 191), (0, 63, 105)]
ibm_tealGreen = [(0, 166, 160), (0, 118, 112), (0, 96, 88), (140, 198, 63), (23, 175, 75), (0, 138, 82)]
ibm_oliveYellow = [(165, 162, 21), (131, 131, 41), (89, 79, 19), (255, 255, 79), (255, 207, 1), (253, 184, 19)]

azure = [(229, 242, 255), (204, 216, 229), (178, 190, 204), (153, 164, 178),
    (127, 138, 153), (102, 113, 127), (76, 87, 102), (51, 61, 76), (25, 35, 51),
    (0, 9, 25)]

salmon = (236, 95, 103)

# Matplotlib requires RGB values normalized by their maxima (255)
for i in range(len(ibm_YellowOrange)):
    r, g, b = ibm_YellowOrange[i]
    ibm_YellowOrange[i] = (r / 255., g / 255., b / 255.)

for i in range(len(ibm_RedOrange)):
    r, g, b = ibm_RedOrange[i]
    ibm_RedOrange[i] = (r / 255., g / 255., b / 255.)

for i in range(len(ibm_PinkPurple)):
    r, g, b = ibm_PinkPurple[i]
    ibm_PinkPurple[i] = (r / 255., g / 255., b / 255.)

for i in range(len(ibm_bluehues)):
    r, g, b = ibm_bluehues[i]
    ibm_bluehues[i] = (r / 255., g / 255., b / 255.)

for i in range(len(ibm_tealGreen)):
    r, g, b = ibm_tealGreen[i]
    ibm_tealGreen[i] = (r / 255., g / 255., b / 255.)

for i in range(len(ibm_oliveYellow)):
    r, g, b = ibm_oliveYellow[i]
    ibm_oliveYellow[i] = (r / 255., g / 255., b / 255.)

for i in range(len(azure)):
    r, g, b = azure[i]
    azure[i] = (r / 255., g / 255., b / 255.)


# ----------------------------------------------------------------------------
# The "Tableau 20"
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120), (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150), (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148), (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199), (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

for i in range(len(tableau20)):    
    r, g, b = tableau20[i]    
    tableau20[i] = (r / 255., g / 255., b / 255.)


# ----------------------------------------------------------------------------
# Swatches a la Nick Felton (Feltron)
felton_navy = [(25 / 255., 49 / 255., 61 / 255.)]
felton_grapefruit = (255 / 255., 100 / 255., 104 / 255.)
felton_bone = [(241 / 255., 241 / 255., 239 / 255.)]
felton_taupe = [(159 / 255., 146 / 255., 138 / 255.)]


# ----------------------------------------------------------------------------
# Oceanic-Next palette https://github.com/voronianski/oceanic-next-color-scheme
oceanbase00 = '#1B2B34'
oceanbase01 = '#343D46'
oceanbase02 = '#4F5B66'
oceanbase03 = '#65737E'
oceanbase04 = '#A7ADBA'
oceanbase05 = '#C0C5CE'
oceanbase06 = '#CDD3DE'
oceanbase07 = '#D8DEE9'
oceanbase08 = '#EC5f67'
oceanbase09 = '#F99157'
oceanbase0A = '#FAC863'
oceanbase0B = '#99C794'
oceanbase0C = '#5FB3B3'
oceanbase0D = '#6699CC'
oceanbase0E = '#C594C5'
oceanbase0F = '#AB7967'

ocean = ['#1B2B34', '#343D46', '#4F5B66', '#65737E', '#A7ADBA', '#C0C5CE',
    '#CDD3DE', '#D8DEE9', '#EC5f67', '#F99157', '#FAC863', '#99C794',
    '#5FB3B3', '#6699CC', '#C594C5', '#AB7967']


# HYBRID PALETTES
# Most color palettes are a list of RGB tuples; the list type supsedes tuple;
# to append tuples from different palettes, need to cast the list as a tuple
# and use parentheses and a comma:
pink_azure = (ibm_PinkPurple[0],) + tuple(azure)
grapefruit_grays = (felton_grapefruit,) + tuple(grays)
pink_grays = (ibm_PinkPurple[0],) + tuple(grays)

# FROM SATELLITE IMAGERY
brilliant_blue = [(57/255., 0/255., 221/255.)]
yellow_sand = [(213/255., 223/255., 161/255.)]
tan_sand = [(207/255., 202/255., 186/255.)]
earthy_brown = [(189/255., 157/255., 100/255.)]
iron_oxide = [(181/255., 127/255., 80/255.)]
deep_blue_gray_deep = [(67/255., 78/255., 95/255.)]
blue_gray_light = [(177/255., 195/255., 197/255.)]
deep_turquoise = [(5/255., 62/255., 62/255.)]
blood = [(100/255., 1/255., 27/255.)]

# ----------------------------------------------------------------------------
# Pre-made plot styles
# plt.style.use('fivethirtyeight')
# plt.style.use('ggplot')
# plt.style.use('bmh')

# #Seaborn color schemes
# sns.set_palette("BuPu_d", desat = 0.6)

# #use _r to reverse the color ramp
# sns.set_palette("BuGn_r", desat = 0.6)
# sns.set_palette("GnBu_r", desat = 0.6)

# #palplot creates an image of the palette
# sns.palplot(sns.light_palette((210, 90, 60), input="husl"))

# Within a globally set plot style, apply a different style for a subset of plots
# with plt.style.context(('fivethirtyeight')):
#    plt.plot()

# ----------------------------------------------------------------------------
# Color maps from a list

# colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]  # R -> G -> B  (184, 71, 27)
# n_bins = [3, 6, 10, 100]  # Discretizes the interpolation into bins
# cmap_name = 'grey_to_orange'
#
# for n_bin, ax in zip(n_bins, axs.ravel()):
#     # Create the colormap
#     cm = LinearSegmentedColormap.from_list(
#         cmap_name, colors, N=n_bin)
#     # Fewer bins will result in "coarser" colomap interpolation
#     im = ax.imshow(Z, interpolation='nearest', origin='lower', cmap=cm)
#     ax.set_title("N bins: %s" % n_bin)
#     fig.colorbar(im, ax=ax)