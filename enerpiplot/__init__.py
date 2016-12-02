# -*- coding: utf-8 -*-
"""
ENERPIPLOT - Common constants & color palettes

"""
ROUND_W = 500
ROUND_KWH = .5

COLOR_REF_RMS = '#FF0077'

# summary (consumption) var plots
COLS_DATA_KWH = ['kWh', 'p_max', 'p_min', 't_ref']
COLORS_DATA_KWH = ['#8C27D3', '#972625', '#f4af38', '#8C27D3']
UNITS_DATA_KWH = ['kWh', 'W', 'W', '']
LABELS_DATA_KWH = ['Consumption', 'Max Power', 'Min Power', 'Sampled']
FMT_TOOLTIP_DATA_KWH = ['{0.000}', '{0}', '{0}', '{0.000}']


def _gen_tableau20():
    # These are the "Tableau 20" colors as RGB.
    tableau = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
               (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
               (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
               (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
               (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]
    # Scale the RGB values to the [0, 1] range, which is the format matplotlib accepts.
    for i in range(len(tableau)):
        r, g, b = tableau[i]
        tableau[i] = (r / 255., g / 255., b / 255.)
    return tableau


# These are the "Tableau 20" colors as RGB.
tableau20 = _gen_tableau20()
