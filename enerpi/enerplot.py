# -*- coding: utf-8 -*-
import datetime as dt
import locale
import matplotlib.dates as mpd
import matplotlib.patches as mp
import matplotlib.pyplot as plt
import numpy as np
import os
# import seaborn as sns

# from prettyprinting import print_red, print_info, print_yellow
from enerpi.base import timeit
from enerpi import DATA_PATH


def _gen_tableau20():
    # # These are the "Tableau 20" colors as RGB.
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


# semaforo_4 = [sns.palettes.crayons[k] for k in ['Green', 'Sea Green', 'Mango Tango', 'Razzmatazz']]
# These are the "Tableau 20" colors as RGB.
tableau20 = _gen_tableau20()
locale.setlocale(locale.LC_ALL, 'es_ES.UTF-8')
# sns.set_style('whitegrid')
DEFAULT_IMG_MASK = 'enerpi_potencia_consumo_ldr_{:%Y%m%d_%H%M}_{:%Y%m%d_%H%M}.png'


def ch_color(x, ch=1., alpha=None):
    new_c = [max(0, min(1., ch * c)) for c in x]
    if alpha is not None:
        if len(x) == 4:
            new_c[3] = alpha
            return new_c
        else:
            return new_c + [alpha]
    return new_c


def round_time(ts=None, delta=dt.timedelta(minutes=1)):
    round_to = delta.total_seconds()
    if ts is None:
        ts = dt.datetime.now()
    seconds = (ts - ts.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return ts + dt.timedelta(0, rounding - seconds, -ts.microsecond)


def fix_time_series_bar_mix(ax_ts, ax_bar, rango_ts, index_ts_bar, bar_width=.8):
    barras = [c for c in ax_bar.get_children() if type(c) is mp.Rectangle]
    delta_ts = rango_ts[1] - rango_ts[0]
    rango = ax_ts.get_xlim()
    ax_ts.set_xlim(rango)
    delta = rango[1] - rango[0]
    pos_start_barras = [((ts_b - rango_ts[0]) / delta_ts) * delta + rango[0] for ts_b in index_ts_bar]
    ax_ts.set_xticks(pos_start_barras)
    ax_bar.set_xticks(pos_start_barras)
    max_width = pos_start_barras[1] - pos_start_barras[0]
    new_width = bar_width * max_width
    if bar_width != 1.:
        pos_start_barras = [x + (1 - bar_width) * max_width / 2 for x in pos_start_barras]
    for b, xs in zip(barras[:-1], pos_start_barras):
        b.set_x(xs)
        b.set_width(new_width)
    ax_bar.set_xlim(rango_ts)
    ax_ts.set_xlim(rango_ts)


def format_timeseries_axis(ax_bar, ax_ts, rango_ts,
                           fmt='%-H:%M', ini=0, fin=None, axis_label='',
                           fmt_mayor=" %-d %h'%y, %-Hh", mayor_divisor=12,
                           fmt_mayor_day="%A %-d %h'%y",
                           delta_ticks=dt.timedelta(minutes=0)):

    def _gen_mayor(x, fmt_label, fmt_label_d):
        label = ' ' + x.strftime(fmt_label_d).capitalize() if x.hour == 0 else x.strftime(fmt_label)
        return mpd.date2num(x), label

    delta_day = dt.timedelta(days=1)
    if fin is None:
        xticks = ax_bar.get_xticks()[ini:]
    else:
        xticks = ax_bar.get_xticks()[ini:fin]
    new_xticks_dt = [round_time(dt.datetime.fromordinal(int(x)) + delta_day * (x % 1), dt.timedelta(hours=1))
                     for x in xticks]
    new_xticks, new_ts_labels = list(zip(*[(mpd.date2num(x + delta_ticks), x.strftime(fmt)) for x in new_xticks_dt]))
    color_mayor = ch_color(tableau20[14], alpha=.7)
    color_minor = ch_color(tableau20[14], 1.1, alpha=.7)
    ax_bar.set_xticks(new_xticks)
    ax_bar.set_xticklabels(new_ts_labels, rotation=0, ha='center', fontweight='bold', fontsize='medium')
    ax_bar.tick_params(axis='x', direction='out', length=2, width=2, pad=5, bottom='on', top='off',
                       color=color_minor, labelcolor=color_minor)
    ax_bar.set_xlabel(axis_label)
    ax_bar.grid(False, axis='x')
    xticks_mayor, ts_labels_mayor = list(zip(*[_gen_mayor(x, fmt_mayor, fmt_mayor_day)
                                               for x in filter(lambda x: x.hour % mayor_divisor == 0, new_xticks_dt)]))
    ax_ts.set_xlabel('')
    ax_ts.xaxis.set_ticks_position('top')
    ax_ts.yaxis.set_ticks_position('right')
    ax_ts.yaxis.set_label_position('right')
    ax_ts.set_xticks(xticks_mayor)
    ax_ts.set_xticklabels(ts_labels_mayor, rotation=0, ha='left', fontweight='bold', fontsize='large')
    ax_ts.tick_params(axis='x', direction='out', length=20, width=2, pad=5,
                      color=color_mayor, labelcolor=color_mayor)
    ax_ts.grid(True, axis='x', color=color_mayor, linestyle='-', linewidth=2, alpha=.7)
    ax_bar.set_xlim(rango_ts)
    ax_ts.set_xlim(rango_ts)


def _gen_image_path(data, filename):
    if type(filename) is str:
        if filename.lower() in ['png', 'svg', 'pdf', 'jpeg']:
            img_name = DEFAULT_IMG_MASK[:-3] + filename
        else:
            img_name = filename
    else:
        img_name = DEFAULT_IMG_MASK
    if len(os.path.splitext(img_name)[1]) == 0:
        img_name += '.png'
    head, tail = os.path.split(img_name)
    if not head:
        img_name = os.path.join(DATA_PATH, img_name)
    masks = img_name.count('{:')
    if masks == 2:
        return img_name.format(data.index[0], data.index[-1])
    elif masks == 1:
        return img_name.format(data.index[0])
    else:
        return img_name


@timeit('plot_potencia_consumo_horas')
def plot_potencia_consumo_horas(potencia, consumo, ldr=None,
                                rs_potencia=None, rm_potencia=None, savefig=None):
    f, ax_bar = plt.subplots(figsize=(16, 9))
    color_potencia = ch_color(tableau20[4], .85, alpha=.9)
    color_consumo = ch_color(tableau20[8], alpha=.9)
    color_ldr = ch_color(tableau20[2], alpha=.6)

    params_bar = dict(ax=ax_bar, kind='bar', color=ch_color(tableau20[9], alpha=.6),
                      lw=1.5, edgecolor=color_consumo)
    ax_bar = consumo.plot(**params_bar)

    ax_pos = ax_bar.get_position()
    ax_ts = f.add_axes(ax_pos, frameon=False, axis_bgcolor=None)

    if rm_potencia is not None:
        potencia = potencia.rolling(rm_potencia).mean()
        if ldr is not None:
            ldr = ldr.rolling(rm_potencia).mean()
    elif rs_potencia is not None:
        potencia = potencia.resample(rs_potencia, label='left').mean()
        if ldr is not None:
            ldr = ldr.resample(rs_potencia, label='left').mean()
    rango_ts = potencia.index[0], potencia.index[-1]
    ax_ts.plot(potencia, lw=1, color=color_potencia)
    ax_ts.fill_between(potencia.index, potencia.values, 0, lw=0, facecolor=ch_color(tableau20[5], alpha=.4),
                       hatch='\\', edgecolor=color_potencia, zorder=1)

    ratio = 2000
    grid_w = 1000
    ylim_c = np.ceil(ax_bar.get_ylim()[1] / .5) * .5
    ylim_p = np.ceil(ax_ts.get_ylim()[1] / grid_w) * grid_w
    ylim = max(ylim_c * ratio, ylim_p)

    if ldr is not None:
        ax_ts.plot(ldr * ylim, lw=1, color=color_ldr, zorder=0)
        ax_ts.fill_between(ldr.index, ldr.values * ylim, 0, lw=0, facecolor=ch_color(tableau20[3], alpha=.1), zorder=0)

    # Remove plot frame
    ax_ts.set_frame_on(False)

    # Formating xticks
    fix_time_series_bar_mix(ax_ts, ax_bar, rango_ts, consumo.index, bar_width=.8)
    format_timeseries_axis(ax_bar, ax_ts, rango_ts)  # , ini=1, fin=-1)

    # Formating Y axes:
    ax_bar.set_ylabel('Consumo eléctrico', ha='center', fontweight='bold', fontsize='x-large', color=color_consumo)
    ax_ts.set_ylabel('Potencia eléctrica', ha='center', fontweight='bold', fontsize='x-large', color=color_potencia,
                     rotation=270, labelpad=20)
    ax_bar.spines['left'].set_color(color_consumo)
    ax_bar.spines['left'].set_linewidth(2)
    ax_bar.spines['right'].set_color(color_potencia)
    ax_bar.spines['right'].set_linewidth(2)
    ax_bar.spines['bottom'].set_linewidth(0)
    ax_ts.hlines(0, *rango_ts, colors=ch_color(tableau20[14], 1.3), lw=5)
    yticks_p = np.linspace(grid_w, ylim, ylim / grid_w)
    yticks_c = np.linspace(grid_w / ratio, ylim / ratio, ylim / grid_w)
    yticklabels_p = ['{:.2g}'.format(x / 1000.) + ' kW' for x in yticks_p]
    yticklabels_c = ['{:.2g}'.format(x) + ' kWh' for x in yticks_c]
    ax_bar.set_yticks(yticks_c)
    ax_bar.set_yticklabels(yticklabels_c, ha='right', fontweight='bold', fontsize='large')
    ax_bar.tick_params(axis='y', direction='out', length=5, width=2, pad=10,
                       color=color_consumo, labelcolor=color_consumo)
    ax_ts.set_yticks(yticks_p)
    ax_ts.set_yticklabels(yticklabels_p, ha='left', fontweight='bold', fontsize='large')
    ax_ts.tick_params(axis='y', direction='out', length=5, width=2, pad=10,
                      color=color_potencia, labelcolor=color_potencia)

    ax_bar.set_position(ax_pos)
    if savefig is not None:
        img_name = _gen_image_path(potencia, savefig)
        # img_name = savefig if type(savefig) is str else 'plot_potencia_consumo_ldr.png'
        f.savefig(img_name, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.1, frameon=False)
        return img_name
    else:
        return f, [ax_bar, ax_ts]

