# -*- coding: utf-8 -*-
import datetime as dt
from io import BytesIO
import locale
import logging
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.dates as mpd
import matplotlib.patches as mp
import numpy as np
import os
import re
from enerpi.base import CONFIG, DATA_PATH, CUSTOM_LOCALE, timeit


IMG_BASEPATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_DATA', 'IMG_BASEPATH'))
DEFAULT_IMG_MASK = CONFIG.get('ENERPI_DATA', 'DEFAULT_IMG_MASK')


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
locale.setlocale(locale.LC_ALL, CUSTOM_LOCALE)
# sns.set_style('whitegrid')

REGEXPR_SVG_HEIGHT = re.compile(r'<svg height="\d{1,4}pt"')
REGEXPR_SVG_WIDTH = re.compile(r' width="(\d{1,4}pt")')

GRIDSPEC_FULL = {'left': 0, 'right': 1, 'bottom': 0, 'top': 1, 'hspace': 0}
# GRIDSPEC_NORMAL = {'left': 0.075, 'right': .925, 'bottom': 0.11, 'top': 0.91, 'hspace': 0}
FONTSIZE = 10
FONTSIZE_TILE = 12
TICK_PARAMS_TILE = dict(direction='in', pad=-15, length=3, width=.5)
font = {'family': 'sans-serif',
        'size': FONTSIZE}  # 'weight' : 'light',
matplotlib.rc('font', **font)


def ch_color(x, ch=1., alpha=None):
    """
    Modify color applying a multiplier in every channel, or setting an alpha value

    :param x:  :tuple: 3/4 channel normalized color (0->1. * 3/4 ch)
    :param ch: :float: change coefficient
    :param alpha: :float: alpha value
    :return: :tuple: modified color

    """
    new_c = [max(0, min(1., ch * c)) for c in x]
    if alpha is not None:
        if len(x) == 4:
            new_c[3] = alpha
            return tuple(new_c)
        else:
            return tuple(new_c + [alpha])
    return tuple(new_c)


def _round_time(ts=None, delta=dt.timedelta(minutes=1)):
    round_to = delta.total_seconds()
    if ts is None:
        ts = dt.datetime.now()
    seconds = (ts - ts.min).seconds
    rounding = (seconds + round_to / 2) // round_to * round_to
    return ts + dt.timedelta(0, rounding - seconds, -ts.microsecond)


def _fix_time_series_bar_mix(ax_ts, ax_bar, rango_ts, index_ts_bar, bar_width=.8):
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


def _format_timeseries_axis(ax_bar, ax_ts, rango_ts,
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
    new_xticks_dt = [_round_time(dt.datetime.fromordinal(int(x)) + delta_day * (x % 1), dt.timedelta(hours=1))
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
        img_name = os.path.join(IMG_BASEPATH, img_name)
    masks = img_name.count('{:')
    if masks == 2:
        return img_name.format(data.index[0], data.index[-1])
    elif masks == 1:
        return img_name.format(data.index[0])
    else:
        return img_name


@timeit('plot_power_consumption_hourly')
def plot_power_consumption_hourly(potencia, consumo, ldr=None, rs_potencia=None, rm_potencia=None, savefig=None):
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
        potencia = potencia.rolling(rm_potencia).mean().fillna(0)
        if ldr is not None:
            ldr = ldr.rolling(rm_potencia).mean().fillna(0)
    elif rs_potencia is not None:
        potencia = potencia.resample(rs_potencia, label='left').mean().fillna(0)
        if ldr is not None:
            ldr = ldr.resample(rs_potencia, label='left').mean().fillna(0)
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
        ax_ts.plot(ldr * ylim / 1000, lw=1, color=color_ldr, zorder=0)
        ax_ts.fill_between(ldr.index, ldr.values * ylim / 1000, 0, lw=0,
                           facecolor=ch_color(tableau20[3], alpha=.1), zorder=0)

    # Remove plot frame
    ax_ts.set_frame_on(False)

    # Formating xticks
    _fix_time_series_bar_mix(ax_ts, ax_bar, rango_ts, consumo.index, bar_width=.8)
    _format_timeseries_axis(ax_bar, ax_ts, rango_ts)  # , ini=1, fin=-1)

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
        f.savefig(img_name, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.1, frameon=False)
        return img_name
    else:
        return f, [ax_bar, ax_ts]


# @timeit('write_fig_to_svg', verbose=True)
def write_fig_to_svg(fig, name_img, preserve_ratio=False):
    """
    Write matplotlib figure to disk in SVG format.

    :param fig: :matplotlib.figure: figure
    :param name_img: :str: desired image path
    :param preserve_ratio: :bool: preserve size ratio on the SVG file (default: False)
    :return: :book: operation ok

    """
    canvas = FigureCanvas(fig)
    output = BytesIO()
    imgformat = 'svg'
    canvas.print_figure(output, format=imgformat, transparent=True)
    svg_out = output.getvalue()
    if not preserve_ratio:
        svg_out = REGEXPR_SVG_WIDTH.sub(' width="100%" preserveAspectRatio="none"',
                                        REGEXPR_SVG_HEIGHT.sub('<svg height="100%"', svg_out.decode())).encode()
    try:
        with open(name_img, 'wb') as f:
            f.write(svg_out)
    except Exception as e:
        logging.error('HA OCURRIDO UN ERROR GRABANDO SVG A DISCO: {}'.format(e))
        print('HA OCURRIDO UN ERROR GRABANDO SVG A DISCO: {}'.format(e))
        return False
    return True


def _tile_figsize(fraction=1.):
    dpi = 72
    # height = 200
    # width = 1.875 * height
    height = 200
    width = 4.5 * height * fraction
    return round(width / dpi, 2), round(height / dpi, 2)


def _prep_axis_tile(color):
    fig, ax = plt.subplots(figsize=_tile_figsize(), dpi=72, gridspec_kw=GRIDSPEC_FULL, facecolor='none')
    fig.patch.set_alpha(0)
    ax.patch.set_alpha(0)
    ax.tick_params(direction='in', pad=-15, length=3, width=.5)
    ax.tick_params(axis='y', length=0, width=0, labelsize=FONTSIZE_TILE)
    ax.tick_params(axis='x', which='both', top='off', labelbottom='off')
    ax.xaxis.grid(True, color=color, linestyle=':', linewidth=1.5, alpha=.6)
    ax.yaxis.grid(True, color=color, linestyle=':', linewidth=1, alpha=.5)
    return fig, ax


def _adjust_tile_limits(name, ylim, date_ini, date_fin, ax):
    ax.set_ylim(ylim)
    ax.set_xlim(left=date_ini, right=date_fin)
    yticks = list(ax.get_yticks())[1:-1]
    yticks_l = [v for v in yticks if (v - ylim[0] < (2 * (ylim[1] - ylim[0]) / 3)) and (v > ylim[0])]
    ax.set_yticks(yticks)
    if name == 'power':
        ax.set_yticklabels([str(round(float(y / 1000.), 1)) + 'kW' for y in yticks_l])
        ax.tick_params(pad=-45)
    elif name == 'ldr':
        ax.set_yticklabels([str(round(float(y / 10.))) + '%' for y in yticks_l])
        ax.tick_params(pad=-40)
    else:
        ax.tick_params(pad=-30)
        ax.set_yticklabels([str(round(y, 4)) for y in yticks_l])
    return ax


@timeit('plot_tile_last_24h')
def plot_tile_last_24h(data_s, barplot=False, ax=None, fig=None, color=(1, 1, 1), alpha=1, alpha_fill=.5):
    matplotlib.rcParams['axes.linewidth'] = 0
    if ax is None:
        fig, ax = _prep_axis_tile(color)
    else:
        ax.patch.set_alpha(0)
        ax.tick_params(direction='in', pad=-15, length=3, width=.5)
        ax.tick_params(axis='y', length=0, width=0, labelsize=FONTSIZE_TILE)
        ax.tick_params(axis='x', which='both', top='off', labelbottom='off')
        ax.xaxis.grid(True, color=color, linestyle=':', linewidth=1.5, alpha=.6)
        ax.yaxis.grid(True, color=color, linestyle=':', linewidth=1, alpha=.5)
    rango_ts = data_s.index[0], data_s.index[-1]
    date_ini, date_fin = [t.to_pydatetime() for t in rango_ts]
    if data_s is not None and not data_s.empty:
        lw = 1.5
        ax.grid(b=True, which='major')
        data_s = data_s.fillna(0)
        if barplot:
            div = .5
            ylim = (0, np.ceil((data_s.max() + div) // div) * div)
            ax.bar(data_s.index, data_s.values, width=1 / 28, edgecolor=color, color=list(color) + [.5], linewidth=lw)
            ax.xaxis.set_major_locator(mpd.HourLocator((0, 12)))
        else:
            if data_s.name == 'power':
                div = 500
                ylim = (0, np.ceil((data_s.max() + div / 5) / div) * div)
            else:  # ldr
                div = 100
                ylim = (0, np.ceil((data_s.max() + div) // div) * div)
            data_s = data_s
            ax.plot(data_s.index, data_s, color=color, linewidth=lw, alpha=alpha)
            ax.fill_between(data_s.index, data_s, color=color, alpha=alpha_fill)
            ax.xaxis.set_major_locator(mpd.HourLocator((0, 12)))
            ax.xaxis.set_minor_locator(mpd.HourLocator(interval=1))
    else:
        ylim = 0, 100
        ax.annotate('NO DATA!', xy=(.35, .3), xycoords='axes fraction',
                    va='center', ha='center', color=ch_color(color, .9), fontsize=25)

    _adjust_tile_limits(data_s.name, ylim, date_ini, date_fin, ax)
    return fig, ax


# TODO Parametrize generated tiles
def gen_svg_tiles(path_dest, catalog, last_hours=(72, 48, 24)):

    def _cut_axes_and_save_svgs(figure, axes, x_lim, delta_total, data_name):
        for lh in last_hours:
            file = os.path.join(path_dest, 'tile_{}_{}_last_{}h.svg'.format('enerpi_data', data_name, lh))
            axes.set_xlim((x_lim[0] + delta_total * (1 - lh / total_hours), x_lim[1]))
            figure.set_figwidth(_tile_figsize(lh / total_hours)[0])
            write_fig_to_svg(figure, name_img=file)

    total_hours = last_hours[0]
    last_data, last_data_c = catalog.get(last_hours=last_hours[0], with_summary=True)
    if (last_data is not None) and (len(last_data) > 2):
        ahora = dt.datetime.now().replace(second=0, microsecond=0)
        xlim = mpd.date2num(ahora - dt.timedelta(hours=last_hours[0])), mpd.date2num(ahora)
        delta = xlim[1] - xlim[0]

        fig, ax = plot_tile_last_24h(catalog.resample_data(last_data.power, rs_data='5min'), barplot=False)
        _cut_axes_and_save_svgs(fig, ax, xlim, delta, last_data.power.name)

        plt.cla()
        fig.set_figwidth(_tile_figsize()[0])
        fig, ax = plot_tile_last_24h(catalog.resample_data(last_data.ldr, rs_data='30s', use_median=True),
                                     barplot=False, ax=ax, fig=fig)
        _cut_axes_and_save_svgs(fig, ax, xlim, delta, last_data.ldr.name)

        if len(last_data_c) > 1:
            fig, ax = plot_tile_last_24h(last_data_c.kWh, barplot=True)
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, last_data_c.kWh.name)

        if fig is not None:
            plt.close(fig)
        return True
    else:
        return False
