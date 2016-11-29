# -*- coding: utf-8 -*-
import datetime as dt
from io import BytesIO
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.dates as mpd
import matplotlib.patches as mp
from matplotlib.colors import hex2color
import numpy as np
import os
import re
from enerpi.base import SENSORS, COLOR_TILES, IMG_BASEPATH, DEFAULT_IMG_MASK, log, check_resource_files, timeit
from enerpiplot import ROUND_W, ROUND_KWH, tableau20


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

    :param str or iterator x:  3/4 channel normalized color (0->1. * 3/4 ch) or hex color
    :param float ch: change coefficient
    :param float alpha: desired alpha value
    :return: modified color
    :rtype: tuple

    """
    if type(x) is str:
        if x.startswith('#'):
            x = hex2color(x)
        elif len(x) == 6:
            x = hex2color('#{}'.format(x))
        else:
            raise ValueError('ch_color: Not a valid color for change: {}, type={}'.format(x, type(x)))
    new_c = [max(0, min(1., ch * c)) for c in x]
    if alpha is not None:
        if len(x) == 4:
            new_c[3] = alpha
            return tuple(new_c)
        else:
            return tuple(new_c + [alpha])
    return tuple(new_c)


def _round_time(ts, delta=dt.timedelta(minutes=1)):
    round_to = delta.total_seconds()
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


def _format_timeseries_axis(ax_bar, ax_ts, rango_ts, num_days_plot,
                            fmt='%-H:%M', ini=0, axis_label='',
                            fmt_mayor=" %-d %h'%y, %-Hh",
                            fmt_mayor_day="%A %-d %h'%y",
                            delta_ticks=dt.timedelta(minutes=0)):

    def _gen_mayor(x, fmt_label, fmt_label_d):
        label = ' ' + x.strftime(fmt_label_d).capitalize() if x.hour == 0 else x.strftime(fmt_label)
        return mpd.date2num(x), label

    def _minor_and_mayor_hours_divisor_and_grid_lw():
        if num_days_plot < 3:
            return 1, 12, 2
        elif num_days_plot < 5:
            return 3, 24, 2
        elif num_days_plot < 8:
            return 6, 24, 1
        elif num_days_plot < 15:
            return 12, 48, 1
        else:
            return 24, 72, .5

    minor_divisor, mayor_divisor, grid_lw = _minor_and_mayor_hours_divisor_and_grid_lw()
    delta_day = dt.timedelta(days=1)
    xticks = ax_bar.get_xticks()[ini:]
    new_xticks_dt = [_round_time(dt.datetime.fromordinal(int(x)) + delta_day * (x % 1),
                                 dt.timedelta(hours=minor_divisor)) for x in xticks]
    new_xticks, new_ts_labels = list(zip(*[(mpd.date2num(x + delta_ticks), x.strftime(fmt)) for x in new_xticks_dt]))
    color_mayor = ch_color(tableau20[14], alpha=.7)
    color_minor = ch_color(tableau20[14], 1.1, alpha=.7)
    ax_bar.set_xticks(new_xticks)
    ax_bar.set_xticklabels(new_ts_labels, rotation=90 if num_days_plot > 1 else 0, ha='center',
                           fontweight='bold' if num_days_plot < 2 else 'light', fontsize='medium')
    ax_bar.tick_params(axis='x', direction='out', length=2, width=grid_lw,
                       pad=5, bottom='on', top='off', color=color_minor, labelcolor=color_minor)
    ax_bar.set_xlabel(axis_label)
    ax_bar.grid(False, axis='x')
    xticks_mayor, ts_labels_mayor = list(zip(*[_gen_mayor(x, fmt_mayor, fmt_mayor_day)
                                               for x in filter(lambda x: x.hour % mayor_divisor == 0, new_xticks_dt)]))
    ax_ts.set_xlabel('')
    ax_ts.xaxis.set_ticks_position('top')
    ax_ts.yaxis.set_ticks_position('right')
    ax_ts.yaxis.set_label_position('right')
    ax_ts.set_xticks(xticks_mayor)
    ax_ts.set_xticklabels(ts_labels_mayor, rotation=0, ha='left', fontweight='bold' if num_days_plot < 2 else 'light',
                          fontsize='large' if num_days_plot < 2 else 'medium')
    ax_ts.tick_params(axis='x', direction='out', length=20, width=grid_lw, pad=5,
                      color=color_mayor, labelcolor=color_mayor)
    ax_ts.grid(True, axis='x', color=color_mayor, linestyle='-', linewidth=grid_lw, alpha=.7)
    ax_bar.set_xlim(rango_ts)
    ax_ts.set_xlim(rango_ts)


def _gen_image_path(index, filename):
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
    n_masks = img_name.count('{:')
    if n_masks == 2:
        return img_name.format(index[0], index[-1])
    elif n_masks == 1:
        return img_name.format(index[0])
    else:
        return img_name


@timeit('plot_power_consumption_hourly', verbose=True)
def plot_power_consumption_hourly(data_rms, data_summary, data_mean_s=None,
                                  rs_data=None, rm_data=None, path_saveimg=None, show=False):
    """
    Matplotlib plot slice of ENERPI raw data with barplot of consumption summary. Optimized for 1-day plot.
    · It can resample or rolling the input data, for output smoothing.
    · Shows or saves image on disk.

    :param pd.DataFrame data_rms:
    :param pd.DataFrame data_summary:
    :param pd.DataFrame data_mean_s:
    :param str rs_data:
    :param str rm_data:
    :param str path_saveimg:
    :param bool show:
    :return: None if show=True, image_path if path_saveimg is not None, or tuple with figure, axes

    """
    # TODO Revisión plot_power_consumption_hourly
    if rm_data is not None:
        data_rms = data_rms.rolling(rm_data).mean().fillna(0)
        if data_mean_s is not None:
            data_mean_s = data_mean_s.rolling(rm_data).mean().fillna(0)
    elif rs_data is not None:
        data_rms = data_rms.resample(rs_data, label='left').mean().fillna(0)
        if data_mean_s is not None:
            data_mean_s = data_mean_s.resample(rs_data, label='left').mean().fillna(0)
    rango_ts = data_rms.index[0], data_rms.index[-1]
    num_days_plot = int(round((rango_ts[1] - rango_ts[0]) / dt.timedelta(days=1)))

    f, ax_bar = plt.subplots(figsize=(16, 9))

    lw_bar = 1.5 if num_days_plot < 7 else .5
    lw_lines = 1 if num_days_plot < 7 else .5
    color_power_axis = ch_color(tableau20[4], .85, alpha=.9)
    color_kwh_axis = ch_color(tableau20[8], alpha=.9)

    params_bar = dict(ax=ax_bar, kind='bar', color=ch_color(tableau20[9], alpha=.6),
                      lw=lw_bar, edgecolor=color_kwh_axis)
    ax_bar = data_summary.plot(**params_bar)

    ax_pos = ax_bar.get_position()
    ax_ts = f.add_axes(ax_pos, frameon=False, axis_bgcolor=None)
    for c in data_rms:
        ax_ts.plot(data_rms[c], lw=lw_lines, color=SENSORS[c].color)
        ax_ts.fill_between(data_rms.index, data_rms[c].values, 0, lw=0, facecolor=ch_color(SENSORS[c].color, alpha=.4),
                           edgecolor=SENSORS[c].color, hatch='\\', zorder=1)
    ratio = 2 * ROUND_W / ROUND_KWH
    ylim_c = np.ceil(ax_bar.get_ylim()[1] / ROUND_KWH) * ROUND_KWH
    ylim = np.ceil(ax_ts.get_ylim()[1] / ROUND_W) * ROUND_W
    ylim = max(ylim_c * ratio, ylim)

    if data_mean_s is not None:
        for c in data_mean_s:
            ax_ts.plot(data_mean_s[c] * ylim / 1000, lw=lw_lines, color=SENSORS[c].color, zorder=0)
            ax_ts.fill_between(data_mean_s.index, data_mean_s[c].values * ylim / 1000, 0, lw=0,
                               facecolor=ch_color(SENSORS[c].color, alpha=.1), zorder=0)

    # Formating xticks
    _fix_time_series_bar_mix(ax_ts, ax_bar, rango_ts, data_summary.index,
                             bar_width=.8 if num_days_plot < 7 else 1.)
    _format_timeseries_axis(ax_bar, ax_ts, rango_ts, num_days_plot)

    # Formating Y axes:
    ax_bar.set_ylabel('Consumo eléctrico', ha='center', fontweight='bold', fontsize='x-large', color=color_kwh_axis)
    ax_ts.set_ylabel('Potencia eléctrica', ha='center', fontweight='bold', fontsize='x-large', color=color_power_axis,
                     rotation=270, labelpad=20)
    ax_bar.spines['left'].set_color(color_kwh_axis)
    ax_bar.spines['left'].set_linewidth(2)
    ax_bar.spines['right'].set_color(color_power_axis)
    ax_bar.spines['right'].set_linewidth(2)
    ax_bar.spines['top'].set_color(ch_color(tableau20[14], alpha=.7))
    ax_bar.spines['top'].set_linewidth(2)
    ax_bar.spines['bottom'].set_linewidth(0)
    ax_ts.hlines(0, *rango_ts, colors=ch_color(tableau20[14], 1.3), lw=5)
    yticks_p = np.linspace(ROUND_W, ylim, ylim / ROUND_W)
    yticks_c = np.linspace(ROUND_W / ratio, ylim / ratio, ylim / ROUND_W)
    yticklabels_p = ['{:.2g}'.format(x / 1000.) + ' kW' for x in yticks_p]
    yticklabels_c = ['{:.2g}'.format(x) + ' kWh' for x in yticks_c]
    ax_bar.set_yticks(yticks_c)
    ax_bar.set_yticklabels(yticklabels_c, ha='right', fontweight='bold', fontsize='large')
    ax_bar.tick_params(axis='y', direction='out', length=5, width=2, pad=10,
                       color=color_kwh_axis, labelcolor=color_kwh_axis)
    ax_ts.set_yticks(yticks_p)
    ax_ts.set_yticklabels(yticklabels_p, ha='left', fontweight='bold', fontsize='large')
    ax_ts.tick_params(axis='y', direction='out', length=5, width=2, pad=10,
                      color=color_power_axis, labelcolor=color_power_axis)
    ax_bar.set_position(ax_pos)
    # Remove plot frame
    ax_ts.set_frame_on(False)

    if show:
        plt.show()
        return None
    elif path_saveimg is not None:
        img_name = _gen_image_path(data_rms.index, path_saveimg)
        check_resource_files(img_name)
        f.savefig(img_name, dpi=300, transparent=True, bbox_inches='tight', pad_inches=0.1, frameon=False)
        return img_name
    else:
        return f, [ax_bar, ax_ts]


def write_fig_to_svg(fig, name_img, preserve_ratio=False):
    """
    Write matplotlib figure to disk in SVG format.

    :param matplotlib.figure fig: figure to export
    :param str name_img: desired image path
    :param bool preserve_ratio: preserve size ratio on the SVG file (default: False)
    :return: operation ok
    :rtype: bool

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
    except OSError as e:
        log('OSError writing SVG on disk: {} [{}]'.format(e, e.__class__), 'error', True)
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
    if name in SENSORS.columns_sensors_rms:
        ax.set_yticklabels([str(round(float(y / 1000.), 1)) + 'kW' for y in yticks_l])
        ax.tick_params(pad=-45)
    elif name in SENSORS.columns_sensors_mean:
        ax.set_yticklabels([str(round(float(y / 10.))) + '%' for y in yticks_l])
        ax.tick_params(pad=-40)
    else:
        ax.tick_params(pad=-30)
        ax.set_yticklabels(['{:.3g}'.format(y) for y in yticks_l])
    return ax


def _plot_sensor_tile(data_s, barplot=False, ax=None, fig=None, color=COLOR_TILES, alpha=1, alpha_fill=.5):
    """Plot sensor evolution with 'tile' style (for webserver svg backgrounds)"""
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
    if (data_s is not None) and not data_s.empty:
        lw = 1.5
        ax.grid(b=True, which='major')
        data_s = data_s.fillna(0)
        if barplot:
            div = .5
            ylim = (0, np.ceil((data_s.max() + div) // div) * div)
            ax.bar(data_s.index, data_s.values, width=1 / 28, edgecolor=color, color=list(color) + [.5], linewidth=lw)
            ax.xaxis.set_major_locator(mpd.HourLocator((0, 12)))
        else:
            if data_s.name in SENSORS.columns_sensors_rms:
                div = 250
                ylim = (0, np.ceil((data_s.max() + div / 5) / div) * div)
            else:
                # assert data_s.name in SENSORS.columns_sensors_mean
                div = 25
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


def gen_svg_tiles(path_dest, catalog, last_hours=(72, 48, 24), color=COLOR_TILES):
    """
    Generate tiles (svg evolution plots of sensor variables) for enerpiweb

    :param str path_dest: IMG_TILES_BASEPATH = os.path.join(BASE_PATH, '..', 'enerpiweb', 'static', 'img', 'generated')
    :param HDFTimeSeriesCatalog catalog: ENERPI data catalog
    :param tuple last_hours: with hour intervals to plot. Default 72, 48, 24 h
    :param tuple color: color for lines in tiles (default: white, (1, 1, 1))
    :return: ok
    :rtype: bool

    """
    def _cut_axes_and_save_svgs(figure, axes, x_lim, delta_total, data_name, preserve_ratio=False):
        for lh in last_hours:
            file = os.path.join(path_dest, 'tile_{}_{}_last_{}h.svg'.format('enerpi_data', data_name, lh))
            axes.set_xlim((x_lim[0] + delta_total * (1 - lh / total_hours), x_lim[1]))
            figure.set_figwidth(_tile_figsize(lh / total_hours)[0])
            write_fig_to_svg(figure, name_img=file, preserve_ratio=preserve_ratio)

    fig = ax = None
    total_hours = last_hours[0]
    last_data, last_data_c = catalog.get(last_hours=last_hours[0], with_summary=True)
    if (last_data is not None) and (len(last_data) > 2):
        ahora = dt.datetime.now().replace(second=0, microsecond=0)
        xlim = mpd.date2num(ahora - dt.timedelta(hours=last_hours[0])), mpd.date2num(ahora)
        delta = xlim[1] - xlim[0]

        for c in SENSORS.columns_sensors_rms:
            fig, ax = _plot_sensor_tile(catalog.resample_data(last_data[c], rs_data='5min'),
                                        barplot=False, ax=ax, fig=fig, color=color)
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, c)
            plt.cla()
            fig.set_figwidth(_tile_figsize()[0])

        for c in SENSORS.columns_sensors_mean:
            fig, ax = _plot_sensor_tile(catalog.resample_data(last_data[c], rs_data='30s', use_median=True),
                                        barplot=False, ax=ax, fig=fig, color=color)
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, c)
            plt.cla()
            fig.set_figwidth(_tile_figsize()[0])

        # (DEBUG Control decay) svg de 'ref'
        # c = SENSORS.ref_rms
        # fig, ax = _plot_sensor_tile(catalog.resample_data(last_data[c], rs_data='30s'),
        #                             barplot=False, ax=ax, fig=fig, color=(0.5922, 0.149, 0.1451))
        # _cut_axes_and_save_svgs(fig, ax, xlim, delta, c, preserve_ratio=True)
        # plt.cla()
        # fig.set_figwidth(_tile_figsize()[0])

        # TODO Revisión tiles kWh
        if len(last_data_c) > 1:
            fig, ax = _plot_sensor_tile(last_data_c.kWh, barplot=True, ax=ax, fig=fig, color=color)
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, last_data_c.kWh.name)

        if fig is not None:
            plt.close(fig)
        return True
    else:
        return False
