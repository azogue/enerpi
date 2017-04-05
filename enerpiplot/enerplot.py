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
# PIL included in pillow
# noinspection PyPackageRequirements
from PIL import Image
from math import sqrt
import re
from enerpi.base import (SENSORS, COLOR_TILES, IMG_BASEPATH, DEFAULT_IMG_MASK, EXPORT_PNG_TILES,
                         log, check_resource_files, timeit)
from enerpiplot import ROUND_W, ROUND_KWH, tableau20
from esiosdata import FacturaElec


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


@timeit('make_radial_gradient_background', verbose=True)
def _make_radial_gradient_background(out_color=(45, 200, 97, 0.27), in_color=(12, 133, 52, 0.83),
                                     imgsize=(834, 556), center=(.5, .5)):
    center_x = int(imgsize[0] * center[0])
    center_y = int(imgsize[1] * center[1])
    new_bg = Image.new('RGBA', size=imgsize)
    # Distance to the center
    distancias = [((x, y), float(sqrt((x - center_x) ** 2 + (y - center_y) ** 2)) / (5 * sqrt(center_y * center_x)))
                  for x in range(imgsize[0]) for y in range(imgsize[1])]

    # Calculate r, g, and b values
    [new_bg.putpixel((x, y),
                     (int(out_color[0] * dc + in_color[0] * (1 - dc)),
                      int(out_color[1] * dc + in_color[1] * (1 - dc)),
                      int(out_color[2] * dc + in_color[2] * (1 - dc)),
                      int(255 * (out_color[3] * dc + in_color[3] * (1 - dc)))))
     for (x, y), dc in distancias]
    return new_bg


def _get_png_tile_background(bg_path, size, c_out, c_in):
    if os.path.exists(bg_path):
        png_img = Image.open(bg_path)
        loaded_size = (png_img.width, png_img.height)
        if loaded_size == size:
            return png_img
        log('Diferent size in PNG TILE: {} == {} -> {}'.format(size, loaded_size, size == loaded_size), 'warn')
    # Generate new tile background
    png_img = _make_radial_gradient_background(out_color=c_out, in_color=c_in, imgsize=size, center=(.3, .3))
    png_img.save(bg_path)
    return png_img


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
    def _cut_axes_and_save_svgs(figure, axes, x_lim, delta_total, data_name,
                                tile_gradient_end, tile_gradient_st, preserve_ratio=False):
        for i, lh in enumerate(last_hours):
            file = os.path.join(path_dest, 'tile_{}_{}_last_{}h.svg'.format('enerpi_data', data_name, lh))
            axes.set_xlim((x_lim[0] + delta_total * (1 - lh / total_hours), x_lim[1]))
            figure.set_figwidth(_tile_figsize(lh / total_hours)[0])
            write_fig_to_svg(figure, name_img=file, preserve_ratio=preserve_ratio)
            # if (i + 1 == len(last_hours)):
            if EXPORT_PNG_TILES and (i + 1 == len(last_hours)):
                path_png = file[:-3] + 'png'
                base_path, name_png = os.path.split(path_png)
                figure.set_dpi(216)
                canvas = FigureCanvas(figure)
                canvas.draw()
                # Fusion data + bg:
                data_img = Image.frombytes('RGBA', (900, 600), canvas.buffer_rgba())
                png_img = _get_png_tile_background(os.path.join(base_path, 'fondo_' + name_png),
                                                   (data_img.width, data_img.height),
                                                   tile_gradient_end, tile_gradient_st)
                png_img.paste(data_img, (0, 0), data_img)
                png_img.save(path_png)

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
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, c, SENSORS[c].tile_gradient_end, SENSORS[c].tile_gradient_st)
            plt.cla()
            fig.set_figwidth(_tile_figsize()[0])

        for c in SENSORS.columns_sensors_mean:
            fig, ax = _plot_sensor_tile(catalog.resample_data(last_data[c], rs_data='30s', use_median=True),
                                        barplot=False, ax=ax, fig=fig, color=color)
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, c, SENSORS[c].tile_gradient_end, SENSORS[c].tile_gradient_st)
            plt.cla()
            fig.set_figwidth(_tile_figsize()[0])

        # (DEBUG Control decay) svg de 'ref'
        # c = SENSORS.ref_rms
        # fig, ax = _plot_sensor_tile(catalog.resample_data(last_data[c], rs_data='30s'),
        #                             barplot=False, ax=ax, fig=fig, color=(0.5922, 0.149, 0.1451))
        # _cut_axes_and_save_svgs(fig, ax, xlim, delta, c, preserve_ratio=True)
        # plt.cla()
        # fig.set_figwidth(_tile_figsize()[0])

        # Consumption tile (€ & kWh)
        if len(last_data_c) > 1:
            fig, ax = _plot_sensor_tile(last_data_c.kWh, barplot=True, ax=ax, fig=fig, color=color)

            # Append cost data:
            fact = FacturaElec(consumo=last_data_c.kWh)
            last_data_coste = fact.reparto_coste().tz_localize(None)
            last_data_coste.index = last_data_coste.index + dt.timedelta(minutes=30)
            coste_max = last_data_coste.max()
            step_c = 0.05
            c_max = max(3 * step_c, np.ceil(coste_max * 100) / 100.)
            yticks = list(np.arange(start=0, stop=c_max, step=step_c))[1:-2]
            ax_2 = plt.twinx(ax)
            ax_2.plot(last_data_coste, color=color, lw=2, ls='-.')
            ax_2.tick_params(axis='y', direction='in', pad=-40, length=3, width=.5, labelsize=FONTSIZE_TILE,
                             labelright=True, colors='k')
            ax_2.set_ylim((0, c_max))
            ax_2.set_yticks(yticks)
            ax_2.set_yticklabels(['{:.02f} €'.format(t) for t in yticks])

            ax.annotate('{:.1f}kWh'.format(last_data_c.kWh.iloc[-25:-1].sum()), (.1, .7),
                        xycoords='axes fraction', verticalalignment='top', alpha=.8,
                        bbox={'edgecolor': color, 'pad': 2, 'linewidth': 1, 'facecolor': color, 'alpha': 0.1},
                        horizontalalignment='left', size=3 * FONTSIZE, color='k')
            ax.annotate('{:.2f}€'.format(last_data_coste.iloc[-25:-1].sum()), (.9, .7),
                        xycoords='axes fraction', verticalalignment='top', alpha=.8,
                        bbox={'edgecolor': color, 'pad': 2, 'linewidth': 1, 'facecolor': color, 'alpha': 0.1},
                        horizontalalignment='right', size=3 * FONTSIZE, color='k')
            _cut_axes_and_save_svgs(fig, ax, xlim, delta, last_data_c.kWh.name,
                                    (191, 160, 245, 0.27), (140, 39, 211, 0.83))

        if fig is not None:
            plt.close(fig)
        return True
    else:
        return False
