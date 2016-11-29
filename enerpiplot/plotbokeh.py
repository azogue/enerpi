# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import bokeh
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import HoverTool, ColumnDataSource, LinearAxis, Range1d
from bokeh.io import reset_output
from enerpi.base import SENSORS
from enerpiplot import (ROUND_W, ROUND_KWH, COLS_DATA_KWH, COLORS_DATA_KWH, COLOR_REF_RMS,
                        UNITS_DATA_KWH, LABELS_DATA_KWH, FMT_TOOLTIP_DATA_KWH)


COLS_SENSORS_RMS = SENSORS.columns_sensors_rms
COLS_SENSORS_MEAN = SENSORS.columns_sensors_mean
COL_MAIN_RMS = COLS_SENSORS_RMS[0]
COL_MAIN_MEAN = COLS_SENSORS_MEAN[0]

COL_TS = SENSORS.ts_column
COLS_DATA = COLS_SENSORS_RMS + COLS_SENSORS_MEAN + [SENSORS.ref_rms]
B_SENSORS_RMS = [True] * len(COLS_SENSORS_RMS) + [False] * len(COLS_SENSORS_MEAN) + [True]
COLORS_DATA = [SENSORS[s].color for s in COLS_DATA[:-1]] + [COLOR_REF_RMS]
UNITS_DATA = [SENSORS[s].unit for s in COLS_DATA[:-1]] + ['']
FMT_TOOLTIP_DATA = ['{0}'] * len(COLS_SENSORS_RMS) + ['{0.0}'] * len(COLS_SENSORS_MEAN) + ['{0}']
LABELS_DATA = SENSORS.descriptions(COLS_DATA)

# TOOLS = "pan,xwheel_zoom,ywheel_zoom,box_zoom,reset,save,crosshair"
TOOLS = "pan,xwheel_zoom,box_zoom,reset,save,crosshair"
P_WIDTH = 900
P_HEIGHT = 500

HTML_TROW = """<tr><td style="font-size: 15px; font-weight: bold;">
<span class="bk-tooltip-color-block" style="background-color: {2}"></span>{0}</td>
<td style="font-size: 16px; font-weight: bold; color: {2};">{1}</td></tr>"""
TOOLTIP_ROWS = """<tr><td style="font-size: 15px; font-weight: bold;">Time:</td>
<td style="font-size: 15px; font-weight: bold;">@time</td></tr>"""


def _append_hover(df, delta_min=0, multi_day=False):
    if multi_day:
        df['time'] = [(x + pd.Timedelta(minutes=delta_min)).strftime("%-d/%b %H:%M:%S") for x in df.index]
    else:
        df['time'] = [(x + pd.Timedelta(minutes=delta_min)).strftime("%H:%M:%S.%f")[:-3] for x in df.index]
    return df


def _titulo_rango_temporal(extremos_dia):
    n_days = (extremos_dia[1] - extremos_dia[0]).days
    if n_days > 1:
        return "{:%-d de %B'%y} --> {:%-d de %B'%y}".format(extremos_dia[0], extremos_dia[1])
    else:
        return '{:%A %-d de %B de %Y}'.format(extremos_dia[0])


def _get_extremos_dia(df):
    extremos = [x.replace(microsecond=0) for x in df.index[[0, -1]]]
    # extremos_dia = [pd.Timestamp(x.date() + pd.Timedelta(days=x.hour // 12)) for x in extremos]
    # return [extremos[0].date(), (extremos[1] - pd.Timedelta(minutes=1) + pd.Timedelta(days=1)).date()]
    return extremos


def get_bokeh_version():
    """
    For jinja2 templating (css & js src's)
    """
    return bokeh.__version__


def _return_html_comps(plots):
    script, divs = components(plots)
    reset_output()
    return script, divs, get_bokeh_version()


def _get_figure_plot(extremos, y_range, **fig_kwargs):
    kwargs = dict(x_range=extremos, y_range=y_range,
                  tools=TOOLS, active_drag=None, plot_width=P_WIDTH, plot_height=P_HEIGHT,
                  x_axis_type="datetime", toolbar_location="right", toolbar_sticky=False,  # "above",
                  title=_titulo_rango_temporal(extremos), responsive=True, **fig_kwargs)
    return figure(**kwargs)


def _format_axis_plot(p, color, label, unit):
    # Axis formatting
    p.axis.major_tick_in = 7
    p.axis.minor_tick_in = 3
    p.axis.major_tick_out = p.axis.minor_tick_out = 0
    p.title.text_font_size = '12pt'
    if unit:
        p.yaxis.axis_label = '{} ({})'.format(label, unit)
    else:
        p.yaxis.axis_label = label
    p.yaxis.axis_label_text_font_size = "11pt"
    p.yaxis.axis_line_color = p.yaxis.major_label_text_color = p.yaxis.axis_label_text_color = color
    p.yaxis.major_tick_line_color = p.yaxis.minor_tick_line_color = color


def _format_legend_plot(p):
    # Legend formatting
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "7pt"
    p.legend.background_fill_alpha = .5
    p.legend.label_height = 12
    p.legend.glyph_height = 10
    p.legend.label_standoff = 5
    p.legend.padding = 8
    p.legend.spacing = 3


def _plot_patch_series(fig_plot, pd_series, **kwargs_patch):
    zero = np.array([0])
    df_patch = pd.Series(pd_series.fillna(method='pad', limit=2).fillna(0).round(2))
    x = np.append(np.insert(df_patch.index.values, 0, df_patch.index.values[0]), df_patch.index.values[-1])
    y = np.append(np.insert(df_patch.values, 0, zero), zero)
    fig_plot.patch(x, y, **kwargs_patch)


def _plot_bokeh_raw(data_plot, **fig_kwargs):
    # Bokeh does not work very well!! with timezones:
    data_plot = data_plot.tz_localize(None)
    for c in COLS_SENSORS_MEAN:
        data_plot[c] /= 10.
    y_range = [0, max(500, int(np.ceil(data_plot[[COL_MAIN_RMS, SENSORS.ref_rms]].max().max() / ROUND_W) * ROUND_W))]
    minmax_ejes = [y_range, [0, 100]]

    extremos = _get_extremos_dia(data_plot)
    ejes = [COL_MAIN_RMS, COL_MAIN_MEAN]
    pos_smean = len(COLS_SENSORS_RMS)

    # Figure
    p = _get_figure_plot(extremos, minmax_ejes[0], **fig_kwargs)

    tooltip_rows = TOOLTIP_ROWS
    for c, color, unit, label, fmt in zip(COLS_DATA, COLORS_DATA, UNITS_DATA, LABELS_DATA, FMT_TOOLTIP_DATA):
        tooltip_rows += HTML_TROW.format('{}:'.format(label), '@{}{} {}'.format(c, fmt, unit), color)
    p.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows)))

    # Axis formatting
    _format_axis_plot(p, COLORS_DATA[0], LABELS_DATA[0], UNITS_DATA[0])
    if len(ejes) > 1:
        positions = (['right', 'left'] * 2)[:len(ejes[1:])]
        for extra_eje, pos, minmax, color, label, unit in zip(ejes[1:], positions, minmax_ejes[1:],
                                                              COLORS_DATA[pos_smean:], LABELS_DATA[pos_smean:],
                                                              UNITS_DATA[pos_smean:]):
            p.extra_y_ranges[extra_eje] = Range1d(*minmax)
            axkw = dict(y_range_name=extra_eje,
                        axis_label='{} ({})'.format(label, unit),
                        axis_label_text_font_size="10pt", axis_line_color=color,
                        major_label_text_color=color, axis_label_text_color=color,
                        major_tick_line_color=color, minor_tick_line_color=color,
                        major_tick_in=5, major_tick_out=0, minor_tick_in=3, minor_tick_out=0)
            p.add_layout(LinearAxis(**axkw), pos)

    # Make data source w/ time hover
    data = ColumnDataSource(_append_hover(data_plot, multi_day=(extremos[1] - extremos[0]).days > 1))

    # Plot lines of analog sensors
    for s, is_rms, color in zip(COLS_DATA, B_SENSORS_RMS, COLORS_DATA):
        kwargs_p = dict(source=data, alpha=.9, line_width=2, color=color)
        if not is_rms:
            kwargs_p.update(dict(y_range_name=ejes[1]))
        p.line(COL_TS, s, **kwargs_p)

    # Plot patch of main rms & normal analog sensors
    kwargs_patch = dict(color=COLORS_DATA[pos_smean], line_alpha=0, fill_alpha=0.10, y_range_name=ejes[1])
    _plot_patch_series(p, data_plot[COL_MAIN_MEAN], **kwargs_patch)

    kwargs_patch = dict(color=COLORS_DATA[0], line_alpha=0, fill_alpha=0.15)
    _plot_patch_series(p, data_plot[COL_MAIN_RMS], **kwargs_patch)

    # Legend formatting
    # _format_legend_plot(p)
    return p


# TODO Arreglar summary plot
def _plot_bokeh_hourly(data_plot, **fig_kwargs):
    # Bokeh does not work very well!! with timezones:
    data_plot = data_plot.tz_localize(None)

    data_plot['ts_mid'] = data_plot.index + pd.Timedelta('30min')
    data_plot.index += pd.Timedelta('30min')

    y_range = [0, max(1, np.ceil(data_plot[COLS_DATA_KWH[0]].max().max() / ROUND_KWH) * ROUND_KWH)]
    minmax_ejes = [y_range, [0, max(500, int(np.ceil(data_plot[COLS_DATA_KWH[1]].max().max() / ROUND_W) * ROUND_W))]]
    extremos = _get_extremos_dia(data_plot)
    ejes = COLS_DATA_KWH[:2]

    # Figure
    p = _get_figure_plot(extremos, minmax_ejes[0], **fig_kwargs)

    tooltip_rows = TOOLTIP_ROWS
    for c, color, unit, label, fmt in zip(COLS_DATA_KWH, COLORS_DATA_KWH,
                                          UNITS_DATA_KWH, LABELS_DATA_KWH, FMT_TOOLTIP_DATA_KWH):
        tooltip_rows += HTML_TROW.format('{}:'.format(label), '@{}{} {}'.format(c, fmt, unit), color)
    p.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows)))

    # Axis formatting
    _format_axis_plot(p, COLORS_DATA_KWH[0], LABELS_DATA_KWH[0], UNITS_DATA_KWH[0])
    if len(ejes) > 1:
        positions = (['right', 'left'] * 2)[:len(ejes[1:])]
        for extra_eje, pos, minmax, color, label, unit in zip(ejes[1:], positions, minmax_ejes[1:], COLORS_DATA_KWH[1:],
                                                              LABELS_DATA_KWH[1:], UNITS_DATA_KWH[1:]):
            p.extra_y_ranges[extra_eje] = Range1d(*minmax)
            axkw = dict(y_range_name=extra_eje,
                        axis_label='{} ({})'.format(label, unit),
                        axis_label_text_font_size="10pt", axis_line_color=color,
                        major_label_text_color=color, axis_label_text_color=color,
                        major_tick_line_color=color, minor_tick_line_color=color,
                        major_tick_in=5, major_tick_out=0, minor_tick_in=3, minor_tick_out=0)
            p.add_layout(LinearAxis(**axkw), pos)

    # Make data source w/ time hover
    data = ColumnDataSource(_append_hover(data_plot, delta_min=-30, multi_day=(extremos[1] - extremos[0]).days > 1))

    # Plot lines
    kwargs_kwh = dict(source=data, x=COL_TS, width=3600000, bottom=0, top=COLS_DATA_KWH[0], legend=LABELS_DATA_KWH[0],
                      fill_alpha=.7, line_alpha=.9, line_width=1, line_join='round', color=COLORS_DATA_KWH[0])
    p.vbar(**kwargs_kwh)

    kwargs_l = dict(source=data, alpha=.95, line_width=1.5, line_join='bevel', color=COLORS_DATA_KWH[1],
                    y_range_name=COLS_DATA_KWH[1], legend=LABELS_DATA_KWH[1])
    p.line(COL_TS, COLS_DATA_KWH[1], **kwargs_l)
    kwargs_l.update(color=COLORS_DATA_KWH[2], legend=LABELS_DATA_KWH[2])
    p.line(COL_TS, COLS_DATA_KWH[2], **kwargs_l)

    # Legend formatting
    _format_legend_plot(p)
    return p


def html_plot_buffer_bokeh(data_plot, is_kwh_plot=False, **fig_kwargs):
    """
    Given a pandas DataFrame (or a list of df's), returns the html components for rendering the graph.
    :return script, divs, bokeh.__version__
    """
    if is_kwh_plot:
        return _return_html_comps([_plot_bokeh_hourly(data_plot, **fig_kwargs)])
    else:
        return _return_html_comps([_plot_bokeh_raw(data_plot, **fig_kwargs)])
