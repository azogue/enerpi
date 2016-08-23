# -*- coding: utf-8 -*-
import locale
# import logging
import numpy as np
import bokeh
from bokeh.plotting import figure
from bokeh.embed import components
from bokeh.models import BoxAnnotation, HoverTool, ColumnDataSource  # , LinearAxis, Range1d  # , NumeralTickFormatter
from bokeh.io import reset_output


ROUND_W = 500

locale.setlocale(locale.LC_ALL, locale.getlocale())
# TOOLS = "pan,xwheel_zoom,box_zoom,reset,save,crosshair"
TOOLS = "pan,xwheel_zoom,box_zoom,reset,save"
P_WIDTH = 900
P_HEIGHT = 500

# tooltips = [("Hora", "@time"), ("Potencia", "$y{0} W")]
html_table_tooltip = """<div><table>{}</table></div>"""
HTML_TROW = """<tr>
<td style="font-size: 15px; font-weight: bold;">{}</td>
<td style="font-size: 15px; font-weight: bold; color: $color[hex];">{}</td>
</tr>"""


def _append_time_hover(df, with_hover, multiple_days=False):
    if with_hover:
        if multiple_days:
            df['time'] = [x.strftime("%-d/%b %H:%M:%S") for x in df.index]  # "%-d %b'%y %H:%M:%S"
        else:
            df['time'] = [x.strftime("%H:%M:%S") for x in df.index]
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


def _append_boxes(p, levels=(500, 3000), horiz_box=True, alpha=0.05, axis=None, is_band=False):
    kwargs_box = dict(plot=p, fill_alpha=alpha)
    if axis is not None:
        kwargs_box.update(y_range_name=axis)
    boxes = []
    if is_band:
        if horiz_box:
            boxes.append(BoxAnnotation(top=levels[0], fill_color='red', **kwargs_box))
            boxes.append(BoxAnnotation(bottom=levels[0], top=levels[1], fill_color='orange', **kwargs_box))
            boxes.append(BoxAnnotation(bottom=levels[1], top=levels[2], fill_color='green', **kwargs_box))
            boxes.append(BoxAnnotation(bottom=levels[2], top=levels[3], fill_color='orange', **kwargs_box))
            boxes.append(BoxAnnotation(bottom=levels[3], fill_color='red', **kwargs_box))
        else:
            boxes.append(BoxAnnotation(right=levels[0], fill_color='red', **kwargs_box))
            boxes.append(BoxAnnotation(left=levels[0], right=levels[1], fill_color='orange', **kwargs_box))
            boxes.append(BoxAnnotation(left=levels[1], right=levels[2], fill_color='green', **kwargs_box))
            boxes.append(BoxAnnotation(left=levels[2], right=levels[3], fill_color='orange', **kwargs_box))
            boxes.append(BoxAnnotation(left=levels[3], fill_color='red', **kwargs_box))
    else:
        if horiz_box:
            boxes.append(BoxAnnotation(top=levels[0], fill_color='green', **kwargs_box))
            boxes.append(BoxAnnotation(bottom=levels[0], top=levels[1], fill_color='orange', **kwargs_box))
            boxes.append(BoxAnnotation(bottom=levels[1], fill_color='red', **kwargs_box))
        else:
            boxes.append(BoxAnnotation(right=levels[0], fill_color='green', **kwargs_box))
            boxes.append(BoxAnnotation(left=levels[0], right=levels[1], fill_color='orange', **kwargs_box))
            boxes.append(BoxAnnotation(left=levels[1], fill_color='red', **kwargs_box))
    p.renderers.extend(boxes)


def get_bokeh_version():
    """
    For templates (css & js src's)
    """
    return bokeh.__version__


def _return_html_comps(plots):
    script, divs = components(plots)
    reset_output()
    return script, divs, get_bokeh_version()


# def _get_axis_boxes_conf(axis):
#     levels = (500, 3000)
#     return {'axis': axis, 'levels': levels, 'horiz_box': True, 'alpha': 0.05, 'is_band': len(levels) > 2}


def _plot_buffer_bokeh(data_plot, color_base, **fig_kwargs):
    # Bokeh does not work very well!! with timezones:
    data_plot = data_plot.tz_localize(None)
    v_round = ROUND_W
    y_range = [0, max(1000, int(np.ceil(data_plot['power'].max().max() / v_round) * v_round))]
    data_plot['ldr_plot'] = data_plot['ldr'] * y_range[1] / 100
    extremos_dia = _get_extremos_dia(data_plot)
    # prim_axis = 'power'
    # prim_axis = ejes[0]

    # Figure
    width, height = P_WIDTH, P_HEIGHT
    # w_boxes = False
    kwargs = dict(x_range=extremos_dia,  y_range=y_range,
                  tools=TOOLS, plot_width=width, plot_height=height,
                  x_axis_type="datetime",
                  title=_titulo_rango_temporal(extremos_dia), responsive=True, **fig_kwargs)
    p = figure(**kwargs)

    html_trow = """<tr><td style="font-size: 15px; font-weight: bold;">
        <span class="bk-tooltip-color-block" style="background-color: {2}"></span>{0}</td>
        <td style="font-size: 16px; font-weight: bold; color: {2};">{1}</td></tr>"""
    tooltip_rows = """<tr><td style="font-size: 15px; font-weight: bold;">Hora:</td>
        <td style="font-size: 15px; font-weight: bold;">@time</td></tr>"""
    for c, color, unit, label in zip(['power', 'ldr'], [color_base, '#F4D83F'], ['W', '%'], ['Power', 'LDR']):
        tooltip_rows += html_trow.format('{}:'.format(label), '@{}{} {}'.format(c, '{0.0}', unit), color)
    p.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows)))

    # Axis formatting
    p.axis.major_tick_in = 7
    p.axis.major_tick_out = 0
    p.axis.minor_tick_in = 3
    p.axis.minor_tick_out = 0
    p.title.text_font_size = '12pt'
    p.yaxis.axis_label = '{} ({})'.format('Power', 'W')
    p.yaxis.axis_label_text_font_size = "11pt"
    p.yaxis.axis_line_color = color_base
    p.yaxis.major_label_text_color = color_base
    p.yaxis.axis_label_text_color = color_base
    p.yaxis.major_tick_line_color = color_base
    p.yaxis.minor_tick_line_color = color_base
    # if len(ejes) > 1:
    #     if len(ejes) == 4:  # TODO Error bokeh plot --> eje x mide 0: los 4 ejes juntos a la izq
    #         positions = ['left', 'left', 'right']
    #     else:
    #         positions = (['right', 'left'] * 2)[:len(ejes[1:])]
    #     for extra_eje, pos, minmax in zip(ejes[1:], positions, minmax_ejes[1:]):
    #         p.extra_y_ranges[extra_eje] = Range1d(*minmax)
    #         axkw = dict(y_range_name=extra_eje,
    #                     axis_label='{} ({})'.format(EJES[extra_eje]['label'], EJES[extra_eje]['unit']),
    #                     axis_label_text_font_size="10pt", axis_line_color=EJES[extra_eje]['color'],
    #                     major_label_text_color=EJES[extra_eje]['color'], axis_label_text_color=color,
    #                     major_tick_line_color=EJES[extra_eje]['color'], minor_tick_line_color=color,
    #                     major_tick_in=5, major_tick_out=0, minor_tick_in=3, minor_tick_out=0)
    #         p.add_layout(LinearAxis(**axkw), pos)

    # Make data source w/ time hover
    multi_day = (extremos_dia[1] - extremos_dia[0]).days > 1
    data = ColumnDataSource(_append_time_hover(data_plot.round(2), True, multiple_days=multi_day))

    # Plot lines
    kwargs_p = dict(source=data, alpha=.9, line_width=2,
                    legend='Power', color=color_base)
    kwargs_l = dict(source=data, alpha=.9, line_width=2,
                    legend='LDR', color='#F4D83F')
    # if axis != prim_axis:
    #     kwargs_p.update(dict(y_range_name=axis))
    p.line('ts', 'power', **kwargs_p)
    p.line('ts', 'ldr_plot', **kwargs_l)

    # Plot patch
    # if columns_fill is not None and c in columns_fill:
    #     # df_patch = df_total_plot[c].fillna(0)
    #     df_patch = data_plot[c].fillna(method='pad', limit=2).dropna(0).round(2)
    #     x = np.append(np.insert(df_patch.index.values, 0, df_patch.index.values[0]), df_patch.index.values[-1])
    #     y = np.append(np.insert(df_patch.values, 0, 0), 0)
    #     kwargs_patch = dict(color="#{}".format(color), line_alpha=0, fill_alpha=0.15)
    #     if axis != prim_axis:
    #         kwargs_patch.update(dict(y_range_name=axis))
    #     p.patch(x, y, **kwargs_patch)
    # if w_boxes:
    #     # TODO especificar axis_boxes desde fuera:
    #     axis_boxes = 'current' if 'current' in ejes else prim_axis
    #     d_box = _get_axis_boxes_conf(axis_boxes)
    #     if d_box['axis'] in ejes:
    #         if d_box['axis'] == prim_axis:
    #             d_box.pop('axis')
    #         _append_boxes(p, **d_box)

    # Legend formatting
    p.legend.location = "top_left"
    p.legend.label_text_font_size = "7pt"
    p.legend.background_fill_alpha = .5
    p.legend.label_height = 12
    p.legend.glyph_height = 10
    p.legend.label_standoff = 5
    p.legend.legend_padding = 8
    p.legend.legend_spacing = 3
    return p


def html_plot_buffer_bokeh(data_plot, color_base, **fig_kwargs):
    """
    Given a pandas DataFrame (or a list of df's), returns the html components for rendering the graph.
    :return script, divs, bokeh.__version__
    """
    return _return_html_comps([_plot_buffer_bokeh(data_plot, color_base, **fig_kwargs)])
