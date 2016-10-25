# -*- coding: utf-8 -*-
# bokeh serve --allow-websocket-origin=localhost:7777 /Users/uge/Dropbox/PYTHON/PYPROJECTS/enerpi/enerpiweb/bokeh_manual_confirmation.py
import locale
import numpy as np
import os
import pandas as pd
import pytz
import random
# import sklearn.cluster as cluster
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
# import lightning
# import hdbscan

import bokeh
from bokeh.io import curdoc
from bokeh.layouts import row, column, widgetbox, Row, Column, WidgetBox, GridSpec, layout, ToolbarBox
from bokeh.models import Icon, ImageURL, ColumnDataSource, HBox, CustomJS, VBox, WidgetBox, Spacer, GlyphRenderer
from bokeh.models.widgets import Div, Slider, TextInput, Select, DatePicker, CheckboxGroup, Button, CellEditor, ButtonGroup
from bokeh.plotting import Figure, figure
from bokeh.models import HoverTool, BoxSelectTool

from enerpi.base import timeit
from enerpiprocess.classevent import TSClassification
# from prettyprinting import *

locale.setlocale(locale.LC_ALL, locale.getlocale())
TOOLS_INTERV = "resize,crosshair,pan,xwheel_zoom,box_zoom,reset,box_select,lasso_select"
TOOLS_CONTEXT = "resize,crosshair,pan,xwheel_zoom,box_zoom,reset"
TZ = pytz.timezone('Europe/Madrid')

ROUND_W = 500
ROUND_KWH = .5
COLS_DATA = ['power', 'ldr', 'ref']
COLOR_POWER = '#0CBB43'
COLORS_DATA = [COLOR_POWER, '#F4D83F', '#8C27D3']
COLORS_LABELED = ['red', 'blue']
UNITS_DATA = ['W', '%', '']
LABELS_DATA = ['Power', 'LDR', 'Samples']
FMT_TOOLTIP_DATA = ['{0}', '{0.0}', '{0}']

COLS_DATA_KWH = ['kWh', 'p_max', 'p_min', 't_ref']
COLORS_DATA_KWH = ['#8C27D3', '#972625', '#f4af38', '#8C27D3']
UNITS_DATA_KWH = ['kWh', 'W', 'W', '']
LABELS_DATA_KWH = ['Consumption', 'Max Power', 'Min Power', 'Sampled']
FMT_TOOLTIP_DATA_KWH = ['{0.000}', '{0}', '{0}', '{0.000}']

locale.setlocale(locale.LC_ALL, locale.getlocale())
TOOLS = "pan,xwheel_zoom,box_zoom,reset,save,crosshair"

HTML_TROW = """<tr><td style="font-size: 15px; font-weight: bold;">
<span class="bk-tooltip-color-block" style="background-color: {2}"></span>{0}</td>
<td style="font-size: 16px; font-weight: bold; color: {2};">{1}</td></tr>"""
TOOLTIP_ROWS = """<tr><td style="font-size: 15px; font-weight: bold;">Time:</td>
<td style="font-size: 15px; font-weight: bold;">@time</td></tr>"""

LABELS_TYPES = ['Iluminación',
                'Climatización',
                'Nevera/Congelador',
                'Cocina',
                'Horno',
                'Lavadora',
                'Secadora',
                'Gran electrodoméstico',
                'TV / Audio',
                'Electrónica',
                'Pequeño electrodoméstico',
                'OTROS']
idx_intervalo = 36


def load_interval_detection_data(file='debug_step_detection.h5'):
    """
    # LOAD RESULTS DATA
    :return: df_subset, df_interv, df_interv_group
    """
    p = os.path.dirname(__file__)
    p_debug = os.path.join(p, '..', 'enerpiprocess', file)
    df_subset = pd.read_hdf(p_debug, 'data')
    df_interv = pd.read_hdf(p_debug, 'interv')
    df_interv_group = pd.read_hdf(p_debug, 'interv_group')
    return df_subset, df_interv, df_interv_group


# def _get_used_labels(clasif):
#
#     mask = '''<a class="bk-bs-btn bk-bs-btn-primary classif-used-tag" role="button"
#     title="{0}"
#     style="padding: 5px; border-radius: 20px;"
#     label_name="{0}" label_type="{1}" label_cont="{2}"
#     onclick="set_form_props(this)">
#     <span><img class="img-fluid" style="max-width: 20px; height: 20px; width: 20px;"
#     src="/enerpi/static/labels_icons/{3}.svg"></span> {0}</a>'''
#
#     # attrs: = l_name, l_type, l_cont, l_icon
#     html_existent_labels = ', '.join([mask.format(*attrs) for attrs in zip(*clasif.label_names(with_all_props=True))])
#     return html_existent_labels


def _get_attrs_evento(labels_evento):
    if len(labels_evento) > 0:
        print(labels_evento)
        label_type, name, is_cont = labels_evento[0].label_type, labels_evento[0].name, labels_evento[0].continuous
    else:
        label_type, name, is_cont = label_types[0], '', False

    # display: flex
    mask_label = '''<a class="bk-bs-btn bk-bs-btn-primary classif-assigned-tag" role="button"
    title="{0} [{1}]" style="padding: 5px; border-radius: 20px;" label_name="{0}" label_type="{1}"
    onclick="format_buttons_existent_labels()">
    <img style="max-width: 20px; height: 20px; width: 20px;" src="/enerpi/static/labels_icons/{2}.svg"> {0}</a>'''
    html_existent_labels = ', '.join([mask_label.format(l.name, l.label_type, l.icon) for l in labels_evento])
    return label_type, name, is_cont, html_existent_labels


def _get_patch(x, y):
    # x = np.append(np.insert(x, 0, x[0]), x[-1])
    # y = np.append(np.insert(y, 0, 0.), 0.)
    x = np.append(np.append(np.insert(x, 0, x[0]), x[-1]), x[0])
    y = np.append(np.append(np.insert(y, 0, 0.), 0.), 0.)
    return x, y


def _str_title_day(ts):
    return "Potencia eléctrica en el {:%A %-d de %B de %Y}".format(ts)


def _get_interv(df_wiener, df_interv, idx, substract_level='level_group'):
    t0, tf = df_interv.loc[idx, 'ts_ini'], df_interv.loc[idx, 'ts_fin']
    intervalo = df_wiener.loc[t0:tf, 'wiener']
    # if substract_level is not None:
    #     intervalo -= df_interv.loc[idx, substract_level]
    # print('** Get interval from {} to {}, {} samples'.format(t0, tf, intervalo.shape[0]))
    return intervalo


def _get_context(df_wiener, intervalo, actual_context=None):
    t0, tf = intervalo.index[0], intervalo.index[-1]
    if (actual_context is None) or (t0 not in actual_context.index) or (tf not in actual_context.index):
        actual_context = df_wiener.loc[t0.date():tf.date() + pd.Timedelta('1D'), 'wiener']
        # print('** Get context from {} to {}, {} samples, head:\n{}\ntail:\n{}'
        #       .format(t0, tf, actual_context.shape[0], actual_context.head(), actual_context.tail()))
        return True, actual_context
    return False, actual_context


def get_data_source_intervalo(intervalo):
    x_width = intervalo.index[-1] - intervalo.index[0]
    y_height = intervalo.values.max() - intervalo.values.min()
    x_rect = [intervalo.index[0] + x_width / 2]
    y_rect = [intervalo.values.min() + int(y_height / 2)]

    x_line = intervalo.index.tolist()
    y_line = intervalo.values.tolist()
    x_patch, y_patch = _get_patch(x_line, y_line)
    time_hover = [x.strftime("%-d/%b %H:%M:%S") for x in x_line]
    data_intervalo = dict(x_line=x_line, y_line=y_line, time=time_hover)
    data_intervalo_rect = dict(x_rect=x_rect, y_rect=y_rect,
                          rect_width=[x_width], rect_height=[y_height]) #, time=time_hover)
    data_intervalo_patch = dict(x_patch=x_patch, y_patch=y_patch) #, time=time_hover)
    return data_intervalo, data_intervalo_rect, data_intervalo_patch


@timeit('get_data_source_context', verbose=True)
def get_data_source_context(data_context):
    x_line = data_context.index.tolist()
    y_line = data_context.values.tolist()
    return dict(x_line=x_line, y_line=y_line, time=[x.strftime("%-d/%b %H:%M:%S") for x in x_line])


def get_data_source_used_labels(clasificacion):
    # names, types, conts, icons = clasificacion.label_names(with_all_props=True)
    names, types, conts, icons, used = clasificacion.label_names(only_used=False, with_all_props=True)
    print('en get_data_source_used_labels!: ', names, types)
    return dict(label_names=names, label_types=types, label_cont=conts, label_icons=icons, label_used=used)


def plot_intervalo_y_contexto():
    # Crea plots:
    color_box = COLORS_LABELED[1] if len(labels_evento) > 0 else COLORS_LABELED[0]
    p_interv = figure(plot_width=400, plot_height=400,
                      title="Visor de eventos", tools=TOOLS_INTERV, x_axis_type="datetime",
                      toolbar_location="right", hidpi=True)  # , toolbar_sticky=False) #, )
    p_interv.rect('x_rect', 'y_rect', 'rect_width', 'rect_height', source=source_r,
                  fill_alpha=.05, line_width=.5, color=color_box)
    p_interv.patch('x_patch', 'y_patch', source=source_p, color=COLOR_POWER, line_width=0, alpha=.15,
                   name='filled_area')
    p_line_interv = p_interv.line('x_line', 'y_line', source=source_l, color=COLOR_POWER)

    p_context = figure(plot_width=700, plot_height=400,  # , responsive=True,
                       title=_str_title_day(intervalo_selec.index[0]), tools=TOOLS_CONTEXT, hidpi=True,
                       x_axis_type="datetime", toolbar_location="right", active_drag='pan', active_scroll='xwheel_zoom')
    p_context.rect('x_rect', 'y_rect', 'rect_width', 'rect_height', source=source_r,
                   fill_alpha=.7, line_width=1, color=color_box)
    p_line_context = p_context.line('x_line', 'y_line', source=source_context,
                                    line_width=.75, line_alpha=.9, color=COLOR_POWER)

    tooltip_rows = TOOLTIP_ROWS
    tooltip_rows += HTML_TROW.format('{}:'.format('Potencia'), '@{}{} {}'.format('y_line', '{0}', 'W'), COLOR_POWER)
    # p_interv.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows)))
    p_interv.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows),
                                 renderers=[p_line_interv]))
    # p_context.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows)))
    p_context.add_tools(HoverTool(tooltips='<div><table>{}</table></div>'.format(tooltip_rows),
                                  renderers=[p_line_context]))
    return p_interv, p_context, p_line_interv


# LOAD DATA
df, df_interv, df_interv_group = load_interval_detection_data(file='debug_step_detection.h5')
df_interv_group = df_interv_group[df_interv_group.big_event]

clasificacion = TSClassification(df_events=df_interv_group, create_init_labels=True, force_regen=False)
label_types = clasificacion.label_types

# Inicializa intervalo seleccionado:
intervalo_selec = _get_interv(df, df_interv_group, idx_intervalo, substract_level='level_group')
evento = clasificacion.get_event_by_id(idx_intervalo)
labels_evento = clasificacion.get_labels_from_event(evento)
print('evento:', evento)

_new_context, context_selec = _get_context(df, intervalo_selec)
assert(_new_context)

data_intervalo, data_intervalo_rect, data_intervalo_patch = get_data_source_intervalo(intervalo_selec)
source_context = ColumnDataSource(data=get_data_source_context(context_selec))
source_l = ColumnDataSource(data=data_intervalo)
source_r = ColumnDataSource(data=data_intervalo_rect)
source_p = ColumnDataSource(data=data_intervalo_patch)

source_used_labels = ColumnDataSource(data=get_data_source_used_labels(clasificacion))

# Crea plots:
p_interv, p_context, p_line_interv = plot_intervalo_y_contexto()

# Set up widgets
input_intervalo = TextInput(title="ID de evento", value=str(idx_intervalo), width=100)
button_ant = Button(label="◀ Anterior", button_type="primary", width=75)
button_sig = Button(label="Siguiente ▶", button_type="primary", width=75)
button_add_label = Button(label="✚ Nueva etiqueta", button_type="success", width=100)
button_remove_label = Button(label="✖ Elimina etiqueta", button_type="danger", width=100)
if len(labels_evento) == 0:
    button_remove_label.disabled = True
# button_update_model = Button(label="☺", button_type="danger", name='button_update_model', width=50)
# button_submit = Button(label="✔ ︎︎Confirma", button_type="warning", width=30)

# Datos del evento seleccionado:
label_type, name, is_cont, html_existent_labels = _get_attrs_evento(labels_evento)
div_labels_event = Div(text=html_existent_labels)
select_label_type = Select(title="Tipo de aparato:", value=label_type, options=label_types, name='selec_label_type')
input_label = TextInput(title="Nombre de aparato eléctrico:", value=name, name='selec_label_name')
check_aparato_uso_permanente = CheckboxGroup(labels=["De uso permanente (24h activo)"],
                                             active=[1 if not is_cont else 0], name='selec_label_continuous')
# print([1 if not is_cont else 0])
# print(check_aparato_uso_permanente.active)

# Clasificación de eventos en time-series.
#         	* Archivo en disco: /Users/uge/Dropbox/PYTHON/PYPROJECTS/enerpi/enerpiprocess/clasificacion_eventos.pickle
#         	* Eventos: 137, de los cuales, 137 son "big_event"'s y 0 son "sub-eventos".
#         	* Etiquetas: 12 (1 usadas)
#         	* Eventos etiquetados: 1

# div_clasif = Div(text='<h5>{}</h5>'
#                  .format('<br>'.join(str(clasificacion).replace('\n', '').split('* ')[2:])), width=1000)
# print('div_clasif: ', div_clasif)

# callback = CustomJS(args=dict(source=source), code="""
#         var data = source.data;
#         var f = cb_obj.value
#         x = data['x']
#         y = data['y']
#         for (i = 0; i < x.length; i++) {
#             y[i] = Math.pow(x[i], f)
#         }
#         source.trigger('change');
#     """)

# Set up callbacks
def update_intervalo(attrname, old, new):
    global idx_intervalo, context_selec, clasificacion, evento, labels_evento
    # print(attrname, type(attrname), old, type(old), new, type(new))
    try:
        id_interv = int(input_intervalo.value)
        if id_interv != idx_intervalo:
            intervalo_selec = _get_interv(df, df_interv_group, id_interv, substract_level='level_group')
            new_data, new_data_rect, new_data_patch = get_data_source_intervalo(intervalo_selec)
            idx_intervalo = id_interv

            evento = clasificacion.get_event_by_id(idx_intervalo)
            labels_evento = clasificacion.get_labels_from_event(evento)
            label_type, name, is_cont, html_existent_labels = _get_attrs_evento(labels_evento)
            select_label_type.value = label_type
            check_aparato_uso_permanente.active = [1 if not is_cont else 0]
            input_label.value = name
            div_labels_event.text = html_existent_labels
            # div_existent_labels.text = _get_used_labels(clasificacion)
            if len(labels_evento) > 0:
                button_remove_label.disabled = False
            else:
                button_remove_label.disabled = True
            boxes = [g for g in filter(lambda x: (type(x) is GlyphRenderer) and
                                                 ('rect_width' in x.data_source.data.keys()),
                                       list(p_interv.renderers) + list(p_context.renderers))]
            color = COLORS_LABELED[1] if len(labels_evento) > 0 else COLORS_LABELED[0]
            for box in boxes:
                box.glyph.fill_color = box.glyph.line_color = color

            source_l.data = new_data
            source_r.data = new_data_rect
            source_p.data = new_data_patch
            str_title = ("Intervalo {}, de {:%d/%b'%y %H:%M:%S} a {:%H:%M:%S}".format(id_interv, new_data['x_line'][0],
                                                                                      new_data['x_line'][-1]))
            p_interv.title.text = str_title

            hay_new_context, new_context_selec = _get_context(df, intervalo_selec, context_selec)
            if hay_new_context:
                source_context.data = get_data_source_context(new_context_selec)
                p_context.title.text = _str_title_day(intervalo_selec.index[0])
    except ValueError as e:
        print('error!! en input: {}'.format(e))
        pass


def update_used_labels(clasificacion):
    global column_buttons_used_labels, source_used_labels, idx_intervalo
    print('** EN UPDATE_USED_LABELS:')
    source_used_labels.data = get_data_source_used_labels(clasificacion)
    column_buttons_used_labels.children = gen_buttons_used_labels()
    idx_intervalo = -1
    update_intervalo('', '', '')
    # column_buttons_used_labels.sizing_mode = 'scale_width'
    # print(column_buttons_used_labels.children)
    # column_buttons_used_labels.update()
    # print(curdoc().roots[0].children[0])
    print('UPDATE_USED_LABELS ok??')


def append_label_to_intervalo(attr, old, new):
    global clasificacion, evento
    select_label_type.update()
    check_aparato_uso_permanente.update()
    input_label.update()

    print('en append_label_to_intervalo: ', input_label.value, select_label_type.value)
    print(check_aparato_uso_permanente)
    print(check_aparato_uso_permanente.active)
    clasificacion.append_label_to_event(evento, input_label.value, select_label_type.value, False, save_change=True)
    print(clasificacion)
    update_used_labels(clasificacion)
    go_to_intervalo_sig(attr, old, new)


def remove_label_from_intervalo(attr, old, new):
    global idx_intervalo, clasificacion, evento, labels_evento
    if len(labels_evento) > 0:
        label_remove = labels_evento[-1]
        print('label_remove: ', label_remove)
        clasificacion.remove_label_from_event(label_remove, evento, save_change=True)
        idx_intervalo = -1
        print(clasificacion)
        update_used_labels(clasificacion)
        update_intervalo(attr, old, new)


def go_to_intervalo_ant(attrname, old, new):
    global df_interv_group
    try:
        input_intervalo.value = str(df_interv_group.index[df_interv_group.index.get_loc(int(input_intervalo.value)) - 1])
        update_intervalo(attrname, old, new)
    except ValueError as e:
        print('error!! en input: {}'.format(e))
        pass


def go_to_intervalo_sig(attrname, old, new):
    global df_interv_group
    try:
        input_intervalo.value = str(df_interv_group.index[df_interv_group.index.get_loc(int(input_intervalo.value)) + 1])
        update_intervalo(attrname, old, new)
    except IndexError as e:
        # Reiniciamos contador:
        input_intervalo.value = str(df_interv_group.index[0])
    except ValueError as e:
        print('error!! en input: {}'.format(e))
        pass


def update_label_default_name(attrname, old, new):
    print('en update_label_default_name (no hace nada):', attrname, old, new, select_label_type.value)
    new_name = clasificacion.get_new_label_name_by_type(select_label_type.value)
    input_label.value = new_name


def update_label_name(attrname, old, new):
    print('en update_label_name (no hace nada):', attrname, old, new, input_label.value)
    # new_name = clasificacion.get_new_label_name_by_type(select_label_type.value)
    # print(input_label)
    # print(input_label.value)
    # input_label.value = input_label.value


def update_label_check(attrname, old, new):
    print('en update_label_check:', attrname, old, new)
    # new_name = clasificacion.get_new_label_name_by_type(select_label_type.value)
    # input_label.value = new_name
    # input_label.value = 'prueba'


def set_labels_params(idx_used_label):
    try:
        select_label_type.value = source_used_labels.data['label_types'][idx_used_label]
        check_aparato_uso_permanente.active = [1 if not source_used_labels.data['label_cont'][idx_used_label] else 0]
        input_label.value = source_used_labels.data['label_names'][idx_used_label]
    except IndexError as e:
        print('IndexError: {}'.format(e))


def create_trigger(func_name, idx_label):
    def func_din(attr, old, new):
        set_labels_params(idx_label)
    func_din.__name__ = func_name
    return func_din


def gen_buttons_used_labels():
    buttons_used_labels = []
    for i, (label_type, name, used, icon) in enumerate(zip(source_used_labels.data['label_types'],
                                                           source_used_labels.data['label_names'],
                                                           source_used_labels.data['label_used'],
                                                           source_used_labels.data['label_icons'])):
        label_b = name if used else label_type
        b = Button(label=label_b, button_type="default", sizing_mode='scale_width')
        if used:
            b._id = 'label_used_{}_{}_icon_{}'.format(i, label_type, icon)
        else:
            b._id = 'label_unused_{}_{}_icon_{}'.format(i, label_type, icon)
        b.on_change('clicks', create_trigger('trigger_label_{}'.format(i), i))
        # b.apply_theme(dict(name='b_label_{}_{}'.format(name, icon)))
        buttons_used_labels.append(b)
    return buttons_used_labels


# row_buttons_used_labels = Row(children=gen_buttons_used_labels()) #, sizing_mode='scale_width')
# row_buttons_used_labels = Column(children=gen_buttons_used_labels(), sizing_mode='scale_width')  # , sizing_mode='stretch_both')
# column_buttons_used_labels = Column(children=gen_buttons_used_labels(), width=200)  # , sizing_mode='stretch_both')
button_ant.on_change('clicks', go_to_intervalo_ant)
button_sig.on_change('clicks', go_to_intervalo_sig)
input_intervalo.on_change('value', update_intervalo)
select_label_type.on_change('value', update_label_default_name)
# input_label.on_change('value', update_label_name)
check_aparato_uso_permanente.on_change('active', update_label_check)
button_add_label.on_change('clicks', append_label_to_intervalo)
button_remove_label.on_change('clicks', remove_label_from_intervalo)
# button_update_model.on_change('clicks', lambda a, b, c: set_labels_params(a, b, 1))

# Set up layouts and add to document
column_buttons_used_labels = Column(children=gen_buttons_used_labels(), width=250)  # , sizing_mode='stretch_both')

label_widgets = widgetbox([select_label_type, check_aparato_uso_permanente, input_label]) #, width=300)
botonera = HBox(children=[button_ant, Spacer(width=50), button_sig], width=300)
botonera_submit = HBox(children=[button_add_label,  Spacer(width=50), button_remove_label], width=300)
column_form = Column(children=[botonera, input_intervalo, div_labels_event, label_widgets, botonera_submit], width=300)
# curdoc().add_root(div_clasif)
# curdoc().add_root(column_buttons_used_labels)
root_layout = Row(children=[column_buttons_used_labels,
                            Column(children=[Row(children=[column_form, p_interv], width=700),
                                             Row(children=[p_context], width=700)])], width=1000)
# root_layout = column(children=[row(children=[column(children=[row(children=[column_form, p_interv], sizing_mode='scale_width'),
#                                                               row(children=[p_context], sizing_mode='scale_width')])], responsive=True),
#                                column_buttons_used_labels],
#                      responsive=True)
    # column(children=[row(children=[column_buttons_used_labels, column_form, p_interv], responsive=True),
    #                            row(children=[p_context], responsive=True)], responsive=True)

# Row(children=[,Column(children=[p_interv, p_context], responsive=True)], sizing_mode='stretch_both')
curdoc().add_root(root_layout)
curdoc().title = "ENERPI - Manual Labeling"
