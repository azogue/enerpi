# -*- coding: utf-8 -*-
from flask import request, redirect, url_for, render_template, jsonify
import json
import os

from enerpi.base import log
from enerpi.api import enerpi_data_catalog
from enerpiplot.plotbokeh import get_bokeh_version
from enerpiweb import app, WITH_ML_SUBSYSTEM
from enerpiweb.rt_stream import stream_is_alive


COLOR_BASE = app.config['BASECOLOR']
BOKEH_VERSION = get_bokeh_version()

if WITH_ML_SUBSYSTEM:
    # from enerpiweb.views_labeling import *
    # TODO Integración con las vistas de enerpiprocess (separadas en otro project por dependencias, por ahora)
    pass
# else:
#     @app.route('/learning')
#     def index_learning():
#         return redirect(url_for('control'))


#############################
# INDEX & ROUTES
#############################
# TODO RESTART APP (RELOAD CONFIG!):
@app.route('/reloadserver/')
def reloadserver():
    log('Se procede a intentar reiniciar el servidor', 'debug', True)
    os.system('sudo service uwsgi-emperor restart')
    log('Ha funcionado??', 'debug', True)
    return redirect(url_for('index'))


# TODO Fix layout of control buttons
@app.route('/control')
def control():
    """
    Admin Control Panel with links to LOG viewing/downloading, hdf_stores download, ENERPI config editor, etc.

    """
    is_sender_active, last = stream_is_alive()
    alerta = request.args.get('alerta', '')
    if alerta:
        alerta = json.loads(alerta)
    cat = enerpi_data_catalog(check_integrity=False)
    df = cat.tree
    if df is not None:
        df = df[df.is_cat & df.is_raw].sort_values(by='ts_ini', ascending=False)
        paths_rel = [(os.path.basename(p), t0.strftime('%d/%m/%y'), tf.strftime('%d/%m/%y'), n)
                     for p, t0, tf, n in zip(df['st'], df['ts_ini'], df['ts_fin'], df['n_rows'])]
    else:
        paths_rel = []
    return render_template('control_panel.html',
                           d_catalog={'path_raw_store': os.path.join(cat.base_path, cat.raw_store),
                                      'path_catalog': os.path.join(cat.base_path, cat.catalog_file),
                                      'ts_init': cat.min_ts, 'ts_catalog': cat.index_ts},
                           d_last_msg=last, is_sender_active=is_sender_active, list_stores=paths_rel, alerta=alerta)


@app.route('/api/monitor')
def only_tiles():
    """
    Url route for showing only the real-time monitoring tiles
    (for emmbed in another place, e.g.)

    """
    ldr = request.args.get('ldr', 'true') == 'true'
    consumption = request.args.get('consumption', 'false') != 'false'
    return render_template('only_tiles.html', include_consumption=consumption, include_ldr=ldr)


@app.route('/api/help', methods=['GET'])
def api_help():
    # TODO hacer tabla en html con url rules:
    endpoints = [rule.rule for rule in app.url_map.iter_rules()
                 if rule.endpoint != 'static']
    return jsonify(dict(api_endpoints=endpoints))


@app.route('/api/bokehplot')
def bokehplot():
    """
    Base webpage for query & show bokeh plots of ENERPI data

    """
    return render_template('bokeh_plot.html', url_stream_bokeh=url_for('bokeh_buffer'), b_version=BOKEH_VERSION)


@app.route('/')
def base_index():
    """
    Redirects to 'index', with real-time monitoring of ENERPI values

    """
    return redirect(url_for('index'))


# @app.route('/allroutes')
# def allroutes():
#     routes = []
#     for rule in app.url_map.iter_rules():
#         # To show how to do it, we are going to filter what type of
#         # request are we going to show, for example here we are only
#         # going to use GET requests
#         if "GET" in rule.methods:
#             url = rule.rule
#             routes.append(url)
#     return render_template('sysadmin/routes.html', routes=routes)
