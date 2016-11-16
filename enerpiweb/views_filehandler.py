# -*- coding: utf-8 -*-
from collections import OrderedDict
from flask import request, redirect, url_for, render_template, send_file, abort
import json
import os
from enerpi.base import (DATA_PATH, CONFIG_FILENAME, FILE_LOGGING, CONFIG,
                         get_lines_file, log, make_ini_file, config_dict_for_web_edit, config_changes)
from enerpi.api import enerpi_data_catalog, get_encryption_key, get_codec
from enerpiweb import app, SERVER_FILE_LOGGING, STATIC_PATH


ENERPI_FILE_LOGGING = FILE_LOGGING
RSC_GEN_FILE_LOGGING = os.path.join(STATIC_PATH, 'enerpiweb_rscgen.log')


# Interesting files / logs to show:
def _get_filepath_from_file_id(file_id):
    if 'flask' == file_id:
        is_logfile, filename = True, SERVER_FILE_LOGGING
    elif 'rsc' == file_id:
        is_logfile, filename = True, RSC_GEN_FILE_LOGGING
    elif 'nginx_err' == file_id:
        is_logfile, filename = True, '/var/log/nginx/error.log'
    elif 'nginx' == file_id:
        is_logfile, filename = True, '/var/log/nginx/access.log'
    elif 'enerpi' == file_id:
        is_logfile, filename = True, ENERPI_FILE_LOGGING
    elif 'uwsgi' == file_id:
        is_logfile, filename = True, '/var/log/uwsgi/enerpiweb.log'
    elif 'config' == file_id:
        is_logfile, filename = False, os.path.join(DATA_PATH, CONFIG_FILENAME)
    elif 'encryption_key' == file_id:
        is_logfile, filename = False, os.path.join(DATA_PATH,
                                                   CONFIG.get('BROADCAST', 'KEY_FILE', fallback='.secret_key'))
    else:  # Fichero derivado del catálogo
        cat = enerpi_data_catalog(check_integrity=False)
        if 'raw_store' == file_id:
            is_logfile, filename = False, os.path.join(cat.base_path, cat.raw_store)
        elif 'catalog' == file_id:
            is_logfile, filename = False, os.path.join(cat.base_path, cat.catalog_file)
        else:
            log('FILE_ID No reconocido: {}'.format(file_id), 'error', False)
            is_logfile, filename = False, SERVER_FILE_LOGGING
            return False, is_logfile, filename
    return True, is_logfile, filename


#############################
# ROUTES for file handling
#############################
@app.route('/api/hdfstores/<relpath_store>', methods=['GET'])
def download_hdfstore_file(relpath_store=None):
    """
    Devuelve el fichero HDFStore del catálogo de ENERPI pasado como ruta relativa (nombre de fichero .h5).
    * File Download *
    :param relpath_store:
    """
    cat = enerpi_data_catalog(check_integrity=False)
    path_file = cat.get_path_hdf_store_binaries(relpath_store)
    if 'as_attachment' in request.args:
        return send_file(path_file, as_attachment=True, attachment_filename=os.path.basename(path_file))
    return send_file(path_file, as_attachment=False)


@app.route('/api/filedownload/<file_id>', methods=['GET'])
def download_file(file_id):
    """
    * File Download *
    :param file_id:
    """
    ok, _, filename = _get_filepath_from_file_id(file_id)
    if ok:
        if os.path.exists(filename):
            if 'as_attachment' in request.args:
                return send_file(filename, as_attachment=True, attachment_filename=os.path.basename(filename))
            return send_file(filename, as_attachment=False)
        else:
            msg = json.dumps({'alert_type': 'danger',
                              'texto_alerta': 'El archivo "{}" ({}) no existe!'.format(filename, file_id)})
            return redirect(url_for('control', alerta=msg))
    return abort(404)


@app.route('/api/showfile')
@app.route('/api/showfile/<file>')
def showfile(file='flask'):
    """
    Página de vista de fichero de texto, con orden ascendente / descendente y/o nº de últimas líneas ('tail' de archivo)
    :param file: file_id to show
    """

    ok, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok:
        delete = request.args.get('delete', '')
        reverse = request.args.get('reverse', False)
        tail_lines = request.args.get('tail', None)
        alerta = request.args.get('alerta', '')
        if alerta:
            alerta = json.loads(alerta)
        if not alerta and delete and is_logfile:
            with open(filename, 'w') as f:
                f.close()
            cad_delete = 'LOGFILE {} DELETED'.format(filename.upper())
            log(cad_delete, 'warn', False)
            return redirect(url_for('showfile', file=file,
                                    alerta=json.dumps({'alert_type': 'warning', 'texto_alerta': cad_delete})))
        data = get_lines_file(filename, tail=tail_lines, reverse=reverse)
        return render_template('text_file.html', titulo='LOG File:' if is_logfile else 'CONFIG File:', file_id=file,
                               subtitulo='<strong>{}</strong>'.format(filename), is_logfile=is_logfile,
                               file_content=data, filename=os.path.basename(filename), alerta=alerta)
    return abort(404)


@app.route('/api/editconfig/', methods=['GET'])
@app.route('/api/editconfig/<file>', methods=['GET', 'POST'])
def editfile(file='config'):
    """
    Config editor, for INI file & encripting key

    :param file: file_id to edit
    """
    # TODO readonly/disabled attr in protected fields of INI file

    ok, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok:
        if is_logfile:
            return redirect(url_for('index'), code=404)

        alerta = request.args.get('alerta', '')
        if alerta:
            alerta = json.loads(alerta)
        without_comments = request.args.get('without_comments', 'False')
        without_comments = without_comments.lower() == 'true'

        if file == 'encryption_key':  # Key edition!! NOT SAFE AT ALL (Debugging purposes!)
            extra_links = [('<i class="fa fa-check-square-o" aria-hidden="true"></i> Edit Configuration',
                            url_for('editfile', file='config'))]
            show_switch_comments = False
            without_comments = True
            variable_name = 'KEY_FILE'
            key = get_encryption_key(filename)
            config_data = OrderedDict(
                [('Encryption KEY', OrderedDict([(variable_name, (key.decode(), 'text', None))]))])
            lines_config = [key.decode()]
            if request.method == 'POST':
                new_key = request.form[variable_name]
                # if new_key.encode() != key:
                if new_key != key.decode():
                    try:
                        _ = get_codec(new_key.encode())
                        if len(new_key) > 10:
                            log('The new KEY ("{}")\nwill be saved in "{}"\n'.format(new_key, filename), 'warning',
                                True)
                            with open(filename, 'wb') as f:
                                f.write(new_key.encode())
                            str_cambios = ('Configuration changes in encryption key:<br>{}<br> New config SAVED!'
                                           .format('- <strong>"{}"</strong> (before={}) -> <strong>"{}"</strong>'
                                                   .format(variable_name, key.decode(), new_key)))
                            alerta = {'alert_type': 'warning', 'texto_alerta': str_cambios}
                            lines_config = [new_key]
                            config_data = OrderedDict([('Encryption KEY',
                                                        OrderedDict([(variable_name, (new_key, 'text', None))]))])
                    except AssertionError as e:
                        alerta = {'alert_type': 'danger', 'texto_alerta': '{}. New KEY was: {}'.format(e, new_key)}
        else:  # INI CONFIG FILE EDITION:
            extra_links = [('<i class="fa fa-user-secret" aria-hidden="true"></i> Edit SECRET Key',
                            url_for('editfile', file='encryption_key'))]
            show_switch_comments = True
            lines_config = get_lines_file(filename)
            config_data = config_dict_for_web_edit(lines_config)
            if request.method == 'POST':
                hay_cambio, vars_updated, config_data_updated = config_changes(request.form, config_data)
                if hay_cambio:
                    str_cambios = ('Configuration changes in:<br>{}<br> New config SAVED!'
                                   .format('<br>'.join(['- <strong>"{}"</strong> (before={}) -> <strong>"{}"</strong>'
                                                       .format(name, ant, new) for name, ant, new in vars_updated])))
                    alerta = {'alert_type': 'warning', 'texto_alerta': str_cambios}
                    lines_config = make_ini_file(config_data_updated, dest_path=filename).splitlines()
                    config_data = config_data_updated
        return render_template('edit_text_file.html', titulo='CONFIGURATION EDITOR', file_id=file,
                               show_switch_comments=show_switch_comments, with_comments=not without_comments,
                               abspath=filename, dict_config_content=config_data, file_lines=lines_config,
                               filename=os.path.basename(filename), alerta=alerta, extra_links=extra_links)
    return abort(404)


@app.route('/api/uploadfile/<file>', methods=['POST'])
def uploadfile(file):
    """
    POST method for VIP files upload & replacement

    :param file: uploaded file_id
    """
    ok, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok:
        if is_logfile:
            print('uploading logfile!!!')
            return redirect(url_for('index'), code=404)
        # filename = secure_filename(filename)
        if len(filename) > 0:
            f = request.files['file']
            if f.mimetype == 'application/octet-stream':
                log('UPLOADED FILE id={}, filename="{}" (dest_path="{}")'.format(file, f.filename, filename), 'warn')
                f.save(filename)
                msg = json.dumps({'alert_type': 'success', 'texto_alerta': ('Uploaded file "{}" as replacement in "{}"!'
                                                                            .format(f.filename, filename))})
                return redirect(url_for('editfile', file=file, alerta=msg))
    return abort(500)


# TODO Implementar generación de PDF reports
# from flask_weasyprint import HTML, render_pdf
#
#
# @app.route('/api/filetopdf/<file>')
# def filetopdf(file='flask'):
#     """
#     Página de vista de fichero de texto, con orden ascendente / descendente y/o nº de últimas líneas
#     ('tail' de archivo)
#     :param file: file_id to show
#     """
#     return render_pdf(url_for('showfile', file=file))
