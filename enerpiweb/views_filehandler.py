# -*- coding: utf-8 -*-
from flask import request, redirect, url_for, render_template, send_file, abort
import json
import os
from enerpi.base import DATA_PATH, FILE_LOGGING, get_lines_file, log, UWSGI_CONFIG_FILE
from enerpi.editconf import web_config_edition_data, check_uploaded_config_file, ENERPI_CONFIG_FILES
from enerpi.api import enerpi_data_catalog
from enerpiweb import app, SERVER_FILE_LOGGING, STATIC_PATH
from enerpiweb.forms import FileForm


def _get_filepath_from_file_id(file_id):
    # Interesting files / logs to show/edit/download/upload:
    if file_id in ENERPI_CONFIG_FILES:
        return True, False, os.path.join(DATA_PATH, ENERPI_CONFIG_FILES[file_id]['filename'])
    elif 'flask' == file_id:
        is_logfile, filename = True, SERVER_FILE_LOGGING
    elif 'rsc' == file_id:
        is_logfile, filename = True, os.path.join(STATIC_PATH, 'enerpiweb_rscgen.log')
    elif 'nginx_err' == file_id:
        is_logfile, filename = True, '/var/log/nginx/error.log'
    elif 'nginx' == file_id:
        is_logfile, filename = True, '/var/log/nginx/access.log'
    elif 'enerpi' == file_id:
        is_logfile, filename = True, FILE_LOGGING
    elif 'uwsgi' == file_id:
        is_logfile, filename = True, '/var/log/uwsgi/{}.log'.format(os.path.splitext(UWSGI_CONFIG_FILE)[0])
    elif 'daemon_out' == file_id:
        is_logfile, filename = True, '/tmp/enerpi_out.txt'
    elif 'daemon_err' == file_id:
        is_logfile, filename = True, '/tmp/enerpi_err.txt'
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
    print('En download_hdfstore_file with path_file: "{}", relpath: "{}", cat:\n{}'
          .format(path_file, relpath_store, cat))
    if (path_file is not None) and os.path.exists(path_file):
        print('Dentro IF')
        if 'as_attachment' in request.args:
            return send_file(path_file, as_attachment=True, attachment_filename=os.path.basename(path_file))
        return send_file(path_file, as_attachment=False)
    return abort(404)


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


@app.route('/api/filedownload/debug/', methods=['POST'])
def download_file_debug():
    """
    * File Download *
    Post request with relative file path under DATA_PATH (debugging purposes)

    """
    form = FileForm()
    if form.validate_on_submit():
        # relpath = relpath.split(':')
        relpath = form.pathfile
        # abspath = os.path.join(DATA_PATH, *relpath)
        abspath = os.path.join(DATA_PATH, relpath)
        if os.path.exists(abspath):
            return send_file(abspath, as_attachment=True, attachment_filename=os.path.basename(abspath))
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
    Configuration editor, for INI file, JSON sensors file & encripting key

    :param file: file_id to edit
    """
    ok, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok and (file in ENERPI_CONFIG_FILES):
        alerta = request.args.get('alerta', '')
        if alerta:
            alerta = json.loads(alerta)
        extra_links = [(ENERPI_CONFIG_FILES[f_id]['text_button'], url_for('editfile', file=f_id))
                       for f_id in sorted(ENERPI_CONFIG_FILES.keys(), key=lambda x: ENERPI_CONFIG_FILES[x]['order'])]
        show_switch_comments = ENERPI_CONFIG_FILES[file]['show_switch_comments']
        if not show_switch_comments:
            without_comments = True
        else:
            without_comments = request.args.get('without_comments', 'False').lower() == 'true'
        d_edit = request.form if request.method == 'POST' else None
        alerta_edit, lines_config, config_data = web_config_edition_data(file, filename, d_edition_form=d_edit)
        if not alerta:
            alerta = alerta_edit
        elif alerta_edit is not None:
            alerta.update(alerta_edit)
        return render_template('edit_text_file.html', titulo='CONFIG EDITOR', file_id=file,
                               show_switch_comments=show_switch_comments, with_comments=not without_comments,
                               abspath=filename, dict_config_content=config_data, file_lines=lines_config,
                               filename=os.path.basename(filename), alerta=alerta, extra_links=extra_links)
    log('Error in editfile with file={}'.format(file), 'error', False)
    return abort(404)


@app.route('/api/uploadfile/<file>', methods=['POST'])
def uploadfile(file):
    """
    POST method for VIP files upload & replacement

    :param file: uploaded file_id
    """
    ok_fileid, is_logfile, filename = _get_filepath_from_file_id(file)
    if ok_fileid:
        if is_logfile:
            log('uploading logfile {} [{}]!!!'.format(file, filename), 'error', True)
            return redirect(url_for('index'), code=404)
        f = request.files['file']
        alert = check_uploaded_config_file(file, f, dest_filepath=filename)
        if alert:
            alert = json.dumps(alert)
        return redirect(url_for('editfile', file=file, alerta=alert))
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
