# -*- coding: utf-8 -*-
from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
from werkzeug.routing import Rule
import jinja2
import os
from enerpi.base import CONFIG, DATA_PATH, check_resource_files


basedir = os.path.abspath(os.path.dirname(__file__))
STATIC_PATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_WEBSERVER', 'STATIC_PATH', fallback='WWW'))
LOGGING_LEVEL_SERVER = CONFIG.get('ENERPI_WEBSERVER', 'LOGGING_LEVEL_WEB', fallback='DEBUG')
SERVER_FILE_LOGGING = os.path.join(STATIC_PATH,
                                   CONFIG.get('ENERPI_WEBSERVER', 'FILE_LOGGING_WEB', fallback='enerpiweb.log'))
PREFIX_WEB = CONFIG.get('ENERPI_WEBSERVER', 'PREFIX_WEB', fallback='/enerpi')
BASECOLOR = '#{}'.format(CONFIG.get('ENERPI_WEBSERVER', 'BASECOLOR_HEX', fallback='0CBB43'))
check_resource_files(STATIC_PATH, os.path.join(basedir, 'static'))

# WITH_WEB = CONFIG.getboolean('ENERPI_WEBSERVER', 'WITH_WEBSERVER', fallback=True)
WITH_ML_SUBSYSTEM = CONFIG.getboolean('ENERPI_WEBSERVER', 'WITH_ML', fallback=False)

# FLASK APP
app = Flask(__name__, static_path=PREFIX_WEB + '/static', static_folder=STATIC_PATH)
app.url_rule_class = lambda path, **options: Rule(PREFIX_WEB + path, **options)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
app.jinja_env.cache = {}
app.config['PREFIX_WEB'] = PREFIX_WEB
app.config['BASECOLOR'] = BASECOLOR
app.config['WITH_ML_SUBSYSTEM'] = WITH_ML_SUBSYSTEM
app.jinja_loader = jinja2.FileSystemLoader(os.path.join(basedir, 'templates'))

# Views
# noinspection PyUnresolvedReferences,PyPep8
from enerpiweb import views, views_filehandler, utils, rt_stream

# wsgi
app.wsgi_app = ProxyFix(app.wsgi_app)
