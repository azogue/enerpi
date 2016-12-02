# -*- coding: utf-8 -*-
"""
ENERPIWEB - Flask web application

"""
from flask import Flask
from flask_autodoc import Autodoc
from flask_mail import Mail
from werkzeug.contrib.fixers import ProxyFix
from werkzeug.routing import Rule
import jinja2
import os
# noinspection PyUnresolvedReferences
from enerpi import __version__
from enerpi.base import CONFIG, DATA_PATH, check_resource_files, GMAIL_ACCOUNT, GMAIL_APP_PASSWORD
from enerpi.api import get_encryption_key


basedir = os.path.abspath(os.path.dirname(__file__))
STATIC_PATH = os.path.join(DATA_PATH, CONFIG.get('ENERPI_WEBSERVER', 'STATIC_PATH', fallback='WWW'))
LOGGING_LEVEL_SERVER = CONFIG.get('ENERPI_WEBSERVER', 'LOGGING_LEVEL_WEB', fallback='DEBUG')
SERVER_FILE_LOGGING = os.path.join(STATIC_PATH,
                                   CONFIG.get('ENERPI_WEBSERVER', 'FILE_LOGGING_WEB', fallback='enerpiweb.log'))
PREFIX_WEB = CONFIG.get('ENERPI_WEBSERVER', 'PREFIX_WEB', fallback='/enerpi')
BASECOLOR = '#{}'.format(CONFIG.get('ENERPI_WEBSERVER', 'BASECOLOR_HEX', fallback='0CBB43'))
check_resource_files(STATIC_PATH, os.path.join(basedir, 'static'), verbose=False)

# WITH_WEB = CONFIG.getboolean('ENERPI_WEBSERVER', 'WITH_WEBSERVER', fallback=True)
WITH_ML_SUBSYSTEM = CONFIG.getboolean('ENERPI_WEBSERVER', 'WITH_ML', fallback=False)

# FLASK APP
app = Flask(__name__, static_path=PREFIX_WEB + '/static', static_folder=STATIC_PATH)
app.url_rule_class = lambda path, **options: Rule(PREFIX_WEB + path, **options)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
app.jinja_env.cache = {}
app.jinja_loader = jinja2.FileSystemLoader(os.path.join(basedir, 'templates'))

# Manual activation of test-mode
# app.config['TESTING'] = True
# app.config['PROPAGATE_EXCEPTIONS'] = True

# Forms protection
app.config['CSRF_ENABLED'] = True
app.config['WTF_CSRF_ENABLED'] = True
app.config['SECRET_KEY'] = get_encryption_key()

app.config['STREAM_MAX_TIME'] = 1800
app.config['BASECOLOR'] = BASECOLOR
app.config['WITH_ML_SUBSYSTEM'] = WITH_ML_SUBSYSTEM

# Plug-ins
# email server
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587  # 465
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USE_SSL'] = False
app.config['MAIL_USERNAME'] = GMAIL_ACCOUNT
app.config['MAIL_DEFAULT_SENDER'] = GMAIL_ACCOUNT
app.config['MAIL_PASSWORD'] = GMAIL_APP_PASSWORD

# API Auto-doc:
auto = Autodoc(app)
mail = Mail(app)

# Views
# noinspection PyUnresolvedReferences,PyPep8
from enerpiweb import views, views_filehandler, utils, rt_stream, emailing

# wsgi
app.wsgi_app = ProxyFix(app.wsgi_app)
