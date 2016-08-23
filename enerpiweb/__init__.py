# -*- coding: utf-8 -*-
from flask import Flask
from werkzeug.contrib.fixers import ProxyFix
from werkzeug.routing import Rule
import jinja2
import os


# FLASK APP
PREFIX_WEB = '/enerpi'
app = Flask(__name__, static_path=PREFIX_WEB + '/static')
app.url_rule_class = lambda path, **options: Rule(PREFIX_WEB + path, **options)
app.jinja_env.trim_blocks = True
app.jinja_env.lstrip_blocks = True
app.jinja_env.cache = {}
app.config['PREFIX_WEB'] = PREFIX_WEB
app.config['BASECOLOR'] = '#0CBB43'
basedir = os.path.abspath(os.path.dirname(__file__))
app.jinja_loader = jinja2.FileSystemLoader(os.path.join(basedir, 'templates'))

SERVER_FILE_LOGGING = os.path.join(basedir, 'static', 'enerpiweb.log')
LOGGING_LEVEL_SERVER = 'DEBUG'


from enerpiweb import views

# wsgi
app.wsgi_app = ProxyFix(app.wsgi_app)
