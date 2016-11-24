# -*- coding: utf-8 -*-
from flask_wtf import Form
from wtforms import StringField
from wtforms.validators import DataRequired


class DummyForm(Form):
    """
    For post submits without any field...
    """
    pass


class FileForm(Form):
    """
    File explorer
    """
    pathfile = StringField('pathfile', validators=[DataRequired()])
