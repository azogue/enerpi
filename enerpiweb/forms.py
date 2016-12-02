# -*- coding: utf-8 -*-
"""
Flask forms

"""
from flask_wtf import FlaskForm
# from wtforms import StringField
# from wtforms.validators import DataRequired


class DummyForm(FlaskForm):
    """
    For post submits without any field...
    """
    pass


# class FileForm(FlaskForm):
#     """
#     File explorer
#     """
#     pathfile = StringField('pathfile', validators=[DataRequired()])
