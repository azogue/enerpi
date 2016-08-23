#!/bin/bash
# -*- coding: utf-8 -*-
# if running bash
if [ -n "$BASH_VERSION" ]; then
    # include .bashrc if it exists
    if [ -f "$HOME/.bashrc" ]; then
	. "$HOME/.bashrc"
    fi
	fi
# set PATH so it includes user's private bin if it exists
if [ -d "$HOME/bin" ] ; then
    PATH="$HOME/bin:$PATH"
fi

PIHOME="/home/pi"
if [ "$HOME" == "/root" ]
then
    echo "Ejecutando como ROOT! (USER: $USER)"
elif [ "$HOME" == "/var/www" ]
then
    echo "Ejecutando como www-data, HOME=$HOME, USER=$USER"
else
    echo "Ejecutando en $HOME (USER: $USER)"
fi

if [ -d "$PIHOME/PYTHON/py35" ]
then
    PYENV="py35"
    source $PIHOME/PYTHON/$PYENV/bin/activate
    export PYTHONPATH=$PYTHONPATH:$PIHOME/PYTHON/:$PIHOME/PYTHON/PIGPIO/:$PIHOME/PYTHON/$PYENV/lib/python3.5/site-packages/:/usr/local/lib/python3.5/dist-packages/:/usr/lib/python3/dist-packages/
else
    PYENV="py34"
    source $PIHOME/PYTHON/$PYENV/bin/activate
    export PYTHONPATH=$PYTHONPATH:$PIHOME/PYTHON/:$PIHOME/PYTHON/PIGPIO/:$PIHOME/PYTHON/$PYENV/lib/python3.4/site-packages/:/usr/local/lib/python3.4/dist-packages/:/usr/lib/python3/dist-packages/
fi
RUTA="$PIHOME/PYTHON/$PYENV/bin/python3"
RUTA_DAEMON="$PIHOME/PYTHON/$PYENV/bin/enerpi-daemon"
echo "PATH: $PATH"
echo "PYTHONPATH: $PYTHONPATH"
cd $PIHOME
$RUTA $RUTA_DAEMON $1
