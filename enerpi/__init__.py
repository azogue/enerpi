# -*- coding: utf-8 -*-
import os

PRETTY_NAME = "︎⚡⚡ ︎ENERPI AC CURRENT SENSOR ⚡⚡"
DESCRIPTION = 'AC Current Meter for Raspberry PI with GPIOZERO and MCP3008'
# TODO LONG_Description del módulo
LONG_DESCRIPTION = 'AC Current Meter for Raspberry PI with GPIOZERO and MCP3008'

# TODO Comparar cargas de CPU/MEM vs params
'''python psaux enerpi
 ==> PS AUX:
FILTRO: "enerpi"
0   USER    PID  %CPU %MEM     VSZ    RSS    TTY STAT  START    TIME  COMMAND
127   pi  10216  61.0  5.7  211372  43064  pts/1  Rl+  ago02  637:53  python enerpi -e -ts 12 -v
124   pi  1859   62.0  6.6  227320  49964  pts/1  Rl+  ago03  1518:56  python enerpi -e --store juan_iborra_3 --temps -v
'''

BASE_PATH = os.path.abspath(os.path.dirname(__file__))

VERSION = '0.8.6'
__version__ = VERSION
