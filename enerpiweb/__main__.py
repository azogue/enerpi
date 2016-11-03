# -*- coding: utf-8 -*-
import logging
from enerpiweb import app as application, SERVER_FILE_LOGGING, LOGGING_LEVEL_SERVER
from enerpiweb.command_enerpiweb import main


# Para enviar el output a /var/log/apache2/error.log, utilizado para debug
# import sys
# sys.stdout = sys.stderr

# Establecemos logging
logging.basicConfig(filename=SERVER_FILE_LOGGING, level=LOGGING_LEVEL_SERVER, datefmt='%d/%m/%Y %H:%M:%S',
                    format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
# logging.debug('->LOG Estableciendo LOGFILE en {}'.format(SERVER_FILE_LOGGING))


if __name__ == '__main__':
    # Para ejecutar a mano
    main()
