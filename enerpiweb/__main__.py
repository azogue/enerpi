# -*- coding: utf-8 -*-
import os
# noinspection PyUnresolvedReferences
from enerpiweb import app as application, SERVER_FILE_LOGGING, LOGGING_LEVEL_SERVER
from enerpiweb.command_enerpiweb import main, check_resource_files
from enerpi.base import set_logging_conf


# Para enviar el output a /var/log/apache2/error.log, utilizado para debug
# import sys
# sys.stdout = sys.stderr

# Establecemos logging
check_resource_files(os.path.dirname(SERVER_FILE_LOGGING))
set_logging_conf(SERVER_FILE_LOGGING, level=LOGGING_LEVEL_SERVER, verbose=True, with_initial_log=True)
# logging.debug('->LOG Estableciendo LOGFILE en {}'.format(SERVER_FILE_LOGGING))


if __name__ == '__main__':
    # Para ejecutar a mano
    main()
