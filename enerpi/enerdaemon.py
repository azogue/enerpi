#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
import os
import sys
from enerpi.base import log, FILE_LOGGING, LOGGING_LEVEL, set_logging_conf
from enerpi.enerpimeter import enerpi_daemon_logger
from enerpi.daemon import Daemon


class EnerPiDaemon(Daemon):
    """
    Clase Daemon para la ejecución de enerpi_main_logger en background
    """
    def run(self):
        """
        Runs the logger
        """
        enerpi_daemon_logger()


def enerpi_daemon(test_mode=False):
    """
    Main logic para el demonio de ejecución de enerpi_main_logger

    Ej de uso:
    sudo -u www-data /home/pi/PYTHON/py35/bin/enerpi-daemon start|stop|status|restart
    enerpi-daemon start|stop|status|restart
    Opciones:
        - start
        - stop
        - status
        - restart
    """
    set_logging_conf(FILE_LOGGING, LOGGING_LEVEL, with_initial_log=False)
    # TODO Eliminar stdout y stderr! (o diferenciar su uso según LOGGING_LEVEL)
    daemon = EnerPiDaemon('/tmp/enerpilogger.pid',
                          test_mode=test_mode,
                          stdout=os.path.expanduser('/tmp/enerpi_out.txt'),
                          stderr=os.path.expanduser('/tmp/enerpi_err.txt'))
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            log('ENERPI Logger daemon started', 'info', False)
            try:
                daemon.start()
            except PermissionError as e:
                sys.stderr.write("PERMISSIONERROR: pidfile can't be registered ({}). Need sudo powers?".format(e))
        elif 'stop' == sys.argv[1]:
            log('ENERPI Logger daemon stopped', 'warn', False)
            daemon.stop()
        elif 'status' == sys.argv[1]:
            log('ENERPI Logger daemon status?', 'debug', False)
            daemon.status()
        elif 'restart' == sys.argv[1]:
            log('ENERPI Logger daemon restarting', 'info', False)
            daemon.restart()
        else:
            log("Unknown command", 'warn', True, False)
            if not test_mode:
                sys.exit(2)
        if not test_mode:
            sys.exit(0)
    else:
        print("usage: %s start|stop|restart|status".format(sys.argv[0]))
        if not test_mode:
            sys.exit(2)


if __name__ == "__main__":
    enerpi_daemon()
