#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
"""
ENERPI - Custom class (inherit from Daemon class) for daemonize the ENERPI logger

"""
import sys
from enerpi.base import log, FILE_LOGGING, LOGGING_LEVEL, set_logging_conf, DAEMON_PIDFILE, DAEMON_STDOUT, DAEMON_STDERR
from enerpi.daemon import Daemon


class EnerPiDaemon(Daemon):
    """
    Clase Daemon para la ejecución de enerpi_main_logger en background
    """
    def run(self):
        """
        Runs the logger
        """
        from enerpi.enerpimeter import enerpi_daemon_logger

        enerpi_daemon_logger()


def enerpi_daemon(test_mode=False):
    """
    Main logic para el demonio de ejecución de enerpi_daemon_logger

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
    daemon = EnerPiDaemon(DAEMON_PIDFILE, test_mode=test_mode, stdout=DAEMON_STDOUT, stderr=DAEMON_STDERR)
    if len(sys.argv) == 2:
        if 'start' == sys.argv[1]:
            try:
                daemon.start()
                log('ENERPI Logger daemon started', 'ok')
            except PermissionError as e:
                sys.stderr.write("PERMISSIONERROR: pidfile can't be registered ({}). Need sudo powers?".format(e))
        elif 'stop' == sys.argv[1]:
            stopped = daemon.stop()
            log('ENERPI Logger daemon stopped:{}'
                .format(stopped), 'warn', True, True)
        elif 'status' == sys.argv[1]:
            log('ENERPI Logger daemon status?', 'debug', True, True)
            daemon.status()
        elif 'restart' == sys.argv[1]:
            daemon.restart()
            log('ENERPI Logger daemon restarting', 'info', True, True)
        else:
            log("Unknown command", 'warn', True, True)
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
