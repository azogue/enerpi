#!/usr/bin/python -tt
# -*- coding: utf-8 -*-
import os
import sys
from enerpi.base import log
from enerpi.command_enerpi import enerpi_main_logger  # , enerpi_main_logger_demo
from enerpi.daemon import Daemon


class MyDaemon(Daemon):
    def run(self):
        enerpi_main_logger()
        # enerpi_main_logger_demo()


def enerpi_daemon():
    # TODO Eliminar stdout y stderr!
    # if sys.platform == 'linux':
    daemon = MyDaemon('/tmp/enerpilogger.pid',
                      stdout=os.path.expanduser('/tmp/enerpi_out.txt'),
                      stderr=os.path.expanduser('/tmp/enerpi_err.txt'))
    # else:
    #     daemon = MyDaemon('/tmp/enerpilogger.pid',
    #                       stdout=os.path.expanduser('~/enerpi_out.txt'),
    #                       stderr=os.path.expanduser('~/enerpi_err.txt'))
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
            sys.exit(2)
        sys.exit(0)
    else:
        print("usage: %s start|stop|restart|status".format(sys.argv[0]))
        sys.exit(2)


if __name__ == "__main__":
    enerpi_daemon()
