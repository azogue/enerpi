#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
ENERPI - Base class for daemonize the ENERPI logger

"""
import atexit
import os
import sys
import time
from signal import SIGTERM, SIGKILL


class Daemon:
    """
    A generic daemon class.

    Usage: subclass the Daemon class and override the run() method
    """

    def __init__(self, pidfile, stdin='/dev/null', stdout='/dev/null', stderr='/dev/null', test_mode=False):
        self.stdin = stdin
        self.stdout = stdout
        self.stderr = stderr
        self.pidfile = pidfile
        self.test_mode = test_mode

    def _get_pid(self):
        """
        Get the pid from the pidfile
        """
        try:
            with open(self.pidfile, 'r') as pf:
                return int(pf.read().strip())
        except IOError:
            return None

    def daemonize(self):
        """
        do the UNIX double-fork magic, see Stevens' "Advanced
        Programming in the UNIX Environment" for details (ISBN 0201563177)
        http://www.erlenstar.demon.co.uk/unix/faq_2.html#SEC16
        """
        try:
            pid = os.fork()
            if pid > 0:
                # exit first parent
                if self.test_mode:
                    return True
                sys.exit(0)
        except OSError as e:
            sys.stderr.write("fork #1 failed: %d (%s)\n" % (e.errno, e.strerror))
            if self.test_mode:
                return False
            sys.exit(1)

        # decouple from parent environment
        os.chdir("/")
        os.setsid()
        os.umask(0)

        # do second fork
        try:
            pid = os.fork()
            if pid > 0:
                # exit from second parent
                if self.test_mode:
                    return True
                sys.exit(0)
        except OSError as e:
            sys.stderr.write("fork #2 failed: %d (%s)\n" % (e.errno, e.strerror))
            if self.test_mode:
                return False
            sys.exit(1)

        # redirect standard file descriptors
        sys.stdout.flush()
        sys.stderr.flush()
        si = open(self.stdin, 'r')
        so = open(self.stdout, 'a+')
        # se = open(self.stderr, 'a+', 0)
        se = open(self.stderr, 'a+', 1)
        os.dup2(si.fileno(), sys.stdin.fileno())
        os.dup2(so.fileno(), sys.stdout.fileno())
        os.dup2(se.fileno(), sys.stderr.fileno())

        # write pidfile
        atexit.register(self.delpid)
        pid = str(os.getpid())
        # try:
        open(self.pidfile, 'w+').write("{}\n".format(pid))
        # except PermissionError as e:
        #     sys.stderr.write("PERMISSIONERROR: pidfile {} can't be written with PID {} ({}). Need sudo powers?\n"
        #                      .format(self.pidfile, pid, e))
        #     print("PERMISSIONERROR: pidfile {} can't be written with PID {} ({}). Need sudo powers?\n"
        #           .format(self.pidfile, pid, e))
        #     sys.exit(-1)

    def delpid(self):
        """
        Remove pidfile from disk

        """
        if os.path.exists(self.pidfile):
            os.remove(self.pidfile)

    def start(self):
        """
        Start the daemon
        """
        # Check for a pidfile to see if the daemon already runs
        pid = self._get_pid()
        if pid:
            message = "pidfile %s already exist. Daemon already running? (PID={})\n".format(pid)
            sys.stderr.write(message % self.pidfile)
            if self.test_mode:
                return False
            sys.exit(1)

        # Start the daemon
        self.daemonize()
        self.run()

    def stop(self):
        """
        Stop the daemon
        """
        # Get the pid from the pidfile
        pid = self._get_pid()
        if not pid:
            message = "pidfile %s does not exist. Daemon not running?\n"
            sys.stderr.write(message % self.pidfile)
            return True  # not an error in a restart

        # Try killing the daemon process
        try:
            retries = 0
            while retries < 3:
                retries += 1
                os.kill(pid, SIGTERM)
                time.sleep(0.2)
            os.kill(pid, SIGKILL)
            time.sleep(0.5)
            os.kill(pid, SIGKILL)
        except (ProcessLookupError, OSError) as err:
            # print('Exception en STOP pid={}: {} [{}]'.format(pid, err, err.__class__))
            err = str(err)
            if (err.find("No such process") > 0) or (err.find("No existe el proceso") > 0):
                self.delpid()
            else:
                print('OSError: ', err)
                if self.test_mode:
                    return False
                sys.exit(1)
        return True

    def status(self):
        """
        Status of the daemon
        """
        pid = self._get_pid()

        if pid:
            message = "pidfile %s exist. Daemon is running with PID={}\n".format(pid)
            sys.stdout.write(message % self.pidfile)
            print('STATUS OK!')
            return True  # not an error in a restart
        else:
            message = "pidfile %s does not exist. Daemon not running?\n"
            sys.stderr.write(message % self.pidfile)
            # if self.test_mode:
            return False
            # sys.exit(1)

    def restart(self):
        """
        Restart the daemon
        """
        self.stop()
        self.start()

    def run(self):
        """
        You should override this method when you subclass Daemon. It will be called after the process has been
        daemonized by start() or restart().
        """
        raise NotImplementedError
