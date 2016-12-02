# -*- coding: utf-8 -*-
"""
ENERPIWEB - CLI methods:

- Install configuration & CRON task for web resources generation
- Uninstall CRON task
- Run Flask server

"""
import argparse
import os
import sys
from enerpi.base import BASE_PATH, DATA_PATH, CONFIG, log, check_resource_files, NGINX_CONFIG_FILE, UWSGI_CONFIG_FILE
from enerpi.prettyprinting import print_secc, print_cyan, print_red, print_magenta


FLASK_WEBSERVER_PORT = CONFIG.getint('ENERPI_WEBSERVER', 'FLASK_WEBSERVER_PORT', fallback=7777)
PERIOD_MINUTES_RSC_GEN = CONFIG.getint('ENERPI_WEBSERVER', 'RSC_GEN_EVERY_MINUTES', fallback=15)
USER_SERVER = CONFIG.get('ENERPI_WEBSERVER', 'USER_SERVER', fallback='www-data')
basedir = os.path.abspath(os.path.dirname(__file__))


def make_cron_command_task_periodic_rscgen():
    """
    CRON task for generate web resources with enerpiplot.mule_rscgen.py
    Example command:
    */15 * * * * sudo -u www-data /home/pi/PYTHON/py35/bin/python
        /home/pi/PYTHON/py35/lib/python3.5/site-packages/enerpiweb/mule_rscgen.py -o

    :return: :str: cron_command
    """
    # cmd_server = '*/{period} * * * * ...'
    cmd_server = 'sudo -u {user_server} {python_pathbin}/python3 {path_enerpiplot}/mule_rscgen.py -o'
    local_params = dict(path_enerpiplot=os.path.abspath(os.path.join(BASE_PATH, '..', 'enerpiplot')),
                        period=PERIOD_MINUTES_RSC_GEN, user_server=USER_SERVER,
                        python_pathbin=os.path.dirname(sys.executable))
    return cmd_server.format(**local_params)


def _make_webserver_config(overwrite=False):
    """
    Genera y escribe en DATA_PATH la configuración de la aplicación web para NGINX y UWSGI-EMPEROR.
    Muestra los comandos para realizar los enlaces correspondientes en /etc/nginx y  /etc/uwsgi-emperor.
    """
    path_config_uwsgi = os.path.join(DATA_PATH, UWSGI_CONFIG_FILE)
    path_config_nginx = os.path.join(DATA_PATH, NGINX_CONFIG_FILE)

    if overwrite or not (os.path.exists(path_config_nginx) and os.path.exists(path_config_uwsgi)):
        nginx_template = os.path.join(basedir, 'templates', 'nginx_conf_mask.txt')
        uwsgi_template = os.path.join(basedir, 'templates', 'uwsgi_ini_mask.txt')
        with open(nginx_template, 'r') as f:
            nginx_raw = f.read()
        with open(uwsgi_template, 'r') as f:
            uwsgi_raw = f.read()
        local_params = dict(file_location=DATA_PATH,
                            filename=UWSGI_CONFIG_FILE,
                            path_enerpiplot=os.path.abspath(os.path.join(basedir, '..', 'enerpiplot')),
                            path_enerpiweb=basedir,
                            path_venv=os.path.abspath(os.path.join(os.path.dirname(sys.executable), '..')))
        uwsgi_conf = uwsgi_raw.format(**local_params)
        nginx_conf = nginx_raw.replace('{file_location}', local_params['file_location']
                                       ).replace('{path_enerpiweb}', local_params['path_enerpiweb']
                                                 ).replace('{filename}', NGINX_CONFIG_FILE)

        # Copy config files to DATA_PATH
        check_resource_files(DATA_PATH, verbose=True)
        with open(os.path.join(DATA_PATH, UWSGI_CONFIG_FILE), 'w') as f:
            f.write(uwsgi_conf)
        with open(os.path.join(DATA_PATH, NGINX_CONFIG_FILE), 'w') as f:
            f.write(nginx_conf)

        # Show Info
        print_secc('NGINX Config generated:')
        print_cyan(nginx_conf)
        print_secc('UWSGI INI Config generated:')
        print_cyan(uwsgi_conf)
    else:
        # Show Info
        print_secc('NGINX Config at "{}":'.format(path_config_nginx))
        print_cyan(open(path_config_nginx, 'r').read())
        print_secc('UWSGI INI Config at "{}":'.format(path_config_uwsgi))
        print_cyan(open(path_config_uwsgi, 'r').read())
    print_red('\n* Append the NGINX config to your actual server, or make the next symlink:\n **"{}"**'
              .format('sudo ln -s {}/{} /etc/nginx/sites-enabled/'.format(DATA_PATH, NGINX_CONFIG_FILE)))
    print_red('* Make a symlink to deposit the UWSGI-Emperor configuration:\n **"{}"**\n'
              .format('sudo ln -s {}/{} /etc/uwsgi-emperor/vassals/'.format(DATA_PATH, UWSGI_CONFIG_FILE)))
    print_magenta('* To start the webserver, restart NGINX & UWSGI-EMPEROR:\n **"{}"**\n **"{}"**\n'
                  .format('sudo service nginx restart', 'sudo service uwsgi-emperor restart'))


def main():
    """
    CLI para ejecutar el webserver a mano vía flask en puerto 7777, o para mostrar/instalar la configuración
    de UWSGI y NGINX para servir la aplicación

    """
    p = argparse.ArgumentParser(description="\033[1m\033[5m\033[32m{}\033[0m\n\n".format('ENERPI Web Server'),
                                epilog='\033[34m\n*** By default, ENERPIWEB starts with flask server ***\n' +
                                       '\033[0m', formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-p', '--port', action='store', type=int, metavar='P', default=FLASK_WEBSERVER_PORT,
                   help='✏  Flask PORT. Default: {}'.format(FLASK_WEBSERVER_PORT))
    p.add_argument('-d', '--debug', action='store_true', help='☕  DEBUG Mode')
    p.add_argument('-i', '--info', action='store_true', help='︎ℹ️  Show config params for NGINX + UWSGI')
    p.add_argument('--install', action='store_true',
                   help='⚒ Install CRON task for WEB RSC generator every {} minutes'.format(PERIOD_MINUTES_RSC_GEN))
    p.add_argument('--uninstall', action='store_true', help='⚒ Uninstall CRON task for WEB RSC generator')
    args = p.parse_args()

    if args.info:
        _make_webserver_config(overwrite=False)
    elif args.install or args.uninstall:
        from enerpi.config.crontasks import set_command_periodic, clear_cron_commands
        # INSTALL / UNINSTALL CRON TASKS & KEY
        cmd_server = make_cron_command_task_periodic_rscgen()
        if args.install:
            log('** (Re-)Create webserver config files:', 'ok', True, False)
            _make_webserver_config(overwrite=True)
            log('** Installing CRON task for web resources generation every {} minutes:\n"{}"'
                .format(PERIOD_MINUTES_RSC_GEN, cmd_server), 'ok', True, False)
            set_command_periodic(cmd_server, comment='Generador de recursos para ENERPIWEB',
                                 minute=PERIOD_MINUTES_RSC_GEN, verbose=True)
        else:
            log('** Deleting CRON task for web resources generation every X minutes:\n"{}"'
                .format(cmd_server), 'warn', True, False)
            clear_cron_commands([cmd_server], verbose=True)
            print_red('\n* To stop the webserver, remove config files from {}:\n **"{}"**\n **"{}"**\n'
                      .format('NGINX & UWSGI-EMPEROR',
                              'sudo rm /etc/uwsgi-emperor/vassals/{}'.format(UWSGI_CONFIG_FILE),
                              'sudo rm /etc/nginx/sites-enabled/{}'.format(NGINX_CONFIG_FILE)))
    else:
        from enerpiweb import app as application

        log('EJECUTANDO FLASK WSGI A MANO en P:{}!'.format(args.port), 'ok', True, False)
        application.run(host="0.0.0.0", port=args.port, processes=1, threaded=True, debug=args.debug)


if __name__ == '__main__':
    main()
