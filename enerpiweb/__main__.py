# -*- coding: utf-8 -*-
import logging
from enerpiweb import app as application, SERVER_FILE_LOGGING, LOGGING_LEVEL_SERVER


# Para enviar el output a /var/log/apache2/error.log, utilizado para debug
# import sys
# sys.stdout = sys.stderr

# Establecemos logging
logging.basicConfig(filename=SERVER_FILE_LOGGING, level=LOGGING_LEVEL_SERVER, datefmt='%d/%m/%Y %H:%M:%S',
                    format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
# logging.debug('->LOG Estableciendo LOGFILE en {}'.format(SERVER_FILE_LOGGING))


# Para ejecutar a mano
def main():
    import argparse
    import os
    from prettyprinting import print_secc, print_cyan, print_red

    flask_webserver_port = 7777
    basepath = os.path.abspath(os.path.dirname(__file__))

    p = argparse.ArgumentParser(description="\033[1m\033[5m\033[32m{}\033[0m\n\n".format('ENERPI Web Server'),
                                epilog='\033[34m\n*** By default, ENERPIWEB starts with flask server ***\n' +
                                       '\033[0m', formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument('-p', '--port', action='store', type=int, metavar='P', default=flask_webserver_port,
                   help='✏  Flask PORT. Default: {}'.format(flask_webserver_port))
    p.add_argument('-d', '--debug', action='store_true', help='☕  DEBUG Mode')
    p.add_argument('-i', '--info', action='store_true', help='︎ℹ️  Show config params for NGINX + UWSGI')
    args = p.parse_args()

    if args.info:
        with open(os.path.join(basepath, 'enerpiweb.ini'), 'r') as f:
            ini_file = f.read()
        with open(os.path.join(basepath, 'enerpiweb_nginx.conf'), 'r') as f:
            conf_file = f.read()
        print_secc('NGINX Config:')
        print_cyan(conf_file)
        print_red('* Append the NGINX config to your actual server, or make the next symlink:\n **"{}"**'
                  .format('sudo ln -s {}/enerpiweb_nginx.conf /etc/nginx/sites-enabled/'.format(basepath)))
        print_secc('UWSGI INI Config:')
        print_cyan(ini_file)
        print_red('* Make a symlink to deposit the UWSGI-Emperor configuration:\n **"{}"**'
                  .format('sudo ln -s {}/enerpiweb.ini /etc/uwsgi-emperor/vassals/'.format(basepath)))
    else:
        print('EJECUTANDO FLASK WSGI A MANO en P:{}!'.format(flask_webserver_port))
        application.run(host="0.0.0.0", port=args.port, processes=1, threaded=True, debug=args.debug)


if __name__ == '__main__':
    main()
