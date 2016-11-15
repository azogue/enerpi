# -*- coding: utf-8 -*-
import datetime as dt
import os
import re
import sys
from enerpi import PRETTY_NAME, DESCRIPTION
from enerpi.base import (BASE_PATH, CONFIG, SENSORS, DATA_PATH, CONFIG_FILENAME, show_pi_temperature,
                         FILE_LOGGING, LOGGING_LEVEL, set_logging_conf, log)
from enerpi.database import (operate_hdf_database, get_ts_last_save, init_catalog, show_info_data,
                             extract_log_file, delete_log_file, HDF_STORE)
from enerpi.enerpimeter import enerpi_logger, receiver, enerpi_raw_data
from enerpi.iobroadcast import UDP_PORT


# Config:
DEFAULT_IMG_MASK = CONFIG.get('ENERPI_DATA', 'DEFAULT_IMG_MASK', fallback='enerpi_plot_{:%Y%m%d_%H%M}.png')
IMG_TILES_BASEPATH = os.path.join(BASE_PATH, '..', 'enerpiweb', 'static', 'img', 'generated')


def _enerpi_arguments():
    """
    CLI Parser
    """
    import argparse

    p = argparse.ArgumentParser(description="\033[1m\033[5m\033[32m{}\033[0m\n{}\n\n".format(PRETTY_NAME, DESCRIPTION),
                                epilog='\033[34m\n*** By default, ENERPI starts as receiver (-r) ***\n' +
                                       '\033[0m', formatter_class=argparse.RawTextHelpFormatter)
    g_m = p.add_argument_group(title='☆  \033[1m\033[4mENERPI Working Mode\033[24m',
                               description='→  Choose working mode between RECEIVER / SENDER')
    g_m.add_argument('-e', '--enerpi', action='store_true', help='⚡  SET ENERPI LOGGER & BROADCAST MODE')
    g_m.add_argument('-r', '--receive', action='store_true', help='⚡  SET Broadcast Receiver mode (by default)')
    g_m.add_argument('--port', '--receiver-port', type=int, action='store', default=UDP_PORT, metavar='XX',
                     help='⚡  SET Broadcast Receiver PORT')
    g_m.add_argument('-d', '--demo', action='store_true', help='☮ SET Demo Mode (broadcast random values)')
    g_m.add_argument('--timeout', action='store', nargs='?', type=int, metavar='∆T', const=60,
                     help='⚡  SET Timeout to finish execution automatically')
    g_m.add_argument('--raw', type=int, action='store', nargs='?', const=5, metavar='∆T',
                     help='☮ SET RAW Data Mode (adquire all samples)')
    g_m.add_argument('--config', action='store_true', help='⚒  Shows configuration in INI file')
    g_m.add_argument('--install', action='store_true', help='⚒  Install CRON task for exec ENERPI LOGGER as daemon')
    g_m.add_argument('--uninstall', action='store_true', help='⚒  Delete all CRON tasks from ENERPI')

    g_p = p.add_argument_group(title='︎ℹ️  \033[4mQUERY & REPORT DATA\033[24m')
    filter_24h = (dt.datetime.now().replace(microsecond=0) - dt.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
    g_p.add_argument('-f', '--filter', action='store', nargs='?', metavar='TS', const=filter_24h,
                     help='✂ Query the HDF Store with pandas-like slicing:'
                          '\n     "2016-01-07 :: 2016-02-01 04:00" --> df.loc["2016-01-07":"2016-02-01 04:00"]'
                          '\n     \t(Pay atention to the double "::"!!)'
                          '\n     · By default, "-f" filters data from 24h ago (.loc[{}:]).\n\n'.format(filter_24h))
    default_img_nomask = DEFAULT_IMG_MASK.replace('{', '{{').replace('}', '}}').replace('%', '%%')
    help_plot = '''⎙  Plot & save image with matplotlib in any compatible format.
     · If not specified, PNG file is generated with MASK:\n           "{}" using datetime data limits.
     · If only specifying image format, default mask is used with the desired format.
     · If image path is passed, the initial (and final, optionally) timestamps of filtered data
     can be used with formatting masks, like:
         "/path/to/image/image_{{:%%c}}_{{:%%H%%M}}.pdf" or "report_{{:%%d%%m%%y}}.svg".'''.format(default_img_nomask)
    g_p.add_argument('-p', '--plot', action='store', metavar='IM', nargs='?', const=DEFAULT_IMG_MASK, help=help_plot)
    g_p.add_argument('-pt', '--plot_tiles', action='store_true', help='⎙  Generate SVG Tiles for enerpiWeb.')

    g_st = p.add_argument_group(title='⚙  \033[4mHDF Store Options\033[24m')
    g_st.add_argument('--store', action='store', metavar='ST', default=HDF_STORE,
                      help='✏️  Set the .h5 file where save the HDF store.\n     Default: "{}"'.format(HDF_STORE))
    g_st.add_argument('--backup', action='store', metavar='BKP', help='☔ Backup the HDF Store')
    g_st.add_argument('--clear', action='store_true', help='☠ \033[31mDelete the HDF Store database\033[39m')
    g_st.add_argument('--clearlog', action='store_true', help='⚠ Delete the LOG FILE at: "{}"'.format(FILE_LOGGING))
    g_st.add_argument('-i', '--info', action='store_true', help='︎ℹ Show data info')
    g_st.add_argument('--last', action='store_true', help='︎ℹ Show last saved data')

    g_d = p.add_argument_group(title='☕  \033[4mDEBUG Options\033[24m')
    g_d.add_argument('--temps', action='store_true', help='♨ Show RPI temperatures (CPU + GPU)')
    g_d.add_argument('-l', '--log', action='store_true', help='☕ Show LOG FILE')
    g_d.add_argument('-s', '--silent', action='store_true', help='‼ Silent mode (Verbose mode ON BY DEFAULT in CLI)')

    g_ts = p.add_argument_group(title='⚒  \033[4mCurrent Meter Sampling Configuration\033[24m')
    g_ts.add_argument('-T', '--delta', type=float, action='store', default=SENSORS.delta_sec_data, metavar='∆T',
                      help='⌚  Set Ts sampling (to database & broadcast), in seconds. Default ∆T: {} s'
                      .format(SENSORS.delta_sec_data))
    g_ts.add_argument('-ts', type=int, action='store', default=SENSORS.ts_data_ms, metavar='∆T',
                      help='⏱  Set Ts raw sampling, in ms. Default ∆T_s: {} ms'.format(SENSORS.ts_data_ms))
    g_ts.add_argument('-w', '--window', type=float, action='store', default=SENSORS.rms_roll_window_sec, metavar='∆T',
                      help='⚖  Set window width in seconds for instant RMS calculation. Default ∆T_w: {} s'
                      .format(SENSORS.rms_roll_window_sec))

    return p.parse_args()


def make_cron_command_task_daemon():
    """
    CRON periodic task for exec ENERPI LOGGER as daemon at every boot
    Example command:
    */15 * * * * sudo -u www-data /home/pi/PYTHON/py35/bin/python
        /home/pi/PYTHON/py35/lib/python3.5/site-packages/enerpiweb/mule_rscgen.py -o

    :return: :str: cron_command
    """
    # cmd_logger = '@reboot sudo -u {user_logger} {python_pathbin}/enerpi-daemon start'
    cmd_logger = 'sudo -u {user_logger} {python_pathbin}/enerpi-daemon start'
    local_params = dict(user_logger=CONFIG.get('ENERPI_DATA', 'USER_LOGGER', fallback='pi'),
                        python_pathbin=os.path.dirname(sys.executable))
    return cmd_logger.format(**local_params)


def enerpi_main_cli(test_mode=False):
    """
    Uso de ENERPI desde CLI

    enerpi -h para mostrar las diferentes opciones

    """
    # Init CLI
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import pandas as pd
    pd.set_option('display.width', 200)

    # CLI Arguments
    args = _enerpi_arguments()
    verbose = not args.silent

    # CONTROL LOGIC
    # Shows RPI Temps
    timer_temps = show_pi_temperature(args.temps, 3)

    if args.install or args.uninstall:
        from enerpi.config.crontasks import set_command_on_reboot, clear_cron_commands
        # INSTALL / UNINSTALL CRON TASKS & KEY
        cmd_logger = make_cron_command_task_daemon()
        if args.install:
            # Logging configuration
            set_logging_conf(FILE_LOGGING, LOGGING_LEVEL, True)

            log('** Installing CRON task for start logger at reboot:\n"{}"'.format(cmd_logger), 'ok', True, False)
            set_command_on_reboot(cmd_logger, verbose=verbose)
            os.chmod(DATA_PATH, 0o777)
            [os.chmod(os.path.join(base, f), 0o777) for base, dirs, files in os.walk(DATA_PATH) for f in files + dirs]
        else:
            log('** Deleting CRON task for start logger at reboot:\n"{}"'.format(cmd_logger), 'warn', True, False)
            clear_cron_commands([cmd_logger], verbose=verbose)
    elif (args.enerpi or args.info or args.backup or args.clear or args.config or args.raw or
            args.last or args.clearlog or args.filter or args.plot or args.plot_tiles):
        # Logging configuration
        set_logging_conf(FILE_LOGGING, LOGGING_LEVEL, True)

        # Shows INI config
        if args.config:
            log('ENERPI Configuration (from INI file in "{}"):'
                .format(os.path.join(DATA_PATH, CONFIG_FILENAME)), 'ok', True, False)
            for s in CONFIG.sections():
                log('* Section {}:'.format(s), 'info', True, False)
                for opt in CONFIG.options(s):
                    log('{:27} -->\t{}'.format(opt.upper(), CONFIG.get(s, opt)), 'debug', True, False)
            log('*' * 80 + '\n', 'ok', True, False)

        # Delete LOG File
        if args.clearlog:
            delete_log_file(FILE_LOGGING, verbose=verbose)

        # Data Store Config
        path_st = operate_hdf_database(args.store, path_backup=args.backup, clear_database=args.clear)

        # Starts ENERPI Logger
        if args.enerpi:
            enerpi_logger(is_demo=False, verbose=verbose, path_st=path_st, delta_sampling=args.delta,
                          roll_time=args.window, sampling_ms=args.ts, timeout=args.timeout)
        elif args.raw:
            # Raw mode
            delta_secs = args.raw
            raw_data = enerpi_raw_data(path_st.replace('.h5', '_raw_sample.h5'), delta_secs=delta_secs,
                                       roll_time=args.window, sampling_ms=args.ts, verbose=verbose)
            t0, tf = raw_data.index[0], raw_data.index[-1]
            log('Showing RAW DATA for {} seconds ({} samples, {:.2f} sps)\n** Real data: from {} to {} --> {:.2f} sps'
                .format(delta_secs, len(raw_data), len(raw_data) / delta_secs,
                        t0, tf, len(raw_data) / (tf-t0).total_seconds()), 'info', verbose, False)
            # TODO revisar config X11 + ssh -X para plot en display local
            raw_data.plot(lw=.5, figsize=(16, 10))
            plt.show()
        # Shows database info
        elif args.info or args.filter or args.plot or args.plot_tiles:
            catalog = init_catalog(raw_file=path_st, check_integrity=False)
            if args.plot_tiles:
                from enerpiplot.enerplot import gen_svg_tiles

                ok = gen_svg_tiles(IMG_TILES_BASEPATH, catalog)
                if ok:
                    log('SVG Tiles generated!', 'ok', verbose, True)
                else:
                    log('No generation of SVG Tiles!', 'error', verbose, True)
            else:
                if args.filter:
                    loc_data = args.filter.split('::')
                    if len(loc_data) > 1:
                        if len(loc_data[0]) > 0:
                            data, consumption = catalog.get(start=loc_data[0], end=loc_data[1], with_summary=True)
                        else:
                            data, consumption = catalog.get(end=loc_data[1], with_summary=True)
                    else:
                        last_hours = re.findall('(\d{1,5})h', loc_data[0], flags=re.IGNORECASE)
                        if last_hours:
                            data, consumption = catalog.get(last_hours=int(last_hours[0]), with_summary=True)
                        else:
                            data, consumption = catalog.get(start=loc_data[0], with_summary=True)
                else:
                    data, consumption = catalog.get(start=catalog.min_ts, with_summary=True)
                if args.info and data is not None and not data.empty:
                    show_info_data(data, consumption)
                if (args.plot and data is not None and not data.empty and consumption is not None and
                        not consumption.empty):
                    from enerpiplot.enerplot import plot_power_consumption_hourly

                    img_name = plot_power_consumption_hourly(data.power, consumption.kWh, ldr=data.ldr,
                                                             rs_potencia=None, rm_potencia=60, savefig=args.plot)
                    log('Image stored in "{}"'.format(img_name), 'ok', verbose, True)
        # Shows database info
        else:  # Shows last 10 entries
            last = get_ts_last_save(path_st, get_last_sample=True, verbose=verbose, n=10)
            log('Showing last 10 entries in {}:\n{}'.format(path_st, last), 'info', verbose, False)
    # Shows & extract info from LOG File
    elif args.log:
        data_log = extract_log_file(FILE_LOGGING, extract_temps=True, verbose=verbose)
        try:
            df_temps = data_log[data_log.temp.notnull()].drop(['tipo', 'msg', 'debug_send'], axis=1).dropna(axis=1)
            if len(set(df_temps['exec'])) == 1:
                df_temps = df_temps.drop(['exec'], axis=1)
            path_csv = os.path.join(DATA_PATH, 'debug_rpitemps.csv')
            if not df_temps.empty:
                df_temps.to_csv(path_csv)
                print('*** Saved temperature data extracted from LOG in {}'.format(path_csv))
        except AttributeError:
            print('No se encuentran datos de Tª de RPI en el LOG')
    # Demo sender
    elif args.demo:
        set_logging_conf(FILE_LOGGING + '_demo.log', LOGGING_LEVEL, True)
        path_st = os.path.join(DATA_PATH, 'debug_buffer_disk.h5')
        enerpi_logger(is_demo=True, verbose=verbose, path_st=path_st, delta_sampling=args.delta,
                      roll_time=args.window, sampling_ms=args.ts, timeout=args.timeout)
    # Receiver
    else:  # elif args.receive:
        log('{}\n   {}'.format(PRETTY_NAME, DESCRIPTION), 'ok', verbose, False)
        receiver(verbose=verbose, timeout=args.timeout, port=args.port)
    log('Exiting from ENERPI CLI...', 'debug', True)
    if timer_temps is not None:
        log('Stopping RPI TEMPS sensing...', 'debug', True)
        timer_temps.cancel()
    if not test_mode:
        sys.exit(0)


if __name__ == '__main__':
    enerpi_main_cli(test_mode=False)
