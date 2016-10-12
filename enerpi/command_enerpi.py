# -*- coding: utf-8 -*-

import datetime as dt
import logging
import os
import re
from threading import Timer
from time import time, sleep
from enerpi import PRETTY_NAME, DESCRIPTION, BASE_PATH
from enerpi.base import CONFIG, log, show_pi_temperature
from enerpi.database import (operate_hdf_database, get_ts_last_save, init_catalog, show_info_data,
                             extract_log_file, delete_log_file, DATA_PATH, HDF_STORE)
from enerpi.enerpimeter import enerpi_logger, receiver, sender_random, DELTA_SEC_DATA, TS_DATA_MS, RMS_ROLL_WINDOW_SEC


# Config:
DEFAULT_IMG_MASK = CONFIG.get('ENERPI_DATA', 'DEFAULT_IMG_MASK', fallback='enerpi_plot_{:%Y%m%d_%H%M}.png')

FILE_LOGGING = CONFIG.get('ENERPI_DATA', 'FILE_LOGGING', fallback='enerpi.log')
FILE_LOGGING = os.path.join(DATA_PATH, FILE_LOGGING)
LOGGING_LEVEL = CONFIG.get('ENERPI_DATA', 'LOGGING_LEVEL', fallback='DEBUG')

IMG_TILES_BASEPATH = os.path.join(BASE_PATH, '..', 'enerpiweb', 'static', 'img', 'generated')


def set_logging_conf(filename=FILE_LOGGING, level=LOGGING_LEVEL, verbose=True):
    # Logging configuration
    os.makedirs(DATA_PATH, exist_ok=True)
    logging.basicConfig(filename=filename, level=level, datefmt='%d/%m/%Y %H:%M:%S',
                        format='%(levelname)s [%(filename)s_%(funcName)s] - %(asctime)s: %(message)s')
    log(PRETTY_NAME, 'ok', verbose)


def _run_logger(is_demo=True, verbose=True, path_st=None, **kwargs_sender):
    # Logging configuration
    logfile = FILE_LOGGING + '_demo.log' if is_demo else FILE_LOGGING

    set_logging_conf(logfile, LOGGING_LEVEL, verbose)
    print(logfile)

    # Data Store Config
    if is_demo:
        path_st = os.path.join(DATA_PATH, 'debug_buffer_disk.h5')
    elif path_st is None:
        path_st = operate_hdf_database(HDF_STORE, path_backup=None, clear_database=False)

    # Starts ENERPI Logger loop
    n_execs = 0
    tic = time()
    while True:
        if is_demo:
            ok = sender_random(path_st=path_st, verbose=verbose, **kwargs_sender)
            # break
        else:
            ok = enerpi_logger(path_st=path_st, verbose=verbose, **kwargs_sender)
        if not ok:
            break
        toc = time()
        log('Waiting 10 s to init again... EXEC={}, ∆T={:.1f} s'.format(n_execs, toc - tic), 'ok', verbose, True)
        sleep(10)
        tic = time()
        n_execs += 1


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
    g_m.add_argument('-e', '-s', '--enerpi', action='store_true', help='⚡  SET ENERPI LOGGER & BROADCAST MODE')
    g_m.add_argument('-r', '--receive', action='store_true', help='⚡  SET Broadcast Receiver mode (by default)')
    g_m.add_argument('-d', '--demo', action='store_true', help='☮️  SET Demo Mode (broadcast random values)')

    g_p = p.add_argument_group(title='︎ℹ️  \033[4mQUERY & REPORT DATA\033[24m')
    filter_24h = (dt.datetime.now().replace(microsecond=0) - dt.timedelta(hours=24)).strftime('%Y-%m-%d %H:%M:%S')
    g_p.add_argument('-f', '--filter', action='store', nargs='?', metavar='TS', const=filter_24h,
                     help='✂️  Query the HDF Store with pandas-like slicing:'
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
    g_st.add_argument('--compact', action='store_true', help='✙✙  Compact the HDF Store database (read, delete, save)')
    g_st.add_argument('--backup', action='store', metavar='BKP', help='☔️  Backup the HDF Store')
    g_st.add_argument('--clear', action='store_true', help='☠  \033[31mDelete the HDF Store database\033[39m')
    g_st.add_argument('--clearlog', action='store_true', help='⚠️  Delete the LOG FILE at: "{}"'.format(FILE_LOGGING))
    g_st.add_argument('-i', '--info', action='store_true', help='︎ℹ️  Show data info')
    g_st.add_argument('--last', action='store_true', help='︎ℹ️  Show last saved data')

    g_d = p.add_argument_group(title='☕  \033[4mDEBUG Options\033[24m')
    g_d.add_argument('--temps', action='store_true', help='♨️  Show RPI temperatures (CPU + GPU)')
    g_d.add_argument('-l', '--log', action='store_true', help='☕  Show LOG FILE')
    g_d.add_argument('--debug', action='store_true', help='☕  DEBUG Mode (save timing to csv)')
    g_d.add_argument('-v', '--verbose', action='store_false', help='‼️  Verbose mode ON BY DEFAULT!')

    g_ts = p.add_argument_group(title='⚒  \033[4mCurrent Meter Sampling Configuration\033[24m')
    g_ts.add_argument('-T', '--delta', type=int, action='store', default=DELTA_SEC_DATA, metavar='∆T',
                      help='⌚  Set Ts sampling (to database & broadcast), in seconds. Default ∆T: {} s'
                      .format(DELTA_SEC_DATA))
    g_ts.add_argument('-ts', type=int, action='store', default=TS_DATA_MS, metavar='∆T',
                      help='⏱  Set Ts raw sampling, in ms. Default ∆T_s: {} ms'.format(TS_DATA_MS))
    g_ts.add_argument('-w', '--window', type=int, action='store', default=RMS_ROLL_WINDOW_SEC, metavar='∆T',
                      help='⚖  Set window width in seconds for instant RMS calculation. Default ∆T_w: {} s'
                      .format(RMS_ROLL_WINDOW_SEC))
    # g_ts.add_argument('--raw', action='store_true',
    #                   help='⚖  Set window width in seconds for instant RMS calculation. Default ∆T_w: {} s'
    #                   .format(RMS_ROLL_WINDOW_SEC))

    return p.parse_args()


def enerpi_main_cli():
    import matplotlib

    matplotlib.use('Agg')
    # CLI Arguments
    args = _enerpi_arguments()

    # CONTROL LOGIC
    if args.temps:
        # Shows RPI Temps
        Timer(3, show_pi_temperature, args=(3,)).start()

    if (args.enerpi or args.info or args.compact or args.backup or args.clear or
            args.last or args.clearlog or args.filter or args.plot or args.plot_tiles):
        # Logging configuration
        set_logging_conf(FILE_LOGGING, LOGGING_LEVEL, args.verbose)

        # Delete LOG File
        if args.clearlog:
            delete_log_file(FILE_LOGGING, verbose=True)

        # Data Store Config
        path_st = operate_hdf_database(args.store, path_backup=args.backup, clear_database=args.clear)

        # Starts ENERPI Logger
        if args.enerpi:
            # TODO Raw mode
            _run_logger(is_demo=False, verbose=args.verbose, path_st=path_st,
                        delta_sampling=args.delta, roll_time=args.window, sampling_ms=args.ts,
                        debug=args.debug)
        # Shows database info
        elif args.info or args.filter or args.plot or args.plot_tiles:
            catalog = init_catalog(base_path=DATA_PATH, raw_file=path_st, check_integrity=False)
            if args.plot_tiles:
                from enerpi.enerplot import gen_svg_tiles

                ok = gen_svg_tiles(IMG_TILES_BASEPATH, catalog)
                if ok:
                    log('SVG Tiles generated!', 'ok', args.verbose, True)
                else:
                    log('No generation of SVG Tiles!', 'error', args.verbose, True)
            else:
                if args.filter:
                    loc_data = args.filter.split('::')
                    if len(loc_data) > 1:
                        if len(loc_data[0]) > 0:
                            data, consumption = catalog.get(start=loc_data[0], end=loc_data[1], with_summary_data=True)
                        else:
                            data, consumption = catalog.get(end=loc_data[1], with_summary_data=True)
                    else:
                        last_hours = re.findall('(\d{1,5})h', loc_data[0], flags=re.IGNORECASE)
                        if last_hours:
                            data, consumption = catalog.get(last_hours=int(last_hours[0]), with_summary_data=True)
                        else:
                            data, consumption = catalog.get(start=loc_data[0], with_summary_data=True)
                else:
                    data, consumption = catalog.get(start=catalog.min_ts, with_summary_data=True)
                if args.info and data is not None and not data.empty:
                    show_info_data(data, consumption)
                if (args.plot and data is not None and not data.empty and consumption is not None and
                        not consumption.empty):
                    from enerpi.enerplot import plot_power_consumption_hourly

                    img_name = plot_power_consumption_hourly(data.power, consumption.kWh, ldr=data.ldr,
                                                             rs_potencia=None, rm_potencia=60, savefig=args.plot)
                    log('Image stored in "{}"'.format(img_name), 'ok', args.verbose, True)
        # Shows database info
        else:  # Shows last 10 entries
            last = get_ts_last_save(path_st, get_last_sample=True, verbose=True, n=10)
            print(last)
    # Shows & extract info from LOG File
    elif args.log:
        data_log = extract_log_file(FILE_LOGGING, extract_temps=True, verbose=True)
        try:
            df_temps = data_log[data_log.temp.notnull()].dropna(axis=1).drop(['tipo', 'msg', 'debug_send'], axis=1)
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
        _run_logger(is_demo=True, verbose=args.verbose, ts_data=args.delta, debug=args.debug)
    # Receiver
    else:  # elif args.receive:
        log('{}\n   {}'.format(PRETTY_NAME, DESCRIPTION), 'ok', args.verbose, False)
        receiver(verbose=args.verbose, debug=args.debug)


def enerpi_main_logger(set_logging=False):
    if set_logging:
        set_logging_conf(verbose=False)
    # Shows RPI Temps
    # Timer(3, show_pi_temperature, args=(3,)).start()
    _run_logger(is_demo=False, verbose=False,
                delta_sampling=DELTA_SEC_DATA, roll_time=RMS_ROLL_WINDOW_SEC, sampling_ms=TS_DATA_MS, debug=False)


def enerpi_main_logger_demo():
    _run_logger(is_demo=True, verbose=True, ts_data=DELTA_SEC_DATA, debug=False)


if __name__ == '__main__':
    enerpi_main_cli()
