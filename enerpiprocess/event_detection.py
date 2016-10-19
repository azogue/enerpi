# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
from scipy.signal import wiener  # , medfilt, firwin
import seaborn as sns

from enerpi.api import enerpi_data_catalog
from enerpi.base import timeit
from enerpiprocess.numba_routines import (_rect_smoothing, _std, _peak_interval,
                                          _fusion_big_events, _new_interval_event_groups)
from enerpiprocess.processplots import plot_intervalos
from prettyprinting import *


MARGEN_ABS = 50
ROLL_WINDOW_STD = 7

PATH_TRAIN_DATA_STORE = '/Users/uge/Dropbox/PYTHON/PYPROJECTS/enerpi/enerpiprocess/train.h5'
TZ = pytz.timezone('Europe/Madrid')
FS = (16, 10)


def _print_types(*args):
    str_out = ''
    for a in args:
        if type(a) is np.ndarray:
            str_out += 'A[{}, ({})]; '.format(a.dtype, len(a))
        else:
            str_out += '[{}]; '.format(a.__class__)
    print(str_out)


# Catálogo y lectura de todos los datos.
@timeit('LOAD ALL DATA', verbose=True)
def load_data():
    cat = enerpi_data_catalog()
    data, data_s = cat.get_all_data()
    print_info(data_s.tail())

    POWER = pd.DataFrame(data.power).tz_localize(TZ)
    print_cyan(POWER.describe().T.astype(int))

    homog_power = POWER.resample('1s').mean().fillna(method='bfill', limit=3).fillna(-1) #.round().astype('int16')
    return data, data_s, POWER, homog_power


@timeit('get_train_data', verbose=True)
def get_train_data(homog_power=None):
    """
    TRAIN DATA: (event detection)
    Subconjunto continuo para entrenamiento:
    De '2016-09-08' a '2016-09-21' (2 semanas completas)
    """
    if homog_power is None:
        try:
            train = pd.read_hdf(PATH_TRAIN_DATA_STORE, 'power')
        except Exception as e:
            print_warn('No se puede cargar ENERPI POWER TRAIN DATA, se regenera y archiva en: {}'
                       .format(PATH_TRAIN_DATA_STORE))
            _, _, _, homog_power = load_data()
            train = homog_power.loc['2016-09-08':'2016-09-21'].copy()
            print_info('* Head:\n{}\n* Tail:\n{}\n* Describe:\n{}\nValores nulos:\n{}'
                       .format(train.head(3), train.tail(3), train.describe().T,
                               train[train.power==-1].groupby(lambda x: x.date).count()))
            train.to_hdf(PATH_TRAIN_DATA_STORE, 'power')
    else:
        train = homog_power.loc['2016-09-08':'2016-09-21'].copy()
    return train


@timeit('get_subsets', verbose=True)
def get_subsets(train_ev):
    # Subset of train data:
    df_1 = train_ev.loc['2016-09-10'].between_time('8:30', '10:30')
    df_2 = train_ev.loc['2016-09-08'].between_time('8:00', '14:00')
    t0, tf = '2016-09-10 19:30', '2016-09-10 23:40'
    df_3 = train_ev.loc[t0:tf]
    df_4 = train_ev.loc['2016-09-18']
    df_5 = train_ev.loc['2016-09-15']
    df_6 = train_ev.loc['2016-09-16']
    df_7 = train_ev.loc['2016-09-17']
    return [df_1, df_2, df_3, df_4, df_5, df_6, df_7]


@timeit('filter_power_data', verbose=True)
def filter_power_data(train_ev, kernel_size_wiener=15):
    """

    :param train_ev:
    :param kernel_size_wiener:
    :return:
    """
    train_ev = train_ev.copy()
    train_ev['wiener'] = wiener(train_ev.power, kernel_size_wiener).round()  # .astype(np.int)
    return train_ev


@timeit('rect_smoothing', verbose=True)
def rect_smoothing(df, roll_window_std_mean=ROLL_WINDOW_STD, margen_abs=MARGEN_ABS, name='wiener'):
    """

    :param df:
    :param roll_window_std_mean:
    :param margen_abs:
    :param name:
    :return:
    """
    num_total = len(df)
    delta_wiener = (df[name] - df[name].shift()).fillna(0)

    shift_roll = -(roll_window_std_mean // 2 + roll_window_std_mean % 2)
    roll = df[name].rolling(roll_window_std_mean, center=True)
    r_mean = roll.mean().shift(shift_roll).fillna(method='ffill').round().values
    r_std = roll.std().shift(shift_roll).fillna(method='ffill').values

    control_int = np.zeros(3, dtype=np.int)
    control_float = np.zeros(2, dtype=np.float)
    is_start_event = np.zeros(num_total, dtype=np.bool)
    change, step_med = np.zeros(num_total, dtype=np.float), np.zeros(num_total, dtype=np.float)

    _rect_smoothing(control_int, control_float, step_med, change, is_start_event,
                    df[name].values,
                    (delta_wiener.abs() > margen_abs).values,
                    r_mean, r_std,
                    delta_wiener.shift(-1).fillna(0).values,
                    # delta_wiener.rolling(7).sum().shift(-6).fillna(0).values)
                    delta_wiener.rolling(9).sum().shift(-8).fillna(0).values)

    df_smooth = pd.DataFrame({name: df[name], 'step_median': step_med, 'is_init': is_start_event},
                             columns=[name, 'step_median', 'is_init'], index=df.index)
    df_smooth['interv_raw'] = df_smooth['is_init'].cumsum().astype(int)
    return df_smooth


@timeit('groupby_intervalos', verbose=True)
def groupby_intervalos(df_step, label_gb='interv_raw', label_raw_big='big_event', label_level='level'):
    """

    :param df_step:
    :param label_gb:
    :param label_raw_big:
    :param label_level:
    :return:
    """
    # TODO std de intervalo init-center-end
    # def _tramos_std(x):
    #     n = x.shape[0]
    #     if n < 10:
    #         return ()

    gb = df_step.tz_convert('UTC').reset_index().groupby(label_gb)

    cols = df_step.columns
    cols_first = ['step_median', 'ts']
    cols_first += list(cols[cols.str.contains('interv') & ~cols.str.contains(label_gb)])
    cols_first += list(cols[cols.str.contains('level') & ~cols.str.contains(label_level)])

    interv_first = gb[cols_first].first().rename(columns=dict(ts='ts_ini'))

    how_wiener = {'n': 'count',
                  'mean_all': 'mean',
                  'median_all': 'median',
                  'std_all': 'std',
                  # 'mean_all': lambda x: _mean(x.values),
                  # 'median_all': lambda x: _median(x.values),
                  # 'std_all': lambda x: _std(x.values),
                  'peak': lambda x: _peak_interval(x.values),
                  'std_0': lambda x: _std(x[:5].values),
                  'std_c': lambda x: _std(x[3:-3].values),
                  'std_f': lambda x: _std(x[5:].values)}
    interv_wiener = gb['wiener'].agg(how_wiener)

    concatenar = [interv_first,
                  interv_first['step_median'].diff().fillna(0).rename('delta'),
                  interv_wiener,
                  gb.ts.last().rename('ts_fin')]
    if label_raw_big in df_step:
        concatenar += [gb[label_raw_big].any().astype(bool)]
    if 'level' in df_step:
        concatenar += [gb['level'].min().astype(int).rename(label_level)]

    df_interv = pd.DataFrame(pd.concat(concatenar, axis=1))
    return df_interv


@timeit('genera_df_intervalos', verbose=True)
def genera_df_intervalos(df_step):
    """

    :param df_step:
    :return:
    """

    def _append_fusion_big_events(df_intervalos, columns):
        # Big events:
        num_total = len(df_intervalos)
        levels = np.zeros(num_total, dtype=np.int)
        intervalo = np.zeros(num_total, dtype=np.int)
        big_events = np.zeros(num_total, dtype=np.bool)
        _fusion_big_events(levels, big_events, intervalo,
                           df_intervalos['n'].values, df_intervalos['delta'].values, df_intervalos['peak'].values,
                           df_intervalos['median_all'].values, df_intervalos['std_all'].values,
                           df_intervalos['std_0'].values, df_intervalos['std_c'].values, df_intervalos['std_f'].values)
        for c, arr in zip(columns, [levels, big_events, intervalo]):
            df_interv[c] = arr
        return df_interv

    def _append_column_ts(df, df_intervalos, col, c_ini='ts_ini', c_fin='ts_fin'):
        df_interv_usar = df_intervalos.reset_index()
        s = pd.Series(pd.concat([df_interv_usar[[c_ini, col]].set_index(c_ini)[col].tz_convert(TZ),
                                 df_interv_usar[[c_fin, col]].set_index(c_fin)[col].tz_convert(TZ)])).sort_index()
        df[col] = s
        df[col] = df[col].fillna(method='ffill').astype(s.dtype)
        return df

    def _agrupa_eventos(intervalos, len_intervalos, is_big_event, n_max_intersticio=30, frac_min_intersticio=1.2):
        np.testing.assert_equal(intervalos.shape, len_intervalos.shape)
        np.testing.assert_equal(intervalos.shape, is_big_event.shape)

        new_intervalos = np.ones(intervalos.shape[0], dtype=bool)
        num_intervalos = np.max(intervalos) - np.min(intervalos) + 1
        result_big_events = np.zeros(num_intervalos, dtype=bool)
        len_events = np.zeros(num_intervalos, dtype=int)
        ini_events = np.zeros(num_intervalos, dtype=int)

        _new_interval_event_groups(new_intervalos, result_big_events, len_events, ini_events,
                                   intervalos, len_intervalos, is_big_event,
                                   n_max_intersticio, frac_min_intersticio)
        return new_intervalos.cumsum()

    # 1º resumen de intervalos. Groupby por intervalos "raw"
    df_interv = groupby_intervalos(df_step, label_gb='interv_raw')

    # Detección de big_events y fusión de consecutivos y con intersticios (breves pausas):
    cols_fusion = ('level', 'big_event', 'intervalo')
    df_interv = _append_fusion_big_events(df_interv, cols_fusion)
    df_interv['interv_group'] = _agrupa_eventos(df_interv[cols_fusion[2]].values,
                                                df_interv['n'].values,
                                                df_interv[cols_fusion[1]].values,
                                                n_max_intersticio=60, frac_min_intersticio=3)

    # Append columnas de agrupación a time-series original:
    df_step = _append_column_ts(df_step, df_interv, cols_fusion[0])
    df_step = _append_column_ts(df_step, df_interv, cols_fusion[1])
    df_step = _append_column_ts(df_step, df_interv, cols_fusion[2])
    df_step = _append_column_ts(df_step, df_interv, 'interv_group')

    # 2º resumen de intervalos. Group-by por intervalos 'agrupados':
    df_interv_group = groupby_intervalos(df_step, label_gb='interv_group', label_level='level_group')
    df_interv = df_interv.join(df_interv_group.reset_index().set_index('interv_raw')[['level_group']])
    df_interv['level_group'] = df_interv['level_group'].fillna(method='ffill')
    df_step = _append_column_ts(df_step, df_interv, 'level_group')

    return df_step, df_interv, df_interv_group


@timeit('test_interval_detection', verbose=True)
def test_interval_detection(regen_train_data=False):
    """
    Detección de eventos y agrupación de los mismos en el subconjunto de entrenamiento.
     - LOAD/REGEN Subconjunto continuo para entrenamiento. De '2016-09-08' a '2016-09-21' (2 semanas completas)
     - Filtrado Wiener
     - Detección de intervalos mediante 'genera_df_intervalos' sobre 'rect_smoothing' del filtrado Wiener.
    Devuelve 3 dataframes: df_step, intervalos_raw, df_interv_group. La 1º comparte índice con la secuencia de
    entrenamiento y posee columnas asociadas al intervalo o al nivel detectado de cada instante. Las 2ª y 3ª son
    resúmenes (group-by-s sobre el intervalo) de los intervalos detectados, en bruto, y agrupados, respectivamente.

    :param regen_train_data: Bool para regenerar el intervalo de entrenamiento
    :return: df_step, intervalos_raw, df_interv_group
    """

    if regen_train_data:
        _data, _data_s, _POWER, homog_power = load_data()
        train = get_train_data(homog_power)
    else:
        train = get_train_data()

    train_ev = filter_power_data(train, kernel_size_wiener=15)

    df_subset = train_ev
    # df_subset = train_ev.loc['2016-09-10':'2016-09-14']
    # df_subset = train_ev.loc['2016-09-10']
    # df_subset = train_ev.loc['2016-09-18':'2016-09-21']

    return genera_df_intervalos(rect_smoothing(df_subset))


def _filter_time(df, t0='2016-09-11 12:15', tf='2016-09-11 15:30', label='ts_ini'):
    return df[(df[label] < tf) & (df[label] > t0)]


def _divide_big_events_en_tramos_para_representacion(df_interv_gr, delta_grupo='1h', delta_plot='3min', verbose=True):
    """
    Agrupamiento de big_events para representación
    """
    delta_grupo = pd.Timedelta(delta_grupo)
    delta_plot = pd.Timedelta(delta_plot)
    df_solo_big = df_interv_gr[df_interv_gr.big_event]
    grupos_con_big_events, grupo = [], []
    inicio = tf_ant = start = df_solo_big.ts_ini[0]
    fin = df_solo_big.ts_fin[-1]
    for i, t0, tf in zip(df_solo_big.index, df_solo_big.ts_ini, df_solo_big.ts_fin):
        if (start + delta_grupo < t0) and (tf_ant + delta_grupo / 10 < t0):
            grupos_con_big_events.append([grupo[0], grupo[-1],
                                          df_interv_gr.loc[grupo[0], 'ts_ini'], df_interv_gr.loc[grupo[-1], 'ts_fin']])
            grupo = [i]
            start = t0
        else:
            grupo.append(i)
        tf_ant = tf
    grupos_con_big_events.append([grupo[0], grupo[-1],
                                  df_interv_gr.loc[grupo[0], 'ts_ini'], df_interv_gr.loc[grupo[-1], 'ts_fin']])
    if verbose:
        print_ok("* Los eventos 'importantes' se dividen en {} grupos, para {} días, de {:%d-%b'%y} a {:%d-%b'%y}"
                .format(len(grupos_con_big_events), (fin - inicio).days, inicio, fin))
    intervs_plot = [df_interv_gr.loc[g[0] - 2:g[1] + 1] for g in grupos_con_big_events]
    xlims_plot = [(g[2] - delta_plot, g[3] + delta_plot) for g in grupos_con_big_events]
    return intervs_plot, xlims_plot, grupos_con_big_events


@timeit('test_export_svgs_groups_interval_detection', verbose=True)
def test_export_svgs_groups_interval_detection(df_subset, df_interv_group,
                                               num_plots_fig=9, export_figs=True, show_fig=False):
    """
    Generación de gráficos y exportación a ficheros 'svg' de intervalos detectados. Requiere como input la salida
    de la función 'test_interval_detection' o similar.
    - Divide los intervalos en tramos, prestando atención únicamente a los 'eventos grandes', con objeto de realizar
    múltiples plots centrados en las zonas con grandes eventos.
    - Se generan figuras con 'num_plots_fig' cada una, y se les da un nombre mediante la plantilla:
        'big_events_detection_{:%Y%m%d}_int{:04d}_to_int{:04d}.svg'
    - Opcionalmente, se muestra cada fig en pantalla (show_fig=True)

    """

    intervs_plot, xlims_plot, big_events_plot = _divide_big_events_en_tramos_para_representacion(df_interv_group)

    N = len(intervs_plot)
    for i in range(N // num_plots_fig + (1 if N % num_plots_fig != 0 else 0)):
        next_g = min(N, (i+1)*num_plots_fig)
        intervs_plot_i = intervs_plot[i*num_plots_fig:next_g]
        if export_figs:
            img_name = os.path.join(p, 'big_events_detection_{:%Y%m%d}_int{:04d}_to_int{:04d}.svg'
                                    .format(intervs_plot_i[0].ts_ini[0],
                                            intervs_plot_i[0].index[0], intervs_plot_i[-1].index[-1]))
        else:
            img_name = None
        plot_intervalos(intervs_plot_i, df_subset, with_level=True, major_fmt='%H:%M',
                        xlim=xlims_plot[i*num_plots_fig:next_g], img_name=img_name)
        if show_fig:
            fig = plt.gcf()
            fig.tight_layout()
            plt.show()
    return True


def show_interval_detection(df_step, intervalos_raw, df_interv_group, verbose=True):
    """
    Shows results
    """
    if verbose:
        print_red(intervalos_raw.dtypes)
        print_red(intervalos_raw.describe())
        print_magenta(df_interv_group.describe())
        print_info(df_step.describe())

        print_cyan(intervalos_raw.head())
        print_ok(df_interv_group.head())

    # Show
    # plot_intervalos(df_interv_group, df_step, with_raw_scatter=False, size=12)

    d_tramo_1 = dict(t0='2016-09-11 12:15', tf='2016-09-11 15:30')
    d_tramo_2 = dict(t0='2016-09-11 19:00', tf='2016-09-11 20:30')

    plot_intervalos([_filter_time(intervalos_raw, **d_tramo_1),
                     _filter_time(df_interv_group, **d_tramo_1),
                     _filter_time(intervalos_raw, **d_tramo_2),
                     _filter_time(df_interv_group, **d_tramo_2)],
                    df_step, with_raw_scatter=False, size=4)
    fig = plt.gcf()
    fig.tight_layout()
    plt.show()
    return True


def save_interval_detection_data(df_subset, df_interv, df_interv_group, file='debug_step_detection.h5'):
    """
    # SAVE DATA
    :return:
    """
    p_debug = os.path.join(p, file)
    with open(p_debug, 'w'):
        pass
    df_subset.to_hdf(p_debug, 'data')
    df_interv.to_hdf(p_debug, 'interv')
    df_interv_group.to_hdf(p_debug, 'interv_group')
    print_yellow('Guardados los resultados del "interval_detection" en {}. Tamaño: {:.2f}'
                 .format(p_debug, os.path.getsize(p_debug) / 1e6))
    return True


def load_interval_detection_data(file='debug_step_detection.h5'):
    """
    # LOAD RESULTS DATA
    :return: df_subset, df_interv, df_interv_group
    """
    p_debug = os.path.join(p, file)
    df_subset = pd.read_hdf(p_debug, 'data')
    df_interv = pd.read_hdf(p_debug, 'interv')
    df_interv_group = pd.read_hdf(p_debug, 'interv_group')
    return df_subset, df_interv, df_interv_group


if __name__ == '__main__':
    # Conf
    import os

    pd.set_option('display.width', 240)
    sns.set_style('ticks')
    p = os.path.dirname(__file__)

    df_subset, df_interv, df_interv_group = test_interval_detection(regen_train_data=False)

    # test_export_svgs_groups_interval_detection(df_subset, df_interv_group, export_figs=True, show_fig=False)
    # test_export_svgs_groups_interval_detection(df_subset, df_interv_group, export_figs=False, show_fig=True)

    # save_interval_detection_data(df_subset, df_interv, df_interv_group, file='debug_step_detection.h5')

    # df_subset, df_interv, df_interv_group = load_interval_detection_data(file='debug_step_detection.h5')
    show_interval_detection(df_subset, df_interv, df_interv_group, verbose=True)

