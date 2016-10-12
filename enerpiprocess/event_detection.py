# -*- coding: utf-8 -*-
import locale
from itertools import cycle
from math import sqrt, exp

import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import seaborn as sns
from enerpi.api import enerpi_data_catalog
from enerpi.base import timeit
from enerpiplot.enerplot import write_fig_to_svg, tableau20
from numba import jit
from prettyprinting import *
from scipy.signal import wiener  # , medfilt, firwin


# NUMBA BUG http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
#   RuntimeError: Failed at nopython (nopython mode backend)
#   LLVM will produce incorrect floating-point code in the current locale
# it means you have hit a LLVM bug which causes incorrect handling of floating-point constants.
# This is known to happen with certain third-party libraries such as the Qt backend to matplotlib.
locale.setlocale(locale.LC_NUMERIC, 'C')

MARGEN_ABS = 50
ROLL_WINDOW_STD = 7

FRAC_PLANO_INTERV = .9  # fracción de elementos planos (sin ∆ apreciable) en el intervalo
N_MIN_FRAC_PLANO = 10  # Nº mínimo de elems en intervalo para aplicar FRAC_PLANO_INTERV
P_OUTLIER = .95
P_OUTLIER_SAFE = .99

MIN_EVENT_T = 7
# MIN_VALIDOS = 3
# MIN_STD = 20

PATH_TRAIN_DATA_STORE = '/Users/uge/Dropbox/PYTHON/PYPROJECTS/enerpi/notebooks/train.h5'
TZ = pytz.timezone('Europe/Madrid')
FS = (16, 10)


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


# TRAIN DATA: (event detection)
@timeit('process_data_for_event_detection', verbose=True)
def process_data_for_event_detection(train, kernel_size_wiener=15, roll_window_std_mean=ROLL_WINDOW_STD, verbose=False):
    train_ev = train.copy()
    train_ev['wiener'] = wiener(train_ev.power, kernel_size_wiener)
    train_ev['delta_wiener'] = (train_ev.wiener - train_ev.wiener.shift()).fillna(0)
    train_ev['abs_ch'] = train_ev['delta_wiener'].abs() > MARGEN_ABS

    roll = train_ev['wiener'].rolling(roll_window_std_mean, center=True)
    shift_roll = -(roll_window_std_mean // 2 + roll_window_std_mean % 2)

    train_ev['r_std'] = roll.std().shift(shift_roll).fillna(method='ffill')
    train_ev['r_mean'] = roll.mean().shift(shift_roll).fillna(method='ffill')
    if verbose:
        train_ev.info()
        print_ok(train_ev.head())
    return train_ev


@timeit('get_subsets', verbose=True)
def get_subsets(train_ev):
    # train_ev = process_data_for_event_detection(train)
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


@jit('f8(f8)', nopython=True, cache=True)
def _phi(x):
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911
    sign = 1
    if x < 0:
        sign = -1
    x = abs(x) / sqrt(2)
    # A&S formula 7.1.26
    t = 1. / (1. + p * x)
    y = 1. - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * exp(-x * x)
    return .5 * (1. + sign * y)


@jit('f8(f8,f8,f8)', nopython=True, cache=True)
def _cdf(p, mean, std):
    return _phi((p - mean) / std)


@jit(['f8(f8[:])', 'f4(f4[:])'], nopython=True, cache=True)
def _std(x):
    s0 = x.shape[0]
    s1 = s2 = 0.
    for i in range(s0):
        if x[i] != np.nan:
            s1 += x[i]
            s2 += x[i] * x[i]
        else:
            s0 -= 1
    if s0 > 0:
        return sqrt((s0 * s2 - s1 * s1)/(s0 * s0))
    else:
        return 0.


@jit(['f4(f4[:])', 'f8(f8[:])'], nopython=True, cache=True)
def _mean(X):
    return np.mean(X)


@jit(['f4(f4[:])', 'f8(f8[:])'], nopython=True, cache=True)
def _median(X):
    return np.median(X)


@jit('b1(f8,f8)', nopython=True, cache=True)
def _es_outlier(pnorm, p_limit=P_OUTLIER):
    if (pnorm < 1. - p_limit) or (pnorm > p_limit):
        return True
    return False


@jit('b1(f8,f8,f8,f8)', nopython=True, cache=True)
def _cambio_futuro(std1, std2, m1, m2):
    d, dc = abs(m1 - m2), abs(std1 - std2)
    if ((dc > 100.) or (dc / ((std1 + std2) / 2.) > 5.)) or ((d > 1000.) or (d / ((m1 + m2) / 2.) > 1.)):
        return True
    return False


@jit('b1(b1,f8,f8,f8,f8,f8,f8,f8,f8)', nopython=True, cache=True)
def _detect_event_change(hay_ch_abs, incr, next_incr,
                         mean_final, std_final, next_mean, next_std,
                         pnorm, pnorm_next):
    # Lógica de detección de eventos:
    next_dif_mean_supera_std = (abs(mean_final - next_mean) > std_final) and (std_final > 5)

    es_outlier_safe = _es_outlier(pnorm, P_OUTLIER_SAFE)
    es_outlier = es_outlier_safe or _es_outlier(pnorm, P_OUTLIER_SAFE)
    next_es_outlier_safe = _es_outlier(pnorm_next, P_OUTLIER_SAFE)

    hay_cambio_futuro = _cambio_futuro(std_final, next_std, mean_final, next_mean)
    hay_incr_considerar = (abs(incr) > 10) or (abs(incr + next_incr) > 15)

    vip = ((hay_ch_abs and next_dif_mean_supera_std and es_outlier and next_es_outlier_safe)
           or ((hay_incr_considerar or hay_cambio_futuro) and es_outlier_safe and next_es_outlier_safe)
           or (next_dif_mean_supera_std and es_outlier_safe and next_es_outlier_safe))
    return vip


# @jit('i8,i8,i8,f8,f8(f8[:],i8[:],b1,b1,i8,i8,f8,i8,b1,f8,f8,i8,f8,f8,f8,f8,i8[:],i8[:],b1)', nopython=True)
@jit(nopython=True, cache=True)
def _process_instant(calc_values, change, hay_ch_abs, hay_ch_min, i, idx_ch_ant, incr, ini_int, is_start_event,
                     last_std, last_step, n, next_incr, next_mean, next_std, p, step, step_med, sufficient_event):
    v_planos = calc_values[ini_int:i][np.nonzero(change[ini_int:i] == 0)[0]]
    n_planos = v_planos.shape[0]
    if (n_planos > 2 * ROLL_WINDOW_STD) and (n > N_MIN_FRAC_PLANO) and (n_planos / n > FRAC_PLANO_INTERV):
        std_final = _std(v_planos[-(2 * ROLL_WINDOW_STD):])
        mean_final = _mean(v_planos[-(2 * ROLL_WINDOW_STD):])
        err_dist_f = std_final / sqrt(len(v_planos[-(2 * ROLL_WINDOW_STD):]))
    elif (n_planos > 3) and (n_planos / n > .7):
        std_final = _std(v_planos)
        mean_final = _mean(v_planos)
        err_dist_f = std_final / sqrt(n_planos)
    else:
        std_final = _std(calc_values[ini_int + 2:i])
        mean_final = _mean(calc_values[ini_int + 2:i])
        err_dist_f = std_final / sqrt(n - 2)

    # Condición de intervalo diferente al anterior para k y k+1:
    p1, p2, m, s = round(p), round(p + next_incr), round(mean_final), round(min(150, std_final + err_dist_f), 1)
    pnorm = _cdf(p1, m, s)
    pnorm_next = _cdf(p2, m, s)
    # Lógica de detección de eventos:
    vip = _detect_event_change(hay_ch_abs, incr, next_incr, mean_final, std_final,
                               next_mean, next_std, pnorm, pnorm_next)
    if sufficient_event and vip:
        change[i] = vip

        new_last_step = _mean(calc_values[ini_int + 2:i - 1])
        new_last_std = _std(calc_values[ini_int + 2:i - 1])
        new_median = _median(calc_values[ini_int + 2:i - 1])

        last_step_integrate = _mean(calc_values[idx_ch_ant + 2:i - 1])
        last_std_integrate = _std(calc_values[idx_ch_ant + 2:i - 1])
        last_median_integrate = _median(calc_values[idx_ch_ant + 2:i - 1])

        # Condición de cierre de intervalo junto al anterior o diferente:
        cambio_median = np.abs(new_median - last_median_integrate) / last_median_integrate > .05

        if (not cambio_median and (abs(np.max(calc_values[ini_int + 2:i]) - last_step_integrate) < 200) and
                (abs(last_std - last_std_integrate) < 10) and
                (abs(last_step - last_step_integrate) < 15) and
                (abs(new_last_step - last_step_integrate) / last_step_integrate < .1)):
            is_start_event[ini_int] = False
            step[idx_ch_ant:i] = int(last_step_integrate)
            step_med[idx_ch_ant:i] = int(last_median_integrate)
            is_start_event[i] = True
            last_step, last_std = last_step_integrate, last_std_integrate
            ini_int = i
            n = 1
        else:
            step[ini_int:i] = int(new_last_step)
            step_med[ini_int:i] = int(_median(calc_values[ini_int:i - 1]))
            is_start_event[i] = True
            idx_ch_ant, last_step, last_std = ini_int, new_last_step, new_last_std
            ini_int = i
            n = 1
    else:
        change[i] = hay_ch_min
        n += 1
    return ini_int, n, idx_ch_ant, last_step, last_std


@jit('(i8[:],i8[:],i1[:],b1[:],f8[:],b1[:],f8[:],f8[:],f8[:],f8[:])', nopython=True, cache=True)
def _rect_smoothing(step_med, step, change, is_start_event,
                    calc_values, abs_ch, r_mean, r_std, delta_shift_1, delta_shift_2):
    N = calc_values.shape[0]
    idx_ch_ant = ini_int = last_step = last_std = n = 0
    for i in range(N):
        (p, hay_ch_abs, incr, next_incr,
         next_std, next_mean) = (calc_values[i], abs_ch[i], delta_shift_1[i], delta_shift_2[i], r_std[i], r_mean[i])
        sufficient_event = n > MIN_EVENT_T
        hay_ch_min = (abs(incr) > 15) or (abs(incr + next_incr) > 20)
        if i == 0:
            n += 1
        elif i ==  - 3:
            change[i] = hay_ch_min
            step[ini_int:] = int(_mean(calc_values[ini_int:]))
            step_med[ini_int:] = int(_median(calc_values[ini_int:]))
        elif not sufficient_event:
            change[i] = hay_ch_min
            n += 1
        else:
            ini_int, n, idx_ch_ant, last_step, last_std = _process_instant(
                calc_values, change, hay_ch_abs, hay_ch_min, i, idx_ch_ant, incr, ini_int,
                is_start_event, last_std, last_step, n, next_incr, next_mean, next_std, p,
                step, step_med, sufficient_event)
    # return step_med, step, change, is_start_event


# Detección de intervalos
@timeit('rect_smoothing', verbose=True)
def rect_smoothing(df):
    N = len(df)
    is_start_event = np.zeros(N, dtype=np.bool)
    change = np.zeros(N, dtype=np.int8)
    step = np.zeros(N, dtype=np.int)
    step_med = np.zeros(N, dtype=np.int)
    _rect_smoothing(step_med, step, change, is_start_event,
                    df.wiener.values, df.abs_ch.values, df.r_mean.values, df.r_std.values,
                    df.delta_wiener.shift(-1).values, df.delta_wiener.rolling(9).sum().shift(-8).values)
    return pd.DataFrame({'step_median': step_med, 'step_mean': step, 'ch': change, 'is_init': is_start_event},
                        columns=['step_median', 'step_mean', 'ch', 'is_init'], index=df.index)


# @jit
def _std_nonzero(x):
    N = x.shape[0]
    # idx_act = np.nonzero(x.ch == 0)[0]

    # interv.ix[np.nonzero(x.ch)[0][-1] + 1:].wiener.std()

    idx_act = np.nonzero(x.ch)[0]
    if len(idx_act) == 0:
        # print_red(x.ch)
        return _std(x.wiener.values)
    elif idx_act[-1] + 1 < N:
        v_usar = x.wiener.values[idx_act[-1] + 1:]
        # print(N, idx_act[-1] + 1, len(v_usar))
        return _std(v_usar)
    else:
        # print(N, idx_act, '\n', x)
        return -1


def _valid_ch(interv_ch):
    idx_act = np.nonzero(interv_ch)[0]
    if (len(idx_act) == 0) or (idx_act[-1] + 1 < len(interv_ch)):
        return True
    else:
        return False


@timeit('genera_df_intervalos', verbose=True)
def genera_df_intervalos(df_data, df_step, verbose=False):
    train_step_t = df_data[['wiener']].join(df_step)
    # train_step_t['interv_simple'] = train_step_t.is_init.cumsum().shift(-1).fillna(method='ffill').astype(int)
    train_step_t['interv_simple'] = train_step_t.is_init.cumsum().fillna(method='ffill').astype(int)
    gb = train_step_t.tz_convert('UTC').reset_index().groupby('interv_simple')
    steps = gb.step_median.first()

    df_interv = pd.DataFrame(pd.concat(
        [gb.ch.apply(lambda x: np.sum(np.abs(x))).rename('n_ch'),
         gb.wiener.count().rename('n'),
         gb.wiener.median().rename('median_all').round(),
         gb.wiener.std().rename('std_all').round(1),
         steps.rename('step_median_0'),
         steps.rename('delta') - steps.shift().fillna(steps.ix[0]).values,
         gb.ts.first().rename('ts_ini'),
         gb.ts.last().rename('ts_fin'),
         gb.wiener.apply(lambda x: _std(x[:5].values)).rename('std_0'),
         gb.wiener.apply(lambda x: _std(x[3:-3].values)).rename('std_c'),
         gb.wiener.apply(lambda x: _std(x[5:].values)).rename('std_f')], axis=1))
    if verbose:
        print_magenta(df_interv)
    return df_interv


@jit(['b1(f8,f8,f8,f8,f8)', 'b1(f4,f4,f4,f4,f4)'], nopython=True, cache=True)
def _condition_big_event(d_acum, delta, std_all, median_all, last_level):
    if (d_acum > 500) or (delta > 500) or ((std_all > 100) and (d_acum > 50)) or (median_all - last_level > 200):
        return True
    return False


@timeit('fusion_big_events', verbose=True)
def _fusion_big_events(df_interv, verbose=False):
    grupos, g_i = [], []
    levels = []
    ant_vip = False
    d_acum = 0
    level = df_interv.median_all[0]
    for i, row in df_interv.iterrows():
        if verbose:
            print_red('{:5} -> {:%H:%M:%S}, d_acum: {:.0f}; delta: {:.0f}; '
                      'std_all: {:.0f}; median_all: {:.0f}; level: {:.0f}. G_i={}'
                      .format(i, row.ts, d_acum, row.delta, row.std_all, row.median_all, level, g_i))
        if _condition_big_event(d_acum, row.delta, row.std_all, row.median_all, level):
            if not ant_vip and (len(g_i) == 0):
                level = row.median_all - row.delta
                d_acum = row.delta
                if verbose:
                    print_info('NEW {} con median_all={:.0f}, delta={:.0f} --> level = {:.0f}'
                               .format(i, row.median_all, row.delta, level))
                if len(g_i) > 0:
                    if verbose:
                        print_info('se cierra anterior: {}'.format(g_i))
                    grupos.append(g_i)
                g_i = [i]
            else:
                d_acum += row.delta
                if abs(d_acum) < 100:
                    if len(g_i) > 0:
                        if verbose:
                            print_cyan('se cierra por d_acum={}, {}'.format(d_acum, g_i))
                        grupos.append(g_i)
                    g_i = []
                    level = row.median_all  # - row.delta
                    d_acum = 0
                else:
                    if verbose:
                        print_red('se sigue acumulando {} en {}, d_acum={}'.format(i, g_i, d_acum))
                    g_i.append(i)
            ant_vip = True
        else:
            if len(g_i) > 0:
                # g_i.append(i)
                grupos.append(g_i)
                if verbose:
                    print_red('se cierra {}, level_ant={:.0f}, ∆={:.0f}, new_level={:.0f}'
                              .format(g_i, level, row.delta, row.median_all))
            g_i = []
            level = row.median_all
            d_acum = 0
            ant_vip = False
        levels.append(level)
    df_interv['level'] = levels
    if len(g_i) > 0:
        grupos.append(g_i)
    if verbose:
        print_red(grupos)
    return df_interv, grupos


@timeit('integrate_step_detection', verbose=True)
def _integrate_step_detection(df, df_step, df_interv, verbose=False):
    """
    Concatena la información de la detección de escalones (eventos) y numera los intervalos.
    """
    df_out = df[['power', 'wiener']].join(df_step[['step_median', 'step_mean', 'is_init']]).tz_convert('UTC')
    df_out['init_event'] = df_out['is_init'] # .shift(-1).fillna(0).astype(bool)
    df_out['big_event'] = False

    # Big events:
    df_interv, grupos = _fusion_big_events(df_interv)
    # df_interv['ts_fin'] = df_interv['ts'].shift(-1).fillna(df.index[-1])
    df_interv['big_event'] = False
    for i, idx in enumerate(grupos):
        df_interv.loc[idx, 'big_event'] = True
        for idxi in idx[1:]:
            df_out.loc[df_interv.loc[idxi, 'ts_ini'], 'init_event'] = False
        df_out.loc[df_interv.loc[idx[0], 'ts_ini'], 'big_event'] = True

    df_out['intervalo'] = df_out['init_event'].cumsum()
    if verbose:
        print_magenta(df_out.tz_convert(TZ))
    return df_out.tz_convert(TZ)


def plot_steps_detection(df_data, df_step):
    ax = df_data.wiener.plot(figsize=FS, lw=.5, alpha=.3, color=tableau20[0])
    # (df_step.is_init.abs() * 1000).plot(ax=ax, lw=.75, alpha=.3, color=tableau20[6])
    ax.vlines(df_step[df_step.is_init].index, 0, 500, lw=.75, alpha=.3, color=tableau20[6])
    df_step.step_median.plot(ax=ax, lw=1, alpha=.9, color=tableau20[4])
    # (df_step.ch.abs() * 500).plot(ax=ax, lw=.75, alpha=.3, color=tableau20[6])
    # df_debug.mean_planos.plot(ax=ax, lw=.75, alpha=.7, color=tableau20[8])
    # df_debug.error_cum.plot(ax=ax, lw=.5, alpha=.9, color=tableau20[2])
    ax.xaxis.set_major_formatter(mpd.DateFormatter('%H:%M:%S', tz=TZ))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax.xaxis.set_tick_params(labelsize=9, pad=3)
    return ax


def _plot_intervals(df_out, with_fill_events=True, with_raw_scatter=True, with_vlines=True):
    """Plot intervalos y big_events por separado, con distintos colores"""
    dp = df_out.reset_index().set_index('intervalo')
    f, ax = plt.subplots(1, 1, figsize=(16, 8))
    # cm = cycle(mpc.get_cmap('viridis').colors[::4])
    cm = cycle(tableau20[::2])
    color_small_ev = 'grey'
    for i in set(dp.index):
        df_i = dp.loc[i].set_index('ts')
        if not df_i.empty:
            is_big = dp.loc[i, 'big_event'].sum() > 0
            color = next(cm) if is_big else 'k'
            lw = 1.25 if is_big else .5
            alpha = .8 if is_big else .5
            legend = 'Ev{:3d}'.format(i) if is_big else ''
            if with_vlines:
                ax.vlines(df_i[df_i.is_init].index, 0, df_i[df_i.is_init].step_median * 2,
                          lw=.25, alpha=.5, color=color_small_ev, label='')
            if is_big:
                if with_fill_events:
                    ax.fill_between(df_i.index, df_i.wiener, y2=0, lw=0, alpha=.2, color=color, label='')
                if with_raw_scatter:
                    ax.scatter(df_i.index, df_i.power, s=10, lw=0, alpha=.4, c=color, label='')
            elif with_raw_scatter:
                ax.scatter(df_i.index, df_i.power, s=6, lw=0, alpha=.3, c=color_small_ev, label='')
            df_i.wiener.plot(ax=ax, lw=lw, alpha=alpha, color=color, label=legend)
    ax.set_ylim((0, dp.wiener.max() + 200))
    ax.xaxis.tick_bottom()
    ax.xaxis.set_major_formatter(mpd.DateFormatter('%H:%M:%S', tz=TZ))
    xl = ax.get_xlim()
    if xl[1] - xl[0] < .25:
        ax.xaxis.set_minor_locator(mpd.MinuteLocator(tz=TZ))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha='center')
    ax.xaxis.set_tick_params(labelsize=9, pad=3)
    plt.legend(loc='best')
    return ax


@timeit('test_interval_detection', verbose=True)
def test_interval_detection(regen_train_data=False):
    """
    # LOAD DATA Subconjunto continuo para entrenamiento. De '2016-09-08' a '2016-09-21' (2 semanas completas)

    * timming:
    get_train_data TOOK: 0.058 s
    process_data_for_event_detection TOOK: 0.331 s
    rect_smoothing TOOK: 1.393 s
    genera_df_intervalos TOOK: 0.855 s
    fusion_big_events TOOK: 0.203 s
    integrate_step_detection TOOK: 18.759 s
    test_interval_detection TOOK: 21.416 s


    :param regen_train_data:
    :return:
    """
    if regen_train_data:
        _data, _data_s, _POWER, homog_power = load_data()
        train = get_train_data(homog_power)
    else:
        train = get_train_data()

    train_ev = process_data_for_event_detection(train, kernel_size_wiener=15, verbose=False)

    df_subset = train_ev.loc['2016-09-10':'2016-09-15']
    print_info(df_subset.count())

    df_step = rect_smoothing(df_subset)
    print_cyan(df_step.count())

    # INTERVALOS:
    df_interv = genera_df_intervalos(df_subset, df_step)
    print_red(df_interv.count())
    # print_ok(df_interv.head(15))
    # print_cyan(df_interv.tail(15))

    # BIG EVENTS
    df_out = _integrate_step_detection(df_subset, df_step, df_interv)
    print_magenta(df_out.count())

    # Show
    # plot_steps_detection(df, df_step)
    _plot_intervals(df_out, with_raw_scatter=False, with_vlines=False)
    fig = plt.gcf()
    fig.tight_layout()
    write_fig_to_svg(fig, 'test_interval_detection.svg')
    # plt.show()


if __name__ == '__main__':
    # Conf
    pd.set_option('display.width', 240)
    sns.set_style('ticks')

    test_interval_detection(regen_train_data=False)

    # # LOAD Subconjunto continuo para entrenamiento. De '2016-09-08' a '2016-09-21' (2 semanas completas)
    # # _data, _data_s, _POWER, homog_power = load_data()
    # # train = get_train_data(homog_power)
    # train = get_train_data()
    # train_ev = process_data_for_event_detection(train, kernel_size_wiener=15, verbose=False)
    #
    # df_1, df_2, df_3, df_4, df_5, df_6, df_7 = get_subsets(train_ev)
    #
    # # Detección de intervalos
    # lista_dfs = [df_1.between_time('8:51', '9:02'), df_2.between_time('13:45', '15:30'), df_2, df_3, df_4]
    # # # lista_dfs = [df_2.between_time('13:45', '15:30')]
    # lista_dfs += [df_5, df_6, df_7]
    # # lista_dfs = [df_5, df_6, df_7]
    # # lista_dfs = [df_5.between_time('9:00', '13:00'), df_5]
    # # # lista_dfs = [df_7.between_time('08:00', '23:00')]
    # # # lista_dfs = [df_7.between_time('10:40', '12:30')]
    # # lista_dfs = [train_ev]
    #
    # for df in lista_dfs:
    #     df_step = rect_smoothing(df)
    #
    #     # INTERVALOS:
    #     df_interv = genera_df_intervalos(df, df_step)
    #     # print_ok(df_interv.head(15))
    #     # print_cyan(df_interv.tail(15))
    #
    #     # BIG EVENTS
    #     df_out = _integrate_step_detection(df, df_step, df_interv)
    #
    #     # Show
    #     # plot_steps_detection(df, df_step)
    #     _plot_intervals(df_out)
    #     plt.gcf().tight_layout()
    #     plt.show()



