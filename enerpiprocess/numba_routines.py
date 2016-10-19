# -*- coding: utf-8 -*-
"""
Colección de rutinas 'tipadas' para su ejecución mediante el decorator jit de NUMBA.
* Fácilmente transformables en pseudo-c para su compilación mediante CYTHON,
  si son requeridas en RPI (sin LLVM usable para utilizar numba, por ahora)

La optimización es muy apreciable. Se pone como ejemplo la detección de intervalos-eventos y agrupación de los mismos
de una secuencia de datos de potencia eléctrica en W (ENERPI data -> power).
    - Se aplica un filtro de Wiener a la señal y se detectan los 'momentos de cambio' asociados a la conexión
    o desconexión de un aparato eléctrico.
    - Se agrupa la señal entre esos momentos de cambio, generando una tabla de intervalos con diversas mediciones.
    - Se clasifican los intervalos (generación de 'big_events') y se fusionan eventos importantes consecutivos
    y con sus intersticios (breves pausas).

* Para secuencias de 1, 5 y 14 días (a 86400 samples /día, Ts=1s).
(get_train_data y filter_power_data se aplican siempre sobre la secuencia completa de 14 días)

- Sin NUMBA:
    get_train_data TOOK: 0.070 s
    filter_power_data TOOK: 0.171 s
    rect_smoothing TOOK: 7.115 s
    groupby_intervalos TOOK: 0.207 s
    groupby_intervalos TOOK: 0.181 s
    genera_df_intervalos TOOK: 0.432 s
    test_interval_detection TOOK: 8.817 s

    get_train_data TOOK: 0.063 s
    filter_power_data TOOK: 0.178 s
    rect_smoothing TOOK: 35.094 s
    groupby_intervalos TOOK: 1.012 s
    groupby_intervalos TOOK: 0.910 s
    genera_df_intervalos TOOK: 2.028 s
    test_interval_detection TOOK: 37.366 s

    get_train_data TOOK: 0.062 s
    filter_power_data TOOK: 0.172 s
    rect_smoothing TOOK: 118.619 s
    groupby_intervalos TOOK: 2.728 s
    groupby_intervalos TOOK: 2.628 s
    genera_df_intervalos TOOK: 5.653 s
    test_interval_detection TOOK: 124.510 s

- *** Con NUMBA ***:
    get_train_data TOOK: 0.066 s
    filter_power_data TOOK: 0.178 s
    rect_smoothing TOOK: 0.412 s
    groupby_intervalos TOOK: 0.093 s
    groupby_intervalos TOOK: 0.066 s
    genera_df_intervalos TOOK: 0.203 s
    test_interval_detection TOOK: 1.888 s

    get_train_data TOOK: 0.058 s
    filter_power_data TOOK: 0.188 s
    rect_smoothing TOOK: 1.865 s
    groupby_intervalos TOOK: 0.390 s
    groupby_intervalos TOOK: 0.361 s
    genera_df_intervalos TOOK: 0.871 s
    test_interval_detection TOOK: 2.986 s

    get_train_data TOOK: 0.057 s
    filter_power_data TOOK: 0.176 s
    rect_smoothing TOOK: 5.519 s
    groupby_intervalos TOOK: 1.187 s
    groupby_intervalos TOOK: 0.953 s
    genera_df_intervalos TOOK: 2.418 s
    test_interval_detection TOOK: 8.176 s

La optimización llega a ser ~ 20x, lo cuál no está nada mal para un simple 'decorator'
"""

from math import sqrt, exp
import numpy as np


ROLL_WINDOW_STD = 7
FRAC_PLANO_INTERV = .9  # fracción de elementos planos (sin ∆ apreciable) en el intervalo
N_MIN_FRAC_PLANO = 10  # Nº mínimo de elems en intervalo para aplicar FRAC_PLANO_INTERV
P_OUTLIER = .95
P_OUTLIER_SAFE = .99

MIN_EVENT_T = 7

JIT_CACHE = True
USE_NUMBA = True


# from time import time
def dummy_jit(*args_dec, **kwargs_dec):
    def _real_deco(function):
        def _wrapper(*args, **kwargs):
            # tic = time()
            out = function(*args, **kwargs)
            # print('{} TOOK: {:.3f} s. args={}, kwargs={}'
            #       .format(function.__name__, time() - tic, args_dec, kwargs_dec))
            return out
        return _wrapper
    return _real_deco


# NUMBA BUG http://numba.pydata.org/numba-doc/dev/user/faq.html#llvm-locale-bug
#   RuntimeError: Failed at nopython (nopython mode backend)
#   LLVM will produce incorrect floating-point code in the current locale
# it means you have hit a LLVM bug which causes incorrect handling of floating-point constants.
# This is known to happen with certain third-party libraries such as the Qt backend to matplotlib.
# locale.setlocale(locale.LC_NUMERIC, 'C')
try:
    from numba import jit
    # import locale
except ImportError:
    USE_NUMBA = False


if not USE_NUMBA:
    jit = dummy_jit

# @jit('f8[:](f4[:],i8)', cache=JIT_CACHE)
# def _wiener_filter(values, kernel_size_wiener):
#     filtered = wiener(values, kernel_size_wiener)
#     return filtered


# def _print_types(*args):
#     str_out = ''
#     for a in args:
#         if type(a) is np.ndarray:
#             str_out += 'A[{}, ({})]; '.format(a.dtype, len(a))
#         else:
#             str_out += '[{}]; '.format(a.__class__)
#     print(str_out)


@jit('f8(f8)', nopython=True, cache=JIT_CACHE)
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


@jit('f8(f8,f8,f8)', nopython=True, cache=JIT_CACHE)
def _cdf(p, mean, std):
    return _phi((p - mean) / (std + .001))


@jit(['f8(f8[:])'], nopython=True, cache=JIT_CACHE)
def _peak_interval(x):
    return np.max(x) - np.min(x)


@jit(['f8(f8[:])'], nopython=True, cache=JIT_CACHE)
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


@jit(['f8(f8[:])'], nopython=True, cache=JIT_CACHE)
def _mean(X):
    return round(np.mean(X))


@jit(['f8(f8[:])'], nopython=True, cache=JIT_CACHE)
def _median(X):
    return round(np.median(X))


@jit('b1(f8,f8)', nopython=True, cache=JIT_CACHE)
def _es_outlier(pnorm, p_limit=P_OUTLIER):
    if (pnorm < 1. - p_limit) or (pnorm > p_limit):
        return True
    return False


@jit('b1(f8,f8,f8,f8)', nopython=True, cache=JIT_CACHE)
def _cambio_futuro(std1, std2, m1, m2):
    d, dc = abs(m1 - m2), abs(std1 - std2)
    if ((dc > 100.) or (dc / ((std1 + std2 + .0001) / 2.) > 5.)) or (d > 1000.) or (d / ((m1 + m2) / 2.) > 1.):
        return True
    return False


@jit(['b1(f8,f8,f8,f8,f8,f8,f8,f8,f8)', 'b1(f4,f4,f4,f4,f4,f4,f4,f4,f4)'], nopython=True, cache=JIT_CACHE)
def _condition_big_event(d_acum, delta, std_all, std_0, std_c, std_f, median_all, peak, last_level):
    if ((d_acum > 300) or (delta > 250) or
            ((peak > 400) and (std_0 + std_c + std_f > 300) and (abs(delta) > 200)) or
            ((std_all > 100) and (d_acum > 50)) or
            ((std_all > 50) and (std_0 + std_c + std_f > 250)) or
            (median_all - last_level > 200)):
        return True
    return False


@jit(['(i8[:],b1[:],i8[:], i8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:],f8[:])'], nopython=True, cache=JIT_CACHE)
def _fusion_big_events(levels, big_events, intervalo, len_intervs, delta,
                       peak, median_all, std_all, std_0, std_c, std_f):
    num_total = delta.shape[0]
    level_ant = median_all[0]
    ant_vip = False
    d_acum = 0
    d_acum_max = 0
    num_intervalo = -1
    for i in range(num_total):
        (len_intervs_i, delta_i, std_all_i, std_0_i, std_c_i, std_f_i, median_all_i, peak_i
         ) = len_intervs[i], delta[i], std_all[i], std_0[i], std_c[i], std_f[i], median_all[i], peak[i]
        if _condition_big_event(d_acum, delta_i, std_all_i,
                                std_0_i, std_c_i, std_f_i, median_all_i, peak_i, level_ant):
            if not ant_vip:
                ant_vip = True
                d_acum = delta_i
                new_level = median_all_i - delta_i
                # Fusión de mini-events de subida delante de eventos + grandes (el inicio de los peaks!)
                if (i > 1) and (not big_events[i - 2]) and (len_intervs[i - 1] < 10) and (std_f[i - 1] > 100):
                    new_level = median_all[i - 1] - delta[i - 1]
                    d_acum += delta[i - 1]
                    num_intervalo -= 1
                    levels[i - 1] = int(new_level)
                    big_events[i - 1] = True
                # Creación de nuevo big_event
                num_intervalo += 1
                levels[i] = int(new_level)
                big_events[i] = True
                d_acum_max = d_acum
            else:
                # Acumulación o cierre en current big_event
                d_acum += delta_i
                d_acum_max = max(d_acum, d_acum_max)
                # if d_acum < 0:
                #     print('* D_acum negativa (={:.0f}): I_{}, num_total={}, Peak={:.0f}, ∆={:.0f},
                # std0={:.1f}, stdc={:.1f}, stdf={:.1f}'
                #           .format(d_acum, i, len_intervs_i, peak_i, delta_i, std_0_i, std_c_i, std_f_i))
                #     # Se corrige el nivel del big_event:
                #     idx_corregir = intervalo == num_intervalo
                #     level_ant += d_acum
                #     level_ant = int(level_ant)
                #     levels[idx_corregir] = level_ant
                if (abs(d_acum) < 150) or (d_acum < -100) or (d_acum < .1 * d_acum_max):
                    # Cierre de big_event
                    levels[i] = int(median_all_i)
                    ant_vip = False
                    # Eventos de tamaño mediano de gran std & peak --> pasan a ser big_events
                    if ((len_intervs_i < 500) and (peak_i > 500) and
                            (abs(delta_i) > 200) and (std_0_i + std_c_i + std_f_i > 300) and (std_c_i > 50)):
                        big_events[i] = True
                        levels[i] = level_ant
                        ant_vip = True
                    elif levels[i] < level_ant:
                        idx_corregir = intervalo == num_intervalo
                        level_ant = int(levels[i])
                        levels[idx_corregir] = level_ant
                    d_acum = 0
                    num_intervalo += 1
                else:
                    levels[i] = level_ant
                    big_events[i] = True
                    ant_vip = True
        else:
            levels[i] = median_all_i
            d_acum = 0
            ant_vip = False
            num_intervalo += 1
        intervalo[i] = num_intervalo
        level_ant = levels[i]


@jit('b1(b1,f8,f8,f8,f8,f8,f8,f8,f8)', nopython=True, cache=JIT_CACHE)
def _detect_event_change(hay_ch_abs, incr, next_incr,
                         mean_final, std_final, next_mean, next_std,
                         pnorm, pnorm_next):
    # Lógica de detección de eventos:
    next_dif_mean_supera_std = (abs(mean_final - next_mean) > std_final) and (std_final > 5)

    es_outlier_safe = _es_outlier(pnorm, P_OUTLIER_SAFE)
    es_outlier = es_outlier_safe or _es_outlier(pnorm, P_OUTLIER_SAFE)
    next_es_outlier_safe = _es_outlier(pnorm_next, P_OUTLIER_SAFE)

    hay_cambio_futuro = _cambio_futuro(std_final, next_std, mean_final, next_mean)
    # TODO Revisar condición repetida
    # hay_incr_considerar = (abs(incr) > 10) or (abs(incr + next_incr) > 15)
    hay_incr_considerar = (abs(incr) > 15) or (abs(incr + next_incr) > 20)

    vip = ((hay_ch_abs and next_dif_mean_supera_std and es_outlier and next_es_outlier_safe)
           or ((hay_incr_considerar or hay_cambio_futuro) and es_outlier_safe and next_es_outlier_safe)
           or (next_dif_mean_supera_std and es_outlier_safe and next_es_outlier_safe))
    return vip


@jit('(i8[:],f8[:],i8,f8[:],f8[:],b1,b1,f8,b1[:],f8,f8,f8,f8,f8[:],b1)', nopython=True, cache=JIT_CACHE)
def _process_instant(control_int, control_float, i,
                     calc_values, change, hay_ch_abs, hay_ch_min,
                     incr, is_start_event,
                     next_incr, next_mean, next_std, p, step_med, sufficient_event):
    ini_int, n, idx_ch_ant = control_int[:]
    last_step, last_std = control_float[:]
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
        cambio_median = np.abs(new_median - last_median_integrate) / (last_median_integrate + .0001) > .05

        if (not cambio_median and (abs(np.max(calc_values[ini_int + 2:i]) - last_step_integrate) < 200) and
                (abs(last_std - last_std_integrate) < 10.) and
                (abs(last_step - last_step_integrate) < 15.) and
                (abs(new_last_step - last_step_integrate) / (last_step_integrate + .0001) < .1)):
            is_start_event[ini_int] = False
            step_med[idx_ch_ant:i] = last_median_integrate
            is_start_event[i] = True
            last_step, last_std = last_step_integrate, last_std_integrate
            ini_int = i
            n = 1
        else:
            step_med[ini_int:i] = _median(calc_values[ini_int:i - 1])
            is_start_event[i] = True
            idx_ch_ant, last_step, last_std = ini_int, new_last_step, new_last_std
            ini_int = i
            n = 1
    else:
        change[i] = hay_ch_min
        n += 1
    control_int[:] = ini_int, n, idx_ch_ant
    control_float[:] = last_step, last_std


@jit('(b1[:],b1[:],i8[:],i8[:], i8[:],i8[:],b1[:], i8,f8)', nopython=True, cache=JIT_CACHE)
def _new_interval_event_groups(new_intervalos, result_big_events, len_events, ini_events,
                               intervalos, len_intervalos, is_big_event,
                               n_max_intersticio, frac_min_intersticio):
    num_total = intervalos.shape[0]
    current_interv = intervalos[0]
    is_big_current = False
    ini_current = len_current = 0
    new_intervalos[0] = False

    for i in range(num_total):
        if intervalos[i] != current_interv:
            # Cambio de intervalo
            result_big_events[current_interv] = is_big_current
            len_events[current_interv] = len_current
            ini_events[current_interv] = ini_current

            # Fusión de intersticios y big_events seguidos, para rellenar 'new_intervalos' en current_interv - 1:
            if current_interv > 1:
                i_fusion = current_interv - 1
                len_fusion = len_events[i_fusion]
                big_event_fusion = result_big_events[i_fusion]
                big_event_fusion_ant = result_big_events[i_fusion - 1]
                big_event_fusion_next = result_big_events[i_fusion + 1]

                # Fusión de big_events consecutivos o fusión de intersticios (eventos pequeños entre eventos grandes)
                if ((big_event_fusion and big_event_fusion_ant) or
                        (not big_event_fusion and
                         (len_fusion < n_max_intersticio) and
                         (float(len_events[i_fusion - 1]) * frac_min_intersticio > float(len_fusion)) and
                         (float(len_events[i_fusion + 1]) * frac_min_intersticio > float(len_fusion)) and
                         big_event_fusion_ant and big_event_fusion_next)):
                    anula = ini_events[i_fusion]
                    new_intervalos[anula] = False
                    result_big_events[i_fusion] = True

            current_interv += 1
            is_big_current = is_big_event[i]
            len_current = len_intervalos[i]
            ini_current = i
        else:
            # Acumula en intervalo
            if not is_big_current:
                is_big_current = is_big_event[i]
            len_current += len_intervalos[i]
            new_intervalos[i] = False
    result_big_events[current_interv] = is_big_current
    len_events[current_interv] = len_current


@jit('(i8[:],f8[:],f8[:],f8[:],b1[:], f8[:],b1[:],f8[:],f8[:],f8[:],f8[:])', nopython=True, cache=JIT_CACHE)
def _rect_smoothing(control_int, control_float, step_med, change, is_start_event,
                    calc_values, abs_ch, r_mean, r_std, delta_shift_1, delta_shift_2):
    num_total = calc_values.shape[0]
    idx_ch_ant = ini_int = n = last_step = last_std = 0
    for i in np.arange(num_total):
        (p, hay_ch_abs, incr, next_incr, next_std, next_mean
         ) = (calc_values[i], abs_ch[i], delta_shift_1[i], delta_shift_2[i], r_std[i], r_mean[i])
        sufficient_event = n > MIN_EVENT_T
        hay_ch_min = (abs(incr) > 15) or (abs(incr + next_incr) > 20)
        if i == 0:
            n += 1
        elif i == - 3:
            change[i] = hay_ch_min
            # step[ini_int:] = _mean(calc_values[ini_int:])
            step_med[ini_int:] = _median(calc_values[ini_int:])
        elif not sufficient_event:
            change[i] = hay_ch_min
            n += 1
        else:
            control_int[:] = ini_int, n, idx_ch_ant
            control_float[:] = last_step, last_std
            _process_instant(control_int, control_float, i,
                             calc_values, change, hay_ch_abs, hay_ch_min, incr,
                             is_start_event, next_incr, next_mean, next_std, p,
                             step_med, sufficient_event)
            ini_int, n, idx_ch_ant = control_int[:]
            last_step, last_std = control_float[:]
