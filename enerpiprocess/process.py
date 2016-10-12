# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import random
import pysolar
import datetime as dt
import pytz
import matplotlib.pyplot as plt
# import matplotlib.dates as mpd
from enerpi.api import enerpi_data_catalog
from enerpi.base import timeit
from prettyprinting import *


LAT, LONG = 38.631463, -0.866402
TZ = pytz.timezone('Europe/Madrid')
# AZIMUT RANGE: 0 --> -360 sentido horario, comienza en SUR. W=-90, N=-180, E=-270, S=0/-360


def _azimut_S(azi):
    if azi + 180 < 0:
        return -(360 + azi)
    return -azi


# Plot day
def _get_rday(data, verbose=True):
    t0 = data.index[0].date()
    tf = data.index[-1].date()
    delta = tf - t0
    rd = random.randrange(0, delta.days + 1, 1)
    day = t0 + dt.timedelta(days=rd)
    if verbose:
        print_cyan('Seleccionado día: {}'.format(day))
    return data.loc[day.strftime('%Y-%m-%d'):day.strftime('%Y-%m-%d')]


def _alt_azi(d):
    return pysolar.solar.get_altitude(LAT, LONG, d), pysolar.solar.get_azimuth(LONG, LAT, d)


@timeit('get_solar_day', verbose=True)
def get_solar_day(day=dt.datetime.today(), step_minutes=5, tz=TZ, step_calc_seg=30):
    if (type(day) is pd.DataFrame) or (type(day) is pd.Series):
        df_sol = pd.DataFrame(day)

        sunrise, sunset = pysolar.util.get_sunrise_sunset(LAT, LONG, df_sol.index[0])
        df_sol.loc[(sunrise > df_sol.index) | (sunset < df_sol.index), 'altitud'] = 0

        idx_solar = df_sol.index[df_sol['altitud'].isnull()][::step_calc_seg]
        alts = [pysolar.solar.get_altitude(LAT, LONG, d) for d in idx_solar]
        # azims = [pysolar.solar.get_azimuth(LAT, LONG, d) for d in idx_solar]
        azims = pd.Series([_azimut_S(pysolar.solar.get_azimuth(LAT, LONG, d)) for d in idx_solar], index=idx_solar)
        irrads = [pysolar.radiation.get_radiation_direct(d, alt) if alt > 0 else 0 for d, alt in
                  zip(idx_solar, alts)]
        df_sol.loc[idx_solar, 'altitud'] = alts
        df_sol.loc[idx_solar, 'azimut'] = azims
        df_sol.loc[idx_solar, 'irradiacion_cs'] = irrads
        df_sol = df_sol.interpolate().fillna(0)
    else:
        day = tz.localize(day).replace(hour=0, minute=0, second=0, microsecond=0)
        tt = [day + dt.timedelta(minutes=m) for m in range(0, 24 * 60, step_minutes)]
        df_sol = pd.DataFrame([_alt_azi(d) for d in tt], columns=['altitud', 'azimut'], index=tt)
        df_sol.altitud = df_sol.altitud.where(df_sol.altitud > 0, other=0)
        df_sol['irradiacion_cs'] = df_sol.apply(lambda x: pysolar.radiation.get_radiation_direct(x.name, x['altitud']),
                                                axis=1)
    return df_sol


@timeit('separa_ldr_artificial_natural', verbose=True)
def separa_ldr_artificial_natural(data_day, resample_inicial='2s', delta_roll_threshold=100):

    def _f_roll(x):
        delta_pos = np.sum(x[x > 0])
        delta_neg = np.sum(x[x < 0])
        return delta_pos + delta_neg

    def _last_on(x):
        if (np.sum(x) == 1.) & (x[-1] == 1.):
            return True
        return False

    # TODO Resolver NaN's
    data_analysis = pd.DataFrame(data_day['ldr'].resample(resample_inicial).mean())

    data_analysis['delta_roll'] = data_analysis.ldr.diff().fillna(0).rolling(3, center=True).apply(_f_roll).fillna(0)

    data_analysis['ch_state'] = 0
    data_analysis.loc[data_analysis[(data_analysis['delta_roll'] > delta_roll_threshold
                                     ).rolling(3).apply(_last_on).fillna(False) > 0].index, 'ch_state'] = 1
    data_analysis.loc[data_analysis[(data_analysis['delta_roll'] < -delta_roll_threshold
                                     ).rolling(3).apply(_last_on).fillna(False) > 0].index, 'ch_state'] = -1

    cambios = data_analysis[data_analysis['ch_state'] != 0]
    print_red(cambios)

    data_analysis['artif_level_max'] = 0
    apagados = data_analysis[data_analysis['ch_state'] == -1].index
    for i_enc in data_analysis[data_analysis['ch_state'] == 1].index:
        apagado = apagados[apagados > i_enc]
        if len(apagado) > 0:
            apagado = apagado[0]
            data_analysis.loc[i_enc:apagado, 'artif_level_max'] = data_analysis.ldr.loc[i_enc:apagado].max()
            print_red('Encendido L={:.0f}, SEGS={:.0f}'.format(data_analysis.ldr.loc[i_enc:apagado].max(),
                                                               (apagado - i_enc).total_seconds()))

    # Reconstrucción LDR natural:
    hay_artificial = data_analysis.artif_level_max > 0
    index_art = data_analysis[hay_artificial.shift(-2) | hay_artificial.shift(2)].index

    rs_natural = data_analysis.ldr.drop(index_art).resample('2min')
    data_analysis_simple = pd.concat([rs_natural.max().rename('nat_max').interpolate(),
                                      rs_natural.min().rename('nat_min').interpolate(),
                                      rs_natural.mean().rename('nat_mean').interpolate()], axis=1)
    # data_analysis_simple = data_analysis_simple.join(data_analysis[['altitud', 'azimut']])
    return data_analysis, data_analysis_simple


if __name__ == '__main__':
    # Catálogo y lectura de todos los datos.
    cat = enerpi_data_catalog()
    data, data_s = cat.get_all_data()
    LDR = pd.DataFrame(data.ldr).tz_localize(TZ)

    # PROCESS DATA
    # data_analysis = _get_rday(LDR)
    data_analysis = LDR.loc['2016-08-27':'2016-08-27']
    data_analysis, data_analysis_simple = separa_ldr_artificial_natural(data_analysis,
                                                                        resample_inicial='2s', delta_roll_threshold=100)
    data_analysis = get_solar_day(data_analysis)

    # Para plot:
    data_analysis.altitud *= 10

    # Reconstrucción LDR natural:
    hay_artificial = data_analysis.artif_level_max > 0
    index_art = data_analysis[hay_artificial.shift(-2) | hay_artificial.shift(2)].index

    ax = data_analysis.ldr.drop(index_art).resample('2min').max().interpolate().plot(figsize=(18, 8),
                                                                                     color='darkorange', lw=2)
    data_analysis.altitud.plot(ax=ax, color='green')
    data_analysis.ldr.plot(ax=ax, color='yellow', lw=1, alpha=.8)
    data_analysis.artif_level_max.plot(ax=ax, color='violet', lw=2)
    plt.show()

    ax = data_analysis_simple.plot(figsize=(18, 8), lw=1.5, alpha=.8)
    plt.show()
