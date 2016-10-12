# -*- coding: utf-8 -*-
import ephem
import math
import numpy as np
import pandas as pd
import pytz
from pysolar.solar import (get_altitude_fast, get_azimuth, get_declination, get_hour_angle,
                           get_projected_radial_distance, get_projected_axial_distance,
                           get_geocentric_latitude, get_geocentric_longitude, get_sun_earth_distance,
                           get_aberration_correction, get_equatorial_horizontal_parallax, get_nutation,
                           get_apparent_sidereal_time, get_true_ecliptic_obliquity, get_apparent_sun_longitude,
                           get_geocentric_sun_right_ascension, get_geocentric_sun_declination, get_local_hour_angle,
                           get_parallax_sun_right_ascension, get_topocentric_local_hour_angle,
                           get_topocentric_sun_declination, get_topocentric_elevation_angle, get_refraction_correction,
                           get_topocentric_azimuth_angle)
from pysolar import time as time
from pysolar.constants import standard_pressure, standard_temperature
from enerpi.base import timeit


TZ = pytz.timezone('Europe/Madrid')
LAT = 38.631463
LONG = -0.866402
ELEV_M = 500


# PYSOLAR Calc both altitude & azimut:
def _pysolar_fast_altitude_azimuth(latitude_deg, longitude_deg, when, fast=True, elevation=0,
                                   temperature=standard_temperature, pressure=standard_pressure):
    if fast:
        day = when.utctimetuple().tm_yday
        declination_rad = math.radians(get_declination(day))
        latitude_rad = math.radians(latitude_deg)
        hour_angle_rad = math.radians(get_hour_angle(when, longitude_deg))
        altitude_rad = math.radians(get_altitude_fast(latitude_deg, longitude_deg, when))
        try:
            azimuth_deg = math.degrees(math.asin(math.cos(declination_rad) * math.sin(hour_angle_rad)
                                                 / math.cos(altitude_rad)))
        except ValueError as e:
            print('Error {} in {}'.format(e, when))
            azimuth_deg = get_azimuth(latitude_deg, longitude_deg, when)
        if math.cos(hour_angle_rad) >= (math.tan(declination_rad) / math.tan(latitude_rad)):
            return math.degrees(altitude_rad), azimuth_deg
        else:
            return math.degrees(altitude_rad), 180 - azimuth_deg
    else:
        # location-dependent calculations
        projected_radial_distance = get_projected_radial_distance(elevation, latitude_deg)
        projected_axial_distance = get_projected_axial_distance(elevation, latitude_deg)

        # time-dependent calculations
        jd = time.get_julian_solar_day(when)
        jde = time.get_julian_ephemeris_day(when)
        jce = time.get_julian_ephemeris_century(jde)
        jme = time.get_julian_ephemeris_millennium(jce)
        geocentric_latitude = get_geocentric_latitude(jme)
        geocentric_longitude = get_geocentric_longitude(jme)
        sun_earth_distance = get_sun_earth_distance(jme)
        aberration_correction = get_aberration_correction(sun_earth_distance)
        equatorial_horizontal_parallax = get_equatorial_horizontal_parallax(sun_earth_distance)
        nutation = get_nutation(jce)
        apparent_sidereal_time = get_apparent_sidereal_time(jd, jme, nutation)
        true_ecliptic_obliquity = get_true_ecliptic_obliquity(jme, nutation)

        # calculations dependent on location and time
        apparent_sun_longitude = get_apparent_sun_longitude(geocentric_longitude, nutation, aberration_correction)
        geocentric_sun_right_ascension = get_geocentric_sun_right_ascension(apparent_sun_longitude,
                                                                            true_ecliptic_obliquity,
                                                                            geocentric_latitude)
        geocentric_sun_declination = get_geocentric_sun_declination(apparent_sun_longitude, true_ecliptic_obliquity,
                                                                    geocentric_latitude)
        local_hour_angle = get_local_hour_angle(apparent_sidereal_time, longitude_deg,
                                                geocentric_sun_right_ascension)
        parallax_sun_right_ascension = get_parallax_sun_right_ascension(projected_radial_distance,
                                                                        equatorial_horizontal_parallax,
                                                                        local_hour_angle,
                                                                        geocentric_sun_declination)
        topocentric_local_hour_angle = get_topocentric_local_hour_angle(local_hour_angle,
                                                                        parallax_sun_right_ascension)
        topocentric_sun_declination = get_topocentric_sun_declination(geocentric_sun_declination,
                                                                      projected_axial_distance,
                                                                      equatorial_horizontal_parallax,
                                                                      parallax_sun_right_ascension,
                                                                      local_hour_angle)
        topocentric_elevation_angle = get_topocentric_elevation_angle(latitude_deg, topocentric_sun_declination,
                                                                      topocentric_local_hour_angle)
        refraction_correction = get_refraction_correction(pressure, temperature, topocentric_elevation_angle)
        altitude = topocentric_elevation_angle + refraction_correction
        azimuth = 180 - get_topocentric_azimuth_angle(topocentric_local_hour_angle, latitude_deg,
                                                      topocentric_sun_declination)
        return altitude, azimuth


def _solar_azimuth(df_solar_pos, column='azimuth', is_pyephem=True, is_pysolar_fast=False):
    """Conversión de Azimut a 'azimut solar':= Sentido horario y 0º en SUR. Este en -90º y Oeste en 90º.
    Para pd.Series o pd.Dataframe, se devuelve el objeto con la columna de azimut modificada.
    Por defecto espera un azimut calculado por pyephem (Sentido horario y 0 en Norte). Pero tb lo hace para las
    orientaciones de pysolar (normal y fast).

    """
    if type(df_solar_pos) is pd.Series:
        column = df_solar_pos.name
    if is_pyephem:
        df_solar_pos[column] -= 180
    elif is_pysolar_fast:
        df_solar_pos[column] = -df_solar_pos[column].where(lambda x: x > 180, df_solar_pos[column] + 360) + 360
    else:
        df_solar_pos[column] = -df_solar_pos[column].where(lambda x: x < -180, df_solar_pos[column] - 360) - 360
    return df_solar_pos


@timeit('sun_position')
def sun_position(index,
                 latitude_deg=LAT, longitude_deg=LONG, elevation=0, observer=None,
                 delta_n_calc=1, south_orient=True, use_pysolar=False):
    """
    Cálculo de posición solar (altitud + azimut) mediante pyephem (+ rápido) o pysolar,
    para objetos pd.DatetimeIndex. Devuelve pd.DataFrame con columnas 'altitude', azimuth' y DatetimeIndex.
    Para añadir a DataFrame existente:
        df = df.join(sun_position(df.index, ...., delta_n_calc=step_calc))

    * Parámetros:
    :param index:          Pandas DatetimeIndex con tz_info.
    :param latitude_deg:   Latitud en grados (float)
    :param longitude_deg:  Longitud en grados (float)
    :param elevation:      Altura sobre el nivel del mar en metros (float). Default: 0
    :param observer:       (OPC) pyephem 'Observer' object (En vez de latitude_deg, longitude_deg, elevation)
    :param delta_n_calc:   ∆ de instantes de cálculo. Con delta_n_calc > 1,
                           equivale a calcular index[::delta_n_calc] y después interpolar intermedios. Default: 1
    :param south_orient:   Devuelve azimut solar (0 en SUR y sentido horario). Default: True
    :param use_pysolar:    Utiliza pysolar para calcular la posición solar, en vez de pyephem. Default: False

    ** Pyephem genera el azimut de 0 a 360 con 0 en Norte y sentido horario (E = 90º, S = 180º)
    ** Pysolar genera el azimut de forma caótica (distinta entre 'normal' y 'fast'!!). Usa orientación Sur y sentido
       antihorario, con la discontinuidad en 0º == -360º (Sur), pero en el método 'fast' en 270º==-90º ¿¿??!!

    :return df:            pd.DataFrame con columnas ['altitude', 'azimuth']
    """

    date_alt_azi = np.ones((len(index), 2)) * np.nan
    if use_pysolar:
        for i, t in enumerate(index):
            if i % delta_n_calc == 0:
                date_alt_azi[i, :] = _pysolar_fast_altitude_azimuth(latitude_deg, longitude_deg, t, fast=True,
                                                                    elevation=elevation,
                                                                    temperature=standard_temperature,
                                                                    pressure=standard_pressure)
                if south_orient and date_alt_azi[i, 1] > 180:
                    date_alt_azi[i, 1] = 360 - date_alt_azi[i, 1]
    else:
        sun = ephem.Sun()
        tt_utc = index.tz_convert('UTC')
        if observer is None:
            observer = ephem.Observer()
            observer.lat = str(latitude_deg)
            observer.lon = str(longitude_deg)
            observer.elevation = elevation
        for i, t in enumerate(tt_utc):
            if i % delta_n_calc == 0:
                observer.date = ephem.Date(t)
                sun.compute(observer)
                date_alt_azi[i, 0] = math.degrees(sun.alt)
                date_alt_azi[i, 1] = math.degrees(sun.az)
                if south_orient:
                    date_alt_azi[i, 1] -= 180
    df = pd.DataFrame(date_alt_azi, columns=['altitude', 'azimuth'], index=index)
    if delta_n_calc > 1:
        return df.interpolate()
    return df


if __name__ == '__main__':
    d_observer = dict(latitude_deg=LAT, longitude_deg=LONG, elevation=ELEV_M)

    str_day = '2016-08-12'
    tt = pd.DatetimeIndex(freq='1s', start=str_day, periods=60 * 24 * 60, tz=TZ)

    df_solar_1 = sun_position(tt, **d_observer, delta_n_calc=120, south_orient=False)
    df_solar_2 = sun_position(tt, **d_observer, delta_n_calc=120, south_orient=False, use_pysolar=True)

    df_solar_1 = _solar_azimuth(df_solar_1, column='azimuth', is_pyephem=True)
    df_solar_2 = _solar_azimuth(df_solar_2, column='azimuth', is_pyephem=False, is_pysolar_fast=True)

    df_solar_1 = sun_position(tt, **d_observer, delta_n_calc=120)
    df_solar_2 = sun_position(tt, **d_observer, delta_n_calc=120, use_pysolar=True)
