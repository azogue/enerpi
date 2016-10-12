# -*- coding: utf-8 -*-
import locale
import re
import subprocess
from time import time, sleep

import ephem
import matplotlib.dates as mpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pysolar
import pytz
import seaborn as sns
import statsmodels.api as sm
from pykeyboard import PyKeyboard
from scipy.signal import medfilt
from enerpi.api import enerpi_data_catalog
from enerpi.base import timeit
from enerpiplot.enerplot import write_fig_to_svg, tableau20
from numpy import pi as PI
from prettyprinting import *


sns.set_style('whitegrid')
KEYBOARD = PyKeyboard()
TZ = pytz.timezone('Europe/Madrid')
LAT = 38.631463
LONG = -0.866402
ELEV_M = 500
lang, codec = locale.getlocale()
use_locale = '{}.{}'.format(lang, codec)
locale.setlocale(locale.LC_ALL, use_locale)

FS = (16, 10)

REGEXPR_SVG_HEIGHT = re.compile(r'<svg height="\d{1,4}pt"')
REGEXPR_SVG_WIDTH = re.compile(r' width="(\d{1,4}pt")')

# Funciones de cálculo relacionadas con ganancias solares para el cálculo horario de EN 13790
i_CONSTANTE_SOLAR_E_0 = 1367  # W/m2
i_COS_OBLICUIDAD_ELIPTICA_RAD = .917477  # cosd(23.44º)
i_SIN_OBLICUIDAD_ELIPTICA_RAD = .397789  # sind(23.44º)

i_DURACION_ANYO_SOLAR = 365.24  # W/m2
i_DIAS_DESDE_SOLST_INV_A_DIA_1 = 10.
i_DIAS_DESDE_DIA_1_A_PERIHELIO = 2.

i_SIGMA_STEFAN_BOLTZMANN = 5.67e-8  # W/(m^2·K^4)

i_VALOR_FORM_FACT_CENIT_PEREZ_K_RAD = 1.041
i_VALOR_COS_85_MINIMO = 0.0871557
i_DEG_A_RAD = PI / 180.

i_EMISIVIDAD_OPACOS_CTE = .9
i_RESISTENCIA_SUPERFICIAL_EXTERIOR = .04
i_PREC_COMPARA = .01

# Tabla 6 (Modeling daylight availability and irradiance components from direct and global irradiance (Perez 1989-90))
# Para valores de epsilon 1, 2, 3, 4, 5, 6, 7, 8:
i_COEFS_MODELO_PEREZ_F11 = [-.008, .13, .33, .568, .873, 1.132, 1.06, .678]
i_COEFS_MODELO_PEREZ_F12 = [.588, .683, .487, .187, -.392, -1.237, -1.6, -.327]
i_COEFS_MODELO_PEREZ_F13 = [-.062, -.151, -.221, -.295, -.362, -.412, -.359, -.25]
i_COEFS_MODELO_PEREZ_F21 = [-.06, -.019, .055, .109, .226, .288, .264, .156]
i_COEFS_MODELO_PEREZ_F22 = [.072, .066, -.064, -.152, -.462, -.823, -1.127, -1.377]
i_COEFS_MODELO_PEREZ_F23 = [-.022, -.029, -.026, -.014, .001, .056, .131, .251]


def _cambia_foco_app_con_command_tab():
    KEYBOARD.press_key('command')
    KEYBOARD.tap_key('tab')
    KEYBOARD.release_key('command')


def i_julian_day_of_year(when):
    return when.dayofyear + when.hour / 24. + when.minute / (24. * 60) + when.second / (24. * 3600)


def i_comprueba_valor_irrad_W_m2(valor_irrad_W_m2):
    # pragma message("VERSION 6000 (\""__FILE__"\"): FERRAN, EUGENIO: Comentado hasta que se decida sobre CANARIAS")
    # UNREFERENCED(valor_irrad_W_m2)
    # assert(valor_irrad_W_m2 < 1500.)
    pass


def i_cos_angulo_incidencia(cenit_solar_deg, inclinacion_deg, azimut_efectivo):
    cos_ang_incidencia_calc = (np.cos(i_DEG_A_RAD * cenit_solar_deg) * np.cos(i_DEG_A_RAD * inclinacion_deg)
                               + np.sin(i_DEG_A_RAD * cenit_solar_deg) * np.sin(i_DEG_A_RAD * inclinacion_deg) * np.cos(i_DEG_A_RAD * azimut_efectivo))
    if cos_ang_incidencia_calc < 0.:
        cos_ang_incidencia_calc = 0.
    return cos_ang_incidencia_calc


def i_cos_cenit_acotado(cenit_solar_deg):
    cos_cenit_acotado = np.cos(i_DEG_A_RAD * cenit_solar_deg)
    if cos_cenit_acotado < i_VALOR_COS_85_MINIMO:
        cos_cenit_acotado = i_VALOR_COS_85_MINIMO
    return cos_cenit_acotado


def i_calcula_irradiacion_solar_maxima_sin_atm(dia_juliano):
    return i_CONSTANTE_SOLAR_E_0 * (1. + .034 * np.cos(2. * PI * dia_juliano / 365.25))


# Ecuation of time
def i_calcula_desviacion_eq_of_time_y_declinacion(dia_juliano):
    """PYSOLAR VERSION:" \
    "returns the number of minutes to add to mean solar time to get actual solar time."
    b = 2 * math.pi / 364.0 * (day - 81)
    return 9.87 * math.sin(2 * b) - 7.53 * math.cos(b) - 1.5 * math.sin(b)"""

    ang_anual_unit_rad = 2. * PI / i_DURACION_ANYO_SOLAR
    A = ang_anual_unit_rad * (dia_juliano + i_DIAS_DESDE_SOLST_INV_A_DIA_1)
    B = A + 2. * .0167 * np.sin(ang_anual_unit_rad * (dia_juliano - i_DIAS_DESDE_DIA_1_A_PERIHELIO))
    sin_declinacion = -i_SIN_OBLICUIDAD_ELIPTICA_RAD * np.cos(B)
    cos_declinacion = np.cos(np.arcsin(sin_declinacion))
    C = (A - np.arctan(np.tan(B) / i_COS_OBLICUIDAD_ELIPTICA_RAD)) / PI
    EoT_opc = 720. * (C - np.floor(C + 0.5 + 0.0000001))
    return sin_declinacion, cos_declinacion, EoT_opc


# Ecuation of time
def i_calcula_declinacion(dia_juliano):
    ang_anual_unit_rad = 2. * PI / i_DURACION_ANYO_SOLAR
    A = ang_anual_unit_rad * (dia_juliano + i_DIAS_DESDE_SOLST_INV_A_DIA_1)
    B = A + 2. * .0167 * np.sin(ang_anual_unit_rad * (dia_juliano - i_DIAS_DESDE_DIA_1_A_PERIHELIO))
    sin_declinacion = -i_SIN_OBLICUIDAD_ELIPTICA_RAD * np.cos(B)
    cos_declinacion = np.cos(np.arcsin(sin_declinacion))
    return sin_declinacion, cos_declinacion


def enhsol_calcula_altura_azimut_en_hora_solar(f_dia_anual, latitud_rad, con_correccion_eot):

    def i_angulo_horario_de_f_dia_anual(f_dia):
        hora_diaria = f_dia % 24.
        return PI * (hora_diaria / 12. - 1.)

    if con_correccion_eot:
        sin_declinacion, cos_declinacion, correccion_EoT_min = i_calcula_desviacion_eq_of_time_y_declinacion(f_dia_anual)
    else:
        correccion_EoT_min = 0
        sin_declinacion, cos_declinacion = i_calcula_declinacion(f_dia_anual)

    angulo_horario = i_angulo_horario_de_f_dia_anual(f_dia_anual)
    sin_alfa_s = np.sin(latitud_rad) * sin_declinacion + np.cos(latitud_rad) * cos_declinacion * np.cos(angulo_horario)
    cos_gamma_s = (sin_declinacion - sin_alfa_s * np.sin(latitud_rad)) / (np.cos(latitud_rad) * np.sqrt(1 - sin_alfa_s**2))

    altura = np.arcsin(sin_alfa_s) / i_DEG_A_RAD

    if altura > 0.:
        azimut = np.arccos(cos_gamma_s) / i_DEG_A_RAD

        if (angulo_horario < 0.):
            azimut = azimut - 180.
        else:
            azimut = 180. - azimut

        if (con_correccion_eot == True):
            azimut += correccion_EoT_min / 4.
    else:
        altura = 0.
        azimut = 0.

    return altura, azimut, 90. - altura  # cenit_solar_deg


def i_interpola_valor_coef_perez(coefs, epsilon, floor_epsilon):
    assert (floor_epsilon > 0)
    if epsilon > 8.:
        return coefs[7]
    elif epsilon - floor_epsilon > .01:
        return coefs[floor_epsilon - 1] + (coefs[floor_epsilon] - coefs[floor_epsilon - 1]) * (epsilon - floor_epsilon)
    else:
        return coefs[floor_epsilon - 1]


def i_asigna_coefs_ec_modelo_perez(epsilon):
    assert (epsilon > .9999)
    f11 = i_interpola_valor_coef_perez(i_COEFS_MODELO_PEREZ_F11, epsilon, np.int(np.floor(epsilon)))
    f12 = i_interpola_valor_coef_perez(i_COEFS_MODELO_PEREZ_F12, epsilon, np.int(np.floor(epsilon)))
    f13 = i_interpola_valor_coef_perez(i_COEFS_MODELO_PEREZ_F13, epsilon, np.int(np.floor(epsilon)))
    f21 = i_interpola_valor_coef_perez(i_COEFS_MODELO_PEREZ_F21, epsilon, np.int(np.floor(epsilon)))
    f22 = i_interpola_valor_coef_perez(i_COEFS_MODELO_PEREZ_F22, epsilon, np.int(np.floor(epsilon)))
    f23 = i_interpola_valor_coef_perez(i_COEFS_MODELO_PEREZ_F23, epsilon, np.int(np.floor(epsilon)))
    return f11, f12, f13, f21, f22, f23


# ASHRAE, Kasten and Young, 1989
def i_relative_optical_air_mass(cenit_solar_deg):
    assert (cenit_solar_deg > -.000000001)
    return 1. / (np.cos(i_DEG_A_RAD * cenit_solar_deg) + .50572 * pow(96.070995 - cenit_solar_deg, -1.6364))


def i_calcula_coefs_epsilon_delta_modelo_perez(
        cenit_solar_deg,
        irrad_directa_normal, irrad_difusa_h, E_0_z):
    assert (irrad_difusa_h > 0.)

    #  Cálculo epsilon (sky's clearness)
    fact_cenit = i_VALOR_FORM_FACT_CENIT_PEREZ_K_RAD * ((i_DEG_A_RAD * cenit_solar_deg) ** 3)
    epsilon = ((irrad_difusa_h + irrad_directa_normal) / irrad_difusa_h + fact_cenit) / (1. + fact_cenit)
    #  Cálculo AM (relative optical airmass)
    air_mass = i_relative_optical_air_mass(cenit_solar_deg)
    #  Cálculo Delta (sky brightness)
    assert (E_0_z > 0.)
    delta = (irrad_difusa_h * air_mass) / E_0_z
    return epsilon, delta


#  Perez Sky Diffuse Irradiance Model
def i_coeficientes_F1_F2_modelo_perez(f_dia_anual, cenit_solar_deg, irrad_directa_normal, irrad_difusa_h):
    E_0_z = i_calcula_irradiacion_solar_maxima_sin_atm(f_dia_anual)
    epsilon, delta = i_calcula_coefs_epsilon_delta_modelo_perez(
        cenit_solar_deg,
        irrad_directa_normal, irrad_difusa_h, E_0_z)

    f11, f12, f13, f21, f22, f23 = i_asigna_coefs_ec_modelo_perez(epsilon)

    F1_circumsolar_brightening_coef = f11 + f12 * delta + i_DEG_A_RAD * cenit_solar_deg * f13
    F1_circumsolar_brightening_coef = np.max(0., F1_circumsolar_brightening_coef)

    F2_horizon_brightening_coef = f21 + f22 * delta + i_DEG_A_RAD * cenit_solar_deg * f23
    return F1_circumsolar_brightening_coef, F2_horizon_brightening_coef


"""
def enhsol_factores_irradiacion_difusa(
                        desfase_horario,
                        latitud_edificio_grados,
                        const struct enhvalor_horas_anyo_t *irradiacion_directa_horiz,
                        const struct enhvalor_horas_anyo_t *irradiacion_difusa_horiz,
                        struct enhvalor_horas_anyo_t **F1_circumsolar_brightening,
                        struct enhvalor_horas_anyo_t **F2_horizon_brightening,
                        struct enhvalor_horas_anyo_t **azimut_solar,
                        struct enhvalor_horas_anyo_t **cenit_solar)
{
    unsigned long i

    assert_no_null(F1_circumsolar_brightening)
    assert_no_null(F2_horizon_brightening)

    *F1_circumsolar_brightening = enhvalor_inicia_horas_anyo_nulo()
    *F2_horizon_brightening = enhvalor_inicia_horas_anyo_nulo()

    *azimut_solar = enhvalor_inicia_horas_anyo_nulo()
    *cenit_solar = enhvalor_inicia_horas_anyo_nulo()

    for (i = 0i < ENHVALOR_NUM_HORAS_ANYO ++i)
    {
        irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora

        irradiacion_directa_horiz_hora = enhvalor_hora_anyo(irradiacion_directa_horiz, i)
        irradiacion_difusa_horiz_hora = enhvalor_hora_anyo(irradiacion_difusa_horiz, i)

        if (irradiacion_directa_horiz_hora > 0. || irradiacion_difusa_horiz_hora > 0.)
        {
            hora_anual
            azimut_solar_hora, cenit_solar_hora
            F1_circumsolar_brightening_coef, F2_horizon_brightening_coef

            hora_anual = i + desfase_horario
            iecbmath_azimut_cenit_y_altura_segun_hora_anual_y_latitud(
                    hora_anual,
                    latitud_edificio_grados,
                    &azimut_solar_hora, &cenit_solar_hora, NULL)

            i_coeficientes_F1_F2_modelo_perez(
                    cambtipo_ulong_a_double(i) + desfase_horario,
                    cenit_solar_hora,
                    irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora,
                    &F1_circumsolar_brightening_coef, &F2_horizon_brightening_coef)

            enhvalor_asigna_hora_anyo(*F1_circumsolar_brightening, i, F1_circumsolar_brightening_coef)
            enhvalor_asigna_hora_anyo(*F2_horizon_brightening, i, F2_horizon_brightening_coef)

            enhvalor_asigna_hora_anyo(*azimut_solar, i, azimut_solar_hora)
            enhvalor_asigna_hora_anyo(*cenit_solar, i, cenit_solar_hora)
        }
    }
}
"""


#  ---------------------------------------------------------------------------
#  Perez Sky Diffuse Irradiance Model
def i_factor_irrad_difusa_modelo_perez(inclinacion_deg, cos_ang_incidencia, cos_cenit_acotado,
                                       F1_circumsolar_brightening_coef, F2_horizon_brightening_coef):
    assert (cos_cenit_acotado > 0.)
    return (.5 * (1. - F1_circumsolar_brightening_coef) * (1. + np.cos(i_DEG_A_RAD * inclinacion_deg))
            + F1_circumsolar_brightening_coef * cos_ang_incidencia / cos_cenit_acotado
            + F2_horizon_brightening_coef * np.sin(i_DEG_A_RAD * inclinacion_deg))


def i_corrige_orientacion_negativa(orientacion_N_deg):
    assert((orientacion_N_deg > -180.00001) | (orientacion_N_deg < 180.00001))
    if orientacion_N_deg < 0.:
        return orientacion_N_deg + 360.
    else:
        return orientacion_N_deg


def enhsol_calcula_irradiacion_inclinada(
        usar_irrad_directa_h_como_irrad_b_directa_normal,
        orientacion_N_deg, inclinacion_deg,
        azimut_solar_deg, cenit_solar_deg,
        irrad_directa_h, irrad_difusa_h, coef_albedo,
        F1_circumsolar_brightening_coef, F2_horizon_brightening_coef):
    assert ((irrad_directa_h > 0.) | (irrad_difusa_h > 0.))

    #  Conversión azimut NORTE-SUR
    orientacion_N_positiva = i_corrige_orientacion_negativa(orientacion_N_deg)
    azimut_efectivo = 180. + azimut_solar_deg - orientacion_N_positiva

    #  Ángulos
    cos_ang_incidencia_calc = i_cos_angulo_incidencia(cenit_solar_deg, inclinacion_deg, azimut_efectivo)
    cos_cenit_acotado = i_cos_cenit_acotado(cenit_solar_deg)

    if (usar_irrad_directa_h_como_irrad_b_directa_normal):
        irrad_b_directa_normal = irrad_directa_h
        irrad_directa_h = irrad_b_directa_normal * cos_cenit_acotado
    else:
        irrad_b_directa_normal = irrad_directa_h / cos_cenit_acotado

    # Difusa (Modelo Pérez)
    factor_irrad_difusa = i_factor_irrad_difusa_modelo_perez(
        inclinacion_deg, cos_ang_incidencia_calc, cos_cenit_acotado,
        F1_circumsolar_brightening_coef, F2_horizon_brightening_coef)

    irrad_d_difusa = factor_irrad_difusa * irrad_difusa_h
    i_comprueba_valor_irrad_W_m2(irrad_d_difusa)

    # Albedo
    factor_albedo = coef_albedo * .5 * (1. - np.cos(i_DEG_A_RAD * inclinacion_deg))
    irrad_r_albedo = factor_albedo * (irrad_difusa_h + irrad_directa_h)
    i_comprueba_valor_irrad_W_m2(irrad_r_albedo)

    # Directa
    # pragma message("NOTA 6001 (\""__FILE__"\"): EUGENIO2: Revisar corrección ang_incidencia por atm")
    # factor_transmitancia = 1. - .0663*acos(a)^2.+.0882*acos(a)^3-.194*acos(a)^2.
    irrad_b_directa = irrad_b_directa_normal * cos_ang_incidencia_calc
    i_comprueba_valor_irrad_W_m2(irrad_b_directa)
    i_comprueba_valor_irrad_W_m2(irrad_b_directa + irrad_d_difusa + irrad_r_albedo)

    return irrad_b_directa, irrad_d_difusa, irrad_r_albedo


#  ---------------------------------------------------------------------------
"""
struct enhvalor_horas_anyo_t *enhsol_calcula_perdidas_solares_al_cielo(
                        inclinacion_deg,
                        transmitancia_U, emisividad, resistencia_R_se,
                        const struct enhvalor_horas_anyo_t *temperatura_seca,
                        const struct enhvalor_horas_anyo_t *temperatura_cielo)
{
    struct enhvalor_horas_anyo_t *perdidas_cielo_W_m2
    factor_forma, dif_medias_temp_ext_cielo
    unsigned long i

    assert_no_null(temperatura_seca)
    assert_no_null(temperatura_cielo)

    perdidas_cielo_W_m2 = enhvalor_inicia_horas_anyo_nulo()

    dif_medias_temp_ext_cielo = enhvalor_valor_medio_anual(temperatura_seca)
    dif_medias_temp_ext_cielo -= enhvalor_valor_medio_anual(temperatura_cielo)

    factor_forma = .5 * (1. + cos(i_DEG_A_RAD * inclinacion_deg))

    for (i = 0 i < ENHVALOR_NUM_HORAS_ANYO i++)
    {
        temperatura_seca_i, temperatura_cielo_i
        perdida_cielo_horaria_por_m2, h_r

        temperatura_seca_i = enhvalor_hora_anyo(temperatura_seca, i)
        temperatura_cielo_i = enhvalor_hora_anyo(temperatura_cielo, i)

        h_r = 4. * emisividad * i_SIGMA_STEFAN_BOLTZMANN * pow(.5 * (temperatura_seca_i + temperatura_cielo_i) + 273.15, 3.)
        perdida_cielo_horaria_por_m2 = resistencia_R_se * transmitancia_U * h_r * factor_forma * dif_medias_temp_ext_cielo

        enhvalor_asigna_hora_anyo(perdidas_cielo_W_m2, i, -perdida_cielo_horaria_por_m2)
    }

    return perdidas_cielo_W_m2
}
"""


def enhsol_calcula_factor_perdidas_al_cielo(temperatura_seca, temperatura_cielo):
    dif_medias_temp_ext_cielo = np.mean(temperatura_seca) - np.mean(temperatura_cielo)
    perdidas_cielo_W_m2 = [4. * i_SIGMA_STEFAN_BOLTZMANN * pow(.5 * (temperatura_seca_i + temperatura_cielo_i) + 273.15,
                                                               3.) * dif_medias_temp_ext_cielo
                           for temperatura_seca_i, temperatura_cielo_i in zip(temperatura_seca, temperatura_cielo)]
    return perdidas_cielo_W_m2


def i_calcula_orto_ocaso(dia_juliano, latitud_deg):
    sin_declinacion, cos_declinacion = i_calcula_declinacion(dia_juliano)
    comp_ang = sin_declinacion / cos_declinacion * np.tan(latitud_deg * i_DEG_A_RAD)

    if comp_ang > 1.:  # día polar
        t_orto = 0.
        t_ocaso = 24.
    elif comp_ang < -1.:  # noche polar
        t_orto = 0.
        t_ocaso = 0.
    else:
        t_orto = 12. * (1. - 1. / PI * np.arccos(-comp_ang))
        t_ocaso = 12. * (1. + 1. / PI * np.arccos(-comp_ang))
    return t_orto, t_ocaso


def enhsol_calcula_orto_ocaso(latitud_deg):
    return np.array([i_calcula_orto_ocaso(float(i) + 1, latitud_deg) for i in range(365)])


'''
#  ---------------------------------------------------------------------------

static i_analiza_valores_de_entrada_irradiacion_con_errores(
                        latitud_deg,
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        en_plano_vertical_sur,
                        const struct enhvalor_horas_anyo_t *irradiacion_directa_horiz,
                        const struct enhvalor_horas_anyo_t *irradiacion_difusa_horiz,
                        const struct enhvalor_horas_anyo_t *azimut_solar,
                        const struct enhvalor_horas_anyo_t *cenit_solar,
                        ArrPuntero(ArrDouble) **valores_entrada_incorrectos_opc)
{
    con_errores
    ArrPuntero(ArrDouble) *valores_incorrectos
    unsigned long i
    inclinacion, desfase_horario, orientacion_N_positiva
    struct enhvalor_horas_anyo_t *F1_circumsolar_brightening, *F2_horizon_brightening
    struct enhvalor_horas_anyo_t *azimut_solar_calc, *cenit_solar_calc
    factor_seguridad

    con_errores = False
    desfase_horario = .5

    if (usar_irrad_directa_h_como_irrad_b_directa_normal == True)
        factor_seguridad = .9
    else:
        factor_seguridad = 1.

    orientacion_N_positiva = 180.      # Al sur
    if (en_plano_vertical_sur == True):
        inclinacion = 90.              # Vertical
    else:
        inclinacion = 0.               # Horizontal

    enhsol_factores_irradiacion_difusa(
                        desfase_horario,
                        latitud_deg,
                        irradiacion_directa_horiz,
                        irradiacion_difusa_horiz,
                        &F1_circumsolar_brightening, &F2_horizon_brightening,
                        &azimut_solar_calc, &cenit_solar_calc)

#pragma message("NOTA 6001 (\""__FILE__"\"): EUGENIO2: Usar también para comparar posición solar calculada vs MET")
    enhvalor_destruye_horas_anyo(&azimut_solar_calc)
    enhvalor_destruye_horas_anyo(&cenit_solar_calc)

    valores_incorrectos = arr_CreaPunteroTD(0, ArrDouble)

    for (i = 0 i < ENHVALOR_NUM_HORAS_ANYO i++)
    {
        irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora

        irradiacion_directa_horiz_hora = enhvalor_hora_anyo(irradiacion_directa_horiz, i)
        irradiacion_difusa_horiz_hora = enhvalor_hora_anyo(irradiacion_difusa_horiz, i)

        if (irradiacion_directa_horiz_hora > 0. || irradiacion_difusa_horiz_hora > 0.)
        {
            hora_anual, E_0_z
            azimut_solar_hora, cenit_solar_hora, cos_cenit_acotado, cos_cenit
            irrad_b_directa_normal_acotada, irrad_b_directa_normal

            hora_anual = i + desfase_horario
            E_0_z = i_calcula_irradiacion_solar_maxima_sin_atm(i_dia_juliano_de_hora_anual(hora_anual))

            azimut_solar_hora = enhvalor_hora_anyo(azimut_solar, i)
            cenit_solar_hora = enhvalor_hora_anyo(cenit_solar, i)
            assert(cenit_solar_hora < 90.0000001)

            cos_cenit_acotado = i_cos_cenit_acotado(cenit_solar_hora)
            cos_cenit = cos(i_DEG_A_RAD * cenit_solar_hora)

            if (usar_irrad_directa_h_como_irrad_b_directa_normal == True)
                irrad_b_directa_normal_acotada = irradiacion_directa_horiz_hora
            else:
                irrad_b_directa_normal_acotada = irradiacion_directa_horiz_hora / cos_cenit_acotado

            if (usar_irrad_directa_h_como_irrad_b_directa_normal == False && fabs(cos_cenit) > 0.)
                irrad_b_directa_normal = irradiacion_directa_horiz_hora / cos_cenit
            else:
                irrad_b_directa_normal = irrad_b_directa_normal_acotada

            if (irrad_b_directa_normal_acotada > factor_seguridad * E_0_z || irrad_b_directa_normal > factor_seguridad * E_0_z)
            {
                Arr*hora_con_errores
                azimut_efectivo, cos_ang_incidencia_calc, I_b_N, error
                unsigned long mes, dia, hora

                con_errores = True

                I_b_N = MAX(irrad_b_directa_normal, irrad_b_directa_normal_acotada)
                error = I_b_N - E_0_z

                azimut_efectivo = 180. + azimut_solar_hora - orientacion_N_positiva
                cos_ang_incidencia_calc = i_cos_angulo_incidencia(cenit_solar_hora, inclinacion, azimut_efectivo)

                enhvalor_datos_fecha_de_hora_anual(i, &mes, &dia, &hora, NULL)

                hora_con_errores = arr_CreaDouble(0)
                arr_AppendDouble(hora_con_errores, (double)i)
                arr_AppendDouble(hora_con_errores, i_dia_juliano_de_hora_anual(hora_anual))
                arr_AppendDouble(hora_con_errores, (double)mes)
                arr_AppendDouble(hora_con_errores, (double)dia)
                arr_AppendDouble(hora_con_errores, (double)hora + desfase_horario)

                arr_AppendDouble(hora_con_errores, azimut_solar_hora)
                arr_AppendDouble(hora_con_errores, cenit_solar_hora)

                arr_AppendDouble(hora_con_errores, cos_ang_incidencia_calc)
                arr_AppendDouble(hora_con_errores, acos(cos_ang_incidencia_calc) / i_DEG_A_RAD)

                arr_AppendDouble(hora_con_errores, irradiacion_directa_horiz_hora)
                arr_AppendDouble(hora_con_errores, irradiacion_difusa_horiz_hora)

                {
                    irrad_b_directa, irrad_d_difusa, irrad_r_albedo
                    F1_circumsolar_brightening_coef, F2_horizon_brightening_coef

                    F1_circumsolar_brightening_coef = enhvalor_hora_anyo(F1_circumsolar_brightening, i)
                    F2_horizon_brightening_coef = enhvalor_hora_anyo(F2_horizon_brightening, i)

                    enhsol_calcula_irradiacion_inclinada(
                            usar_irrad_directa_h_como_irrad_b_directa_normal,
                            orientacion_N_positiva, inclinacion,
                            azimut_solar_hora, cenit_solar_hora,
                            irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora, .2, # coef_albedo,
                            F1_circumsolar_brightening_coef, F2_horizon_brightening_coef,
                            &irrad_b_directa, &irrad_d_difusa, &irrad_r_albedo)

                    arr_AppendDouble(hora_con_errores, irrad_b_directa)
                    arr_AppendDouble(hora_con_errores, irrad_d_difusa)
                    arr_AppendDouble(hora_con_errores, irrad_r_albedo)
                    arr_AppendDouble(hora_con_errores, irrad_b_directa + irrad_d_difusa + irrad_r_albedo)

                }

                arr_AppendDouble(hora_con_errores, E_0_z)
                arr_AppendDouble(hora_con_errores, I_b_N)
                arr_AppendDouble(hora_con_errores, error)

                arr_AppendPunteroTD(valores_incorrectos, hora_con_errores, ArrDouble)
            }
        }
    }

    enhvalor_destruye_horas_anyo(&F1_circumsolar_brightening)
    enhvalor_destruye_horas_anyo(&F2_horizon_brightening)

    if (con_errores == True)
    {
        ASIGNA_OPC_DARRTD(valores_entrada_incorrectos_opc, valores_incorrectos, ArrDouble, arr_DestruyeDouble)
    }
    else:
        arr_DestruyeEstructurasTD(&valores_incorrectos, ArrDouble, arr_DestruyeDouble)

    return con_errores
}

#  ---------------------------------------------------------------------------

enhsol_analisis_valores_de_entrada_irradiacion_en_ficheros_met(
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        en_plano_vertical_sur,
                        enum enhtipos_situacion_t situacion,
                        enum enhtipos_zona_invierno_t zona_invierno,
                        enum enhtipos_zona_verano_t zona_verano,
                        ArrPuntero(ArrDouble) **valores_entrada_incorrectos_opc)
{
    con_errores
    struct enhfichmet_datos_met_t *datos_met

    datos_met = enhdatcli_obten_datos_climaticos_cte(situacion, zona_invierno, zona_verano)
    assert_no_null(datos_met)

    con_errores = i_analiza_valores_de_entrada_irradiacion_con_errores(
                        datos_met->latitud,
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        en_plano_vertical_sur,
                        datos_met->irradiacion_directa_horiz,
                        datos_met->irradiacion_difusa_horiz,
                        datos_met->azimut_solar,
                        datos_met->cenit_solar,
                        valores_entrada_incorrectos_opc)

    enhfichmet_destruye_datos_met(&datos_met)

    return con_errores
}

#  ---------------------------------------------------------------------------

#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ! Cálculos de ganancias solares
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
#  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

def i_calcula_irradiacion_inclinada(
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        inclinacion,
                        orientacion,
                        const struct enhvalor_horas_anyo_t *azimut_solar,
                        const struct enhvalor_horas_anyo_t *cenit_solar,
                        const struct enhvalor_horas_anyo_t *F1_circumsolar_brightening,
                        const struct enhvalor_horas_anyo_t *F2_horizon_brightening,
                        const Arr*porcentaje_radiacion_directa,
                        coeficiente_albedo,
                        unsigned long i,
                        irradiacion_directa_horiz_hora,
                        irradiacion_difusa_horiz_hora,
                        *irradiacion_inclinada_con_sombras,
                        *irradiacion_inclinada_sin_sombras)
{
    azimut_solar_hora, cenit_solar_hora
    F1_circumsolar_brightening_coef, F2_horizon_brightening_coef
    irrad_b_directa, irrad_d_difusa, irrad_r_albedo

    assert_no_null(irradiacion_inclinada_con_sombras)
    assert_no_null(irradiacion_inclinada_sin_sombras)

    azimut_solar_hora = enhvalor_hora_anyo(azimut_solar, i)
    cenit_solar_hora = enhvalor_hora_anyo(cenit_solar, i)

    F1_circumsolar_brightening_coef = enhvalor_hora_anyo(F1_circumsolar_brightening, i)
    F2_horizon_brightening_coef = enhvalor_hora_anyo(F2_horizon_brightening, i)

    assert(cenit_solar_hora < 90 + i_PREC_COMPARA)
    enhsol_calcula_irradiacion_inclinada(
                usar_irrad_directa_h_como_irrad_b_directa_normal,
                orientacion, inclinacion,
                azimut_solar_hora, cenit_solar_hora,
                irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora,
                coeficiente_albedo,
                F1_circumsolar_brightening_coef, F2_horizon_brightening_coef,
                &irrad_b_directa, &irrad_d_difusa, &irrad_r_albedo)

    *irradiacion_inclinada_sin_sombras = irrad_b_directa + irrad_d_difusa + irrad_r_albedo

    if (porcentaje_radiacion_directa == NULL)
    {
        *irradiacion_inclinada_con_sombras = *irradiacion_inclinada_sin_sombras
    }
    else
    {
        porcen_radiacion_directa

        porcen_radiacion_directa = arr_GetDouble(porcentaje_radiacion_directa, i)

        if (porcen_radiacion_directa > .000001)
            *irradiacion_inclinada_con_sombras = *irradiacion_inclinada_sin_sombras * porcen_radiacion_directa
        else
            *irradiacion_inclinada_con_sombras = irrad_d_difusa + irrad_r_albedo
    }
}

#  ---------------------------------------------------------------------------

def i_append_ganancia_solar_translucido(
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        superficie,
                        inclinacion, orientacion,
                        const struct enhdouble_t *factor_reductor_dispositivo_sombra,
                        const struct enhdouble_t *factor_solar,
                        const struct enhctemp_t *datos_correccion_temporal,
                        const struct enhvalor_horas_anyo_t *irradiacion_directa_horiz,
                        const struct enhvalor_horas_anyo_t *irradiacion_difusa_horiz,
                        const struct enhvalor_horas_anyo_t *azimut_solar,
                        const struct enhvalor_horas_anyo_t *cenit_solar,
                        const struct enhvalor_horas_anyo_t *F1_circumsolar_brightening,
                        const struct enhvalor_horas_anyo_t *F2_horizon_brightening,
                        const Arr*porcentaje_radiacion_directa,
                        coeficiente_albedo,
                        struct enhvalor_horas_anyo_t *ganancias_solares_calculo_zona,
                        *ganancia_solar_anual_aportada,
                        *ganancia_solar_anual_aportada_sin_sombras)
{
    unsigned long i

    assert(superficie > 0.)
    assert_no_null(ganancia_solar_anual_aportada)
    assert_no_null(ganancia_solar_anual_aportada_sin_sombras)

    if (porcentaje_radiacion_directa is not None)
        assert(arr_NumElemsDouble(porcentaje_radiacion_directa) == ENHVALOR_NUM_HORAS_ANYO)

    *ganancia_solar_anual_aportada = 0.
    *ganancia_solar_anual_aportada_sin_sombras = 0.

    for (i = 0 i < ENHVALOR_NUM_HORAS_ANYO i++)
    {
        irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora

        irradiacion_directa_horiz_hora = enhvalor_hora_anyo(irradiacion_directa_horiz, i)
        irradiacion_difusa_horiz_hora = enhvalor_hora_anyo(irradiacion_difusa_horiz, i)

        if (irradiacion_directa_horiz_hora > 0. || irradiacion_difusa_horiz_hora > 0.)
        {
            irradiacion_inclinada_con_sombras, irradiacion_inclinada_sin_sombras
            ganancia_solar_calculada_con_sombras, ganancia_solar_calculada_sin_sombras

            i_calcula_irradiacion_inclinada(
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        inclinacion, orientacion,
                        azimut_solar, cenit_solar,
                        F1_circumsolar_brightening, F2_horizon_brightening,
                        porcentaje_radiacion_directa,
                        coeficiente_albedo,
                        i,
                        irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora,
                        &irradiacion_inclinada_con_sombras, &irradiacion_inclinada_sin_sombras)

            {
                valor_factor_reductor_dispositivo_sombra, valor_factor_solar, factor_ganancia

                valor_factor_reductor_dispositivo_sombra = enhdouble_valor_hora_anyo(i, factor_reductor_dispositivo_sombra, datos_correccion_temporal)
                valor_factor_solar = enhdouble_valor_hora_anyo(i, factor_solar, datos_correccion_temporal)

                factor_ganancia = valor_factor_reductor_dispositivo_sombra * valor_factor_solar * superficie

                ganancia_solar_calculada_con_sombras = factor_ganancia * irradiacion_inclinada_con_sombras
                ganancia_solar_calculada_sin_sombras = factor_ganancia * irradiacion_inclinada_sin_sombras
            }

            enhvalor_incrementa_hora_anyo(ganancias_solares_calculo_zona, i, ganancia_solar_calculada_con_sombras)
            *ganancia_solar_anual_aportada += ganancia_solar_calculada_con_sombras
            *ganancia_solar_anual_aportada_sin_sombras += ganancia_solar_calculada_sin_sombras
        }
    }
}

#  ---------------------------------------------------------------------------

def i_append_ganancia_solar_opaco(
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        superficie,
                        inclinacion, orientacion,
                        const struct enhdouble_t *transmitancia_termica,
                        const struct enhctemp_t *datos_correccion_temporal,
                        const struct enhvalor_horas_anyo_t *irradiacion_directa_horiz,
                        const struct enhvalor_horas_anyo_t *irradiacion_difusa_horiz,
                        const struct enhvalor_horas_anyo_t *azimut_solar,
                        const struct enhvalor_horas_anyo_t *cenit_solar,
                        const struct enhvalor_horas_anyo_t *F1_circumsolar_brightening,
                        const struct enhvalor_horas_anyo_t *F2_horizon_brightening,
                        const struct enhvalor_horas_anyo_t *factor_perdidas_al_cielo,
                        const Arr*porcentaje_radiacion_directa,
                        absortividad,
                        coeficiente_albedo,
                        struct enhvalor_horas_anyo_t *ganancias_solares_calculo_zona,
                        *ganancia_solar_anual_aportada,
                        *ganancia_solar_anual_aportada_sin_sombras)
{
    unsigned long i

    assert(superficie > 0.)
    assert_no_null(ganancia_solar_anual_aportada)
    assert_no_null(ganancia_solar_anual_aportada_sin_sombras)

    if (porcentaje_radiacion_directa is not None)
        assert(arr_NumElemsDouble(porcentaje_radiacion_directa) == ENHVALOR_NUM_HORAS_ANYO)

    *ganancia_solar_anual_aportada = 0.
    *ganancia_solar_anual_aportada_sin_sombras = 0.

    for (i = 0 i < ENHVALOR_NUM_HORAS_ANYO i++)
    {
        irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora

        irradiacion_directa_horiz_hora = enhvalor_hora_anyo(irradiacion_directa_horiz, i)
        irradiacion_difusa_horiz_hora = enhvalor_hora_anyo(irradiacion_difusa_horiz, i)

        if (irradiacion_directa_horiz_hora > 0. || irradiacion_difusa_horiz_hora > 0.)
        {
            irradiacion_inclinada_con_sombras, irradiacion_inclinada_sin_sombras
            ganancia_solar_calculada_con_sombras, ganancia_solar_calculada_sin_sombras

            i_calcula_irradiacion_inclinada(
                        usar_irrad_directa_h_como_irrad_b_directa_normal,
                        inclinacion, orientacion,
                        azimut_solar, cenit_solar,
                        F1_circumsolar_brightening, F2_horizon_brightening,
                        porcentaje_radiacion_directa,
                        coeficiente_albedo,
                        i,
                        irradiacion_directa_horiz_hora, irradiacion_difusa_horiz_hora,
                        &irradiacion_inclinada_con_sombras, &irradiacion_inclinada_sin_sombras)

            {
                valor_transmitancia_termica, valor_factor_perdidas_al_cielo, factor_ganancia, factor_forma

                valor_transmitancia_termica = enhdouble_valor_hora_anyo(i, transmitancia_termica, datos_correccion_temporal)
                factor_ganancia = i_RESISTENCIA_SUPERFICIAL_EXTERIOR * valor_transmitancia_termica * superficie

                factor_forma = .5 * (1. + cos(inclinacion * i_DEG_A_RAD))
                valor_factor_perdidas_al_cielo = enhvalor_hora_anyo(factor_perdidas_al_cielo, i)

                ganancia_solar_calculada_con_sombras = factor_ganancia * (absortividad * irradiacion_inclinada_con_sombras - factor_forma * i_EMISIVIDAD_OPACOS_CTE * valor_factor_perdidas_al_cielo)
                ganancia_solar_calculada_sin_sombras = factor_ganancia * (absortividad * irradiacion_inclinada_sin_sombras - factor_forma * i_EMISIVIDAD_OPACOS_CTE * valor_factor_perdidas_al_cielo)
            }

            enhvalor_incrementa_hora_anyo(ganancias_solares_calculo_zona, i, ganancia_solar_calculada_con_sombras)
            *ganancia_solar_anual_aportada += ganancia_solar_calculada_con_sombras
            *ganancia_solar_anual_aportada_sin_sombras += ganancia_solar_calculada_sin_sombras
        }
    }
}

'''


@timeit('local_maxima_minima', verbose=True)
def local_maxima_minima(serie, window_delta='2h', verbose=True):

    def _is_local_min(x):
        if np.argmin(x) == len(x) // 2:
            return True
        return False

    def _is_local_max(x):
        if np.argmax(x) == len(x) // 2:
            return True
        return False

    roll_w = int(pd.Timedelta(window_delta) / serie.index.freq)
    if roll_w % 2 == 0:
        roll_w += 1
    if verbose:
        print_secc('LOCAL MAX-MIN de {} valores. Centered window of {} samples (for ∆T={}, freq={})'
                   .format(len(serie), roll_w, window_delta, serie.index.freq))
    maximos = serie.rolling(roll_w, center=True).apply(_is_local_max).fillna(0).astype(bool)
    minimos = serie.rolling(roll_w, center=True).apply(_is_local_min).fillna(0).astype(bool)
    if verbose:
        print_info(serie[maximos])
        print_cyan(serie[minimos])
    return maximos, minimos


def _cos_incidencia(data, orientacion_N_deg, inclinacion_deg=90,
                    alt='altitud', azi='azimut', irrad=None):  # 'irradiacion_cs'):
    cenit_deg = 90 - data[alt]
    azimut_efectivo = 180 + data[azi] - orientacion_N_deg
    # Ángulos de incidencia
    cos_theta = (np.cos(cenit_deg * i_DEG_A_RAD) * np.cos(inclinacion_deg * i_DEG_A_RAD)
                 + np.sin(cenit_deg * i_DEG_A_RAD)
                 * np.sin(inclinacion_deg * i_DEG_A_RAD) * np.cos(azimut_efectivo * i_DEG_A_RAD))
    cos_incidencia = cos_theta.apply(lambda x: max(0, x))
    if irrad is not None:
        if type(irrad) is str:
            return cos_incidencia * data[irrad]
        else:
            return cos_incidencia * irrad
    # cos_cenit_acotado = cosd(cenit_deg).apply(lambda x: max(COS85_MIN, x))
    return cos_incidencia


@timeit('ajuste_ldr_con_barrido_irrad_inc', verbose=True)
def ajuste_ldr_con_barrido_irrad_inc(solar_data, step_deg=5, inclinacion_deg=90):
    min_ldr = 0
    cols_interes = ['ldr', 'altitud', 'azimut', 'irradiacion_cs']
    artificial = solar_data[(solar_data.artif_level_max > 0)]['ldr'].resample('30s').max()
    residual = solar_data[((solar_data.artif_level_max == 0) & (solar_data.altitud <= 0)
                           )]['ldr'].resample('2min').mean()
    if not residual.empty:
        min_ldr = residual.mode()[0]
        residual = residual.fillna(min_ldr).astype(int)

    data_solar_p_raw = solar_data[(solar_data.artif_level_max == 0) & (solar_data.altitud > 0)][cols_interes].copy()
    data_solar_p = data_solar_p_raw.resample('1min').mean()
    data_solar_p['ldr'] = data_solar_p_raw.ldr.resample('1min').max() - min_ldr

    print_magenta(residual.describe())
    print_ok(min_ldr)

    labels = []
    for ang_N in range(-180, 180, step_deg):
        label = 'irrad_{}'.format(ang_N)
        data_solar_p[label] = _cos_incidencia(data_solar_p, ang_N, irrad=cols_interes[-1],
                                              inclinacion_deg=inclinacion_deg)
        labels.append(label)

    X, Y = data_solar_p['ldr'], data_solar_p.drop(cols_interes, axis=1)
    try:
        model = sm.OLS(X, Y).fit()
    except (np.linalg.linalg.LinAlgError, ValueError) as e:
        print_err(e)
        model = None
    # print_cyan(model.summary())
    # print_red(model.pvalues[model.pvalues > .5])

    f = plt.figure(figsize=(16, 9), facecolor='None')
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2, colspan=1)

    ax = solar_data['irradiacion_cs'].resample('1min').mean().plot(ax=ax, lw=3, color='violet', alpha=.7)
    ax = solar_data['ldr'].resample('1min').mean().plot(ax=ax, lw=1, color='darkorange', alpha=.7)
    if not data_solar_p.empty:
        ax = (data_solar_p['ldr'] + min_ldr).plot(ax=ax, lw=3, color='darkgreen', alpha=.7)
        for c in labels:
            data_solar_p[c].plot(ax=ax, lw=.5, color='grey')
    if model is not None:
        reconstr = (model.params * Y).sum(axis=1).rename('Reconstr_OLS') + min_ldr
        reconstr.plot(ax=ax, color='darkred', lw=5, alpha=.5)
    if not artificial.empty:
        plt.fill_between(artificial.index, artificial.values, 0, color='orange', lw=2, alpha=.5, label='Lights')
    if not residual.empty:
        plt.hlines(min_ldr, solar_data.index[0], solar_data.index[-1], color='yellow', lw=3, alpha=.8, label='Residual')
    dia_plot = solar_data.index[0].date()
    plt.title("LDR ({:%d-%b'%y})".format(dia_plot))
    # plt.fill_between(residual.index, residual.values, 0, color='yellow', alpha=.5, label='Residual')
    # plt.legend()

    if model is not None:
        ax = plt.subplot2grid((3, 1), (2, 0), rowspan=1, colspan=1)
        orient_idx = model.params.index.str.slice(6).astype(int)
        params = pd.Series(model.params.values, index=orient_idx)
        pvalues = pd.Series(model.pvalues.values, index=orient_idx)
        params.plot(ax=ax, lw=3, color='green', label='PARAMS')
        (pvalues * ax.get_ylim()[1]).plot(ax=ax, color='red', label='p-values')
    else:
        params = None
    plt.tight_layout()

    # plt.suptitle("LDR ({:%d-%b'%y})".format(dia_plot))
    # plt.savefig('/Users/uge/Desktop/LDR_analysis/LDR_análisis_día_{:%Y_%m_%d}.svg'.format(dia_plot))
    write_fig_to_svg(f, '/Users/uge/Desktop/LDR_analysis/LDR_análisis_día_{:%Y_%m_%d}.svg'.format(dia_plot))
    # plt.show()
    return params


def plot_data_ldr(ldr_raw, dplot, pois=None, ax=None, with_vlines=True):
    if ax is not None:
        ldr_raw.iloc[::5].plot(ax=ax, lw=1, color='darkorange', alpha=.9)
    else:
        ax = ldr_raw.iloc[::5].plot(figsize=FS, lw=1, color='darkorange', alpha=.9)
    (dplot['altitud'] * 12).plot(ax=ax, lw=4, color='y', alpha=.6)
    dplot['median_filter'].plot(ax=ax, color='darkred', lw=2, alpha=.7)
    if (pois is not None) and with_vlines:
        ylim = ax.get_ylim()[1]
        pois_no_label = pois.index[~pois.is_poi_to_label]
        pois_label = pois.index[pois.is_poi_to_label]
        if len(pois_no_label) > 0:
            ax.vlines(pd.DatetimeIndex(pois_no_label), 0, ylim,
                      lw=.75, color='darkgrey', alpha=.7, linestyle='--')
        if len(pois_label) > 0:
            ax.vlines(pd.DatetimeIndex(pois_label), 0, ylim,
                      lw=1.5, color='darkblue', alpha=.7, linestyle='--')
    # ax.xaxis.set_major_locator(mpd.HourLocator())
    # ax.xaxis.set_major_formatter(mpd.DateFormatter('%H:%M'))
    return ax


def set_sun_times(df, delta_rs, tz=TZ, lat_deg=LAT, long_deg=LONG, offset='5min'):
    delta = pd.Timedelta(offset)
    if df.index[-1].date() == df.index[0].date():
        # Same day:
        df['sun'] = False
        ts_day = pd.Timestamp(df.index[0].date(), tz=tz)
        sunrise, sunset = pysolar.util.get_sunrise_sunset(lat_deg, long_deg, ts_day + pd.Timedelta('12h'))
        sunrise, sunset = [sunrise - delta, sunset + delta]
        df.loc[sunrise:sunset, 'sun'] = True
    else:
        suntimes, states = [], []
        freq = delta_rs / pd.Timedelta('1D')
        days = pd.DatetimeIndex(df.iloc[::1000].index.date).drop_duplicates().tz_localize(tz)
        for ts_day in days:
            sunrise, sunset = pysolar.util.get_sunrise_sunset(lat_deg, long_deg, ts_day + pd.Timedelta('12h'))
            sunrise, sunset = [sunrise - delta, sunset + delta]
            suntimes += [mpd.num2date(round(mpd.date2num(sunrise) / freq) * freq, tz=tz),
                         mpd.num2date(round(mpd.date2num(sunset) / freq) * freq, tz=tz)]
            states += [1, 0]
        df_sun = pd.DataFrame(states, index=suntimes, columns=['sun']).loc[df.index[0]:df.index[-1]]
        df = df.join(df_sun).tz_convert(tz)
        df['sun'] = df['sun'].fillna(method='ffill').fillna(1 - df_sun.sun.iloc[0])
        return df
    return df


@timeit('identifica_pois_LDR', verbose=True)
def identifica_pois_ldr(data_raw, kernel_size=75):
    def _es_max_local(x):
        # delta_max_roll = np.max(x) - np.min(x)
        # _es_max_local).fillna(False).astype(bool)
        mid = len(x) // 2
        # if (np.argmax(x) == mid) and (x[mid] > DELTA_MIN_PARA_CONSIDERACION_MAX_LOCAL):
        if (np.argmax(x) == mid) and (x[mid] - np.min(x) > DELTA_MIN_PARA_CONSIDERACION_MAX_LOCAL):
            return True
        elif (np.argmin(x) == mid) and (np.max(x) - x[mid] > DELTA_MIN_PARA_CONSIDERACION_MAX_LOCAL):
            return True
        return False

    def _pendiente(x):
        return (x[-1] - x[0]) / len(x)

    def _pendiente_ini(x):
        if len(x) > 100:
            return _pendiente(x.values[1:101])
        return _pendiente(x.values)

    def _pendiente_fin(x):
        if len(x) > 100:
            return _pendiente(x.values[-100:-1])
        return _pendiente(x.values)

    def _pendiente_last_gen(x):
        limit = min(len(x), 600)
        if limit > 1:
            return _pendiente(x.values[-limit:-1])
        # print_red(x)
        return x.iloc[0]

    def _get_alt(d):
        alt = pysolar.solar.get_altitude_fast(LAT, LONG, d)
        if alt > 0:
            return alt
        return 0

    print_info("IDENTIFICACIÓN POI's LDR.\n * {} raw points. De {:%d-%m-%y} a {:%d-%m-%y} ({:.1f} horas)"
               .format(len(data_raw), data_raw.index[0], data_raw.index[-1], len(data_raw) / 3600.))
    delta_rs = pd.Timedelta('5s')
    DELTA_MIN_PARA_CONSIDERACION_MAX_LOCAL = 20

    # Resampling 1s
    # data_homog = data_raw.resample('1s').mean().interpolate().dropna()
    data_homog = data_raw.resample('1s').mean().fillna(method='bfill', limit=5).fillna(-1)
    _info_tiempos('RS 1s')

    # Median filter
    data_homog['median_filter'] = medfilt(data_homog.ldr, kernel_size=[kernel_size])
    _info_tiempos('MedFilt')
    mf_resampled = pd.DataFrame(data_homog['median_filter']) #.resample(delta_rs).max())
    _info_tiempos('Resample MedFilt')
    mf_resampled = set_sun_times(mf_resampled, delta_rs, tz=TZ, lat_deg=LAT, long_deg=LONG, offset='10min')
    _info_tiempos('Sun times')

    # Altitud sol
    subset_pysolar = mf_resampled.iloc[::60].index
    mf_resampled = mf_resampled.join(pd.Series(subset_pysolar, index=subset_pysolar, name='altitud'
                                               ).apply(_get_alt)).interpolate()
    _info_tiempos('Sun ALT')

    # BUSCA POI's
    # Construye intervalos entre POI's
    roll_window = 9
    roll_mf = mf_resampled.median_filter.rolling(roll_window, center=True)
    mf_resampled['intervalo'] = (roll_mf.max() - roll_mf.min()
                                 ).fillna(method='bfill').rolling(roll_window).apply(_es_max_local).cumsum()
    mf_resampled['intervalo'] = mf_resampled['intervalo'].fillna(method='bfill').fillna(method='ffill')
    print_red(mf_resampled.head(20))
    print_red(mf_resampled.tail(20))
    _info_tiempos('POIs & intervals')

    # DATOS POIS:
    gb_poi = mf_resampled.tz_localize(None).reset_index().groupby('intervalo')
    pois = pd.DataFrame(pd.concat([gb_poi.sun.apply(lambda x: np.sum(x) / len(x)).rename('fr_sun'),
                                   # gb_poi.sun.first().rename('ini_sun'),
                                   # gb_poi.sun.last().rename('fin_sun'),
                                   gb_poi.sun.count().rename('seconds'), # * delta_rs.total_seconds(),
                                   gb_poi.altitud.first().round().rename('alt_sun_ini'),
                                   gb_poi.altitud.last().round().rename('alt_sun_fin'),
                                   gb_poi.altitud.mean().round(),
                                   gb_poi.median_filter.mean().round().rename('ldr_mean'),
                                   gb_poi.median_filter.min().round().rename('ldr_min'),
                                   gb_poi.median_filter.max().round().rename('ldr_max'),
                                   gb_poi.median_filter.median().round().rename('ldr_median'),
                                   gb_poi.median_filter.std().rename('ldr_std'),
                                   gb_poi.median_filter.apply(lambda x: (x.values[-1] - x.values[0]) / len(x)
                                                              ).rename('pendiente_tot'),
                                   gb_poi.median_filter.apply(lambda x: np.round(np.median(x.values[1:10]))
                                                              ).rename('inicio'),
                                   gb_poi.median_filter.apply(lambda x: np.round(np.median(x.values[-10:-1]))
                                                              ).rename('fin'),
                                   gb_poi.median_filter.apply(_pendiente_ini).rename('pendiente_i'),
                                   gb_poi.median_filter.apply(_pendiente_fin).rename('pendiente_f'),
                                   gb_poi.median_filter.apply(_pendiente_last_gen).rename('pendiente_last_mf'),
                                   gb_poi.ts.first().rename('ts_ini'),
                                   gb_poi.ts.last().rename('ts_fin')], axis=1))
    _info_tiempos('Data POIS')
    # pois['step'] = (pois['inicio'] - pois['fin'].shift())  # .fillna(0)
    pois['step'] = pois['fin'] - pois['inicio'].shift(-1)
    pois['pendiente_salto'] = pois['pendiente_f'] - pois['pendiente_i'].shift(-1)
    pois = pois.iloc[1:-1]
    # ['seconds', 'fin_sun', 'fin', 'solo_natural', 'artificial_princ', 'step_salto', 'pendiente_salto']]

    pois['is_poi_to_label'] = (((pois['step'].abs() > 30) | (pois['seconds'] > 7200))
                               & (((pois['seconds'] > 300) & (pois['fr_sun'] > .5))
                                  | ((pois['seconds'] > 10) & (pois['fr_sun'] <= .5))))
    _info_tiempos('POIs for manual labeling')
    print_magenta('Encontrados {} Intervalos [{:.0f} para etiquetar!] con una distribución de tiempos de:\n{} '
                  .format(len(pois), pois['is_poi_to_label'].sum(), pois.seconds.describe().T))
    print_cyan(pois.head(7))
    return data_homog, mf_resampled, pois


@timeit('genera_pois_de_all_data_LDR', verbose=True)
def genera_pois_de_all_data_LDR(path_st='/Users/uge/Desktop/LDR_analysis/POIS_ident.h5'):
    global tic

    tic = time()
    # Catálogo y lectura de todos los datos.
    cat = enerpi_data_catalog()
    data, data_s = cat.get_all_data()
    # print_info(data_s.tail())
    LDR = pd.DataFrame(data.ldr).tz_localize(TZ)
    _info_tiempos('LDR')

    # print_cyan(LDR.describe())
    # LDR.hist(bins=(LDR.max() - LDR.min() + 1).values[0] // 5, log=True, figsize=(18, 8))
    # plt.show()

    data_homog, data_rs, pois = identifica_pois_ldr(LDR)
    print_red(len(pois))
    if path_st is not None:
        pois.to_hdf(path_st, 'POIS')
        data_rs.to_hdf(path_st, 'data_rs')
        data_homog.to_hdf(path_st, 'data_homog')
        _info_tiempos('SAVE POIs')
    return data_homog, data_rs, pois


def etiquetado_intervalos_LDR(data_homog_all, data_rs_all, pois, path_st='/Users/uge/Desktop/LDR_analysis/POIS.h5'):

    def _str_to_bool(resp):
        resp = resp.lower().rstrip().lstrip()
        if resp in ['si', 'sí', 's', '1', 'yes', 'y']:
            return True
        return False

    def _get_input():
        es_nat = _str_to_bool(input('Es sólo luz natural? '))
        es_art = _str_to_bool(input('Es interruptor de luz principal? '))
        corr = input('REVISA: LUZ_NATURAL={}, INTERRUPTOR_P={}'.format(es_nat, es_art))
        if len(corr) == 0:
            return es_nat, es_art
        return _get_input()

    try:
        pois_saved = pd.read_hdf(path_st, 'POIS').sort_index()
        pois_dias = [pois_saved]
    except Exception as e:
        print_err(e)
        pois_saved = pd.DataFrame([], columns=['is_poi_to_label'])
        pois_dias = []

    path_image = '/Users/uge/Desktop/LDR_analysis/LDR_IDENT_STEP.svg'
    data_rs_all = data_rs_all.tz_localize(None)
    data_homog_all = data_homog_all.tz_localize(None)

    pois_tsfin = pois.reset_index().set_index('ts_fin').sort_index()
    gbpois = pois_tsfin.groupby(lambda x: x.date)

    offset_step = pd.Timedelta('1min')
    f, axes = plt.subplots(2, 1, figsize=(10, 16))

    for day in sorted(gbpois.groups.keys()):
        idx_day = gbpois.groups[day]
        ts_day = pd.Timestamp(day)
        day = '{:%Y-%m-%d}'.format(ts_day)
        print_secc('{}\n'.format(day))

        pois = pois_tsfin.loc[idx_day].copy()
        print_red(pois[pois.is_poi_to_label].drop('is_poi_to_label', axis=1))
        pois['solo_natural'] = np.nan
        pois['artificial_princ'] = np.nan

        data_homog = data_homog_all.loc[day:day]
        d_calc = data_rs_all.loc[day:day]

        # Comprueba si ya están etiquetados todos:
        pois_prev = pois_saved.loc[day:day]
        if pois_prev.empty or not pois_prev[pois_prev.is_poi_to_label & pois_prev.solo_natural.isnull()].empty:
            plt.sca(axes[0])
            plt.cla()
            ax = plot_data_ldr(data_homog.ldr, d_calc, pois=pois, ax=axes[0])
            ylim = ax.get_ylim()[1]
            ax.set_ylim((0, ylim))
            inter_plot = pois[pois.seconds > 600].copy()
            inter_plot['ts'] = inter_plot['ts_ini'] + (inter_plot.index.values - inter_plot['ts_ini']) / 2
            for t, i in zip(inter_plot['ts'], inter_plot['intervalo']):
                ax.annotate(str(int(i)), (t, ylim - 50), color='red', ha='center')
            plt.title("LDR ({:%d-%b'%y})".format(ts_day))

            try:
                for ts_fin, row in pois.iterrows():
                    interv = row.intervalo
                    if row.is_poi_to_label and np.isnan(row.solo_natural):
                        plt.sca(axes[1])
                        plt.cla()
                        print_info('Intervalo {:.0f}. De {:%d/%m/%y %H:%M:%S} a {:%H:%M:%S} ({:.0f} segundos)'
                                   .format(interv, row.ts_ini, ts_fin, row.seconds))
                        delta_interv = ts_fin - row.ts_ini
                        offset = delta_interv / 10
                        ax = plot_data_ldr(data_homog.ldr.loc[row.ts_ini - offset: ts_fin + offset],
                                           d_calc.loc[row.ts_ini - offset: ts_fin + offset], pois=pois, ax=axes[1])
                        ax.fill_betweenx([0, ylim], row.ts_ini, ts_fin, alpha=.3,
                                         color=tableau20[3])
                        ax.fill_betweenx([0, ylim], ts_fin - offset_step, ts_fin + offset_step, alpha=.4,
                                         color=tableau20[1])
                        ax.set_ylim((0, ylim))
                        plt.title("Intervalo {:.0f} (día {:%d-%b'%y})".format(interv, ts_day))

                        write_fig_to_svg(f, path_image)
                        subprocess.call(['open', '-a', 'safari', path_image])
                        sleep(.5)
                        _cambia_foco_app_con_command_tab()

                        es_nat, es_art = _get_input()
                        pois.loc[ts_fin, 'solo_natural'] = es_nat
                        pois.loc[ts_fin, 'artificial_princ'] = es_art

            except KeyboardInterrupt:
                print_magenta('¿Desea dejar de etiquetar el día {:%d/%m/%y}?'.format(ts_day))
                cod_exit = input('Introduzca "EXIT" para dejar de etiquetar días, guardar y salir: ').lower()
                print_red(cod_exit)
                print_red(cod_exit == 'exit')
                if cod_exit == 'exit':
                    print_red('HACIENDO BREAK')
                    pois_dias.append(pois.reset_index())
                    pois_totales = pd.concat(pois_dias).reset_index()
                    pois_totales.to_hdf(path_st, 'POIS')
                    break
            pois_dias.append(pois)
            # Por seguridad, se graba cada día:
            pois_totales = pd.DataFrame(pd.concat(pois_dias, axis=0)).sort_index()
            pois_totales.to_csv(path_st + '_.csv')
            pois_totales.to_hdf(path_st, 'POIS')
        else:
            print_info("Ya existen POI's ({}) para el día: {:%d-%m-%Y}!!".format(len(pois_prev), ts_day))
    _info_tiempos('FIN')

    pois_saved = pd.read_hdf(path_st, 'POIS')
    _info_tiempos(pois_saved)
    return pois_saved


def vista_etiquetado_intervalos_LDR(data_homog_all, data_rs_all, path_st='/Users/uge/Desktop/LDR_analysis/POIS.h5'):

    pois_saved = pd.read_hdf(path_st, 'POIS').sort_index()

    path_image = '/Users/uge/Desktop/LDR_analysis/LDR_IDENT_STEP.svg'
    data_rs_all = data_rs_all.tz_localize(None)
    data_homog_all = data_homog_all.tz_localize(None)

    f, axes = plt.subplots(1, 1, figsize=(10, 10))
    gbpois = pois_saved.groupby(lambda x: x.date)
    for day in sorted(gbpois.groups.keys()):
        idx_day = gbpois.groups[day]
        ts_day = pd.Timestamp(day)
        day = '{:%Y-%m-%d}'.format(ts_day)
        print_secc('{}\n'.format(day))

        pois = pois_saved.loc[idx_day].copy()
        data_homog = data_homog_all.loc[day:day]
        d_calc = data_rs_all.loc[day:day]
        # print_red(pois[pois.is_poi_to_label].drop('is_poi_to_label', axis=1))

        plt.sca(axes)
        plt.cla()
        ax = plot_data_ldr(data_homog.ldr, d_calc, pois=pois, ax=axes, with_vlines=False)
        ylim = ax.get_ylim()[1]
        ax.set_ylim((0, ylim))
        inter_plot = pois[pois.seconds > 3000].copy()
        inter_plot['ts'] = inter_plot['ts_ini'] + (inter_plot.index.values - inter_plot['ts_ini']) / 2
        for t, i in zip(inter_plot['ts'], inter_plot['intervalo']):
            ax.annotate(str(int(i)), (t, ylim - 50), color='red', ha='center')

        # Labeled:
        labeled = pois[pois.solo_natural.notnull()]
        for tf, row in labeled.iterrows():
            t0, is_nat, is_interr_pr = row.ts_ini, row.solo_natural, row.artificial_princ
            if is_nat:
                color, alpha = tableau20[2], .1
            else:
                color, alpha = tableau20[0], .4
            ax.fill_betweenx([0, ylim], t0, tf, alpha=alpha, color=color, lw=0)
            if is_interr_pr:
                # ax.annotate([tf], 0, ylim, alpha=.9, color=tableau20[4])
                ax.annotate('', xy=(tf, row.inicio), xycoords='data',
                            xytext=(tf, row.fin), textcoords='data', color=tableau20[4],
                            arrowprops=dict(arrowstyle="->", color=tableau20[4], linewidth=2))
            else:
                ax.annotate('', xy=(tf, row.inicio), xycoords='data',
                            xytext=(tf, row.fin), textcoords='data', color=tableau20[-6],
                            arrowprops=dict(arrowstyle="->", color=tableau20[-1], linewidth=1))

                # ax.vlines([tf], 0, ylim, alpha=.9, color=tableau20[4])
        plt.title("LDR ({:%d-%b'%y})".format(ts_day))

        write_fig_to_svg(f, path_image)
        subprocess.call(['open', '-a', 'safari', path_image])
        sleep(.5)
        _cambia_foco_app_con_command_tab()
        input('Continuar? ')


if __name__ == '__main__':
    import datetime as dt

    def _info_tiempos(cad):
        global tic
        toc = time()
        print_ok('--> {:35}\t OK\t [TOOK {:.3f} seconds]'.format(cad, toc - tic))
        tic = toc

    # dia_juliano
    # i_calcula_orto_ocaso(dia_juliano, latitud_deg)
    # t_orto, t_ocaso

    # from enerpi.process import separa_ldr_artificial_natural, get_solar_day
    # def _plot_ldr_day(day):
    #     # print_secc('PLOT_LDR_DAY: {}\n\n'.format(day))
    #     data = LDR.loc[day:day]
    #     if len(data) > 100:
    #         data, data_simple = separa_ldr_artificial_natural(data, resample_inicial='2s', delta_roll_threshold=100)
    #         # data_solar = get_solar_day(data_simple)
    #         solar = get_solar_day(data)
    #         return ajuste_ldr_con_barrido_irrad_inc(solar, step_deg=5)
    #     else:
    #         print_warn('No hay suficientes datos:\n{}'.format(data))
    #         return None

    pd.set_option('display.width', 120)
    tic = time()


    # ERROR PYSOLAR AZIMUT:
    def _debug_azi(d):
        try:
            return pysolar.solar.get_azimuth_fast(LAT, LONG, d)
        except ValueError as e:
            print_err(e)
            return pysolar.solar.get_azimuth(LAT, LONG, d)

    print_ok(pysolar.solar.get_altitude(42.364908, -71.112828, dt.datetime(2007, 2, 18, 20, 13, 1, 130320)))
    print_ok(pysolar.solar.get_altitude_fast(42.364908, -71.112828, dt.datetime(2007, 2, 18, 20, 13, 1, 130320)))
    print_ok(pysolar.solar.get_azimuth(42.364908,-71.112828,dt.datetime(2007, 2, 18, 20, 18, 0, 0)))


    tt = pd.DatetimeIndex(freq='1min', start=pd.Timestamp('2016-09-05'), periods=24 * 60, tz=TZ)

    # altura = pd.Series([pysolar.solar.get_altitude_fast(LAT, LONG, d) for d in tt], index=tt, name='altura')
    # altura_detalle = pd.Series([pysolar.solar.get_altitude(LAT, LONG, d) for d in tt], index=tt, name='altura')
    # #azimut = pd.Series([_debug_azi(d) for d in tt], index=tt, name='azimut')
    #
    # ax = pd.Series([_debug_azi(d) for d in tt], index=tt, name='azimut_fast').plot(figsize=FS)
    # pd.Series([pysolar.solar.get_azimuth(LAT, LONG, d) for d in tt], index=tt, name='azimut').plot(ax=ax)
    # altura.plot(ax=ax)
    # altura_detalle.plot(ax=ax)
    # plt.legend()
    # plt.show()

    obs = ephem.Observer()
    print(obs)
    obs.lat = 38
    print(obs.lat)
    print(obs)

    # ax = (altura_detalle - altura).plot(figsize=FS)
    # plt.show()

    # tic = time()
    # df = pd.DataFrame([pysolar.solar.get_altitude_azimuth_fast(LAT, LONG, d) for d in tt], index=tt,
    #                   columns=['altura_f', 'azimut_f'])
    # print_yellow('TOOK {:.3f} s'.format(time() - tic))
    #
    # tic = time()
    # df_detail = pd.DataFrame([pysolar.solar.get_altitude_azimuth_fast(LAT, LONG, d, fast=False) for d in tt], index=tt,
    #                           columns=['altura', 'azimut'])
    # print_yellow('TOOK {:.3f} s'.format(time() - tic))
    # df = df.join(df_detail)
    #
    # print_cyan(df.iloc[::100].head())
    #
    # df[['altura', 'altura_f']].plot(figsize=FS, lw=1, alpha=.7)
    # plt.legend()
    # plt.show()
    #
    # df[['azimut', 'azimut_f']].plot(figsize=FS, lw=1, alpha=.7)
    # plt.legend()
    # plt.show()
    #
    # (df['azimut'] - df['azimut_f']).plot(figsize=FS, lw=1, alpha=.7)
    # plt.show()


    # # POI's
    # path_st = '/Users/uge/Desktop/LDR_analysis/POIS_ident_1s.h5'
    # data_homog, data_rs, pois = genera_pois_de_all_data_LDR(path_st)
    #
    # # pois = pd.read_hdf(path_st, 'POIS')
    # # data_rs = pd.read_hdf(path_st, 'data_rs')
    # # data_homog = pd.read_hdf(path_st, 'data_homog')
    # _info_tiempos('LOAD POIs')
    #
    # # ETIQUETADO MANUAL DE SAMPLES
    # # path_st_labeled = '/Users/uge/Desktop/LDR_analysis/POIS_labeled_1s.h5'
    # # pois_labeled = etiquetado_intervalos_LDR(data_homog, data_rs, pois, path_st=path_st_labeled)
    # # print_ok(pois_labeled)
    #
    # # vista_etiquetado_intervalos_LDR(data_homog, data_rs, path_st=path_st_labeled)

