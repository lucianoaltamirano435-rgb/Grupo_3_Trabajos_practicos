#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 15:21:30 2026

@author: lucianoaltamirano
"""

# =============================================================================
# PAQUETES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# Para k-modas se requiere el paquete 'kmodes' (instalar una sola vez):
# import sys
# !{sys.executable} -m pip install kmodes
from kmodes.kmodes import KModes

print("Paquetes cargados correctamente.")

# Seteo de directorio:
# os.chdir(r"C:\Users\gmpas\OneDrive\Escritorio\Seminario Programación\TP2")
os.chdir(r"/Volumes/ADATA HD330/Maestría Economía Aplicada UBA/Taller de programación/Trabajos prácticos/TP2.5 (preliminar)")

print(os.getcwd())

pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# PARTE I: CREACION DE VARIABLES
# =============================================================================

# %% PARTE I

# %% 0.1 Carga de datos:

# 0.1.1. Definicion de las columnas necesarias para el TP2:
"""
Respecto del TP1 se agregan CH12, CH13 y CH14 (necesarias para 'educ') y
PP3E_TOT y PP3F_TOT (necesarias para 'horastrab').
"""

'''
columnas_necesarias = [
    "CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "PONDERA",
    "CH03", "CH04", "CH06", "CH07", "CH08",
    "CH12", "CH13", "CH14",
    "NIVEL_ED", "ESTADO", "CAT_OCUP", "PP07H", "PP04C", "PP04C99",
    "EMPLEO", "SECTOR", "PP04D_COD", "P21", "P47T", "REGION",
    "PP07K", "PP07L", "PP07M", "PP03D", "PP3E_TOT", "PP3F_TOT",
    "CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "PONDERA", 
    "CH03", "CH04", "CH06", "CH07", "CH08", "NIVEL_ED", "ESTADO", 
    "CAT_OCUP", "PP07H", "PP04C", "PP04C99", "EMPLEO", 
    "SECTOR", "PP04D_COD", "P21", "P47T", "REGION", 
    "PP07K", "PP07L", "PP07M", "PP03D"
]

# 0.1.2. Importar la base de datos y transformarla a parquet (una sola vez):
bd_24 = pd.read_excel("usu_individual_T424.xlsx", usecols = columnas_necesarias)
bd_25 = pd.read_excel("usu_individual_T425.xlsx", usecols = columnas_necesarias)

bd_24.to_parquet("bd_24.parquet")
bd_25.to_parquet("bd_25.parquet")
'''

# 0.1.3. Cargar los archivos parquet:
bd_24 = pd.read_parquet("bd_24.parquet")
bd_25 = pd.read_parquet("bd_25.parquet")

print(bd_24.shape, bd_25.shape)

# %% 0.2 Correccion de valores sin sentido (heredado del TP1):

# 0.2.1. Ingresos - P21 y P47T: -9 se trata como NaN:
for base in [bd_24, bd_25]:
    base["P21"]  = base["P21"].replace(-9, np.nan)
    base["P47T"] = base["P47T"].replace(-9, np.nan)

# 0.2.2. Edad - CH06: -1 se trata como NaN:
for base in [bd_24, bd_25]:
    base["CH06"] = base["CH06"].replace(-1, np.nan)

# 0.2.3. Horas trabajadas - PP3E_TOT y PP3F_TOT: 999 (Ns/Nr) se trata como NaN:
for base in [bd_24, bd_25]:
    base["PP3E_TOT"] = base["PP3E_TOT"].replace(999, np.nan)
    base["PP3F_TOT"] = base["PP3F_TOT"].replace(999, np.nan)

# 0.2.4. Tamano del establecimiento - PP04C: 99 (Ns/Nr) se trata como NaN:
"""
CORRECCIÓN (i). En la version preliminar del TP2 se usaba PP04C en crudo
(bajo el nombre 'tam_estab_raw') tanto en la matriz de correlaciones como
en el PCA y en k-medias. PP04C codifica con 99 la no respuesta, de modo que
esos casos entraban al analisis como si se tratara del establecimiento mas
grande posible. Al ser aproximadamente 5.400 observaciones, distorsionaba
la correlación con informalidad, la dirección de los componentes del PCA y
la posición de los centroides. Se limpia aca, antes de cualquier cálculo.
"""
for base in [bd_24, bd_25]:
    base["PP04C"] = base["PP04C"].replace(99, np.nan)

# 0.2.5. Cantidad de ocupaciones adicionales - PP03D: 9 (Ns/Nr) se trata como NaN:
for base in [bd_24, bd_25]:
    base["PP03D"] = base["PP03D"].replace(9, np.nan)
    
#%% 0.3. Crear una categórica de año para la base de datos final:
bd_24['año'] = 2024
bd_25['año'] = 2025

# %% 0.3 Union de bases de datos:
bd = pd.concat([bd_24, bd_25], ignore_index = True)

print("Base unificada:", bd.shape)
print(bd["ANO4"].value_counts())

# %% 0.4 Ajuste de los ingresos por inflacion (pesos de 2024 a pesos de 2025):

"""
CORRECCION (iii). En el TP1 los ingresos de 2024 se llevaron a pesos de
2025 multiplicando por 1.314 (variacion del IPC de noviembre 2024 a
noviembre 2025, INDEC; se toma noviembre por ser el mes central del cuarto
trimestre). En la versión preliminar del TP2 este paso no estaba, con lo
cual la descriptiva, la matriz de correlaciones, el PCA y los clústeres
mezclaban pesos de distinto poder adquisitivo. Como el ingreso es una de
las variables con mayor varianza, esto desplazaba los centroides y el
primer componente principal.

El ajuste es ademas la condicion que habilita a poolear ambos anos en la
Parte II: sin el, agrupar observaciones de 2024 y 2025 en un mismo
analisis meclaria unidades monetarias distintas.

Se incluye una guarda ('ingresos_ajustados') porque en Spyder es habitual
reejecutar celdas: sin ella, correr esta celda dos veces aplicaria el
factor dos veces.
"""

factor = 1.314   # 31.4% de inflacion entre nov-2024 y nov-2025

if "ingresos_ajustados" not in globals():
    bd.loc[bd["ANO4"] == 2024, "P21"]  = bd.loc[bd["ANO4"] == 2024, "P21"]  * factor
    bd.loc[bd["ANO4"] == 2024, "P47T"] = bd.loc[bd["ANO4"] == 2024, "P47T"] * factor
    ingresos_ajustados = True
    print("Ingresos de 2024 ajustados por inflacion.")
else:
    print("Los ingresos ya habian sido ajustados; no se vuelve a aplicar el factor.")

print(bd.groupby("ANO4")[["P21", "P47T"]].mean().round(0))

# %% 0.5 Variable dummy_menor5 (heredada del TP1):

"""
Identifica las viviendas con al menos un hijo/a o nieto/a (CH03 = 3 o 5)
de hasta 5 anos, y luego generaliza el valor a todos los miembros de esa
vivienda. Se calcula sobre 'bd' y no sobre 'ocupados' porque necesita ver
a todos los integrantes del hogar, incluidos los menores, que por
definicion no estan en la base de ocupados.
"""

viviendas_con_menor = bd[
    (bd["CH03"].isin([3, 5])) &
    (bd["CH06"] <= 5)
]["CODUSU"].unique()

bd["dummy_menor5"] = bd["CODUSU"].isin(viviendas_con_menor).astype(int)

print(bd["dummy_menor5"].value_counts())

# %% 1. Creacion de variables

# %% 1.1 edad2 (edad al cuadrado):
bd["edad2"] = bd["CH06"] ** 2

print(bd[["CH06", "edad2"]].describe().round(1))

# %% 1.2 educ (anos de educacion formal):

"""
Se construye a partir de CH12 (nivel mas alto cursado), CH13 (si lo
finalizo) y CH14 (ultimo ano aprobado, para quienes no lo finalizaron).

Supuestos de duracion de cada nivel (estructura 6+6, coherente con el
ejemplo de la consigna: Secundario finalizado en "sexto" => educ = 12).
"""

anos_base_nivel = {
    0: np.nan,   # Valores no validos
    1: 0,        # Jardin / preescolar
    2: 0,        # Primario
    3: 0,        # EGB
    4: 6,        # Secundario   (Primario completo previo)
    5: 9,        # Polimodal    (EGB completo previo)
    6: 12,       # Terciario    (Secundario / Polimodal completo previo)
    7: 12,       # Universitario
    8: 17,       # Posgrado     (Universitario completo previo)
    9: np.nan    # Educacion especial
}

duracion_nivel = {
    0: np.nan,
    1: 0,
    2: 6,
    3: 9,
    4: 6,
    5: 3,
    6: 3,
    7: 5,
    8: 2,
    9: np.nan
}

bd["_anos_base"] = bd["CH12"].map(anos_base_nivel)
bd["_duracion"]  = bd["CH12"].map(duracion_nivel)

# CH14: 98 (educacion especial) y 99 (Ns/Nr) se tratan como NaN:
ch14_limpio = bd["CH14"].replace({98: np.nan, 99: np.nan})

# CH13: 9 (Ns/Nr) se pasa a NaN explicito para que no caiga en el default:
ch13_limpio = bd["CH13"].replace({9: np.nan})

condiciones = [
    ch13_limpio == 1,   # Finalizo el nivel  --> base + duracion completa
    ch13_limpio == 2,   # No finalizo        --> base + ultimo ano aprobado
]
resultados = [
    bd["_anos_base"] + bd["_duracion"],
    bd["_anos_base"] + ch14_limpio,
]

bd["educ"] = np.select(condiciones, resultados, default = np.nan)
bd = bd.drop(columns = ["_anos_base", "_duracion"])

# Verificacion de valores extremos y de consistencia por nivel:
print(bd["educ"].describe().round(2))
print("\nAnos de educacion promedio segun nivel mas alto cursado (CH12):")
print(bd.groupby("CH12")["educ"].mean().round(1))

# %% 1.3 horastrab (jefe/a de hogar) y horastrabj (extendida al hogar):

"""
horastrab: total de horas trabajadas por el jefe/a del hogar, como suma de
las horas en la ocupacion principal (PP3E_TOT) y en otras ocupaciones
(PP3F_TOT). Queda definida solamente para CH03 = 1.

horastrabj: la misma cantidad, pero asignada a todos los miembros del
hogar, de manera que pueda usarse como caracteristica del hogar en los
analisis a nivel individuo.
"""

bd["horastrab"] = np.where(
    bd["CH03"] == 1,
    bd["PP3E_TOT"] + bd["PP3F_TOT"],
    np.nan
)

# Tabla auxiliar con las horas del jefe/a de cada hogar:
horas_jefe = bd.loc[bd["CH03"] == 1, ["CODUSU", "NRO_HOGAR", "horastrab"]] \
               .drop_duplicates(subset = ["CODUSU", "NRO_HOGAR"]) \
               .rename(columns = {"horastrab": "horastrabj"})

bd = bd.merge(horas_jefe, on = ["CODUSU", "NRO_HOGAR"], how = "left")

# %% 1.4 nhogar (cantidad de miembros por hogar):
bd["nhogar"] = bd.groupby(["CODUSU", "NRO_HOGAR"])["COMPONENTE"].transform("nunique")

print(bd[["horastrab", "horastrabj", "nhogar"]].describe().round(2))

# %% 1.5 Creacion de la base de ocupados (heredada del TP1):

print(bd["ESTADO"].value_counts(normalize = True) * 100)

respondieron   = bd[bd["ESTADO"] != 0].copy()
norespondieron = bd[bd["ESTADO"] == 0].copy()

ocupados = respondieron[respondieron["ESTADO"] == 1].copy()

print("Base de ocupados:", ocupados.shape)

# %% 1.6 Creacion de variables dicotomicas (heredadas del TP1):

"""
CORRECCION (iv). La version preliminar del TP2 solo reconstruia cat_ocup2,
desc_jubilatorio y tam_estab. Se recuperan aca todas las dicotomicas del
TP1, porque son exactamente el input que pide el item 6.a del TP2:

CH04     --> sexo           (Masculino / Femenino)
CH07     --> estado_civil   (Pareja / Sin pareja)
SECTOR   --> sector_2       (Formal / Informal)
NIVEL_ED --> nivel_ed2      (Basico o sin nivel / Superior)
CAT_OCUP --> cat_ocup2      (No asalariado / Asalariado)
CH08     --> cobertura_med  (Cobertura / No cobertura)

Se conservan ademas las categoricas etiquetadas del TP1 que intervienen
en la definicion de informalidad (desc_jubilatorio, tam_estab_agrup).
"""

"sexo"
ocupados["sexo"] = ocupados["CH04"].map({
    1: "Masculino",
    2: "Femenino"
})

print(ocupados["sexo"].value_counts(normalize = True) * 100)

"estado_civil"
ocupados["estado_civil"] = ocupados["CH07"].map({
    1: "Pareja",
    2: "Pareja",
    3: "Sin pareja",
    4: "Sin pareja",
    5: "Sin pareja",
    9: np.nan
})

print(ocupados["estado_civil"].value_counts())

"sector_2"
ocupados["sector_2"] = ocupados["SECTOR"].map({
    1: "Formal",
    2: "Informal",
    3: np.nan,
    9: np.nan
})

print(ocupados["sector_2"].value_counts())

"nivel_ed2"
ocupados["nivel_ed2"] = ocupados["NIVEL_ED"].map({
    1: "Basico/No_nivel",
    2: "Basico/No_nivel",
    3: "Basico/No_nivel",
    4: "Basico/No_nivel",
    5: "Superior",
    6: "Superior",
    7: "Basico/No_nivel",
    9: np.nan
})

print(ocupados["nivel_ed2"].value_counts(normalize = True) * 100)

"cat_ocup2"
ocupados["cat_ocup2"] = ocupados["CAT_OCUP"].map({
    1: "No asalariado",
    2: "No asalariado",
    3: "Asalariado",
    4: np.nan,
    9: np.nan
})

print(ocupados["cat_ocup2"].value_counts(normalize = True) * 100)

"cobertura_med"
ocupados["cobertura_med"] = ocupados["CH08"].map({
    1:   "Cobertura",
    2:   "Cobertura",
    3:   "Cobertura",
    12:  "Cobertura",
    13:  "Cobertura",
    23:  "Cobertura",
    123: "Cobertura",
    4:   "No cobertura",
    9:   np.nan
})

print(ocupados["cobertura_med"].value_counts())

"desc_jubilatorio"
ocupados["desc_jubilatorio"] = ocupados["PP07H"].map({
    1: "Si",
    2: "No",
    9: np.nan
})

print(ocupados["desc_jubilatorio"].value_counts())

"tam_estab_agrup"
ocupados["tam_estab_agrup"] = ocupados["PP04C99"].map({
    1: "Hasta 5",
    2: "De 6 a 40",
    3: "Mas de 40",
    9: np.nan
})

print(ocupados["tam_estab_agrup"].value_counts())

# %% 1.7 Renombre de variables (heredado del TP1):

renombres = {
    "CODUSU":     "cod_vivienda",
    "NRO_HOGAR":  "nro_hogar",
    "COMPONENTE": "componente",
    "ANO4":       "ano",
    "TRIMESTRE":  "trimestre",
    "PONDERA":    "ponderador",
    "CH06":       "edad",
    "NIVEL_ED":   "nivel_ed",
    "ESTADO":     "cond_actividad",
    "CAT_OCUP":   "cat_ocup",
    "PP04C":      "tam_estab",
    "PP04C99":    "tam_estab_cod",
    "EMPLEO":     "tipo_empleo",
    "SECTOR":     "tipo_sector",
    "PP04D_COD":  "cod_ocupacion",
    "P21":        "ingreso_ppal",
    "P47T":       "ingreso_total",
    "REGION":     "region",
    "PP07H":      "desc_jub_cod",
    "PP07K":      "comprobante_sal",
    "PP07L":      "alcance_recibo",
    "PP07M":      "parte_sueldo",
    "PP03D":      "cant_ocupaciones_ad",
}

ocupados     = ocupados.rename(columns = renombres)
respondieron = respondieron.rename(columns = renombres)

"""
Nota sobre 'tam_estab': en la version preliminar del TP2 esta variable
aparecia como 'tam_estab_raw'. Se unifica bajo el nombre 'tam_estab' (ya
depurada de los 99) y se conserva en formato numerico, porque el PCA,
k-medias y el cluster jerarquico la necesitan como variable cuantitativa.
Las versiones etiquetadas se crean aparte, con sufijo '_cat', para no
pisar la informacion numerica (ver 1.8).
"""

# Verificacion de duplicados de nombres tras el renombre:
print(ocupados.columns.duplicated().sum())
print(ocupados.columns[ocupados.columns.duplicated()].tolist())

# %% 1.8 Etiquetado de variables categoricas (heredado del TP1):

"""
A diferencia del TP1, aca las etiquetas se guardan en columnas nuevas con
sufijo '_cat'. En el TP1 el mapeo pisaba la columna original y la
convertia en texto, lo cual mepidia volver a usarla en calculos numericos.
Como en el TP2 las mismas variables se necesitan en PCA y en cluster, se
conservan ambas versiones.
"""

"nivel_ed_cat"
ocupados["nivel_ed_cat"] = ocupados["nivel_ed"].map({
    1: "Primario incompleto",
    2: "Primario completo",
    3: "Secundario incompleto",
    4: "Secundario completo",
    5: "Superior incompleto",
    6: "Superior completo",
    7: "Sin instruccion",
    9: np.nan
})

"region_cat"
ocupados["region_cat"] = ocupados["region"].map({
    1:  "Gran Buenos Aires",
    40: "Noroeste",
    41: "Noreste",
    42: "Cuyo",
    43: "Pampeana",
    44: "Patagonia"
})

"cat_ocup_cat"
ocupados["cat_ocup_cat"] = ocupados["cat_ocup"].map({
    1: "Patron",
    2: "Cuenta propia",
    3: "Obrero o empleado",
    4: "Trab. familiar sin remuneracion",
    9: np.nan
})

"tipo_empleo_cat"
ocupados["tipo_empleo_cat"] = ocupados["tipo_empleo"].map({
    1: "Formal",
    2: "Informal",
    9: np.nan
})

"tipo_sector_cat"
ocupados["tipo_sector_cat"] = ocupados["tipo_sector"].map({
    1: "Formal",
    2: "Informal",
    3: "Hogares",
    9: np.nan
})

"comprobante_sal_cat"
ocupados["comprobante_sal_cat"] = ocupados["comprobante_sal"].map({
    1: "Recibo_sello",
    2: "Recibo_nosello",
    3: "Factura",
    4: "Nada",
    5: "Ad_honorem"
})

"alcance_recibo_cat"
ocupados["alcance_recibo_cat"] = ocupados["alcance_recibo"].map({
    1: "Totalidad",
    2: "Solo una parte",
    0: np.nan,
    9: np.nan
})

"parte_sueldo_cat"
ocupados["parte_sueldo_cat"] = ocupados["parte_sueldo"].map({
    1: "Totalidad",
    2: "Parte",
    0: np.nan,
    9: np.nan
})

# %% 1.9 Construccion del indicador de informalidad (heredado del TP1):

"""
Definicion asignada al grupo: se considera informal a quien es asalariado,
no tiene descuento jubilatorio y trabaja en un establecimiento de hasta
5 personas. El condicional sobre el tamano usa PP04C (codigos 1 a 5, que
en ese tramo coinciden con la cantidad exacta de personas) o, en su
defecto, la version agrupada PP04C99.
"""

ocupados["informal"] = (
    (ocupados["cat_ocup2"] == "Asalariado") &
    (ocupados["desc_jubilatorio"] == "No") &
    ((ocupados["tam_estab"] <= 5) | (ocupados["tam_estab_agrup"] == "Hasta 5"))
).map({True: "Informal", False: "Formal"})

print(ocupados["informal"].value_counts(normalize = True).round(4) * 100)
print(pd.crosstab(ocupados["ano"], ocupados["informal"], normalize = "index").round(4) * 100)

#%% 1.10 Creación de las bases de datos anuales:

# 1.10.1. Crear las bases de datos por año:
ocupados_X_2024 = ocupados[ocupados['ano'] == 2024]
ocupados_X_2025 = ocupados[ocupados['ano'] == 2025]

# 1.10.2. Separar las variables de respuesta:
y_2024 = ocupados_X_2024['informal']
y_2025 = ocupados_X_2025['informal']

# 1.10.3. Crear un código para hacer el 'join':
for df in [ocupados_X_2024, ocupados_X_2025]:
    df['id'] = df['cod_vivienda'].astype(str) + df['nro_hogar'].astype(str) + df['componente'].astype(str)
    
#%% 1.11 Hacer la unificación de la base de datos:

# 1.11.1. Añadir la variable id a las bases que se van a modelar:
y_2024_df = pd.DataFrame({
    'id': ocupados_X_2024['id'],
    'informal_2024': y_2024
})

# 1.11.2. Unificar la base de datos:
ocupados_X_2025_MM = ocupados_X_2025.merge(
    y_2024_df,
    on  = 'id',
    how = 'inner'
).drop(columns = ['id'])
print("Estructura de la base de datos de 2024 original:")
print(ocupados_X_2025.shape)

print("Estructura de la base de datos de 2025 con el 'join':")
print(ocupados_X_2025_MM.shape)
