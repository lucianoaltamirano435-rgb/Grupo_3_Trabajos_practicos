# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 17:05:51 2026

@author: gmpas
"""

"""
(*) El trabajo se desarrollo de forma conjunta.

Parte I y Parte II.A (PCA) --> Luciano Altamirano
Parte II.B (Cluster)       --> Gonzalo Pasiche

NOTA GENERAL SOBRE LA CONTINUIDAD CON EL TP1
--------------------------------------------
Este script reproduce la limpieza y la construccion de variables del TP1
(porque la base 'ocupados' no se guardo como archivo intermedio) y luego
avanza con las consignas del TP2. Se mantiene la numeracion y el formato
del TP1 para que ambos codigos sean legibles en conjunto.

Cuatro correcciones respecto de la version preliminar del TP2, cada una
documentada en el lugar donde se aplica:
  (i)   PP04C = 99 es "Ns/Nr" y debe ir a NaN antes de cualquier calculo.
  (ii)  PP04C es un codigo ordinal de tramos, no un conteo de empleados.
  (iii) El ajuste por inflacion de los ingresos de 2024 estaba ausente.
  (iv)  Faltaban las variables dicotomicas creadas en el TP1, que son
        justamente el input del item 6.a.

CRITERIO DE APERTURA POR ANO
----------------------------
La Parte I mantiene la apertura por ano en el item 3, que es donde la
consigna la pide explicitamente y donde ademas pregunta si los
coeficientes cambian entre 2024 y 2025. La Parte II, en cambio, trabaja
sobre la base completa (2024 + 2025 pooleados), porque ninguna de sus
consignas solicita separacion temporal: piden aplicar los metodos "de la
base de ocupados". Poolear duplica el tamano muestral, estabiliza las
curvas de codo y evita duplicar la salida grafica, algo relevante dado el
tope de cinco paginas del informe. La estabilidad de los resultados entre
anos se verifica igualmente, cruzando los mefistofelicos clusteres obtenidos contra el
ano de la observacion (ver items 4.a.6 y 6.a.5).
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
os.chdir(r"/Volumes/ADATA HD330/Maestría Economía Aplicada UBA/Taller de programación/Trabajos prácticos/Grupo_3_Trabajos_practicos/TP2")

print(os.getcwd())

pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# PARTE I: CREACION DE VARIABLES Y MAS ESTADISTICA DESCRIPTIVA
# (Luciano Altamirano)
# =============================================================================

# %% PARTE I

# %% 0.1 Carga de datos:

# 0.1.1. Definicion de las columnas necesarias para el TP2:
"""
Respecto del TP1 se agregan CH12, CH13 y CH14 (necesarias para 'educ') y
PP3E_TOT y PP3F_TOT (necesarias para 'horastrab').
"""

columnas_necesarias = [
    "CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "PONDERA",
    "CH03", "CH04", "CH06", "CH07", "CH08",
    "CH12", "CH13", "CH14",
    "NIVEL_ED", "ESTADO", "CAT_OCUP", "PP07H", "PP04C", "PP04C99",
    "EMPLEO", "SECTOR", "PP04D_COD", "P21", "P47T", "REGION",
    "PP07K", "PP07L", "PP07M", "PP03D", "PP3E_TOT", "PP3F_TOT"
]

# 0.1.2. Importar la base de datos y transformarla a parquet (una sola vez):
'''
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

# %% 2. Estadistica descriptiva:

"""
Se reportan las variables continuas limpias en el TP1 y las nuevas
variables numericas creadas en el TP2, sobre la base completa de ocupados.
"""

# 2.1. Definicion de los dos grupos de variables:
var_continuas_tp1 = ["edad", "ingreso_ppal", "ingreso_total",
                     "cant_ocupaciones_ad", "tam_estab"]
var_nuevas_tp2    = ["edad2", "educ", "horastrab", "horastrabj", "nhogar"]

var_continuas = var_continuas_tp1 + var_nuevas_tp2

# 2.2. Definicion de los percentiles a reportar:
percentiles = [0.25, 0.50, 0.75]

# 2.3. Construccion de la tabla de estadisticos descriptivos:
etiquetas_desc = {
    "edad":                "Edad",
    "ingreso_ppal":        "Ingreso ocupación principal",
    "ingreso_total":       "Ingreso total individual",
    "cant_ocupaciones_ad": "Cantidad de ocupaciones adicionales",
    "tam_estab":           "Tamano del establecimiento (código)",
    "edad2":               "Edad al cuadrado",
    "educ":                "Anos de educación",
    "horastrab":           "Horas trabajadas (jefe/a)",
    "horastrabj":          "Horas trabajadas del jefe/a (hogar)",
    "nhogar":              "Miembros del hogar",
}

tabla_desc = ocupados[var_continuas].describe(percentiles = percentiles).T
tabla_desc = tabla_desc.rename(columns = {
    "count": "N", "mean": "Promedio", "std": "Desvio Est.",
    "min": "Min", "25%": "P25", "50%": "P50", "75%": "P75", "max": "Max"
})
tabla_desc = tabla_desc[["N", "Promedio", "Desvio Est.", "Min",
                         "P25", "P50", "P75", "Max"]].round(2)
tabla_desc = tabla_desc.rename(index = etiquetas_desc)

print(tabla_desc)

# 2.4. Exportacion de la tabla a Excel:
with pd.ExcelWriter("tabla_descriptiva_TP2.xlsx") as writer:
    tabla_desc.to_excel(writer, sheet_name = "Descriptiva_TP2")

print("Guardada: tabla_descriptiva_TP2.xlsx")

# %% 3. Matriz de correlaciones (2024 y 2025) - base de ocupados:

"""
Este es el unico item del TP2 que se abre por ano, porque la consigna lo
pide de manera explicita y pregunta si los coeficientes cambian entre
2024 y 2025.

Las variables solicitadas son informal, edad, edad2, educ, horastrabj,
nhogar, P21 (ingreso_ppal) y PP04C (tam_estab). Se usa 'tam_estab' ya
depurada de los codigos 99 y 'ingreso_ppal' ya expresada en pesos de 2025.
La informalidad se codifica como 0 = Formal y 1 = Informal, de modo que un
coeficiente positivo se lee como asociacion con mayor informalidad.
"""
from scipy import stats

# 3.1. Definición de las variables y sus etiquetas:
variables_corr = ["informal", "edad", "edad2", "educ", "horastrabj",
                  "nhogar", "ingreso_ppal", "tam_estab"]
etiquetas_corr = {
    "informal":     "Informalidad",
    "edad":         "Edad",
    "edad2":        "Edad al cuadrado",
    "educ":         "Años de educación",
    "horastrabj":   "Horas trabajadas (jefe)",
    "nhogar":       "Miembros del hogar",
    "ingreso_ppal": "Ingreso ocupación principal",
    "tam_estab":    "Tamaño del establecimiento",
}

# 3.2. Función auxiliar para calcular matriz de valores p:
def calcular_pvalues(df):
    df = df.dropna()
    # n = len(df)
    cols = df.columns
    pvalues = pd.DataFrame(np.ones((len(cols), len(cols))), columns=cols, index=cols)
    for i in cols:
        for j in cols:
            if i != j:
                _, p = stats.pearsonr(df[i], df[j])
                pvalues.loc[i, j] = p
    return pvalues

# 3.3. Cálculo de las matrices de correlación y valores p por año:
matrices_corr  = {}
matrices_pval  = {}

for ano_val in [2024, 2025]:
    subset = ocupados[ocupados["ano"] == ano_val][variables_corr].copy()
    subset["informal"] = subset["informal"].map({"Formal": 0, "Informal": 1})

    # Matriz de correlaciones
    corr_matrix = subset.corr()
    corr_matrix = corr_matrix.rename(index=etiquetas_corr, columns=etiquetas_corr)
    matrices_corr[ano_val] = corr_matrix

    # Matriz de valores p
    subset_renombrado = subset.rename(columns=etiquetas_corr)
    pval_matrix = calcular_pvalues(subset_renombrado)
    matrices_pval[ano_val] = pval_matrix

    print(f"\nMatriz de correlación - Ocupados {ano_val}")
    print(corr_matrix.round(2))
    print(f"\nMatriz de valores p - Ocupados {ano_val}")
    print(pval_matrix.round(3))

# 3.4. Diferencia entre las matrices de ambos años (apoyo para el comentario):
dif_corr = (matrices_corr[2025] - matrices_corr[2024]).round(3)
print("\nDiferencia de coeficientes (2025 - 2024):")
print(dif_corr)

# 3.5. Exportación de las matrices a Excel:
with pd.ExcelWriter("matriz_correlaciones_TP2.xlsx") as writer:
    for ano_val, matriz in matrices_corr.items():
        matriz.round(2).to_excel(writer, sheet_name=f"Corr_{ano_val}")
    for ano_val, pval in matrices_pval.items():
        pval.round(3).to_excel(writer, sheet_name=f"Pval_{ano_val}")
    dif_corr.to_excel(writer, sheet_name="Diferencia")

# 3.5. Heatmaps de las matrices de correlacion:
for ano_val in [2024, 2025]:

    corr_matrix = matrices_corr[ano_val]

    # Mascara para mostrar solo el triangulo inferior:
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))

    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (10, 8))

    sns.heatmap(
        corr_matrix,
        mask = mask,
        vmin = -1, vmax = 1,
        annot = True, fmt = ".2f",
        cmap = "coolwarm",
        linewidths = 0.5,
        ax = ax,
        cbar_kws = {"label": "Correlacion de Pearson", "shrink": 0.8}
    )

    ax.set_title(
        f"Matriz de correlaciones - Ocupados {ano_val}",
        fontsize = 13, fontweight = "bold", loc = "center", pad = 15
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 35, ha = "right", fontsize = 9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 9)

    plt.tight_layout()
    plt.savefig(f"heatmap_correlaciones_{ano_val}.png", dpi = 300, bbox_inches = "tight")
    plt.show()

# =============================================================================
# PARTE II: METODOS NO SUPERVISADOS
# =============================================================================

# %% Definicion comun de variables y preparacion de los datos (base completa):

"""
Los items 1 a 5 del TP2 trabajan con el mismo conjunto de siete variables
sobre la base completa de ocupados (2024 + 2025). Se prepara una sola vez
el objeto 'datos_pool' con la base filtrada y su versión estandarizada,
para no repetir el bloque en cada item.

La estandarizacion es imprescindible en PCA, k-medias y cluster jerarquico
porque las siete variables estan en escalas muy distintas: el ingreso se
mide en cientos de miles de pesos y la cantidad de miembros del hogar en
unidades. Sin estandarizar, el ingreso dominaria por completo tanto la
direccion de los componentes como la distancia euclidea.
"""

variables_cluster = ["edad", "edad2", "educ", "horastrabj",
                     "nhogar", "ingreso_ppal", "tam_estab"]

etiquetas_cluster = {
    "edad":         "Edad",
    "edad2":        "Edad2",
    "educ":         "Años de educación",
    "horastrabj":   "Horas trab. (jefe)",
    "nhogar":       "Miembros del hogar",
    "ingreso_ppal": "Ingreso principal",
    "tam_estab":    "Tamaño establec.",
}

# (a) Filtro de observaciones completas:
datos_pool = ocupados.dropna(subset = variables_cluster + ["informal"]).copy()

# (b) Estandarizacion de las variables:
scaler_pool = StandardScaler()
X_pool = scaler_pool.fit_transform(datos_pool[variables_cluster])

print(f"Observaciones completas: {datos_pool.shape[0]:,} de {len(ocupados):,} ocupados.")
print("\nDistribucion por ano de la muestra de analisis:")
print(datos_pool["ano"].value_counts().sort_index())
print("\nDistribucion de la informalidad:")
print(datos_pool["informal"].value_counts(normalize = True).round(4) * 100)

# %% Verificacion del solapamiento muestral entre 2024 y 2025:

"""
La EPH tiene un esquema de rotacion de viviendas por el cual una parte de
los hogares encuestados en el cuarto trimestre de 2024 vuelve a ser
encuestada en el cuarto trimestre de 2025. Al poolear ambos anos, esas
personas aparecen dos veces y no constituyen observaciones independientes.

Para un analisis de clustering descriptivo esto no invalida los
resultados, pero corresponde cuantificarlo y mencionarlo en el informe.
La identificacion de una misma persona se hace por la combinacion de
vivienda, hogar y numero de componente.
"""

solape = datos_pool.groupby(["cod_vivienda", "nro_hogar", "componente"])["ano"].nunique()

n_personas   = len(solape)
n_en_ambos   = (solape == 2).sum()
pct_en_ambos = n_en_ambos / n_personas * 100

print(f"Personas distintas en la muestra:   {n_personas:,}")
print(f"Presentes en ambos anos:            {n_en_ambos:,}")
print(f"Porcentaje de solapamiento:         {pct_en_ambos:.1f}%")

# =============================================================================
# PARTE II.A: ANALISIS DE COMPONENTES PRINCIPALES (PCA)
# (Luciano Altamirano)
# =============================================================================

# %%---- 1. PCA: scores del primer y segundo componente -----

"""
Se aplica PCA sobre las siete variables estandarizadas y se grafican los
scores de los dos primeros componentes, distinguiendo por color a las
personas formales e informales. La informalidad no participa del calculo:
se usa unicamente para colorear, de modo que el grafico muestra si las
siete variables contienen por si solas un eje de variacion que separe a
ambos grupos.
"""

"1.1 Ajuste del PCA sobre los datos estandarizados:"
pca    = PCA(n_components = len(variables_cluster), random_state = 42)
scores = pca.fit_transform(X_pool)

datos_pool["PC1"] = scores[:, 0]
datos_pool["PC2"] = scores[:, 1]

var_pc1 = pca.explained_variance_ratio_[0] * 100
var_pc2 = pca.explained_variance_ratio_[1] * 100

print(f"Varianza explicada por PC1: {var_pc1:.1f}%")
print(f"Varianza explicada por PC2: {var_pc2:.1f}%")

"1.2 Grafico de dispersion de los scores (PC1 vs PC2):"
sns.set_style("white")
fig, ax = plt.subplots(figsize = (9, 7))

sns.scatterplot(
    data = datos_pool,
    x = "PC1", y = "PC2",
    hue = "informal",
    palette = {"Formal": "#2196F3", "Informal": "#F44336"},
    alpha = 0.35, s = 16, edgecolor = "none",
    ax = ax
)

ax.set_title("PCA - Scores de los dos primeros componentes\nOcupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel(f"Componente 1 ({var_pc1:.1f}% de la varianza)")
ax.set_ylabel(f"Componente 2 ({var_pc2:.1f}% de la varianza)")
ax.axhline(0, color = "gray", lw = 0.8, ls = "--")
ax.axvline(0, color = "gray", lw = 0.8, ls = "--")
ax.legend(title = "Condicion")

plt.tight_layout()
plt.savefig("pca_scores.png", dpi = 300, bbox_inches = "tight")
plt.show()

"1.3 Identificacion de outliers (mas de 5 desvios en PC1 o PC2):"
lim_pc1 = 5 * datos_pool["PC1"].std()
lim_pc2 = 5 * datos_pool["PC2"].std()

outliers = datos_pool[(datos_pool["PC1"].abs() > lim_pc1) |
                      (datos_pool["PC2"].abs() > lim_pc2)]

print(f"\nOutliers detectados (|score| > 5 desvios): {len(outliers)}")

if len(outliers) > 0:
    print("\nPerfil de los outliers:")
    print(outliers[variables_cluster].describe().round(1))
    print("\nComposicion segun informalidad:")
    print(outliers["informal"].value_counts())

"1.4 Comparacion de los scores promedio por condicion (apoyo interpretativo):"
print("\nScores promedio por condicion:")
print(datos_pool.groupby("informal")[["PC1", "PC2"]].mean().round(3))

# %%---- 2. PCA: grafico de loadings (biplot) -----

"""
Se superponen los scores de los individuos y las flechas de los
ponderadores (loadings) de cada variable sobre los dos primeros
componentes. Las flechas se reescalan para que sean visibles junto a la
nube de puntos; lo interpretable es su direccion y su longitud relativa,
no su magnitud absoluta.
"""

"2.1 Recuperacion de los loadings:"
loadings = pca.components_.T   # filas = variables, columnas = componentes

"2.2 Factor de escala para que las flechas sean comparables con los scores:"
escala = np.abs(scores[:, :2]).max() * 0.8

"2.3 Grafico del biplot:"
sns.set_style("white")
fig, ax = plt.subplots(figsize = (10, 8))

colores = datos_pool["informal"].map({"Formal": "#2196F3", "Informal": "#F44336"})
ax.scatter(scores[:, 0], scores[:, 1], c = colores, alpha = 0.2,
           s = 12, edgecolor = "none")

# Offsets manuales para separar etiquetas solapadas (dx, dy en unidades del gráfico):
offsets_etiquetas = {
    "edad":         (0.3,  0.0),
    "edad2":        (0.3, -0.3),
    "educ":         (-0.2, -0.6),
    "tam_estab":    (-0.2,  0.0),
    "horastrabj":   (-0.3,  0.3),
    "nhogar":       (-0.2, -0.3),
    "ingreso_ppal": ( 0.2,  0.3),
    "P21":          ( 0.2,  0.3),
}

for i, var in enumerate(variables_cluster):
    lx = loadings[i, 0] * escala
    ly = loadings[i, 1] * escala
    ax.arrow(0, 0, lx, ly,
             color="black", width=0.01, head_width=0.15,
             length_includes_head=True)
    dx, dy = offsets_etiquetas.get(var, (0.0, 0.0))
    ax.text(lx * 1.12 + dx,
            ly * 1.12 + dy,
            etiquetas_cluster[var],
            color="black", fontsize=10, fontweight="bold",
            ha="center", va="center")

ax.set_title("PCA - Biplot de scores y loadings\nOcupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel(f"Componente 1 ({var_pc1:.1f}% de la varianza)")
ax.set_ylabel(f"Componente 2 ({var_pc2:.1f}% de la varianza)")
ax.axhline(0, color = "gray", lw = 0.8, ls = "--")
ax.axvline(0, color = "gray", lw = 0.8, ls = "--")

# Leyenda construida a mano, porque el scatter se hizo con colores directos:
handles = [Line2D([0], [0], marker = "o", color = "w", label = "Formal",
                  markerfacecolor = "#2196F3", markersize = 8),
           Line2D([0], [0], marker = "o", color = "w", label = "Informal",
                  markerfacecolor = "#F44336", markersize = 8)]
ax.legend(handles = handles, title = "Condicion", loc = "best")

plt.tight_layout()
plt.savefig("pca_biplot.png", dpi = 300, bbox_inches = "tight")
plt.show()

"2.4 Tabla de loadings de los dos primeros componentes:"
tabla_loadings = pd.DataFrame(
    loadings[:, :2],
    index = [etiquetas_cluster[v] for v in variables_cluster],
    columns = ["Componente 1", "Componente 2"]
).round(3)

print("\nLoadings de los dos primeros componentes:")
print(tabla_loadings)

with pd.ExcelWriter("pca_loadings_TP2.xlsx") as writer:
    tabla_loadings.to_excel(writer, sheet_name = "Loadings")

# %%---- 3. PCA: proporcion de varianza explicada -----

"""
Se grafica la proporcion de varianza explicada por cada uno de los siete
componentes, junto con la varianza acumulada. La lectura debe hacerse en
conjunto con la matriz de correlaciones del item 3 de la Parte I: cuanto
mas correlacionadas estan las variables originales, mas varianza logran
concentrar los primeros componentes. Si las correlaciones son bajas, la
varianza se reparte de manera pareja entre los siete componentes y PCA no
consigue reducir dimensiones.
"""

"3.1 Recuperacion de la varianza explicada:"
var_ratio   = pca.explained_variance_ratio_ * 100
var_acum    = np.cumsum(var_ratio)
componentes = np.arange(1, len(var_ratio) + 1)

"3.2 Grafico de barras (individual) y linea (acumulada):"
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = (9, 6))

ax.bar(componentes, var_ratio, color = "steelblue",
       edgecolor = "white", label = "Varianza explicada")
ax.plot(componentes, var_acum, marker = "o", color = "#F44336",
        label = "Varianza acumulada")

for x, y in zip(componentes, var_ratio):
    ax.text(x, y + 1.5, f"{y:.1f}", ha = "center", fontsize = 9)

ax.set_title("PCA - Proporcion de varianza explicada\nOcupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel("Componente principal")
ax.set_ylabel("Porcentaje de la varianza (%)")
ax.set_xticks(componentes)
ax.set_ylim(0, 105)
ax.legend(loc = "center right")

plt.tight_layout()
plt.savefig("pca_varianza_explicada.png", dpi = 300, bbox_inches = "tight")
plt.show()

"3.3 Tabla de varianza explicada e acumulada:"
tabla_varianza = pd.DataFrame({
    "Varianza explicada (%)": var_ratio.round(1),
    "Varianza acumulada (%)": var_acum.round(1)
}, index = [f"PC{i}" for i in componentes])

print("\nVarianza explicada por componente:")
print(tabla_varianza)

# =============================================================================
# PARTE II.B: ANALISIS CLUSTER
# (Gonzalo Pasiche)
# =============================================================================

# %% Funcion auxiliar para evaluar la correspondencia cluster - informalidad:

"""
Como el cluster no supervisado no asigna etiquetas, cada particion se
evalua con dos medidas:

- Pureza: porcentaje de casos correctamente ubicados si a cada cluster se
  le asigna la categoria (Formal / Informal) mas frecuente en su interior.
- Linea de base: porcentaje de la categoria mayoritaria en la muestra. Es
  el resultado que se obtendria clasificando a todos en un unico grupo, y
  por lo tanto el piso contra el cual hay que comparar la pureza.

Si la pureza no supera claramente a la linea de base, el algoritmo no esta
recuperando la informalidad, aunque el porcentaje suene alto.
"""

def evaluar_cluster(etiquetas, y_real, titulo = ""):
    tabla  = pd.crosstab(etiquetas, y_real)
    pureza = tabla.max(axis = 1).sum() / tabla.values.sum() * 100
    base   = y_real.value_counts(normalize = True).max() * 100

    print(f"\n{titulo}")
    print(tabla)
    print(f"Pureza:          {pureza:.2f}%")
    print(f"Linea de base:   {base:.2f}%  (categoria mayoritaria)")
    print(f"Ganancia:        {pureza - base:+.2f} puntos porcentuales")

    return tabla, pureza, base

# %%---- 4.a Cluster k-medias con k = 2 -----

"""
Se corre k-medias con k = 2 y n_init = 20 sobre las siete variables
estandarizadas. El valor de k queda fijado por la consigna: la pregunta es
si los dos grupos que el algoritmo encuentra por su cuenta coinciden con
la particion entre formales e informales.

n_init = 20 significa veinte arranques con centroides iniciales distintos,
conservando la mejor solucion. K-medias converge a optimos locales y el
resultado depende de la inicializacion, de modo que repetir el arranque es
la proteccion estandar contra una solucion desafortunada.
"""

"4.a.1 Entrenamiento del k-medias (k = 2, n_init = 20):"
kmeans = KMeans(n_clusters = 2, n_init = 20, random_state = 42)
datos_pool["cluster_kmedias"] = kmeans.fit_predict(X_pool)

print(datos_pool["cluster_kmedias"].value_counts().sort_index())

"4.a.2 Grafico de los resultados (2 de las 7 variables):"
sns.set_style("white")
fig, axs = plt.subplots(1, 2, figsize = (14, 6))

# Panel (a): coloreado por cluster asignado
sns.scatterplot(
    data = datos_pool, x = "edad", y = "ingreso_ppal",
    hue = "cluster_kmedias", palette = "Set1",
    alpha = 0.35, s = 16, edgecolor = "none", ax = axs[0]
)
axs[0].set_title("Clusteres asignados por k-medias")
axs[0].set_xlabel("Edad")
axs[0].set_ylabel("Ingreso ocupacion principal")
axs[0].legend(title = "Cluster")

# Panel (b): coloreado por informalidad observada
sns.scatterplot(
    data = datos_pool, x = "edad", y = "ingreso_ppal",
    hue = "informal",
    palette = {"Formal": "#2196F3", "Informal": "#F44336"},
    alpha = 0.35, s = 16, edgecolor = "none", ax = axs[1]
)
axs[1].set_title("Informalidad observada")
axs[1].set_xlabel("Edad")
axs[1].set_ylabel("Ingreso ocupacion principal")
axs[1].legend(title = "Condicion")

fig.suptitle("K-medias (k = 2) - Ocupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold")

plt.tight_layout()
plt.savefig("kmedias_cluster.png", dpi = 300, bbox_inches = "tight")
plt.show()

"4.a.3 Comparacion de los clusteres con la informalidad observada:"
tabla_km, pureza_km, base_km = evaluar_cluster(
    datos_pool["cluster_kmedias"], datos_pool["informal"],
    titulo = "K-medias (k = 2) vs informalidad - Ocupados 2024 + 2025"
)

"4.a.4 Composicion de cada cluster segun informalidad:"
composicion_km = pd.crosstab(datos_pool["cluster_kmedias"],
                             datos_pool["informal"], normalize = "index") * 100

print("\nComposicion de cada cluster (%):")
print(composicion_km.round(2))

"4.a.5 Perfil de los centroides (en las unidades originales):"
centroides = pd.DataFrame(
    scaler_pool.inverse_transform(kmeans.cluster_centers_),
    columns = [etiquetas_cluster[v] for v in variables_cluster]
).round(1)
centroides.index.name = "Cluster"
centroides["N"] = datos_pool["cluster_kmedias"].value_counts().sort_index().values

print("\nCentroides en unidades originales:")
print(centroides.T)

"4.a.6 Estabilidad de los clusteres entre 2024 y 2025:"
"""
Control que reemplaza a la apertura por ano: si la composicion de cada
cluster es similar en ambos anos, los grupos encontrados son estables en
el tiempo y el pooleo no esta ocultando un cambio estructural.
"""
estabilidad_km = pd.crosstab(datos_pool["cluster_kmedias"],
                             datos_pool["ano"], normalize = "columns") * 100

print("\nComposicion de los clusteres por ano (%) - k-medias:")
print(estabilidad_km.round(2))

# %%---- 4.b Metodo del codo (Elbow) para k-medias -----

"""
Se grafica la inercia (suma de cuadrados intra-cluster) para k = 1 hasta
k = 40. La inercia siempre decrece al agregar clusteres, y con k igual a
la cantidad de observaciones vale cero, de modo que el criterio no es
minimizarla sino localizar el punto a partir del cual la mejora marginal
se vuelve pequena.
"""

rango_k = range(1, 41)

"4.b.1 Calculo de la inercia para cada valor de k:"
inercias = []

for k in rango_k:
    km_elbow = KMeans(n_clusters = k, n_init = 20, random_state = 42)
    km_elbow.fit(X_pool)
    inercias.append(km_elbow.inertia_)
    if k % 10 == 0:
        print(f"K-medias: completado hasta k = {k}")

"4.b.2 Grafico de la curva del codo:"
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = (10, 6))

ax.plot(rango_k, inercias, marker = "o", markersize = 4, color = "steelblue")

ax.set_title("Metodo del codo (Elbow) - K-medias\nOcupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel("Numero de clusteres (k)")
ax.set_ylabel("Inercia (WCSS)")
ax.set_xticks(range(0, 41, 5))

plt.tight_layout()
plt.savefig("elbow_kmedias.png", dpi = 300, bbox_inches = "tight")
plt.show()

"4.b.3 Reduccion porcentual de la inercia al pasar de k a k+1:"
"""
El codo suele ser ambiguo a simple vista. Esta tabla lo hace explicito:
muestra cuanto se reduce la inercia al agregar un cluster mas, de modo que
la eleccion de k pueda justificarse con un numero y no solo con la
inspeccion visual.
"""
serie_inercia   = pd.Series(inercias, index = rango_k)
reduccion_iner  = (serie_inercia.diff(-1) / serie_inercia * 100).round(2)

print("\nReduccion porcentual de la inercia al pasar de k a k+1 (primeros 12):")
print(reduccion_iner.head(12))

# %%---- 5. Cluster jerarquico -----

"""
Un dendrograma es la representacion grafica de un procedimiento de
agrupamiento aglomerativo: se parte de tantos grupos como observaciones y
en cada paso se fusionan los dos grupos mas parecidos, hasta llegar a un
unico grupo. La altura de cada union indica la disimilitud a la que se
produjo la fusion, de modo que cortar el arbol a una altura determinada
equivale a elegir un numero de clusteres. Se usa el metodo de Ward, que
fusiona los grupos minimizando el aumento de la varianza interna, criterio
analogo al de k-medias.

El algoritmo requiere calcular y almacenar la matriz de distancias entre
todos los pares de observaciones, cuyo tamano crece con el cuadrado de la
cantidad de casos: con las casi 40.000 observaciones de la base pooleada
serian mas de mil millones de distancias, inviable en memoria y con un
grafico ilegible. Por eso se trabaja con una muestra aleatoria de 300
casos, fijada con random_state para que el resultado sea reproducible.
"""

"5.1 Extraccion de una muestra aleatoria:"
muestra = datos_pool.sample(n = 300, random_state = 42)

print("Composicion de la muestra por ano:")
print(muestra["ano"].value_counts().sort_index())

"5.2 Estandarizacion de las variables sobre la muestra:"
scaler_m  = StandardScaler()
X_muestra = scaler_m.fit_transform(muestra[variables_cluster])

"5.3 Calculo de las uniones jerarquicas (metodo de Ward):"
enlaces = linkage(X_muestra, method = "ward")

"5.4 Grafico del dendrograma:"
sns.set_style("white")
fig, ax = plt.subplots(figsize = (14, 7))

dendrogram(enlaces, ax = ax, no_labels = True, color_threshold = 0)

ax.set_title("Dendrograma - Clustering jerarquico (Ward)\nMuestra de 300 ocupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel("Observaciones (muestra aleatoria de 300 ocupados)")
ax.set_ylabel("Distancia de fusion (disimilitud)")

plt.tight_layout()
plt.savefig("dendrograma.png", dpi = 300, bbox_inches = "tight")
plt.show()

"5.5 Corte del arbol en 2 grupos y comparacion con la informalidad:"
grupos_jer = fcluster(enlaces, t = 2, criterion = "maxclust")

tabla_jer, pureza_jer, base_jer = evaluar_cluster(
    pd.Series(grupos_jer, index = muestra.index),
    muestra["informal"],
    titulo = "Jerarquico (corte en 2) vs informalidad - Muestra de 300 casos"
)

"5.6 Perfil de los grupos jerarquicos (promedios en unidades originales):"
muestra_perfil = muestra.copy()
muestra_perfil["grupo_jerarquico"] = grupos_jer

perfil_jer = muestra_perfil.groupby("grupo_jerarquico")[variables_cluster].mean().round(1)
perfil_jer = perfil_jer.rename(columns = etiquetas_cluster)

print("\nPerfil de los grupos jerarquicos (promedios):")
print(perfil_jer.T)

# %%---- 6.a Cluster k-modas con k = 2 -----

"""
K-modas es la adaptacion de k-medias a datos categoricos: reemplaza la
media por la moda como representante de cada grupo y la distancia
euclidea por la distancia de Hamming, que cuenta simplemente en cuantos
atributos difieren dos observaciones. Por eso las variables entran como
categorias y no hace falta convertirlas en dummies 0/1: si se dummificara,
cada categoria pesaria como una variable independiente y las variables con
mas niveles quedarian sobrerrepresentadas en la distancia.

INPUT (segun el item 6.a):
  (1) Las variables dicotomicas creadas en el TP1: sexo, estado_civil,
      sector_2, nivel_ed2, cat_ocup2, cobertura_med y dummy_menor5.
  (2) Las variables numericas nuevas del TP2, categorizadas en tramos:
      educ, horastrabj y nhogar.
  (3) Se agregan ademas, categorizadas, la edad y las dos continuas
      centrales del TP1 (ingreso de la ocupacion principal y tamano del
      establecimiento), que aportan informacion y son las mismas que se
      usaron en PCA y k-medias.

Se excluye la variable objetivo 'informal', tal como pide la consigna. La
razon de fondo es que el clustering no supervisado solo tiene sentido si
el algoritmo no ve la respuesta: si la variable objetivo entrara al input,
"descubrirla" seria trivial y circular.

Sobre 'edad2': categorizada en tramos produce exactamente la misma
particion que 'edad', porque el cuadrado es una transformacion monotona
para valores positivos. Incluir ambas duplicaria el peso de la edad en la
distancia de Hamming sin aportar informacion, de modo que se conserva
unicamente 'edad_cat'.
"""

# %% 6.a.1 Categorizacion de las variables numericas:

"""
CORRECCION (ii) - sobre los cortes de 'tam_estab'. PP04C no registra la
cantidad exacta de empleados sino un codigo ordinal de tramos: los valores
1 a 5 corresponden a 1 a 5 personas, pero de ahi en adelante cada codigo
representa un intervalo (6 = "6 a 10", 7 = "11 a 25", 8 = "26 a 40",
9 = "41 a 100", y asi hasta 12 = "mas de 500"). Cortar en 40 o en 1000
como si fueran cantidades de personas agruparia mal los casos. Los cortes
que se usan aca (1-5, 6-8, 9-12) reproducen exactamente las categorias de
PP04C99, que es la version agrupada que publica el INDEC.
"""

"edad_cat"
ocupados["edad_cat"] = pd.cut(
    ocupados["edad"],
    bins = [0, 25, 35, 45, 60, 120],
    labels = ["Hasta 24", "25 a 34", "35 a 44", "45 a 59", "60 y mas"]
)

print(ocupados["edad_cat"].value_counts().sort_index())

"educ_cat"
ocupados["educ_cat"] = pd.cut(
    ocupados["educ"],
    bins = [-1, 6, 11, 12, 16, 30],
    labels = ["Primario o menos", "Secundario incompleto", "Secundario completo",
              "Superior incompleto", "Superior completo"]
)

print(ocupados["educ_cat"].value_counts().sort_index())

"horastrabj_cat"
"""
Cortes basados en la normativa laboral y en las definiciones del INDEC:
hasta 20 horas es una jornada marginal, de 21 a 34 corresponde a
subocupacion horaria, de 35 a 45 es jornada completa y mas de 45 es
sobreocupacion.
"""
ocupados["horastrabj_cat"] = pd.cut(
    ocupados["horastrabj"],
    bins = [-1, 20, 34, 45, 300],
    labels = ["Marginal", "Parcial", "Completa", "Extendida"]
)

print(ocupados["horastrabj_cat"].value_counts().sort_index())

"nhogar_cat"
ocupados["nhogar_cat"] = pd.cut(
    ocupados["nhogar"],
    bins = [0, 1, 2, 4, 50],
    labels = ["Unipersonal", "2 miembros", "3 a 4 miembros", "5 o mas"]
)

print(ocupados["nhogar_cat"].value_counts().sort_index())

"ingreso_ppal_cat"
"""
Se usan quintiles calculados sobre los ingresos positivos de la base
pooleada, lo cual es valido porque los valores de 2024 ya fueron llevados
a pesos de 2025. Los ingresos iguales a cero (trabajadores sin
remuneracion, principalmente familiares) se apartan como categoria propia,
porque no son un nivel bajo de ingreso sino una situacion cualitativamente
distinta.
"""
ocupados["ingreso_ppal_cat"] = pd.qcut(
    ocupados["ingreso_ppal"].where(ocupados["ingreso_ppal"] > 0),
    q = 5,
    labels = ["Q1 (mas bajo)", "Q2", "Q3", "Q4", "Q5 (mas alto)"]
)
ocupados["ingreso_ppal_cat"] = ocupados["ingreso_ppal_cat"].cat.add_categories("Sin ingreso")
ocupados.loc[ocupados["ingreso_ppal"] == 0, "ingreso_ppal_cat"] = "Sin ingreso"

print(ocupados["ingreso_ppal_cat"].value_counts())

"tam_estab_cat2"
ocupados["tam_estab_cat2"] = pd.cut(
    ocupados["tam_estab"],
    bins = [0, 5, 8, 12],
    labels = ["Hasta 5", "De 6 a 40", "Mas de 40"]
)

print(ocupados["tam_estab_cat2"].value_counts().sort_index())

# Verificacion cruzada contra la version agrupada del INDEC (PP04C99):
print("\nConsistencia entre los cortes propios y PP04C99:")
print(pd.crosstab(ocupados["tam_estab_cat2"], ocupados["tam_estab_agrup"]))

# %% 6.a.2 Construccion de la matriz de entrada:

"""
Tratamiento de los faltantes: k-modas no admite NaN. En lugar de eliminar
las observaciones incompletas se los convierte en una categoria propia
("Sin dato"). Es el criterio habitual en clustering de datos categoricos
por dos motivos. Primero, porque un nivel adicional no le cuesta nada al
algoritmo, que trabaja sobre etiquetas y no sobre magnitudes; en k-medias,
en cambio, habria que imputar un numero. Segundo, porque en la EPH la no
respuesta es informativa: no declarar ingresos o descuento jubilatorio se
asocia a perfiles laborales particulares, de modo que descartarla sesga la
muestra. Con dropna() se perderia alrededor de un 30% de los casos, sobre
todo por ingreso_ppal y horastrabj.

Como consecuencia, k-modas trabaja con la totalidad de los ocupados,
mientras que k-medias y el jerarquico se limitan a las observaciones
completas. La diferencia de tamano muestral entre metodos se reporta en la
tabla de sintesis del item 6.c.
"""

vars_kmodas = [
    # (1) Dicotomicas creadas en el TP1:
    "sexo", "estado_civil", "sector_2", "nivel_ed2", "cat_ocup2",
    "cobertura_med", "dummy_menor5",
    # (2) Nuevas variables numericas del TP2, categorizadas:
    "educ_cat", "horastrabj_cat", "nhogar_cat",
    # (3) Variables continuas del TP1, categorizadas:
    "edad_cat", "ingreso_ppal_cat", "tam_estab_cat2",
]

etiquetas_kmodas = {
    "sexo":             "Sexo",
    "estado_civil":     "Estado civil",
    "sector_2":         "Sector (EPH)",
    "nivel_ed2":        "Nivel educativo",
    "cat_ocup2":        "Categoria ocupacional",
    "cobertura_med":    "Cobertura medica",
    "dummy_menor5":     "Hogar con menor de 5",
    "educ_cat":         "Anos de educacion",
    "horastrabj_cat":   "Horas trabajadas (jefe)",
    "nhogar_cat":       "Miembros del hogar",
    "edad_cat":         "Edad",
    "ingreso_ppal_cat": "Ingreso ocupacion principal",
    "tam_estab_cat2":   "Tamano del establecimiento",
}

# 6.a.2.1. Diagnostico previo de faltantes y de cantidad de categorias:
diagnostico = pd.DataFrame({
    "N_categorias":  ocupados[vars_kmodas].nunique(),
    "N_faltantes":   ocupados[vars_kmodas].isna().sum(),
    "Pct_faltantes": (ocupados[vars_kmodas].isna().mean() * 100).round(1)
})
diagnostico.index = [etiquetas_kmodas[v] for v in diagnostico.index]

print("Diagnostico del input de k-modas:")
print(diagnostico)

# 6.a.2.2. Conversion a texto y reemplazo de faltantes por "Sin dato":
X_cat = ocupados[vars_kmodas].copy()

# dummy_menor5 se etiqueta para que quede legible en la tabla de modas:
X_cat["dummy_menor5"] = X_cat["dummy_menor5"].map({0: "Sin menor", 1: "Con menor"})

# Todas las columnas pasan a texto y los NaN a una categoria propia:
X_cat = (
    X_cat
    .astype(object)     # libera el tipo CategoricalDtype antes de fillna
    .fillna("Sin dato") # atrapa cualquier tipo de NaN
    .astype(str)        # homogeneiza todo a string
)

y_cat = ocupados["informal"]

print(f"\nInput de k-modas: {X_cat.shape[0]:,} filas y {X_cat.shape[1]} variables "
      f"(sin perdida de observaciones).")

# %% 6.a.3 Entrenamiento del k-modas con k = 2:

"""
Se usa la inicializacion de Huang, que es la propuesta por el autor del
algoritmo y la que mejor se comporta con variables de pocos niveles, y
n_init = 20 para replicar el criterio adoptado en k-medias.
"""

"6.a.3.1 Entrenamiento del modelo:"
km_modas = KModes(n_clusters = 2, init = "Huang", n_init = 20,
                  random_state = 42, verbose = 0)
clusters_kmodas = km_modas.fit_predict(X_cat)

ocupados["cluster_kmodas"] = clusters_kmodas

print(pd.Series(clusters_kmodas).value_counts().sort_index())

"6.a.3.2 Comparacion con la informalidad observada:"
tabla_kmo, pureza_kmo, base_kmo = evaluar_cluster(
    pd.Series(clusters_kmodas, index = X_cat.index), y_cat,
    titulo = "K-modas (k = 2) vs informalidad - Ocupados 2024 + 2025"
)

"6.a.3.3 Perfil de cada cluster (moda de cada variable):"
modas = pd.DataFrame(km_modas.cluster_centroids_, columns = vars_kmodas)
modas.columns = [etiquetas_kmodas[v] for v in vars_kmodas]
modas.index.name = "Cluster"
modas["N"] = pd.Series(clusters_kmodas).value_counts().sort_index().values

print("\nPerfil de los clusteres (moda de cada variable):")
print(modas.T)

"6.a.3.4 Composicion de cada cluster segun informalidad:"
composicion_kmo = pd.crosstab(pd.Series(clusters_kmodas, index = X_cat.index),
                              y_cat, normalize = "index") * 100

print("\nComposicion de cada cluster (%):")
print(composicion_kmo.round(2))

"6.a.3.5 Estabilidad de los clusteres entre 2024 y 2025:"
estabilidad_kmo = pd.crosstab(ocupados["cluster_kmodas"],
                              ocupados["ano"], normalize = "columns") * 100

print("\nComposicion de los clusteres por ano (%) - k-modas:")
print(estabilidad_kmo.round(2))

"6.a.3.6 Exportacion de los perfiles a Excel:"
with pd.ExcelWriter("kmodas_perfiles_TP2.xlsx") as writer:
    modas.T.to_excel(writer, sheet_name = "Modas")
    composicion_kmo.round(2).to_excel(writer, sheet_name = "Composicion")
    estabilidad_kmo.round(2).to_excel(writer, sheet_name = "Estabilidad_por_ano")

print("Guardada: kmodas_perfiles_TP2.xlsx")

# %% 6.a.4 Grafico de la composicion de los clusteres de k-modas:

sns.set_style("whitegrid")
fig, axs = plt.subplots(1, 2, figsize = (13, 5.5))

# Panel (a): composicion de cada cluster segun informalidad
composicion_kmo.plot(kind = "bar", stacked = True, ax = axs[0],
                     color = ["#2196F3", "#F44336"], edgecolor = "white")

for container in axs[0].containers:
    axs[0].bar_label(container, fmt = "%.1f", label_type = "center", fontsize = 9)

axs[0].set_title("K-modas: composicion de cada cluster")
axs[0].set_xlabel("Cluster asignado")
axs[0].set_ylabel("Composicion (%)")
axs[0].set_xticklabels(composicion_kmo.index, rotation = 0)
axs[0].legend(title = "Condicion", loc = "lower right")

# Panel (b): comparacion con k-medias
comparacion = pd.DataFrame({
    "Linea de base": [base_kmo, base_km],
    "Pureza":        [pureza_kmo, pureza_km]
}, index = ["K-modas", "K-medias"])

comparacion.plot(kind = "bar", ax = axs[1],
                 color = ["#B0BEC5", "steelblue"], edgecolor = "white")

for container in axs[1].containers:
    axs[1].bar_label(container, fmt = "%.1f", fontsize = 9)

axs[1].set_title("Pureza alcanzada frente a la linea de base")
axs[1].set_xlabel("")
axs[1].set_ylabel("Porcentaje (%)")
axs[1].set_xticklabels(comparacion.index, rotation = 0)
axs[1].set_ylim(0, 105)
axs[1].legend(loc = "lower right")

fig.suptitle("K-modas (k = 2) - Ocupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold")

plt.tight_layout()
plt.savefig("kmodas_composicion.png", dpi = 300, bbox_inches = "tight")
plt.show()

# %%---- 6.b Metodo del codo (Elbow) para k-modas -----

"""
La medida de disimilitud de k-modas es el costo: la suma, sobre todas las
observaciones, de la cantidad de atributos en los que cada caso difiere de
la moda de su cluster. Cumple el mismo papel que la inercia en k-medias
(decrece monotonamente al agregar clusteres), pero se mide en cantidad de
desacuerdos y no en distancia al cuadrado.

Nota de implementacion: k-modas es considerablemente mas lento que
k-medias, porque en cada iteracion debe recalcular modas sobre variables
categoricas. Correr cuarenta valores de k con n_init = 20 sobre las 40.000
observaciones tardaria horas. Se trabaja entonces sobre una muestra
aleatoria de 5.000 casos y con n_init = 5, suficiente para identificar la
forma de la curva. La muestra se fija con random_state para que el
resultado sea reproducible.
"""

rango_k_modas = range(1, 41)
n_muestra     = 5000

"6.b.1 Extraccion de la muestra:"
X_sub = X_cat.sample(n = min(n_muestra, len(X_cat)), random_state = 42)

print(f"Muestra para el elbow de k-modas: {X_sub.shape[0]:,} casos.")

"6.b.2 Calculo del costo para cada valor de k:"
costos = []

for k in rango_k_modas:
    km_elbow = KModes(n_clusters = k, init = "Huang", n_init = 5,
                      random_state = 42, verbose = 0)
    km_elbow.fit(X_sub)
    costos.append(km_elbow.cost_)
    if k % 5 == 0:
        print(f"K-modas: completado hasta k = {k}")

"6.b.3 Grafico de la curva del codo para k-modas:"
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = (10, 6))

ax.plot(rango_k_modas, costos, marker = "o", markersize = 4, color = "#F44336")

ax.set_title("Metodo del codo (Elbow) - K-modas\nMuestra de 5.000 ocupados, T4 2024 y T4 2025",
             fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel("Numero de clusteres (k)")
ax.set_ylabel("Costo (suma de disimilitudes de Hamming)")
ax.set_xticks(range(0, 41, 5))

plt.tight_layout()
plt.savefig("elbow_kmodas.png", dpi = 300, bbox_inches = "tight")
plt.show()

"6.b.4 Reduccion porcentual del costo al pasar de k a k+1:"
serie_costo    = pd.Series(costos, index = rango_k_modas)
reduccion_cost = (serie_costo.diff(-1) / serie_costo * 100).round(2)

print("\nReduccion porcentual del costo al pasar de k a k+1 (primeros 12):")
print(reduccion_cost.head(12))

# %% 6.b.5 Comparacion de las dos curvas de codo (k-medias vs k-modas):

"""
Las dos medidas no son comparables en nivel, porque estan expresadas en
unidades distintas: la inercia se mide en distancias euclideas al cuadrado
sobre variables estandarizadas y el costo en cantidad de atributos
discrepantes. Para poder contrastar la forma de ambas curvas se las
normaliza al valor que toman en k = 1, de modo que ambas arranquen en 1 y
lo que se lea sea la caida relativa.
"""

sns.set_style("whitegrid")
fig, axs = plt.subplots(1, 2, figsize = (14, 6))

# Panel (a): curvas normalizadas superpuestas
inercia_norm = np.array(inercias) / inercias[0]
costo_norm   = np.array(costos) / costos[0]

axs[0].plot(rango_k, inercia_norm, marker = "o", markersize = 4,
            color = "steelblue", label = "K-medias (inercia)")
axs[0].plot(rango_k_modas, costo_norm, marker = "s", markersize = 4,
            color = "#F44336", label = "K-modas (costo)")

axs[0].set_title("Curvas normalizadas al valor en k = 1")
axs[0].set_xlabel("Numero de clusteres (k)")
axs[0].set_ylabel("Medida de disimilitud (k = 1 igual a 1)")
axs[0].set_xticks(range(0, 41, 5))
axs[0].legend()

# Panel (b): reduccion marginal de cada metodo
axs[1].plot(rango_k[:15], reduccion_iner.head(15).values, marker = "o",
            markersize = 4, color = "steelblue", label = "K-medias")
axs[1].plot(rango_k_modas[:15], reduccion_cost.head(15).values, marker = "s",
            markersize = 4, color = "#F44336", label = "K-modas")

axs[1].set_title("Reduccion marginal al agregar un cluster")
axs[1].set_xlabel("Numero de clusteres (k)")
axs[1].set_ylabel("Reduccion de la disimilitud (%)")
axs[1].set_xticks(range(0, 16, 2))
axs[1].legend()

fig.suptitle("Comparacion de las curvas de codo: k-medias vs k-modas",
             fontsize = 13, fontweight = "bold")

plt.tight_layout()
plt.savefig("elbow_comparacion.png", dpi = 300, bbox_inches = "tight")
plt.show()

# %% 6.c Resumen comparativo de los tres metodos de cluster:

"""
Tabla de sintesis para el informe. La columna decisiva es la ganancia en
puntos porcentuales sobre la linea de base: es la que responde a la
pregunta de si el algoritmo recupera o no la particion entre formales e
informales. Una pureza alta con ganancia proxima a cero significa
simplemente que la categoria mayoritaria es grande, no que el metodo haya
identificado la informalidad.
"""

resumen = pd.DataFrame([
    {"Metodo": "K-medias (k = 2)", "N": int(tabla_km.values.sum()),
     "Pureza (%)": round(pureza_km, 2), "Linea de base (%)": round(base_km, 2),
     "Ganancia (pp)": round(pureza_km - base_km, 2)},
    {"Metodo": "Jerarquico (Ward, corte en 2)", "N": int(tabla_jer.values.sum()),
     "Pureza (%)": round(pureza_jer, 2), "Linea de base (%)": round(base_jer, 2),
     "Ganancia (pp)": round(pureza_jer - base_jer, 2)},
    {"Metodo": "K-modas (k = 2)", "N": int(tabla_kmo.values.sum()),
     "Pureza (%)": round(pureza_kmo, 2), "Linea de base (%)": round(base_kmo, 2),
     "Ganancia (pp)": round(pureza_kmo - base_kmo, 2)},
])

print("\nResumen comparativo de los metodos de cluster:")
print(resumen)

with pd.ExcelWriter("resumen_cluster_TP2.xlsx") as writer:
    resumen.to_excel(writer, sheet_name = "Resumen", index = False)

print("Guardada: resumen_cluster_TP2.xlsx")