# =============================================================================
# PAQUETES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from matplotlib.lines import Line2D
import statsmodels.api as sm
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
os.chdir(r"/Volumes/ADATA HD330/Maestría Economía Aplicada UBA/Taller de programación/Trabajos prácticos/Grupo_3_Trabajos_practicos/TP3")

print(os.getcwd())

pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# PARTE A: CREACION DE VARIABLES Y ANÁLISIS DESCRIPTIVO
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
    # Identificación (Base Personas, pág. 14-15 del diccionario)
    "CODUSU",       # Identifica vivienda (permite aparear Hogar/Personas y seguir en el tiempo)
    "NRO_HOGAR",    # Identifica hogar dentro de la vivienda
    "COMPONENTE",   # Identifica persona dentro del hogar
    "ANO4",         # Año de relevamiento
    "TRIMESTRE",    # Trimestre de relevamiento
    "PONDERA",      # Ponderador (factor de expansión)

    # Características de los miembros del hogar (pág. 16-18)
    "CH03",         # Relación de parentesco (para identificar al jefe/a de hogar)
    "CH04",         # Sexo
    "CH06",         # Edad
    "CH07",         # Estado civil
    "CH08",         # Cobertura médica

    # Educación (pág. 17) — insumo de la variable 'educ' del TP2
    "CH12",         # Nivel más alto que cursa/cursó
    "CH13",         # Si finalizó ese nivel
    "CH14",         # Último año aprobado

    # Cuestionario Individual (pág. 18)
    "NIVEL_ED",     # Nivel educativo agrupado
    "ESTADO",       # Condición de actividad (para separar ocupados/respondieron)
    "CAT_OCUP",     # Categoría ocupacional
    "EMPLEO",       # Formal/Informal (variable EPH de referencia)
    "SECTOR",       # Formal/Informal/Hogares

    # Ocupación principal (pág. 20-21)
    "PP04C",        # Tamaño del establecimiento (cantidad de personas, numérico)
    "PP04C99",      # Tamaño del establecimiento agrupado
    "PP04D_COD",    # Código de ocupación (CNO)

    # Ocupación principal - asalariados (pág. 24, 26)
    "PP07H",        # Descuento jubilatorio (clave para la definición de informalidad)
    "PP07K",        # Tipo de comprobante de pago (recibo con sello, sin sello, factura, nada)
    "PP07L",        # Si el recibo abarca la totalidad o solo parte del sueldo
    "PP07M",        # Qué parte del sueldo no está en el recibo

    # Ocupados que trabajaron en la semana de referencia (pág. 20)
    "PP03D",        # Cantidad de ocupaciones
    "PP3E_TOT",     # Horas trabajadas en la ocupación principal
    "PP3F_TOT",     # Horas trabajadas en otras ocupaciones

    # Ingreso de la ocupación principal (pág. 30) e ingreso total individual (pág. 31)
    "P21",          # Monto de ingreso de la ocupación principal
    "P47T",         # Monto de ingreso total individual

    # Identificación geográfica (pág. 5, 14)
    "REGION",       # Código de región
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

# %% 0.4 Union de bases de datos:
bd = pd.concat([bd_24, bd_25], ignore_index = True)

print("Base unificada:", bd.shape)
print(bd["ANO4"].value_counts())

# %% 0.5 Ajuste de los ingresos por inflacion (pesos de 2024 a pesos de 2025):

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

# %% 0.6 Variable dummy_menor5 (heredada del TP1):

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
    "CH03":       "parentesco",
    "CH12":       "nivel_ed_alto",
    "CH13":       "finalizacion_nivel",
    "CH14":       "nivel_ed_aprb",
    "PP3E_TOT":   "horas_s_principal",
    "PP3F_TOT":   "horas_s_otras_act"
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

"parentesco"
ocupados["parentesco"] = ocupados["parentesco"].map({
    1:  "Jefe",
    2:  "Cónyuge o pareja",
    3:  "Hijo o hijastro",
    4:  "Yerno o nuera",
    5:  "Nieto",
    6:  "Madre o padre",
    7:  "Suegro",
    8:  "Hermano",
    9:  "Otros",
    10: "No son familiares",
    0:  np.nan,
    99: np.nan
})

"nivel_ed_alto (CH12 — nivel más alto cursado o cursando)"
ocupados["nivel_ed_alto"] = ocupados["nivel_ed_alto"].map({
    1: "Jardín/preescolar",
    2: "Primario",
    3: "EGB",
    4: "Secundario",
    5: "Polimodal",
    6: "Terciario",
    7: "Universitario",
    8: "Posgrado universitario",
    9: "Educación especial (discapacidad)"
})

"nivel_ed_aprb (CH14 — último año aprobado)"
ocupados["nivel_ed_aprb"] = ocupados["nivel_ed_aprb"].map({
    0:  "Ninguno",
    1:  "Primero",
    2:  "Segundo",
    3:  "Tercero",
    4:  "Cuarto",
    5:  "Quinto",
    6:  "Sexto",
    7:  "Séptimo",
    8:  "Octavo",
    9:  "Noveno",
    98: "Educación especial",
    99: np.nan
})

"finalizacion_nivel"
ocupados["finalizacion_nivel"] = ocupados["finalizacion_nivel"].map({
    1:  "Sí",
    2:  "No",
    9:  np.nan
})

"ingresos"
ocupados['ingreso_ppal']  = np.log(ocupados['ingreso_ppal'].replace(0, np.nan))
ocupados['ingreso_total'] = np.log(ocupados['ingreso_total'].replace(0, np.nan))

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

# 1.10. Suprimir las variables duplicadas de 'ocupados':
duplicadas = ocupados.columns[ocupados.columns.duplicated()].tolist()
print("Columnas duplicadas detectadas:", duplicadas)
ocupados = ocupados.loc[:, ~ocupados.columns.duplicated(keep = "first")]
print("Estructura luego de suprimir duplicadas:", ocupados.shape)
print("Duplicadas restantes:", ocupados.columns.duplicated().sum())

#%% 1.11. Creacion de las bases de datos anuales (Consigna A.1):

# 1.11.1. Crear las bases de datos por año (todavia con 'informal' e
# identificadores, que se necesitan para los pasos siguientes):
ocupados_X_2024 = ocupados[ocupados['ano'] == 2024].copy()
ocupados_X_2025 = ocupados[ocupados['ano'] == 2025].copy()

# 1.11.2. Separar las variables de respuesta (texto "Formal"/"Informal"):
y_2024 = ocupados_X_2024['informal']
y_2025 = ocupados_X_2025['informal']

# 1.11.3. Crear un codigo para hacer el 'join':
for df in [ocupados_X_2024, ocupados_X_2025]:
    df['id'] = df['cod_vivienda'].astype(str) + df['nro_hogar'].astype(str) + df['componente'].astype(str)

#%% 1.12. Unificacion de la base de datos (Consigna A.1, extension Maurizio & Monsalvo):

# 1.12.1. Añadir la variable id a la base que aporta el estatus 2024:
y_2024_df = pd.DataFrame({
    'id':            ocupados_X_2024['id'],
    'informal_2024': y_2024
})

# 1.12.2. Unificar la base de datos mediante un inner join por 'id':
"""
'informal' (el estatus 2025, variable objetivo del Modelo 2) se
mantiene en ocupados_X_2025_MM hasta el paso 1.12.4, donde se extrae
directamente desde esta misma tabla. Esto evita el error de alineacion
que ocurre al intentar reconstruir y_2025_MM con un 'reindex' sobre
'ocupados_X_2025_MM.index': 'DataFrame.merge' NO conserva el indice
original de 'ocupados_X_2025', sino que genera un RangeIndex nuevo, por
lo que reindexar contra 'y_2025' (que si conserva el indice original)
producia practicamente todo NaN.
"""
ocupados_X_2025_MM = ocupados_X_2025.merge(
    y_2024_df,
    on  = 'id',
    how = 'inner'
).drop(columns = ['id', 'cod_vivienda', 'nro_hogar', 'componente', 'ano', 'trimestre'])

print("Estructura de ocupados_X_2025 (original):", ocupados_X_2025.shape)
print("Estructura de ocupados_X_2025_MM (con el 'join'):", ocupados_X_2025_MM.shape)

# 1.12.3. Suprimir las variables duplicadas resultantes del 'join':
duplicadas_MM = ocupados_X_2025_MM.columns[ocupados_X_2025_MM.columns.duplicated()].tolist()
print("Columnas duplicadas detectadas en ocupados_X_2025_MM:", duplicadas_MM)
ocupados_X_2025_MM = ocupados_X_2025_MM.loc[:, ~ocupados_X_2025_MM.columns.duplicated(keep = "first")]
print("Duplicadas restantes:", ocupados_X_2025_MM.columns.duplicated().sum())

# 1.12.4. Extraer y_2025_MM DESDE ocupados_X_2025_MM (alineacion garantizada
# por ser la misma tabla) y recien despues eliminar 'informal' de X:
y_2025_MM = ocupados_X_2025_MM['informal']
ocupados_X_2025_MM = ocupados_X_2025_MM.drop(columns = 'informal')

print("Observaciones de y_2025_MM:", y_2025_MM.shape[0])
print("Indices alineados con ocupados_X_2025_MM:",
      y_2025_MM.index.equals(ocupados_X_2025_MM.index))

#%% 1.13. Supresion de variables redundantes, tautologicas y no
#        caracteristicas (aplicado por igual a las tres bases):

"""
Se suprimen tres grupos de columnas ANTES del one-hot encoding, de
manera que ninguna de sus dummies llegue a generarse:

(i)   Identificadoras / de agrupacion, sin valor como caracteristica
      individual una vez separadas las bases por año.
(ii)  Crudas numericas redundantes: codigos originales de la EPH que
      ya tienen su version etiquetada con sufijo '_cat' (o dummy en
      1.6). Quedaron en la base como numericas, por lo que el
      renombre (1.7) no las convirtio en texto ni las elimino.
(iii) Definitorias de 'informal' (paso 1.9): su inclusion en X
      implica tautologia, ya que el modelo terminaria "prediciendo"
      la variable objetivo a partir de sus propios insumos de
      construccion, y no de caracteristicas independientes del
      individuo.
(iv)  No caracteristicas: el ponderador muestral y la condicion de
      actividad (constante, todos son ocupados).
"""

id_y_agrupacion = [
    'id', 'cod_vivienda', 'nro_hogar', 'componente',
    'ano', 'año', 'trimestre',
]

crudas_redundantes = [
    'CH04', 'CH07', 'CH08',
    'nivel_ed', 'cat_ocup', 'tipo_sector', 'tipo_empleo',
    'region', 'comprobante_sal', 'alcance_recibo', 'parte_sueldo',
    'desc_jub_cod', 'tam_estab_cod',
]

definitorias_informal = [
    'cat_ocup2', 'desc_jubilatorio',
    'tam_estab', 'tam_estab_agrup',
    'sector_2', 'tipo_sector_cat', 'tipo_empleo_cat',
]

no_caracteristicas = ['ponderador', 'cond_actividad']

vars_a_suprimir = (
    id_y_agrupacion + crudas_redundantes +
    definitorias_informal + no_caracteristicas
)

for nombre in ['ocupados_X_2024', 'ocupados_X_2025', 'ocupados_X_2025_MM']:
    df = globals()[nombre]
    presentes = [c for c in vars_a_suprimir if c in df.columns]
    globals()[nombre] = df.drop(columns = presentes)
    print(f"{nombre}: suprimidas {len(presentes)} columnas -> "
          f"estructura {globals()[nombre].shape}")

# 1.13.1. Suprimir 'informal' de ocupados_X_2024 y ocupados_X_2025
# (ya se extrajo como y_2024 e y_2025 en 1.11.2):
ocupados_X_2024 = ocupados_X_2024.drop(columns = 'informal', errors = 'ignore')
ocupados_X_2025 = ocupados_X_2025.drop(columns = 'informal', errors = 'ignore')

#%% 1.14. One hot encoding de las variables cualitativas:

# 1.14.1. Encoding conjunto de ocupados_X_2024 y ocupados_X_2025:
"""
Para que la tabla de diferencia de medias (Consigna A.2) compare
exactamente las mismas variables en ambos años, el one-hot encoding se
aplica sobre las dos bases EN CONJUNTO: si una categoria estuviera
presente solo en un año, encodear por separado generaria columnas de
dummies distintas y no comparables entre si.
"""

ocupados_X_2024['_origen'] = '2024'
ocupados_X_2025['_origen'] = '2025'

union_24_25 = pd.concat([ocupados_X_2024, ocupados_X_2025], ignore_index = True)

cualitativas_24_25 = union_24_25.select_dtypes(include = 'object').columns.tolist()
cualitativas_24_25 = [c for c in cualitativas_24_25 if c != '_origen']

print("Variables cualitativas identificadas (2024/2025):", cualitativas_24_25)

union_24_25 = pd.get_dummies(
    union_24_25,
    columns    = cualitativas_24_25,
    drop_first = True,
    dummy_na   = False
)

bool_cols_24_25 = union_24_25.select_dtypes(include = bool).columns.tolist()
union_24_25[bool_cols_24_25] = union_24_25[bool_cols_24_25].astype(int)

ocupados_X_2024 = union_24_25[union_24_25['_origen'] == '2024'].drop(columns = '_origen')
ocupados_X_2025 = union_24_25[union_24_25['_origen'] == '2025'].drop(columns = '_origen')

print("Estructura ocupados_X_2024:", ocupados_X_2024.shape)
print("Estructura ocupados_X_2025:", ocupados_X_2025.shape)
print("Mismas columnas en ambas bases:",
      set(ocupados_X_2024.columns) == set(ocupados_X_2025.columns))

# 1.14.2. Encoding de ocupados_X_2025_MM (por separado, porque incluye
# el regresor adicional 'informal_2024' que no existe en las otras dos):
cualitativas_MM = ocupados_X_2025_MM.select_dtypes(include = 'object').columns.tolist()

print("Variables cualitativas identificadas (2025_MM):", cualitativas_MM)

for var in cualitativas_MM:
    n_cat = ocupados_X_2025_MM[var].nunique(dropna = True)
    naturaleza = "binaria" if n_cat == 2 else "nominal (>2 categorias)"
    print(f"{var}: {n_cat} categorias -> {naturaleza}")

ocupados_X_2025_MM = pd.get_dummies(
    ocupados_X_2025_MM,
    columns    = cualitativas_MM,
    drop_first = True,
    dummy_na   = False
)

bool_cols_MM = ocupados_X_2025_MM.select_dtypes(include = bool).columns.tolist()
ocupados_X_2025_MM[bool_cols_MM] = ocupados_X_2025_MM[bool_cols_MM].astype(int)

print("Estructura ocupados_X_2025_MM:", ocupados_X_2025_MM.shape)

#%% 2. Analisis descriptivo y diferencia de medias (Consigna A.2):

# 2.1. Estadisticos descriptivos de ocupados_X_2025_MM:
ocupados_X_2025_MM_to_excel = ocupados_X_2025_MM.describe().T

ocupados_X_2025_MM_to_excel.to_excel(
    "ocupados_X_2025_MM_describe.xlsx",
    sheet_name   = "Descriptiva",
    float_format = "%.4f"
)

# 2.2. Tabla de diferencia de medias entre ocupados_X_2024 y ocupados_X_2025:
"""
Balance Table Train vs Test Sample: para cada caracteristica comun a
ambas bases (ya con las mismas columnas tras el encoding conjunto de
1.14.1) se calcula la media 2024, la media 2025, la diferencia
(2025 - 2024) y el estadistico t de Welch (varianzas no necesariamente
iguales) de un contraste de diferencia de medias.
"""

from scipy import stats

columnas_comunes = [
    c for c in ocupados_X_2024.columns
    if c in ocupados_X_2025.columns
]

resultados_medias = []

for var in columnas_comunes:
    x_24 = ocupados_X_2024[var].dropna()
    x_25 = ocupados_X_2025[var].dropna()

    media_24   = x_24.mean()
    media_25   = x_25.mean()
    diferencia = media_25 - media_24

    t_stat, p_valor = stats.ttest_ind(x_25, x_24, equal_var = False, nan_policy = "omit")

    resultados_medias.append({
        "variable":      var,
        "media_2024":    media_24,
        "media_2025":    media_25,
        "diferencia":    diferencia,
        "t_estadistico": t_stat,
        "p_valor":       p_valor
    })

tabla_diferencia_medias = pd.DataFrame(resultados_medias).set_index("variable")

print(tabla_diferencia_medias.round(4))

tabla_diferencia_medias.to_excel(
    "tabla_diferencia_medias.xlsx",
    sheet_name   = "Diferencia_medias",
    float_format = "%.4f"
)

# =============================================================================
# PARTE B: MODELO DE REGRESION LOGISTICA
# =============================================================================

#%% 1. Estimacion de los modelos de Regresion Logistica (Consigna B.1,
#      extension de Maurizio & Monsalvo, 2021):

"""
Modelo 1 (ocupados_X_2024): logit de informalidad 2024 sobre las
caracteristicas X de 2024, sin informalidad rezagada.

Modelo 2 (ocupados_X_2025_MM): logit de informalidad 2025 sobre las
caracteristicas X de 2025 mas 'informal_2024_Informal' (dummy de
informalidad rezagada, incorporada en el 'join' de 1.12 y encodeada en
1.14.2), siguiendo la especificacion de Maurizio & Monsalvo (2021), que
introduce el estatus de informalidad del periodo anterior como regresor
adicional (Seccion 4.1 del paper, ecuaciones 1 y 2).

Ambas matrices de caracteristicas ya llegan limpias de 1.13 (sin
identificadores, sin crudas redundantes, sin variables definitorias de
'informal' y sin ponderador/condicion de actividad), por lo que aca solo
resta binarizar el vector objetivo, alinear observaciones, depurar
representaciones redundantes, estandarizar las variables continuas y
estimar.
"""

# 1.1. Vectores objetivo binarios (0 = Formal, 1 = Informal):
y_2024_bin    = y_2024.map({'Formal': 0, 'Informal': 1})
y_2025_MM_bin = y_2025_MM.map({'Formal': 0, 'Informal': 1})

print("Distribucion y_2024_bin:")
print(y_2024_bin.value_counts())

print("Distribucion y_2025_MM_bin:")
print(y_2025_MM_bin.value_counts())

# 1.2. Alinear observaciones y eliminar NaN (statsmodels no admite NaN):
datos_2024 = pd.concat(
    [y_2024_bin.rename('y'), ocupados_X_2024], axis = 1
).dropna()
y_2024_final = datos_2024['y']
X_2024_raw   = datos_2024.drop(columns = 'y')

datos_2025_MM = pd.concat(
    [y_2025_MM_bin.rename('y'), ocupados_X_2025_MM], axis = 1
).dropna()
y_2025_MM_final = datos_2025_MM['y']
X_2025_MM_raw   = datos_2025_MM.drop(columns = 'y')

print("Observaciones Modelo 1 (2024):", X_2024_raw.shape)
print("Observaciones Modelo 2 (2025_MM):", X_2025_MM_raw.shape)

# 1.3. Excluir representaciones redundantes de educacion y otras
# colinealidades identificadas en el diagnostico de errores estandar:

"""
El modelo incluye cinco representaciones del mismo concepto educativo
(educ, nivel_ed2, nivel_ed_cat, nivel_ed_alto, nivel_ed_aprb). Su
inclusion simultanea genera multicolinealidad severa que impide la
inversion de la Hessiana y produce errores estandar igual a NaN. Se
conserva unicamente 'educ' (anos de educacion formal, continua) por ser
la especificacion de Maurizio & Monsalvo (2021). Se elimina tambien
'horastrabj' (colineal con 'horastrab') y 'cod_ocupacion' (categorica
de alta cardinalidad que no aporta como regresora directa en el logit).
"""

redundantes_logit = [
    # Representaciones redundantes de educacion (se conserva 'educ'):
    'nivel_ed2_Superior',
    'nivel_ed_cat_Primario incompleto',
    'nivel_ed_cat_Secundario completo',
    'nivel_ed_cat_Secundario incompleto',
    'nivel_ed_cat_Sin instruccion',
    'nivel_ed_cat_Superior completo',
    'nivel_ed_cat_Superior incompleto',
    'nivel_ed_alto_Jardín/preescolar',
    'nivel_ed_alto_Polimodal',
    'nivel_ed_alto_Posgrado universitario',
    'nivel_ed_alto_Primario',
    'nivel_ed_alto_Secundario',
    'nivel_ed_alto_Terciario',
    'nivel_ed_alto_Universitario',
    'nivel_ed_aprb_Ninguno',
    'nivel_ed_aprb_Noveno',
    'nivel_ed_aprb_Octavo',
    'nivel_ed_aprb_Primero',
    'nivel_ed_aprb_Quinto',
    'nivel_ed_aprb_Segundo',
    'nivel_ed_aprb_Sexto',
    'nivel_ed_aprb_Séptimo',
    'nivel_ed_aprb_Tercero',
    'ingreso_ppal',
    'finalizacion_nivel_Sí',
    'cat_ocup_cat_Patron',
    # Colineal con las variables de horas de trabajo:
    'horastrabj',
    'horastrab',
    # Alta cardinalidad sin valor directo como regresora:
    'cod_ocupacion',
]

X_2024_raw = X_2024_raw.drop(
    columns = [c for c in redundantes_logit if c in X_2024_raw.columns]
)
X_2025_MM_raw = X_2025_MM_raw.drop(
    columns = [c for c in redundantes_logit if c in X_2025_MM_raw.columns]
)

print("Columnas en X_2024_raw tras excluir redundantes:",    X_2024_raw.shape[1])
print("Columnas en X_2025_MM_raw tras excluir redundantes:", X_2025_MM_raw.shape[1])

# 1.4. Eliminar columnas con varianza cero o cuasi-cero (std < 0.01):

"""
Las columnas con varianza cero son constantes en la muestra y no
aportan informacion al modelo; ademas causan singularidad en la
Hessiana. Se usa un umbral de std < 0.01 para incluir tambien las
cuasi-constantes (categorias con menos del 1% de los casos, como
ciertas dummies de parentesco o nivel educativo extremo).
"""

def limpiar_varianza(df, umbral_std = 0.01):
    cols_bajas = df.columns[df.std() < umbral_std].tolist()
    print("Columnas eliminadas por varianza baja:", cols_bajas)
    return df.drop(columns = cols_bajas)

X_2024_raw    = limpiar_varianza(X_2024_raw)
X_2025_MM_raw = limpiar_varianza(X_2025_MM_raw)

# 1.5. Estandarizar las variables continuas (media 0, desv. std 1):

"""
Las variables continuas (edad, edad2, educ, horas, ingresos, nhogar)
tienen escalas muy distintas a las dummies (0/1). Sin estandarizacion
el algoritmo BFGS no puede calcular gradientes utiles y los coeficientes
colapsan a cero o la Hessiana no puede invertirse. Se estandarizan solo
las continuas para mantener la interpretabilidad de las dummies.
El scaler de 2024 se ajusta ('fit') sobre 2024 y se aplica ('transform')
sobre 2025_MM, de manera que la escala de referencia sea siempre la
muestra de entrenamiento.
"""

continuas = [
    'edad', 'edad2', 'educ', 'horastrab',
    'horas_s_principal', 'horas_s_otras_act',
    'nhogar', 'ingreso_ppal', 'ingreso_total',
    'cant_ocupaciones_ad',
]

scaler = StandardScaler()

cols_cont_2024 = [c for c in continuas if c in X_2024_raw.columns]
X_2024_raw = X_2024_raw.copy()
X_2024_raw[cols_cont_2024] = scaler.fit_transform(X_2024_raw[cols_cont_2024])

cols_cont_MM = [c for c in continuas if c in X_2025_MM_raw.columns]
X_2025_MM_raw = X_2025_MM_raw.copy()
X_2025_MM_raw[cols_cont_MM] = scaler.transform(X_2025_MM_raw[cols_cont_MM])

# 1.6. Diagnostico de rango final:
from numpy.linalg import matrix_rank

X_2024_final    = sm.add_constant(X_2024_raw)
X_2025_MM_final = sm.add_constant(X_2025_MM_raw)

print("Rango / columnas Modelo 1:",
      matrix_rank(X_2024_final), "/", X_2024_final.shape[1])
print("Rango / columnas Modelo 2:",
      matrix_rank(X_2025_MM_final), "/", X_2025_MM_final.shape[1])

# 1.7. Estimacion del Modelo 1 (2024):
modelo_2024 = sm.Logit(y_2024_final, X_2024_final).fit(
    method  = 'bfgs',
#   maxiter = 1000,
    disp    = True
)
print(modelo_2024.summary())

# 1.8. Estimacion del Modelo 2 (2025_MM, especificacion Maurizio & Monsalvo):
modelo_2025_MM = sm.Logit(y_2025_MM_final, X_2025_MM_final).fit(
    method  = 'bfgs',
#   maxiter = 1000,
    disp    = True
)
print(modelo_2025_MM.summary())

#%% 1.9. Tabla de coeficientes, EMP, errores estandar y odds-ratios
#        (Consigna B.1 items i, ii y iii):

"""
Para cada modelo se extraen de statsmodels:
- Coeficientes estimados (params)
- Errores estandar (bse)
- Odds-ratios: exp(coef), interpretados como el cambio multiplicativo
  en los odds de ser informal ante un incremento unitario en X.
- Efectos Marginales Promedio (EMP): se calcula el efecto marginal
  en cada observacion y luego se promedia sobre la muestra
  (get_margeff() con method='dydx', at='mean' calcula el efecto en
  la media de X; se usa at='overall' para el promedio sobre toda la
  distribucion observada, que es el EMP propiamente dicho).
"""

# 1.9.1. Funcion auxiliar para construir la tabla de un modelo:
def tabla_logit(modelo, nombre_modelo):
    margeff = modelo.get_margeff(at = 'overall', method = 'dydx')

    # coef, bse y odds: excluir 'const' para alinear con los EMP
    # (get_margeff no calcula efecto marginal de la constante):
    vars_sin_const = [v for v in modelo.params.index if v != 'const']

    coef  = modelo.params[vars_sin_const]
    bse   = modelo.bse[vars_sin_const]
    odds  = np.exp(coef)
    emp   = pd.Series(margeff.margeff,    index = vars_sin_const)
    emp_se = pd.Series(margeff.margeff_se, index = vars_sin_const)

    tabla = pd.DataFrame({
        'Coeficiente':    coef,
        'Error estandar': bse,
        'Odds-ratio':     odds,
        'EMP':            emp,
        'EMP std. error': emp_se,
    })
    tabla.index.name = 'Variable'
    tabla.columns = pd.MultiIndex.from_tuples(
        [(nombre_modelo, c) for c in tabla.columns]
    )
    return tabla

# 1.9.2. Construir la tabla para cada modelo:
tabla_m1 = tabla_logit(modelo_2024,    'Modelo 1 (2024)')
tabla_m2 = tabla_logit(modelo_2025_MM, 'Modelo 2 (2025 MM)')

# 1.9.3. Unir ambas tablas en columnas:
tabla_conjunta = tabla_m1.join(tabla_m2, how = 'outer').round(4)

print(tabla_conjunta.to_string())

# 1.9.4. Exportar a Excel:
tabla_conjunta.to_excel(
    "tabla_coef_logit.xlsx",
    sheet_name   = "Coef_EMP_OR",
    float_format = "%.4f"
)
