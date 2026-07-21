# -*- coding: utf-8 -*-
"""
Universidad de Buenos Aires - Taller de Programación
Trabajo Práctico N° 2: Métodos No Supervisados usando la EPH

Grupo 3:
    - Luciano Altamirano (Parte I y Parte II.A)
    - Gonzalo Pasiche   (Parte II.B)

Fecha de entrega: viernes 24 de julio de 2026 a las 13:00 h.
"""

# =============================================================================
# PAQUETES
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage

# =============================================================================
# PARTE I: CREACIÓN DE VARIABLES Y MÁS ESTADÍSTICA DESCRIPTIVA
# (Luciano Altamirano)
# =============================================================================

# Seteo de directorio
os.chdir(r"/Volumes/ADATA HD330/Maestría Economía Aplicada UBA/Taller de programación/Trabajos prácticos/Grupo_3_Trabajos_practicos/TP2")
os.chdir(r"C:\Users\gmpas\OneDrive\Escritorio\Seminario Programación\TP2")

print(os.getcwd())

pd.set_option("display.float_format", "{:,.2f}".format)

#%% PARTE I

#%% 0.1 Carga de datos:

#%% 0.1 Carga de datos:

# 0.1.1. Cargar de datos en formato excel:
'''
columnas_necesarias = [
    "CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "PONDERA",
    "CH03", "CH04", "CH06", "CH07", "CH08",
    "CH12", "CH13", "CH14",
    "NIVEL_ED", "ESTADO", "CAT_OCUP", "PP07H", "PP04C", "PP04C99",
    "EMPLEO", "SECTOR", "PP04D_COD", "P21", "P47T", "REGION",
    "PP07K", "PP07L", "PP07M", "PP03D", "PP3E_TOT", "PP3F_TOT"
]

# 0.1.2. Importar la base de datos:
bd_24 = pd.read_excel("usu_individual_T424.xlsx", usecols = columnas_necesarias)
bd_25 = pd.read_excel("usu_individual_T425.xlsx", usecols = columnas_necesarias)


# 0.1.3. Transformar en formato parquet, para que no pese mucho:
bd_24.to_parquet("bd_24.parquet")
bd_25.to_parquet("bd_25.parquet")
'''

# 0.1.4. Cargar los archivos parquet:
bd_24 = pd.read_parquet("bd_24.parquet")
bd_25 = pd.read_parquet("bd_25.parquet")

#%% 0.2 Verificación de columnas necesarias para el TP2:
columnas_necesarias = [
    "CODUSU", "NRO_HOGAR", "COMPONENTE", "ANO4", "TRIMESTRE", "PONDERA",
    "CH03", "CH04", "CH06", "CH07", "CH08",
    "CH12", "CH13", "CH14",
    "NIVEL_ED", "ESTADO", "CAT_OCUP", "PP07H", "PP04C", "PP04C99",
    "EMPLEO", "SECTOR", "PP04D_COD", "P21", "P47T", "REGION",
    "PP07K", "PP07L", "PP07M", "PP03D", "PP3E_TOT", "PP3F_TOT"
]

#%% 0.3 Corrección de valores sin sentido (heredado del TP1):

# 0.3.1. Ingresos: -9 se trata como NaN:
bd_24["P21"] = bd_24["P21"].replace(-9, np.nan)
bd_24["P47T"] = bd_24["P47T"].replace(-9, np.nan)
bd_25["P21"] = bd_25["P21"].replace(-9, np.nan)
bd_25["P47T"] = bd_25["P47T"].replace(-9, np.nan)

# 0.3.2. Edad - CH06: -1 se trata como NaN:
bd_24["CH06"] = bd_24["CH06"].replace(-1, np.nan)
bd_25["CH06"] = bd_25["CH06"].replace(-1, np.nan)

# 0.3.3. Horas trabajadas - PP3E_TOT y PP3F_TOT: 999 (Ns/Nr) se trata como NaN:
for base in [bd_24, bd_25]:
    base["PP3E_TOT"] = base["PP3E_TOT"].replace(999, np.nan)
    base["PP3F_TOT"] = base["PP3F_TOT"].replace(999, np.nan)

#%% 0.4 Unión de bases de datos:
bd = pd.concat([bd_24, bd_25], ignore_index = True)

#%% 1. Creación de variables

#%% 1.1 .edad2:
bd["edad2"] = bd["CH06"] ** 2

#%% 1.2. educ (años de educación formal):

"""
Se construye a partir de CH12 (nivel más alto cursado), CH13 (si lo finalizó)
y CH14 (último año aprobado, para quienes no finalizaron el nivel).

Supuestos de duración de cada nivel (estructura 6+6, coherente con el
ejemplo de la consigna: Secundario finalizado en "sexto" => educ=12):
"""

anos_base_nivel = {
    0: np.nan, # Valores no válidos
    1: 0,      # Jardín/preescolar
    2: 0,      # Primario
    3: 0,      # EGB
    4: 6,      # Secundario  (Primario completo previo)
    5: 9,      # Polimodal   (EGB completo previo)
    6: 12,     # Terciario   (Secundario/Polimodal completo previo)
    7: 12,     # Universitario
    8: 17,     # Posgrado    (Universitario completo previo)
    9: np.nan  # Educación especial
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

# CH14: 98 (educación especial) y 99 (Ns/Nr) → NaN
ch14_limpio = bd["CH14"].replace({98: np.nan, 99: np.nan})

# CH13 = 9 (Ns/Nr) → NaN explícito para evitar que caiga en default
ch13_limpio = bd["CH13"].replace({9: np.nan})

condiciones = [
    ch13_limpio == 1,   # finalizó el nivel → base + duración completa
    ch13_limpio == 2,   # no finalizó       → base + último año aprobado
]
resultados = [
    bd["_anos_base"] + bd["_duracion"],
    bd["_anos_base"] + ch14_limpio,
]

bd["educ"] = np.select(condiciones, resultados, default=np.nan)
bd = bd.drop(columns=["_anos_base", "_duracion"])

# Verificación de valores extremos
print(bd["educ"].describe())
print("\nDistribución por nivel educativo (CH12) y educ media:")
print(bd.groupby("CH12")["educ"].mean().round(1))

#%% 1.c horastrab (solo jefe/a de hogar) y horastrabj (broadcast al hogar):
bd["horastrab"] = np.where(
    bd["CH03"] == 1,
    bd["PP3E_TOT"] + bd["PP3F_TOT"],
    np.nan
)

# Tabla auxiliar con las horas del jefe/a por hogar
horas_jefe = bd.loc[bd["CH03"] == 1, ["CODUSU", "NRO_HOGAR", "horastrab"]] \
               .drop_duplicates(subset=["CODUSU", "NRO_HOGAR"]) \
               .rename(columns={"horastrab": "horastrabj"})

bd = bd.merge(horas_jefe, on=["CODUSU", "NRO_HOGAR"], how="left")

#%% 1.d nhogar (cantidad de miembros por hogar):
bd["nhogar"] = bd.groupby(["CODUSU", "NRO_HOGAR"])["COMPONENTE"].transform("nunique")

print(bd[["horastrab", "horastrabj", "nhogar"]].describe())

#%% 1.5 Reconstrucción de variables auxiliares del TP1 (para 'informal')
bd["ESTADO"].value_counts(normalize=True) * 100

respondieron = bd[bd["ESTADO"] != 0].copy()
ocupados = respondieron[respondieron["ESTADO"] == 1].copy()

"cat_ocup2"
ocupados["cat_ocup2"] = ocupados["CAT_OCUP"].map({
    1: "No asalariado",
    2: "No asalariado",
    3: "Asalariado",
    4: np.nan,
    9: np.nan
})

"desc_jubilatorio"
ocupados["desc_jubilatorio"] = ocupados["PP07H"].map({
    1: "Si",
    2: "No",
    9: np.nan
})

"tam_estab / tam_estab_agrup"
ocupados["tam_estab"] = ocupados["PP04C"].replace(99, np.nan)
ocupados["tam_estab_agrup"] = ocupados["PP04C99"].map({
    1: "Hasta 5",
    2: "De 6 a 40",
    3: "Mas de 40",
    9: np.nan
})

#%% 1.6 Renombres:
renombres = {
    "CODUSU":     "cod_vivienda",
    "NRO_HOGAR":  "nro_hogar",
    "ANO4":       "año",
    "TRIMESTRE":  "trimestre",
    "PONDERA":    "ponderador",
    "CH06":       "edad",
    "NIVEL_ED":   "nivel_ed",
    "ESTADO":     "cond_actividad",
    "CAT_OCUP":   "cat_ocup",
    "PP04C":      "tam_estab_raw",
    "EMPLEO":     "tipo_empleo",
    "SECTOR":     "tipo_sector",
    "PP04D_COD":  "cod_ocupacion",
    "P21":        "ingreso_ppal",
    "P47T":       "ingreso_total",
    "REGION":     "region",
    "PP03D":      "cant_ocupaciones_ad",
}

ocupados = ocupados.rename(columns=renombres)

#%% 1.7 Construcción del indicador de informalidad:
ocupados["informal"] = (
    (ocupados["cat_ocup2"] == "Asalariado") &
    (ocupados["desc_jubilatorio"] == "No") &
    ((ocupados["tam_estab"] <= 5) | (ocupados["tam_estab_agrup"] == "Hasta 5"))
).map({True: "Informal", False: "Formal"})

#%% 2. Estadística descriptiva:
var_continuas_tp1 = ["edad", "ingreso_ppal", "ingreso_total", "cant_ocupaciones_ad", "tam_estab"]
var_nuevas_tp2 = ["edad2", "educ", "horastrab", "horastrabj", "nhogar"]

# 2.1. Unificar las listas de variables:
var_continuas = var_continuas_tp1 + var_nuevas_tp2

# 2.2. Deifinir los percentiles:
percentiles = [0.25, 0.50, 0.75]

# 2.3. Presentar los estadísticos descriptivos:
tabla_desc = ocupados[var_continuas].describe(percentiles = percentiles).T
tabla_desc = tabla_desc.rename(columns = {
    "count": "N", "mean": "Promedio", "std": "Desvio Est.",
    "min": "Min", "25%": "P25", "50%": "P50", "75%": "P75", "max": "Max"
})
tabla_desc = tabla_desc[["N", "Promedio", "Desvio Est.", "Min", "P25", "P50", "P75", "Max"]].round(2)
print(tabla_desc)

# 2.4. Exportar a Excel tabla con estadísticos descriptivos:
with pd.ExcelWriter("tabla_descriptiva_TP2.xlsx") as writer:
    tabla_desc.to_excel(writer, sheet_name = "Descriptiva_TP2")

#%% 3. Matriz de correlaciones (2024 y 2025) - base de ocupados:
variables_corr = ["informal", "edad", "edad2", "educ", "horastrabj", "nhogar", "ingreso_ppal", "tam_estab_raw"]
etiquetas_corr = {
    "informal":     "Informalidad",
    "edad":         "Edad",
    "edad2":        "Edad al cuadrado",
    "educ":         "Años de educación",
    "horastrabj":   "Horas trabajadas (jefe)",
    "nhogar":       "Miembros del hogar",
    "ingreso_ppal": "Ingreso ocupación principal",
    "tam_estab_raw":"Tamaño del establecimiento",
}

# 3.1. Crear un objeto para guardar las matrices de correlación:
matrices_corr = {}

for año in [2024, 2025]:
    subset = ocupados[ocupados["año"] == año][variables_corr].copy()
    subset["informal"] = subset["informal"].map({"Formal": 1, "Informal": 2})

    corr_matrix = subset.corr()
    corr_matrix = corr_matrix.rename(index = etiquetas_corr, columns = etiquetas_corr)
    matrices_corr[año] = corr_matrix

    print(f"\nMatriz de correlación - Ocupados {año}")
    print(corr_matrix.round(2))

with pd.ExcelWriter("matriz_correlaciones_TP2.xlsx") as writer:
    for año, matriz in matrices_corr.items():
        matriz.round(2).to_excel(writer, sheet_name = f"Corr_{año}")
        
# 3.1. Crear heatmaps para representar el análisis correlación:
for año in [2024, 2025]:

    # Filtro de la muestra de ocupados y codificación binaria de informalidad:
    subset = ocupados[ocupados["año"] == año][variables_corr].copy()
    subset["informal"] = subset["informal"].map({"Formal": 0, "Informal": 1})

    # Cálculo de la matriz de correlaciones y renombre de etiquetas:
    corr_matrix = subset.corr()
    corr_matrix = corr_matrix.rename(index = etiquetas_corr, columns = etiquetas_corr)

    # Máscara para mostrar solo el triángulo inferior:
    mask = np.triu(np.ones_like(corr_matrix, dtype = bool))

    # Inicialización del lienzo:
    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))

    # Heatmap de correlaciones:
    sns.heatmap(
        corr_matrix,
        mask = mask,
        vmin = -1, vmax = 1,
        annot = True, fmt = ".2f",
        cmap = "coolwarm",
        linewidths = 0.5,
        ax = ax,
        cbar_kws = {"label": "Correlación de Pearson", "shrink": 0.8}
    )

    # Título y formato de los ejes:
    ax.set_title(
        f"Ocupados {año}",
        fontsize = 13, fontweight = "bold", loc = "center", pad = 15
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 35, ha = "right", fontsize = 9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation = 0, fontsize = 9)

    # Exportación del gráfico:
    plt.tight_layout()
    plt.savefig(f"heatmap_correlaciones_{año}.png", dpi = 300, bbox_inches = "tight")
    plt.show()

# =============================================================================
# PARTE II: MÉTODOS NO SUPERVISADOS
# =============================================================================

# =============================================================================
# PARTE II.A: ANÁLISIS DE COMPONENTES PRINCIPALES (PCA)
# (Luciano Altamirano)
# =============================================================================


# =============================================================================
# PARTE II.B: ANÁLISIS CLÚSTER
# (Gonzalo Pasiche)
# =============================================================================

ocupados.info()

ocupados[["edad", "edad2", "educ", "horastrabj", "nhogar", "ingreso_ppal", "tam_estab_raw"]].info()

#%%---- 1. Clúster k medias-----
"1.1 Variables para entrenar el modelo:"
variables_cluster = ["edad", "edad2", "educ", "horastrabj",
                     "nhogar", "ingreso_ppal", "tam_estab_raw"]

for año in [2024, 2025]:

    "1.2 Preparación de los datos (filtro por año y eliminación de faltantes):"
    datos = ocupados[ocupados["año"] == año].copy()
    datos = datos.dropna(subset = variables_cluster + ["informal"])
    X = datos[variables_cluster]

    "1.3 Estandarización de las variables:"
    scaler = StandardScaler()
    X_estandarizado = scaler.fit_transform(X)

    "1.4 Entrenamiento del K-means (k = 2, n_init = 20):"
    kmeans = KMeans(n_clusters = 2, n_init = 20, random_state = 42)
    datos["cluster"] = kmeans.fit_predict(X_estandarizado)

    "1.5 Gráfico de los resultados (2 variables de las 7):"
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (9, 7))
    sns.scatterplot(
        data = datos,
        x = "edad", y = "ingreso_ppal",
        hue = "cluster",
        palette = "Set1",
        alpha = 0.5,
        ax = ax
    )
    ax.set_title(f"K-means (k = 2) - Ocupados {año}",
                fontsize = 13, fontweight = "bold", pad = 15)
    ax.set_xlabel("Edad")
    ax.set_ylabel("Ingreso ocupación principal")
    ax.legend(title = "Cluster")
    plt.tight_layout()
    plt.savefig(f"kmeans_cluster_{año}.png", dpi = 300, bbox_inches = "tight")
    plt.show()

    "1.6 Comparación de los clusters con la informalidad real:"
    tabla = pd.crosstab(datos["cluster"], datos["informal"])
    print(f"\nComparación cluster vs informalidad - Ocupados {año}")
    print(tabla)

#%%---- 2. Método del codo (Elbow)-----

"2.1 Cálculo de la inercia para k = 1 hasta k = 40:"
inercias = []
rango_k = range(1, 41)

# Usamos los mismos datos estandarizados del último año procesado.
# Si querés un año puntual, volvé a filtrar y estandarizar acá.
for k in rango_k:
    kmeans = KMeans(n_clusters = k, n_init = 20, random_state = 42)
    kmeans.fit(X_estandarizado)
    inercias.append(kmeans.inertia_)

"2.2 Gráfico de la curva del codo:"
sns.set_style("whitegrid")
fig, ax = plt.subplots(figsize = (10, 6))
ax.plot(rango_k, inercias, marker = "o", color = "steelblue")
ax.set_title("Método del codo (Elbow)", fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel("Número de clusters (k)")
ax.set_ylabel("Inercia (WCSS)")
plt.tight_layout()
plt.savefig("elbow_method.png", dpi = 300, bbox_inches = "tight")
plt.show()


#%%---- 3. Clúster jerárquico-----
"3.1 Preparación de una muestra (el jerárquico no escala a toda la base):"
# Tomamos una muestra aleatoria para que el dendrograma sea legible y no sature la memoria.
muestra = ocupados.dropna(subset = variables_cluster).sample(
    n = 300, random_state = 42
)
X_muestra = muestra[variables_cluster]

"3.2 Estandarización de las variables:"
scaler = StandardScaler()
X_muestra_estandarizado = scaler.fit_transform(X_muestra)

"3.3 Cálculo de las uniones jerárquicas (método de Ward):"
# 'ward' agrupa minimizando la varianza interna de cada cluster.
enlaces = linkage(X_muestra_estandarizado, method = "ward")

"3.4 Gráfico del dendrograma:"
fig, ax = plt.subplots(figsize = (14, 7))
dendrogram(enlaces, ax = ax, no_labels = True, color_threshold = 0)
ax.set_title("Dendrograma - Clustering jerárquico (Ward)",
            fontsize = 13, fontweight = "bold", pad = 15)
ax.set_xlabel("Observaciones (muestra de ocupados)")
ax.set_ylabel("Distancia (disimilitud)")
plt.tight_layout()
plt.savefig("dendrograma.png", dpi = 300, bbox_inches = "tight")
plt.show()
