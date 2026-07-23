# -*- coding: utf-8 -*-

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
os.chdir(r"C:\Users\gmpas\OneDrive\Escritorio\Seminario Programación\TP2")

print(os.getcwd())

pd.set_option("display.float_format", "{:,.2f}".format)

# =============================================================================
# PARTE I: CREACION DE VARIABLES Y MAS ESTADISTICA DESCRIPTIVA
# (Luciano Altamirano)
# =============================================================================



# =============================================================================
# PARTE II: METODOS NO SUPERVISADOS
# =============================================================================
# %% Definicion comun de variables y preparacion de los datos (base completa):

"""
Los items 1 a 5 del TP2 trabajan con el mismo conjunto de siete variables
sobre la base completa de ocupados (2024 + 2025). Se prepara una sola vez
el objeto 'datos_pool' con la base filtrada y su version estandarizada,
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
    "educ":         "Anos de educacion",
    "horastrabj":   "Horas trab. (jefe)",
    "nhogar":       "Miembros del hogar",
    "ingreso_ppal": "Ingreso ppal.",
    "tam_estab":    "Tamano establec.",
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
X_cat = X_cat.astype(str).replace({"nan": "Sin dato", "NaN": "Sin dato"})

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


