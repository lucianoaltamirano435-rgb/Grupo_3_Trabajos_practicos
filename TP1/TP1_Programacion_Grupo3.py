
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import seaborn as sns

# Cargar la base de datos:
os.chdir(r"/Volumes/ADATA HD330/Maestría Economía Aplicada UBA/Taller de programación/Trabajos prácticos/TP1 (preliminar)")
os.chdir(r"C:\Users\gmpas\OneDrive\Escritorio\Seminario Programación\TP1")

print(os.getcwd())

pd.set_option("display.float_format", "{:,.00f}".format)

"""
(*) El trabajo se desarrolló de forma conjunta, las ediciones de las 
parte 1 y 2 aparecen con el users de Luciano Altamarano dado 
que él fue quien creó el repositorio

parte 1-2 --> Gonzalo Pasiche
parte 3 --> Luciano Altamirano
"""

#%% PARTE 1
#%% 1.1 Carga de datos

"(I) Carga de datos en formato excel:"
# variables = ["CODUSU", "NRO_HOGAR", "ANO4", "TRIMESTRE", "PONDERA", 
#              "CH03", "CH04", "CH06", "CH07", "CH08", "NIVEL_ED", "ESTADO", 
#              "CAT_OCUP", "PP07H", "PP04C", "PP04C99", "EMPLEO", 
#              "SECTOR", "PP04D_COD", "P21", "P47T", "REGION", 
#              "PP07K", "PP07L", "PP07M", "PP03D"]

# bd_24 = pd.read_excel("usu_individual_T424.xlsx", usecols=variables)
# bd_25 = pd.read_excel("usu_individual_T425.xlsx", usecols=variables)


"""
(II) Transformar en formato parquet, para que no pese mucho: 
la bd"
"""
# bd_24.to_parquet("bd_24.parquet")
# bd_25.to_parquet("bd_25.parquet")

bd_24 = pd.read_parquet("bd_24.parquet")
bd_25 = pd.read_parquet("bd_25.parquet")

#%% 1.2 Corrección de valores sin sentido

"(I) Ingresos"
"Los -9 se tratan en la EPH como NaN"
bd_24[["P21", "P47T"]].describe()
bd_25[["P21","P47T"]].describe()

bd_24["P21"] = bd_24["P21"].replace(-9, np.nan)
bd_24["P47T"] = bd_24["P47T"].replace(-9, np.nan)

bd_25["P21"] = bd_25["P21"].replace(-9,np.nan)
bd_25["P47T"] = bd_25["P47T"].replace(-9,np.nan)

"(II) Edad - CH06"
"""
En la EPH (-1) es consignada para las no respuestas, 
se reemplazaran por NaN"
"""
bd_24[["CH06"]].describe()
bd_25[["CH06"]].describe()

bd_24["CH06"] = bd_24["CH06"].replace(-1, np.nan)
bd_25["CH06"] = bd_25["CH06"].replace(-1, np.nan)


"(III) Pondera - PONDERA"
"No puede existir valores cero o negativos"
bd_24[["PONDERA"]].describe()
bd_25[["PONDERA"]].describe()

#%% 1.3. Union de bases de datos
"""
Luego de la revisión se unirán ambas bases de cada año
la cual se llamará como bd
"""
bd = pd.concat([bd_24, bd_25], ignore_index=True)

# Identificar viviendas con al menos un hijo/nieto menor de 5 años
viviendas_con_menor = bd[
    (bd["CH03"].isin([3, 5])) & 
    (bd["CH06"] <= 5)
]["CODUSU"].unique()

# Generalizar el valor a todos los miembros de esa vivienda
bd["dummy_menor5"] = bd["CODUSU"].isin(viviendas_con_menor).astype(int)
print(bd["dummy_menor5"].value_counts())

#%% 1.4 Variable ESTADO y guardar bases (respondieron - norespondieron)

bd["ESTADO"].value_counts(normalize=True)*100

respondieron = bd[bd["ESTADO"] != 0]
norespondieron = bd[bd["ESTADO"] == 0]

#%% 1.5. Creación de base ocupados

ocupados = respondieron[respondieron["ESTADO"] == 1]

ocupados["ESTADO"].describe()


#%% 1.6 Creación de variables dicotómicas

"""
Se crearán las siguientes dicotómicas, con sus
respectivas etiquetas:

CH04 --> sexo (1=Masculino ; 2=Femenino)
CH07 --> estado_civil (1=Pareja (<3) ; 2=Sin pareja (>=3))
SECTOR--> sector_2 (1=Formal ; 2=Informal) , lo otro lo paso a NaN
NIVEL_ED --> nivel_ed2 (1=Basico (!5 y 6) ; 2 = Superior)
CAT_OCUP --> cat_ocup2 (1= No asalariado (1,2 ,4) ; 2=Asalarioado (3))
ESTADO --> ya todo es como 1 (OCUPADO)
CH06 --> años cumplidos (continua)
CH08 --> cobertura_med (1=cobertura ; 2= no_cobertura)
"""

"sexo"
ocupados["sexo"] = ocupados["CH04"].map(
                {1:"Masculino",2:"Femenino"})

ocupados["sexo"].value_counts(normalize=True)*100

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

ocupados["nivel_ed2"].value_counts(normalize=True)*100


"cat_ocup2"
ocupados["cat_ocup2"] = ocupados["CAT_OCUP"].map({
    1: "No asalariado",
    2: "No asalariado",
    3: "Asalariado",
    4: np.nan,
    9: np.nan
})

ocupados["cat_ocup2"].value_counts(normalize=True)*100

"cobertura_med"
ocupados["cobertura_med"] = ocupados["CH08"].map({
    1: "Cobertura",
    2: "Cobertura",
    3: "Cobertura",
    12: "Cobertura",
    13: "Cobertura",
    23: "Cobertura",
    123: "Cobertura",
    4: "No cobertura",
    9: np.nan
})

print(ocupados["cobertura_med"].value_counts())


"""
Para que el gráfico heatmap tenga sentido tamnbien se 
renombran estas variables
"""

respondieron = respondieron.rename(columns={
    "CH04": "sexo",
    "CH07": "estado_civil",
    "CH08": "cobertura_med"
})


#%% 1.7 Nivel de ingresos (pesos 2024 a pesos 2025)

"""
El valor de inflacion se sacó del informe del INDEC 
de noviembre 2025, se usó noviembre por que es el mes 
central del trimestre analizado
"""
factor = 1.314  # 31.4% inflación nov2024 a nov2025

ocupados.loc[ocupados["ANO4"] == 2024, "P21"]  = ocupados.loc[ocupados["ANO4"] == 2024, "P21"]  * factor
ocupados.loc[ocupados["ANO4"] == 2024, "P47T"] = ocupados.loc[ocupados["ANO4"] == 2024, "P47T"] * factor

print(ocupados.groupby("ANO4")[["P21", "P47T"]].mean().round(0))

#%% 1.8 Renombre de variables (respondieron y ocupados)

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
    "PP07H":      "desc_jubilatorio",
    "PP04C":      "tam_estab",
    "PP04C99":    "tam_estab_agrup",
    "EMPLEO":     "tipo_empleo",
    "SECTOR":     "tipo_sector",
    "PP04D_COD":  "cod_ocupacion",
    "P21":        "ingreso_ppal",
    "P47T":       "ingreso_total",
    "REGION":     "region",
    "PP07K":      "comprobante_sal",
    "PP07L":      "alcance_recibo",
    "PP07M":      "parte_sueldo",
    "PP03D":      "cant_ocupaciones_ad"
}

ocupados = ocupados.rename(columns=renombres)
respondieron = respondieron.rename(columns=renombres)

"""
Como ya se habían creado variables en la seccion 6
ser verifica si existen duplicados
"""

print(ocupados.columns.duplicated().sum())
print(ocupados.columns[ocupados.columns.duplicated()].tolist())

print(respondieron.columns.duplicated().sum())
print(respondieron.columns[respondieron.columns.duplicated()].tolist())



#%% 1.9 Heatmap 

variables = list(renombres.values()) + ["sexo", "estado_civil", "cobertura_med"]

def calcular_nan(base, nombre_base):
    años = base["año"].unique()
    resultado = {}
    for año in años:
        subset = base[base["año"] == año][variables]
        resultado[f"{nombre_base} {año}"] = subset.isnull().mean() * 100
    return pd.DataFrame(resultado)

nan_respondieron = calcular_nan(respondieron, "Respondieron")
nan_ocupados = calcular_nan(ocupados, "Ocupados")

# Asignar nombres autoexplicativos de las variables:
etiquetas_legibles = {
    "cod_vivienda":         "Código de vivienda",
    "nro_hogar":            "Número de hogar",
    "año":                  "Año",
    "trimestre":            "Trimestre",
    "ponderador":           "Ponderador",
    "edad":                 "Edad",
    "nivel_ed":             "Nivel educativo",
    "cond_actividad":       "Condición de actividad",
    "cat_ocup":             "Categoría ocupacional",
    "desc_jubilatorio":     "Descuento jubilatorio",
    "tam_estab":            "Tamaño del establecimiento",
    "tam_estab_agrup":      "Tamaño del establec. (agrupado)",
    "tipo_empleo":          "Tipo de empleo",
    "tipo_sector":          "Sector (formal/informal)",
    "cod_ocupacion":        "Código de ocupación",
    "ingreso_ppal":         "Ingreso ocupación principal",
    "ingreso_total":        "Ingreso total individual",
    "region":               "Región",
    "comprobante_sal":      "Comprobante de sueldo",
    "alcance_recibo":       "Alcance del recibo",
    "parte_sueldo":         "Pago en parte del sueldo",
    "cant_ocupaciones_ad":  "Cantidad de ocupaciones adicionales",
    "sexo":                 "Sexo",
    "estado_civil":         "Estado civil",
    "cobertura_med":        "Cobertura médica",
}

# Preparar los datos a reportarse en el gráfico:
df_heatmap = pd.concat([nan_respondieron, nan_ocupados], axis = 1)
df_heatmap = df_heatmap.rename(index = etiquetas_legibles)

n_obs = {
    "Respondieron 2024": len(respondieron[respondieron["año"] == 2024]),
    "Respondieron 2025": len(respondieron[respondieron["año"] == 2025]),
    "Ocupados 2024":     len(ocupados[ocupados["año"] == 2024]),
    "Ocupados 2025":     len(ocupados[ocupados["año"] == 2025]),
}

# Crear el gráfico:
fig, ax = plt.subplots(figsize=(14, 11))

sns.heatmap(df_heatmap, annot = True, fmt = ".1f", cmap = "YlOrRd",
            linewidths = 0.5, ax = ax, cbar_kws = {"label": "% NaN"})

ax.set_title("Valores faltantes por variable, base y año",
             fontsize = 13, fontweight = "bold", loc = "left", pad = 25)

subtitulo = "(porcentaje) | " + " | ".join([f"{k}: n={v:,}" for k, v in n_obs.items()])
ax.annotate(subtitulo, xy = (0, 1.02), xycoords = "axes fraction",
            fontsize = 9, color = "gray", ha = "left")

ax.set_xlabel("Base y año", fontsize = 11)
ax.set_ylabel("Variables", fontsize = 11)

plt.tight_layout()
plt.savefig("heatmap.png", dpi = 300, bbox_inches = "tight")
plt.show()

#%% PARTE 2

"""
El analisis de correlación se hará usando tanto 
variables discretas como dicotómicas creadas:

CH04 --> sexo (1=Masculino ; 2=Femenino)
CH07 --> estado_civil (1=Pareja (<3) ; 2=Sin pareja (>=3))
SECTOR--> sector_2 (1=Formal ; 2=Informal) , lo otro lo paso a NaN
NIVEL_ED --> nivel_ed2 (1=Basico (!5 y 6) ; 2 = Superior)
CAT_OCUP --> cat_ocup2 (1= No asalariado (1,2 ,4) ; 2=Asalarioado (3))
ESTADO --> ya todo es como 1 (OCUPADO)
CH06 --> años cumplidos
CH08 --> cobertura_med (1=cobertura ; 2= no_cobertura)
"""

#%% 2.1 Matriz de correlaciones

"""
Este gráfico sólo servirá para verificar la inoperancia de la
correlación de spearmen cuando tenermos más de 1 nivel en variables
categóricas, por ello en 1.2 se muestra la matriz para 
las dicotómicas creadas
"""

# Crear un índice de etiquetas:
etiquetas_corr = {
    "tipo_sector":    "Sector (formal/informal)",
    "cond_actividad": "Condición de actividad",
    "cat_ocup":       "Categoría ocupacional",
    "ingreso_ppal":   "Ingreso ocupación principal",
    "ingreso_total":  "Ingreso total individual",
    "sexo":           "Sexo",
    "edad":           "Edad",
    "estado_civil":   "Estado civil",
    "cobertura_med":  "Cobertura médica",
    "nivel_ed":       "Nivel educativo",
}

# Se define el grupo de variables para el análisis:
variables_corr = ["tipo_sector", "cond_actividad", "cat_ocup", "ingreso_ppal", "ingreso_total", 
                  "sexo", "edad", "estado_civil", "cobertura_med", "nivel_ed"]

# Estructuración del gráfico:
for año in [2024, 2025]:
    subset = ocupados[ocupados["año"] == año][variables_corr].copy()
    
    # Convertir categóricas a número solo para el gráfico
    for col in subset.select_dtypes(include = "object").columns:
        subset[col] = pd.Categorical(subset[col]).codes
        subset[col] = subset[col].replace(-1, np.nan)  # -1 es NaN en codes
    
    corr_matrix = subset.corr()
    corr_matrix = corr_matrix.rename(index = etiquetas_corr, columns = etiquetas_corr)
    upp_mat = np.triu(corr_matrix)
    
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (10, 8))
    sns.heatmap(corr_matrix, vmin = -1, vmax = 1, annot = True, fmt = ".2f",
                cmap = "coolwarm", mask = upp_mat, ax = ax)
    ax.set_title(f"Matriz de correlación - Variables originales {año}")
    plt.tight_layout()
    plt.show()

#%% 2.2 Matriz de correlaciones (con variables dicotómicas):
# Convertir temporalmente las variables de texto a números solo para la correlación.
ocupados_num = ocupados.copy()

# Crear un índice de etiquetas:
etiquetas_corr2 = {
    "sexo":         "Sexo",
    "estado_civil": "Estado civil",
    "sector_2":     "Sector (formal/informal)",
    "nivel_ed2":    "Nivel educativo",
    "cat_ocup2":    "Categoría ocupacional",
    "cobertura_med":"Cobertura médica",
    "ingreso_ppal": "Ingreso ocupación principal",
    "ingreso_total":"Ingreso total individual",
    "edad":         "Edad",
}

# Dicotómicas creadas (texto a número):
ocupados_num["sexo"] = ocupados_num["sexo"].map({"Masculino": 1, "Femenino": 2})
ocupados_num["estado_civil"] = ocupados_num["estado_civil"].map({"Pareja": 1, "Sin pareja": 2})
ocupados_num["sector_2"] = ocupados_num["sector_2"].map({"Formal": 1, "Informal": 2})
ocupados_num["nivel_ed2"] = ocupados_num["nivel_ed2"].map({"Basico/No_nivel": 1, "Superior": 2})
ocupados_num["cat_ocup2"] = ocupados_num["cat_ocup2"].map({"No asalariado": 1, "Asalariado": 2})
ocupados_num["cobertura_med"] = ocupados_num["cobertura_med"].map({"Cobertura": 1, "No cobertura": 2})

# Continuas ya renombradas: edad, ingreso_ppal, ingreso_total:
variables_corr = ["sexo", "estado_civil", "sector_2", "nivel_ed2", "cat_ocup2",
                  "cobertura_med", "ingreso_ppal", "ingreso_total", "edad"]

# Crear el gráfico:
for año in [2024, 2025]:
    subset = ocupados_num[ocupados_num["año"] == año][variables_corr]

    corr_matrix = subset.corr()
    corr_matrix = corr_matrix.rename(index=etiquetas_corr2, columns=etiquetas_corr2)
    upp_mat = np.triu(corr_matrix)

    sns.set_style("white")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, vmin=-1, vmax=1, annot=True, fmt=".2f",
                cmap="coolwarm", mask=upp_mat, ax=ax)
    ax.set_title(f"Matriz de correlación - Variables {año}")
    plt.tight_layout()
    plt.savefig(f"matriz_corr_{año}.png", dpi=300, bbox_inches="tight")
    plt.show()


#%% 2.1 Tablas descriptivas

#%% 2.1.1 Etiquetado de variables

ocupados["region"].value_counts()

"nivel educativo"
ocupados["nivel_ed"] = ocupados["nivel_ed"].map({
    1: "Primario incompleto",
    2: "Primario completo",
    3: "Secundario incompleto",
    4: "Secundario completo",
    5: "Superior incompleto",
    6: "Superior completo",
    7: "Sin instrucción",
    9: np.nan
})

print(ocupados["nivel_ed"].value_counts())


"desc_jubilatorio"
ocupados["desc_jubilatorio"] = ocupados["desc_jubilatorio"].map({
    1: "Si",
    2: "No",
    9: np.nan
})

print(ocupados["desc_jubilatorio"].value_counts())


"tam_estab_agrup"
ocupados["tam_estab_agrup"] = ocupados["tam_estab_agrup"].map({
    1: "Hasta 5",
    2: "De 6 a 40",
    3: "Mas de 40",
    9: np.nan
})

print(ocupados["tam_estab_agrup"].value_counts())


"tipo_empleo"
ocupados["tipo_empleo"] = ocupados["tipo_empleo"].map({
    1: "Formal",
    2: "Informal",
    9: np.nan
})

print(ocupados["tipo_empleo"].value_counts())


"region"
ocupados["region"] = ocupados["region"].map({
    1: "Gran Buenos Aires",
    40: "Noroeste",
    41: "Noreste",
    42: "Cuyo",
    43: "Pampeana",
    44: "Patagonia"
})

print(ocupados["region"].value_counts())


"comprobante_sal"
ocupados["comprobante_sal"] = ocupados["comprobante_sal"].map({
    1: "Recibo_sello",
    2: "Recibo_nosello",
    3: "Factura",
    4: "Nada",
    5: "Ad_honorem"
})

print(ocupados["comprobante_sal"].value_counts())


"alcance_recibo"
ocupados["alcance_recibo"] = ocupados["alcance_recibo"].map({
    1: "Totalidad",
    2: "Solo una parte",
    0: np.nan,
    9: np.nan
})

print(ocupados["alcance_recibo"].value_counts())


"parte_sueldo"
ocupados["parte_sueldo"] = ocupados["parte_sueldo"].map({
    1: "Totalidad",
    2: "Parte",
    0: np.nan,
    9: np.nan
})

print(ocupados["parte_sueldo"].value_counts())


"tipo_sector"
ocupados["tipo_sector"] = ocupados["tipo_sector"].map({
    1: "Formal",
    2: "Informal",
    3: "Hogares",
    9: np.nan
})

print(ocupados["tipo_sector"].value_counts())

"cat_ocup"
ocupados["cat_ocup"] = ocupados["cat_ocup"].map({
    1: "Patron",
    2: "Cuenta propia",
    3: "Obrero o empleado",
    4: "Trab. familiar sin remuneracion",
    9: np.nan
})

print(ocupados["cat_ocup"].value_counts())

"tam_estab"
ocupados["tam_estab"] = ocupados["tam_estab"].replace(99, np.nan)

"cant_ocupaciones_ad"
ocupados["cant_ocupaciones_ad"] = ocupados["cant_ocupaciones_ad"].replace(9, np.nan)
#%% 2.1.2 Tablas descriptivas

"""
(*) descuento jubilatorio 
(*) comprobante_sal
(*) alcance_recibo 
(*) parte_sueldo

--> solo tiene si eres obrero o empleado (asalariado)"

"""

var_continuas = ["edad", "ingreso_ppal", "ingreso_total", 
                 "cant_ocupaciones_ad", "tam_estab"]

var_categoricas = ["sexo", "estado_civil", "cobertura_med", "nivel_ed", "cat_ocup",
                   "desc_jubilatorio", "tam_estab_agrup", "tipo_empleo", "tipo_sector",
                   "region", "comprobante_sal", "alcance_recibo", "parte_sueldo"]

percentiles = [0.01, 0.25, 0.50, 0.75, 0.99]

# Tabla continuas
tabla_cont = ocupados[var_continuas].describe(percentiles=percentiles).T
tabla_cont = tabla_cont.rename(columns={
    "count": "N", "mean": "Promedio", "std": "Desvio Est.",
    "min": "Min", "1%": "P1", "25%": "P25", "50%": "P50",
    "75%": "P75", "99%": "P99", "max": "Max"
})
tabla_cont = tabla_cont[["N", "Promedio", "Desvio Est.", "Min", "P1", "P25", "P50", "P75", "P99", "Max"]].round(2)

# Tabla categóricas
lista_freq = []
for var in var_categoricas:
    freq = ocupados[var].value_counts()
    pct = ocupados[var].value_counts(normalize=True).mul(100).round(2)
    nulos = ocupados[var].isnull().sum()
    for val in freq.index:
        lista_freq.append({
            "Variable": var,
            "Valor": val,
            "Frecuencia": freq[val],
            "Porcentaje": pct[val],
            "NaN": nulos
        })
tabla_cat = pd.DataFrame(lista_freq)

# Guardar
with pd.ExcelWriter("tabla_descriptiva_total.xlsx") as writer:
    tabla_cont.to_excel(writer, sheet_name="Continuas")
    tabla_cat.to_excel(writer, sheet_name="Categoricas", index=False)

print("Guardada: tabla_descriptiva_total.xlsx")

#%% PARTE 3

#%% 3.1. Construcción del indicador de informalidad:
ocupados["informal"] = (
    (ocupados["cat_ocup2"] == 'Asalariado') &
    (ocupados["desc_jubilatorio"] == 'No') &
    ((ocupados["tam_estab"] <= 5) | (ocupados["tam_estab_agrup"] == "Hasta 5"))
).map({True: "Informal", False: "Formal"})

#%% 3.2. Construcción de la tabla de doble entrada:
    
#%% 3.2.1. Tabla de frecuencias absolutas para el año 2024:
ocupados_24 = ocupados[ocupados["año"] == 2024]
tabla_2024 = pd.crosstab(
    ocupados_24["informal"],
    ocupados_24["sexo"],
    values = ocupados_24["ponderador"],
    aggfunc = "sum",
    margins = True,
    margins_name = "Total"
)
print(tabla_2024)

#%% 3.2.2. Tabla de frecuencias relativas para el año 2024:
tabla_pond_2024 = pd.crosstab(
    ocupados_24["informal"],
    ocupados_24["sexo"],
    values = ocupados_24["ponderador"],
    aggfunc = "sum"
)
tabla_pond_pct_2024 = (tabla_pond_2024.div(tabla_pond_2024.sum(axis = 0), axis = 1) * 100)
print(tabla_pond_pct_2024.map(lambda x: f"{x:.2f}"))

#%% 3.2.3. Tabla de frecuencias absolutas para el año 2025:
ocupados_25 = ocupados[ocupados["año"] == 2025]
tabla_2025 = pd.crosstab(
    ocupados_25["informal"],
    ocupados_25["sexo"],
    values = ocupados_25["ponderador"],
    aggfunc = "sum",
    margins = True,
    margins_name = "Total"
)
print(tabla_2025)

#%% 3.2.4. Tabla de frecuencias relativas para el año 2025:
tabla_pond_2025 = pd.crosstab(
    ocupados_25["informal"],
    ocupados_25["sexo"],
    values = ocupados_25["ponderador"],
    aggfunc = "sum"
)
tabla_pond_pct_2025 = (tabla_pond_2025.div(tabla_pond_2025.sum(axis = 0), axis = 1) * 100)
print(tabla_pond_pct_2025.map(lambda x: f"{x:.2f}"))

#%% 3.3. Gráfico de barras de la informalidad por sexo y año (hogares con hijos menores de 5 años):

filtro_menor5 = ocupados[ocupados["dummy_menor5"] == 1]

tabla_informal_menor5 = filtro_menor5.groupby(["sexo", "año"]).apply(
    lambda x: np.average(x["informal"] == "Informal", weights = x["ponderador"]) * 100
).unstack()

tabla_informal_menor5 = tabla_informal_menor5.rename(index = {"Femenino": "Mujeres", "Masculino": "Hombres"})

fig, ax = plt.subplots(figsize = (8, 5))

tabla_informal_menor5.plot(
    kind = "bar",
    ax = ax,
    color = ["#2196F3", "#F44336"],
    edgecolor = "white"
)

for container in ax.containers:
    ax.bar_label(container, fmt = "%.2f", fontsize = 10)

ax.set_xlabel("")
ax.set_ylabel("Porcentaje (%)")
ax.set_xticklabels(tabla_informal_menor5.index, rotation = 0)
ax.legend(title="Año", loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol = 2)

plt.subplots_adjust(bottom=0.2)
plt.savefig("grafico_informalidad_menor5.png", dpi=300, bbox_inches="tight")
plt.show()

#%% 3.4. Gráfico de distribución por sexo:
ocupados['ln_ingreso_ppal'] = np.log(ocupados['ingreso_ppal'])
sns.set(style = "whitegrid")
fig, axs = plt.subplots(2, 2, figsize = (12, 8))

# (a) Histograma de ingreso_ppal (P21):
sns.histplot(data = ocupados, x = "ln_ingreso_ppal", hue = "informal",
             element = "step", stat = "density", common_norm = False, ax = axs[0, 0])
axs[0, 0].set_title("Histograma de ingreso principal (P21)")
axs[0, 0].set_xlabel("Ingreso ocupación principal (ln)")
axs[0, 0].set_ylabel("Densidad")

# (b) Kernel de ingreso_ppal (P21):
sns.kdeplot(data = ocupados, x = "ln_ingreso_ppal", hue = "informal",
            common_norm = False, ax = axs[0, 1])
axs[0, 1].set_title("Distribución de kernel de ingreso principal (P21)")
axs[0, 1].set_xlabel("Ingreso ocupación principal (ln)")
axs[0, 1].set_ylabel("Densidad")

# (c) Histograma de ingreso_total (P47T):
ocupados['ln_ingreso_total'] = np.log(ocupados['ingreso_total'])
sns.histplot(data = ocupados, x = "ln_ingreso_total", hue = "informal",
             element = "step", stat = "density", common_norm = False, ax = axs[1, 0])
axs[1, 0].set_title("Histograma de ingreso total (P47T)")
axs[1, 0].set_xlabel("Ingreso total individual (ln)")
axs[1, 0].set_ylabel("Densidad")

# (d) Kernel de ingreso_total (P47T):
sns.kdeplot(data = ocupados, x = "ln_ingreso_total", hue = "informal",
            common_norm = False, ax = axs[1, 1])
axs[1, 1].set_title("Distribución de kernel de ingreso total (P47T)")
axs[1, 1].set_xlabel("Ingreso total individual (ln)")
axs[1, 1].set_ylabel("Densidad")

# (e) Quitar el título de la leyenda:
for ax in axs.flat:
    ax.get_legend().set_title("")
    
# (f) Mostrar la figura:
fig.tight_layout()
plt.show()
