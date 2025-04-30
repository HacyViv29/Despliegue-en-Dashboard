#Creamos el archivo de la App en el interprete principal

#####################################
#Importamos las librerias necesarias
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from funpymodeling import freq_tbl
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import plotly.graph_objects as go
#######################################

#Definimos la instancia
@st.cache_resource

#########################################
#Creamos la funcion de carga de archivos
def load_data():
    #Lectura del archivo csv
    df = pd.read_csv('Grecia.csv', index_col='name').drop(columns=['Unnamed: 0'], errors='ignore')

    #Seleccionamos las columnas tipo numericas del dataframe
    numeric_df = df.select_dtypes(include=['int', 'float']) #Devuelve columnas
    numeric_cols = numeric_df.columns #Devuelve lista de columnas

    #Seleccionamos las columnas de tipo texto
    text_df = df.select_dtypes(include=['object']) #Devuelve columnas
    text_cols = text_df.columns #Devuelve lista de columnas

    #Seleccionamos algunas columnas categoricas de valores para desplegar en diferentes
    categorical_colum_host_since = df['host_since']
    #Obtengo los valores unicos de la columna categorica seleccionada
    unique_categories_host_since = categorical_colum_host_since.unique()

    return df, numeric_cols, text_cols, unique_categories_host_since, numeric_df
#######################################################
#Cargo los datos obtenidos de la funcion load_data
df, numeric_cols, text_cols, unique_categories_host_since, numeric_df = load_data()

#######################################################
#Creación de Dashboard

st.sidebar.title("Dataset Airbnb: Atenas, Grecia")
# Widget 1: Select box
#Menu desplegable de opciones de las paginas seleccionadas
View = st.sidebar.selectbox(label="Opciones", options=["Análisis Categorico", "Regresión Lineal Simple", "Regresión Lineal Multiple", "Regresión Logistica"])

#Contenido de la vista 1
if View == "Análisis Categorico":
    #Generamos los encabezados para el dashboard
    st.title("Análisis Univariado de Variables Categoricas")

    ############################################################
    #Generamos los encabezados para la barra lateral (sidebar)

    st.sidebar.header("Sidebar de opciones")
    st.sidebar.subheader("Selección de variables")
    #Widget 3: Select box
    #Menu desplegable de opciones de la variable categorica seleccionada
    category_selected = st.sidebar.selectbox(label="Variable Categorica", options=text_cols)

    frecuencia = freq_tbl(df[category_selected])
    frecuencia_index = frecuencia.set_index(category_selected)
    frecuencia_limpio = frecuencia_index.drop(columns=["percentage","cumulative_perc"], axis=1)
    frecuencia_mayor_1 = frecuencia_limpio[frecuencia_limpio["frequency"] > 1]
    frecuencia_index_sorted = frecuencia_mayor_1.sort_values(by="frequency", ascending=False)
    top_n = 10  # Número de categorías a mostrar
    frecuencia_index_top = frecuencia_index_sorted.head(top_n)


##################################################################
    #Widget 2: Checkbox
    #Generamos un cuadro de selección (checkbox) en una barra lateral (sidebar) para mostrar el dataset
    st.sidebar.subheader("Selección de tablas y graficas")
    opcion = st.sidebar.radio(
    "Selecciona una opción:",
    (
        "Análisis Univariado del Dataframe",
        "Análisis Univariado de la Variable Seleccionada",
        "Tabla de frecuencia",
        "Gráfico de frecuencia",
        "Gráfico de dispersión",
        "Gráfico de área",
        "Gráfico hexagonal",
        "Gráfico de pastel"
    )
)

    ###############################################
    #Condicional para que aparezca el dataframe
    if opcion == "Análisis Univariado del Dataframe":
        st.subheader("- Análisis del Dataframe Completo.")
        for col in df.columns:
            st.subheader(f"Frecuencia de '{col}'")

            # Obtener tabla de frecuencia
            freq_table = df[col].value_counts(dropna=False).reset_index()
            freq_table.columns = [col, 'frequency']
            freq_table['percentage'] = freq_table['frequency'] / freq_table['frequency'].sum()
            freq_table['cumulative_perc'] = freq_table['percentage'].cumsum()

            # Mostrar tabla
            st.dataframe(freq_table)

    #####################################################

    elif opcion == "Análisis Univariado de la Variable Seleccionada":	
        st.subheader("- Análisis Univariado de la Variable Seleccionada.")
        #Mostramos el dataframe con las columnas seleccionadas
        st.dataframe(frecuencia_index)

    ####################################################

    elif opcion == "Tabla de frecuencia":
        st.subheader("- Tablas de Frecuencia.")
        #Mostramos el dataframe con las columnas seleccionadas
        st.dataframe(frecuencia_mayor_1)

#########################################################
    elif opcion == "Gráfico de frecuencia":
        st.subheader("- Grafico de frecuencia.")

        #Realizamos un grafico de frecuencia de nuestro dataframe con indice
        grafico1 = plt.figure(figsize=(10, 6))
        frecuencia_index_top["frequency"].plot(kind='bar', color='gray')
        plt.title("Top 10 categorías más frecuentes" if len(frecuencia_index_top) >= 10 else "Categorías más frecuentes")
        st.pyplot(grafico1)

    ##########################################################
    elif opcion == "Gráfico de dispersión":
        st.subheader("- Grafico de dispersión.")
        dispersion = px.scatter(frecuencia, x="frequency", y="cumulative_perc")
        st.plotly_chart(dispersion)

    ##########################################################
    elif opcion == "Gráfico de área":
        st.subheader("- Grafico de área.")
        ax = frecuencia.plot(kind="area", alpha=0.5, figsize=(10,4))
        area = ax.get_figure()  # obtenemos el Figure desde el Axes
        st.pyplot(area)

    ###########################################################
    elif opcion == "Gráfico hexagonal":
        st.subheader("- Grafico de hexagonal.")
        fig, ax = plt.subplots(figsize=(10,5))
        hb = frecuencia.plot.hexbin("frequency", "cumulative_perc", ax=ax)
        plt.colorbar(hb.collections[0], ax=ax)  # asignar el colorbar al gráfico
        st.pyplot(fig)

    ############################################################
    elif opcion == "Gráfico de pastel":
        st.subheader("- Gráfico de pastel")
        pastel, ax = plt.subplots(figsize=(10,5))  # Cambié el nombre de la variable a fig
        ax.pie(frecuencia_index_top["frequency"],labels=frecuencia_index_top.index, autopct="%0.1f %%", shadow=False)
        st.pyplot(pastel)

#############################################################
#Vista 2
if View == "Regresión Lineal Simple":
    st.title("Regresión Lineal Simple")

    #Generamos los encabezados para la barra lateral (sidebar)
    st.sidebar.header("Sidebar de opciones")
    st.sidebar.subheader("- Mapa de calor")

    #Widget 1: Checkbox
    #Generamos un cuadro de selección (checkbox) en una barra lateral (sidebar) para mostrar el dataset
    check_box_heatmap_total = st.sidebar.checkbox(label = "Mapa de calor de todo el Dataframe")

    if not check_box_heatmap_total:
        st.sidebar.subheader("Variables a seleccionar para el mapa de calor")
        #WIDGET 3: Multiselect box
        #Generamos un cuadro de selección múltiple (multiselect) para elegir las columnas a graficar
        numerics_vars_selected = st.sidebar.multiselect(label="Variables graficadas", options=numeric_cols)

        if len(numerics_vars_selected) > 0:
            min_valor = 2 if len(numerics_vars_selected) == 1 else len(numerics_vars_selected)
            max_valor = 13

            # Nos aseguramos de que el rango no sea inválido
            if min_valor > max_valor:
                st.sidebar.warning("Has seleccionado demasiadas opciones para generar un rango válido.")
            else:
                opciones_numeros = list(range(min_valor, max_valor + 1))
                num_vars = st.sidebar.selectbox("Elige el numero de variables:", opciones_numeros)
        else:
            st.sidebar.info("Selecciona al menos una variable para ver más controles.")

    button_heatmap = st.sidebar.button(label = "Mostrar Mapa de calor")

    if button_heatmap:
        if check_box_heatmap_total:
            df_to_plot = numeric_df
        elif len(numerics_vars_selected) > 0:
            if len(numerics_vars_selected) == num_vars:
                df_to_plot = numeric_df[numerics_vars_selected]
            else:
                # Calculamos la matriz de correlación absoluta
                corr_matrix = numeric_df.corr().abs()

                # Variables ya seleccionadas
                already_selected = set(numerics_vars_selected)

                # Calculamos la media de correlación con las seleccionadas
                correlated_candidates = (
                    corr_matrix[numerics_vars_selected]
                    .mean(axis=1)
                    .sort_values(ascending=False)
                    .drop(labels=already_selected, errors='ignore')  # Quitamos las ya seleccionadas
                )

                # Elegimos las que mejor se correlacionan
                additional_vars = correlated_candidates.head(num_vars - len(numerics_vars_selected)).index.tolist()

                # Construimos el DataFrame final
                df_to_plot = numeric_df[numerics_vars_selected + additional_vars]

        else:
            st.warning("No se han seleccionado variables.")
            st.stop()

        st.subheader("- Mapa de calor")
        num_labels = df_to_plot.shape[1]
        tick_fontsize = int(np.interp(num_labels, [5, 30], [14, 6]))  # Ajusta 5-30 variables a tamaño 14-6
        annot_fontsize = int(np.interp(num_labels, [5, 30], [14, 6]))  # Ajusta 5-30 variables a tamaño 14-6

        corr_Factors1 = df_to_plot.corr()

        # Graficar el heatmap
        mapaCalor, ax = plt.subplots(figsize=(22, 22))

        # Mapa de calor
        sns.heatmap(
            corr_Factors1, 
            cmap="coolwarm",
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            square=True,
            annot_kws={"size": annot_fontsize},
            cbar_kws={"shrink": 0.8},
            ax=ax
        )

        plt.xticks(rotation=90, fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)

        # Mostrar en dashboard
        st.pyplot(mapaCalor)

    #####################################################################

    st.sidebar.subheader("Selección de variables para Regresión Lineal")
    #Widget 5: Select box
    #Menu desplegable de opciones de la variable dependiente seleccionada
    dependent_variable = st.sidebar.selectbox(label="Variable dependiente", options=numeric_cols)

    #Widget 6: Select box
    #Menu desplegable de opciones de la variable independiente seleccionada
    independent_variable = st.sidebar.selectbox(label="Variable independiente", options=numeric_cols)

    #Generamos un botón (button) en una barra lateral (sidebar) para mostrar el dataset
    button_mapa_dispersion = st.sidebar.button(label = "Mostrar Mapa de Dispersión original de las variables")
    button_regresion_simple = st.sidebar.button(label = "Mostrar Regresión Lineal Simple")

    if button_mapa_dispersion:
        if independent_variable == dependent_variable:
            st.warning("No se puede graficar la misma variable.")
            st.stop()
        else:
            #Realizamos un grafico de dispersión de las variables seleccionadas
            st.subheader(f"- Grafcico de dispersión de {independent_variable} y {dependent_variable}")
            disper = px.scatter(numeric_df, x=independent_variable, y=dependent_variable)
            st.plotly_chart(disper)

    if button_regresion_simple:
        if independent_variable == dependent_variable:
            st.warning("No se puede graficar la misma variable.")
            st.stop()
        else:
            st.subheader("- Datos del modelo")
            var_indep = numeric_df[[independent_variable]]
            var_dep = numeric_df[[dependent_variable]]
            st.write(f"Variable independiente: {independent_variable} y variable dependiente: {dependent_variable}")

            modelo = LinearRegression()
            modelo.fit(X=var_indep, y=var_dep)
            y_pred = modelo.predict(X=numeric_df[[independent_variable]])
            coef_Deter = modelo.score(var_indep, var_dep)
            coef_Correl = np.sqrt(coef_Deter)
            st.write(f"Coeficiente de determinación: {coef_Deter} y coeficiente de correlación: {coef_Correl}")
            numeric_df.insert(29, f"predicción de {independent_variable}", y_pred)

            st.subheader(f"Comparación de predicción de {independent_variable}")
            #Creamos la tabla comparativa con las dos columnas
            tabla_comparativa = numeric_df[[independent_variable, f"predicción de {independent_variable}"]].copy()


            # Mostramos la tabla en el dashboard
            st.write("Comparativa: Valor real vs. Valor predicho")
            st.dataframe(tabla_comparativa)

            st.subheader("- Gráfico de dispersión Comparativo")
            disp_comp = go.Figure()

            # Scatter de los valores reales
            disp_comp.add_trace(go.Scatter(
                x=numeric_df[dependent_variable],
                y=numeric_df[independent_variable],
                mode='markers',
                name=f'{independent_variable} Real',
                marker=dict(color='blue')
            ))

            # Scatter de los valores predichos
            disp_comp.add_trace(go.Scatter(
                x=numeric_df[dependent_variable],
                y=numeric_df[f"predicción de {independent_variable}"],
                mode='markers',
                name=f'{independent_variable} Predicho',
                marker=dict(color='red')
            ))

            # Títulos y etiquetas
            disp_comp.update_layout(
                title=f"Gráfico de dispersión comparativo de {independent_variable} real VS {independent_variable} predicho",
                xaxis_title=dependent_variable,
                yaxis_title=independent_variable,
                legend=dict(x=1, y=0.5)
            )

            # Mostrar en el dashboard
            st.plotly_chart(disp_comp)

            numeric_df.drop(columns=[f"predicción de {independent_variable}"], inplace=True)


