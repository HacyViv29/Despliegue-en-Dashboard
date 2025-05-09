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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, recall_score
#######################################

#######################################
#Encabezado de la app
ciudad = "Atenas"
site_name = f"Airbnb en la ciudad {ciudad}"
st.set_page_config(page_title=site_name, page_icon="🏠", layout="wide")

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

    # Obtener las variables dicotómicas con valores "t" y "f" o "1" y "0"
    binary_cols = [
        col for col in df.columns
        if df[col].nunique() == 2 and (
            set(df[col].dropna().unique()) <= {"1", "0"} or 
            set(df[col].dropna().unique()) <= {"t", "f"}
        )
    ]

    binary_df = df[binary_cols]


    return df, numeric_cols, text_cols, unique_categories_host_since, numeric_df, binary_cols, binary_df
#######################################################
#Cargo los datos obtenidos de la funcion load_data
df, numeric_cols, text_cols, unique_categories_host_since, numeric_df, binary_cols, binary_df= load_data()

#######################################################
#Creación de Dashboard

# Logo Airbnb desde URL
logo_url = "https://bigcleanswitch.org/wp-content/uploads/2019/08/190806-Airbnb-Logo-White.png"

# Estilo CSS para fondo y texto de la sidebar
st.markdown("""
    <style>
    [data-testid="stSidebar"] {
        background-color: #FF5A5F;
        color: white;
        text-align: center;
        align-items: center;
    }

    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5 {
        color: white;
    }

    [data-testid="stSidebar"] .stButton > button {
        background-color: white;
        color: #FF5A5F;
        border: none;
        padding: 0.5em 1em;
        border-radius: 5px;
        font-weight: bold;
        transition: background-color 0.3s, color 0.3s;
    }

    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #ffe5e6;
        color: #FF5A5F;
    }

    /* Slider - cambiar color de las pistas */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:first-child {
        background: black !important; /* parte antes del valor seleccionado */
    }

    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:nth-child(2) {
        background: white !important; /* parte seleccionada */
    }

    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:nth-child(3) {
        background: black !important; /* parte después del valor seleccionado */
    }

    /* Thumb (círculo) del slider */
    [data-testid="stSidebar"] .stSlider [role="slider"] {
        background-color: #ffffff !important;
        border: 2px solid #FF5A5F !important;
    }

    .center-logo {
        display: flex;
        justify-content: center;
    }
    </style>
""", unsafe_allow_html=True)

# Mostrar logo y título en la sidebar
with st.sidebar:
    st.markdown(f"<div class='center-logo'><img src='{logo_url}' width='120'></div>", unsafe_allow_html=True)

st.sidebar.title("Dataset Airbnb: Atenas, Grecia")
#Menu desplegable de opciones de las paginas seleccionadas
View = st.sidebar.selectbox(label="Opciones", options=["Datos Generales","Mapa de residencias","Análisis Univariado", "Regresión Lineal Simple", "Regresión Lineal Multiple", "Regresión Logistica"])

#Contenido de la vista 1
if View == "Datos Generales":
    st.sidebar.title("Datos Generales")
    st.sidebar.markdown("Filtra los datos según tus preferencias")
    # Filtro por tipo de habitación
    room_type = st.sidebar.selectbox("Tipo de habitación", df['room_type'].unique())

    # Filtro por noches mínimas
    min_nights = st.sidebar.slider("Noches mínimas", min_value=int(df['minimum_nights'].min()), max_value=int(df['minimum_nights'].max()), value=(int(df['minimum_nights'].min()), int(df['minimum_nights'].max())))

    # Filtro por noches máximas
    max_nights = st.sidebar.slider("Noches máximas", min_value=int(df['maximum_nights'].min()), max_value=int(df['maximum_nights'].max()), value=(int(df['maximum_nights'].min()), int(df['maximum_nights'].max())))

    # Filtro por rango de precio
    price_range = st.sidebar.slider("Rango de precio", min_value=int(df['price'].min()), max_value=int(df['price'].max()), value=(int(df['price'].min()), int(df['price'].max())))

    # Filtrado del DataFrame según los valores seleccionados
    df_filtrado = df[
        (df['room_type'] == room_type) &
        (df['minimum_nights'] >= min_nights[0]) & (df['minimum_nights'] <= min_nights[1]) &
        (df['maximum_nights'] >= max_nights[0]) & (df['maximum_nights'] <= max_nights[1]) &
        (df['price'] >= price_range[0]) & (df['price'] <= price_range[1])
    ] 

    #Generamos los encabezados para el dashboard
    st.title(f"{ciudad} - Datos Generales")
    st.markdown("## 📋 Resumen del Dataset filtrado")
    st.write(df_filtrado.head())

    st.markdown("## 🧾 Resumen del Dataset filtrado")

    # Sección: Información General
    with st.container():
        st.markdown("### 📌 Información General")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Tiempo más frecuente de respuesta", df_filtrado['host_response_time'].mode()[0])
            st.metric("Tipo de alquiler seleccionado", df_filtrado['room_type'].mode()[0])
        with col2:
            st.metric("Disponibilidad Media (365 días)", f"{df_filtrado['availability_365'].mean():.0f} días")
            if 'review_scores_rating' in df_filtrado.columns and df_filtrado['review_scores_rating'].notnull().any():
                st.metric("Calificación promedio", f"{df_filtrado['review_scores_rating'].mean():.2f} / 5.0")

    # Sección: Precios
    with st.container():
        st.markdown("### 💰 Estadísticas de Precio")
        col3, col4, col5 = st.columns(3)
        col3.metric("Precio promedio", f"${df_filtrado['price'].mean():.2f}")
        col4.metric("Precio mínimo", f"${df_filtrado['price'].min():.2f}")
        col5.metric("Precio máximo", f"${df_filtrado['price'].max():.2f}")

    # Sección: Actividad
    with st.container():
        st.markdown("### ⭐ Actividad de los Anuncios")
        col6 = st.columns(1)[0]
        col6.metric("Promedio de Reseñas por Anuncio", f"{df_filtrado['number_of_reviews'].mean():.1f}")

######################################################################
#Contenido de la vista 2
if View == "Mapa de residencias":
    #Sidebar
    st.sidebar.title("Mapa de residencias")
    st.sidebar.markdown("Busca residencias dependiendo del tipo y su precio")

    #Selección del tipo de habitación
    seleccion = st.sidebar.selectbox(label="Tipo de habitación", options=df["room_type"].unique())

    precio = st.sidebar.slider(label="Rango de precio buscado", min_value=float(df["price"].min()), max_value=float(df["price"].max()), value=(float(df["price"].min()),float(df["price"].max())), step=1.0) 

    df_resultado = df[(df["room_type"] == seleccion) & (df["price"].between(precio[0], precio[1]))]

    #Dashboard
    st.title(f"{ciudad} - Mapa de Alojamientos")
    # Mapa de Alojamientos
    fig3 = px.scatter_map(
        df_resultado,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        color_continuous_scale=px.colors.cyclical.IceFire,
        height=500
    )
    fig3.update_layout(
        map=dict(
            style="carto-positron",
            zoom=11,
            center=dict(lat=df_resultado["latitude"].mean(), lon=df_resultado["longitude"].mean())
        )
    )
    st.plotly_chart(fig3, use_container_width=True)

######################################################################
#Contenido de la vista 3
if View == "Análisis Univariado":
    #Generamos los encabezados para el dashboard
    st.title(f"{ciudad} - Análisis Univariado de Variables Categóricas")

    ############################################################
    #Generamos los encabezados para la barra lateral (sidebar)

    st.sidebar.header("Sidebar de opciones")
    #Generamos un cuadro de selección (checkbox) en una barra lateral (sidebar) para mostrar el dataset
    st.sidebar.subheader("Selección de tablas y graficas")

    # Selección de variable
    variable = st.sidebar.selectbox("Selecciona una variable:", text_cols)

    opcion = st.sidebar.radio(
    "Selecciona una opción:",
    (
        "Análisis Univariado",
        "Tablas de frecuencia",
        "Gráficos de frecuencia",
        "Gráficos de dispersión",
        "Gráficos de área",
        "Gráficos hexagonales",
        "Gráficos de pastel"
    )
    )

    frecuencia = freq_tbl(df[variable])
    frecuencia_index = frecuencia.set_index(variable)
    frecuencia_limpio = frecuencia_index.drop(columns=["percentage","cumulative_perc"], axis=1)
    frecuencia_mayor_1 = frecuencia_limpio[frecuencia_limpio["frequency"] > 1]
    frecuencia_index_sorted = frecuencia_mayor_1.sort_values(by="frequency", ascending=False)
    top_n = 10  # Número de categorías a mostrar
    frecuencia_index_top = frecuencia_index_sorted.head(top_n)

    ###############################################
    #Condicional para que aparezca el dataframe
    if opcion  == "Análisis Univariado":
        st.subheader(f"- {opcion}")

        st.subheader(f"Análisis de '{variable}'")

        # Obtener tabla de frecuencia
        freq_table = df[variable].value_counts(dropna=False).reset_index()
        freq_table.columns = [variable, 'frequency']
        freq_table['percentage'] = freq_table['frequency'] / freq_table['frequency'].sum()
        freq_table['cumulative_perc'] = freq_table['percentage'].cumsum()

        # Mostrar tabla
        st.dataframe(freq_table)

    ####################################################

    elif opcion == "Tablas de frecuencia":
        st.subheader(f"- {opcion}")

        st.subheader(f"Tabla de frecuencia de '{variable}'")

        #Mostramos el dataframe con las columnas seleccionadas
        st.dataframe(frecuencia_mayor_1)

#########################################################
    elif opcion == "Gráficos de frecuencia":
        st.subheader(f"- {opcion}")

        st.subheader(f"Gráfico de frecuencia de '{variable}'")	

        #Realizamos un Gráfico de frecuencia de nuestro dataframe con indice
        Gráfico1 = plt.figure(figsize=(10, 6))
        frecuencia_index_top["frequency"].plot(kind='bar', color='gray')
        plt.title("Top 10 categorías más frecuentes" if len(frecuencia_index_top) >= 10 else "Categorías más frecuentes")
        st.pyplot(Gráfico1)

    ##########################################################
    elif opcion == "Gráficos de dispersión":
        st.subheader(f"- {opcion}")

        st.subheader(f"Gráfico de dispersión de '{variable}'")

        dispersion = px.scatter(frecuencia, x="frequency", y="cumulative_perc")
        st.plotly_chart(dispersion)

    ##########################################################
    elif opcion == "Gráficos de área":
        st.subheader(f"- {opcion}")

        st.subheader(f"Gráfico de área de '{variable}'")

        ax = frecuencia.plot(kind="area", alpha=0.5, figsize=(10,4))
        area = ax.get_figure()  # obtenemos el Figure desde el Axes
        st.pyplot(area)

    ###########################################################
    elif opcion == "Gráficos hexagonales":
        st.subheader(f"- {opcion}")

        st.subheader(f"Gráfico hexagonal de '{variable}'")

        fig, ax = plt.subplots(figsize=(10,5))
        hb = frecuencia.plot.hexbin("frequency", "cumulative_perc", ax=ax)
        plt.colorbar(hb.collections[0], ax=ax)  # asignar el colorbar al gráfico
        st.pyplot(fig)

    ############################################################
    elif opcion == "Gráficos de pastel":
        st.subheader(f"- {opcion}")

        st.subheader(f"Gráfico de pastel de '{variable}'")

        pastel, ax = plt.subplots(figsize=(10,5))  # Cambié el nombre de la variable a fig
        ax.pie(frecuencia_index_top["frequency"],labels=frecuencia_index_top.index, autopct="%0.1f %%", shadow=False)
        st.pyplot(pastel)

#############################################################
#Vista 4
if View == "Regresión Lineal Simple":
    st.title(f"{ciudad} - Regresión Lineal Simple")

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

    st.sidebar.subheader("Selección de variables para Regresión Lineal Simple")
    #Widget 5: Select box
    #Menu desplegable de opciones de la variable dependiente seleccionada
    dependent_variable = st.sidebar.selectbox(label="Variable dependiente", options=numeric_cols)

    #Widget 6: Select box
    #Menu desplegable de opciones de la variable independiente seleccionada
    independent_option = [var for var in numeric_cols if var != dependent_variable]
    independent_variable = st.sidebar.selectbox(label="Variable independiente", options=independent_option, index=0)

    # Validar que al menos una variable independiente fue seleccionada
    if not independent_variable:
        st.warning("Por favor selecciona al menos una variable independiente.")
        st.stop()

    #Generamos un botón (button) en una barra lateral (sidebar) para mostrar el dataset
    button_regresion_simple = st.sidebar.button(label = "Mostrar Regresión Lineal Simple")

    if not button_regresion_simple and not button_heatmap:
        if independent_variable == dependent_variable:
            st.warning("No se puede graficar la misma variable.")
            st.stop()
        else:
            #Realizamos un Gráfico de dispersión de las variables seleccionadas
            st.subheader(f"- Grafcico de dispersión de {independent_variable} y {dependent_variable}")
            disper = px.scatter(numeric_df, x=independent_variable, y=dependent_variable)
            st.plotly_chart(disper)

    if button_regresion_simple:
        if independent_variable == dependent_variable:
            st.warning("No se puede graficar la misma variable.")
            st.stop()
        else:
            var_indep = numeric_df[[independent_variable]]
            var_dep = numeric_df[[dependent_variable]]

            modelo = LinearRegression()
            modelo.fit(X=var_indep, y=var_dep)
            y_pred = modelo.predict(X=numeric_df[[independent_variable]])
            coef_Deter = modelo.score(var_indep, var_dep)
            coef_Correl = np.sqrt(coef_Deter)
            numeric_df.insert(29, f"predicción de {dependent_variable}", y_pred)

            #Creamos la tabla comparativa con las dos columnas
            tabla_comparativa = numeric_df[[dependent_variable, f"predicción de {dependent_variable}"]].copy()


            st.subheader("- Gráfico de dispersión Comparativo")
            disp_comp = go.Figure()

            # Scatter de los valores reales
            disp_comp.add_trace(go.Scatter(
                x=numeric_df[independent_variable],
                y=numeric_df[dependent_variable],
                mode='markers',
                name=f'{dependent_variable} Real',
                marker=dict(color='blue')
            ))

            # Scatter de los valores predichos
            disp_comp.add_trace(go.Scatter(
                x=numeric_df[independent_variable],
                y=numeric_df[f"predicción de {dependent_variable}"],
                mode='markers',
                name=f'{dependent_variable} Predicho',
                marker=dict(color='red')
            ))

            # Títulos y etiquetas
            disp_comp.update_layout(
                title=f"Gráfico de dispersión comparativo de {independent_variable} real VS {independent_variable} predicho",
                xaxis_title=independent_variable,
                yaxis_title=dependent_variable,
                legend=dict(x=1, y=0.5)
            )

            # Mostrar en el dashboard
            st.plotly_chart(disp_comp)
            st.write(f"Coeficiente de determinación: {coef_Deter} y coeficiente de correlación: {coef_Correl}")

            numeric_df.drop(columns=[f"predicción de {dependent_variable}"], inplace=True)


        #############################################################
#Vista 5
if View == "Regresión Lineal Multiple":
    st.title(f"{ciudad} - Regresión Lineal Multiple")

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

        # Calculamos la matriz de correlación
        corr_Factors1 = df_to_plot.corr()

        # Graficar el heatmap
        mapaCalor, ax = plt.subplots(figsize=(22, 22))

        # Mapa de calor con anotaciones
        sns.heatmap(
            corr_Factors1, 
            cmap="coolwarm",         # Usamos el mapa de colores "coolwarm"
            annot=True,              # Aseguramos que todos los valores estén anotados
            fmt=".2f",               # Formato de los valores (2 decimales)
            annot_kws={"size": annot_fontsize},  # Tamaño de las anotaciones
            linewidths=0.5,          # Espacio entre celdas
            square=True,             # Hace la matriz cuadrada
            cbar_kws={"shrink": 0.8},  # Ajusta el tamaño de la barra de color
            ax=ax                    # Asignamos el gráfico al eje
        )

        # Aseguramos que los ejes tengan los tamaños de fuente adecuados
        plt.xticks(rotation=90, fontsize=tick_fontsize)
        plt.yticks(rotation=0, fontsize=tick_fontsize)

        # Mostrar en el dashboard
        st.pyplot(mapaCalor)

    #####################################################################

    st.sidebar.subheader("Selección de variables para Regresión Lineal Multiple")
    #Widget 5: Select box
    #Menu desplegable de opciones de la variable dependiente seleccionada
    dependent_variable = st.sidebar.selectbox(label="Variable dependiente", options=numeric_cols)

    #Widget 6: Select box
    #Menu desplegable de opciones de la variable independiente seleccionada
    independent_option = [var for var in numeric_cols if var != dependent_variable]
    independent_variables = st.sidebar.multiselect(label="Variables independientes", options=independent_option, default=independent_option[:1])

    # Validar que al menos una variable independiente fue seleccionada
    if not independent_variables:
        st.warning("Por favor selecciona al menos una variable independiente.")
        st.stop()

    # Validar máximo de 5 variables
    if len(independent_variables) > 5:
        st.sidebar.warning("Solo puedes seleccionar hasta 5 variables.")
        st.stop()

    #Generamos un botón (button) en una barra lateral (sidebar) para mostrar el dataset
    button_regresion_multiple = st.sidebar.button(label = "Mostrar Regresión Lineal Multiple")

    if not button_regresion_multiple and not button_heatmap:
        # Convertimos a formato largo: cada variable independiente como una columna "variable"
        df_melted = pd.melt(
            numeric_df,
            id_vars=[dependent_variable],
            value_vars=independent_variables,
            var_name="Variable independiente",
            value_name="Valor"
        )

        # Gráfico de dispersión combinado
        st.subheader(f"- Gráfico de dispersión combinado contra {dependent_variable}")
        fig = px.scatter(
            df_melted,
            x="Valor",
            y=dependent_variable,
            color="Variable independiente",
            labels={"Valor": "Valor de variable independiente"},
            trendline="ols"
        )
        st.plotly_chart(fig)

    if button_regresion_multiple:
        var_indep = numeric_df[independent_variables]
        var_dep = numeric_df[[dependent_variable]]

        modelo = LinearRegression()
        modelo.fit(X=var_indep, y=var_dep)
        y_pred = modelo.predict(X=numeric_df[independent_variables])
        coef_Deter = modelo.score(var_indep, var_dep)
        coef_Correl = np.sqrt(coef_Deter)
        numeric_df.insert(29, f"predicción de {dependent_variable}", y_pred)

        #Creamos la tabla comparativa con las dos columnas
        tabla_comparativa = numeric_df[[dependent_variable, f"predicción de {dependent_variable}"]].copy()
        indi_var = independent_variables[0]

        st.subheader("- Gráfico de dispersión Comparativo")
        disp_comp = go.Figure()

        # Scatter de los valores reales
        disp_comp.add_trace(go.Scatter(
            x=numeric_df[indi_var],
            y=numeric_df[dependent_variable],
            mode='markers',
            name=f'{dependent_variable} Real',
            marker=dict(color='blue')
        ))

        # Scatter de los valores predichos
        disp_comp.add_trace(go.Scatter(
            x=numeric_df[indi_var],
            y=numeric_df[f"predicción de {dependent_variable}"],
            mode='markers',
            name=f'{dependent_variable} Predicho',
            marker=dict(color='red')
        ))

        # Títulos y etiquetas
        disp_comp.update_layout(
            title=f"Gráfico de dispersión comparativo de {dependent_variable} real VS {dependent_variable} predicho",
            xaxis_title=indi_var,
            yaxis_title=dependent_variable,
            legend=dict(x=1, y=0.5)
        )

        # Mostrar en el dashboard
        st.plotly_chart(disp_comp)

        st.write(f"Coeficiente de determinación: {coef_Deter} y coeficiente de correlación: {coef_Correl}")

        numeric_df.drop(columns=[f"predicción de {dependent_variable}"], inplace=True)

#########################################################################################################
#Vista 6
if View == "Regresión Logistica":
    st.sidebar.subheader("Selección de variables para Regresión Lineal")
    #Widget 5: Select box
    #Menu desplegable de opciones de la variable dependiente seleccionada
    dependent_variable = st.sidebar.selectbox(label="Variable dependiente", options=binary_cols)

    #Widget 6: Select box
    #Menu desplegable de opciones de la variable independiente seleccionada
    independent_option = [var for var in numeric_cols if var != dependent_variable]
    independent_variables = st.sidebar.multiselect(label="Variables independientes", options=independent_option, default=independent_option[:1])
    # Validar que al menos una variable independiente fue seleccionada
    if not independent_variables:
        st.warning("Por favor selecciona al menos una variable independiente.")
        st.stop()

    opcRegLog = st.sidebar.radio(
    "Selecciona una opción a analizar:",
    (
        "Predicción",
        "Matriz de Confusión",
        "Metricas del modelo"
    )
    )

    st.title(f"Regresión Logistica - {ciudad}")
    varsIndep = numeric_df[independent_variables]
    varDep = binary_df[[dependent_variable]]

    X= varsIndep 
    y= varDep

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state= None)

    escalar = StandardScaler()
    X_train = escalar.fit_transform(X_train)
    X_test = escalar.transform(X_test)

    modelo = LogisticRegression()
    modelo.fit(X_train, y_train)

    # Predecir
    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)

    if opcRegLog == "Predicción":
        pred_df = X_test.copy()
        pred_df = pd.DataFrame(pred_df, columns=independent_variables)
        pred_df["Probabilidad False"] = y_proba[:, 0]
        pred_df["Probabilidad True"] = y_proba[:, 1]
        pred_df["Predicción"] = y_pred

        st.subheader("Predicciones del modelo")
        st.dataframe(pred_df.head(10))  # Mostrar primeras 10 filas

    elif opcRegLog == "Matriz de Confusión":
        st.subheader("Matriz de confusión")
        matriz = confusion_matrix(y_test, y_pred)
        st.write("Matriz de Confusión:")
        st.write(pd.DataFrame(matriz, index=["Positivo", "Negativo"], columns=["Predicho 0", "Predicho 1"]))

    elif opcRegLog == "Metricas del modelo":
        st.subheader("Métricas del Modelo")


        precisionf = precision_score(y_test, y_pred, average="binary", pos_label="f")
        precisiont = precision_score(y_test, y_pred, average="binary", pos_label="t")
        exactitud = accuracy_score(y_test, y_pred)
        sensibilidadf = recall_score(y_test, y_pred, average="binary", pos_label="f")
        sensibilidadt = recall_score(y_test, y_pred, average="binary", pos_label="t")

        # Crear tabla con precisión y sensibilidad por clase, y exactitud en una sola columna
        metricas_df = pd.DataFrame({
            "f": [precisionf, sensibilidadf, exactitud],
            "t": [precisiont, sensibilidadt, exactitud]  # o "" si prefieres cadena vacía
        }, index=["Precisión", "Sensibilidad", "Exactitud"])

        # Reemplazar NaN por una cadena vacía o algún valor si lo prefieres
        metricas_df = metricas_df.fillna("")  # Reemplazar NaN por ""

        # Asegurarse de que los valores sean numéricos antes de aplicar el formato
        # Convertir la columna 'f' y 't' a números (float) si es posible
        metricas_df["f"] = pd.to_numeric(metricas_df["f"], errors='coerce')
        metricas_df["t"] = pd.to_numeric(metricas_df["t"], errors='coerce')

        # Mostrar tabla formateada
        st.table(metricas_df.style.format("{:.2f}"))

