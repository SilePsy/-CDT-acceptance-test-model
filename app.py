"""
Aplicación Streamlit para Predicción de Incumplimiento de Depósitos
Modelo: LightGBM (Sin Variables de Bajo IV)
Autor: Sistema de ML
Fecha: 2026
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
import warnings
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================
st.set_page_config(
    page_title="Predictor de Incumplimiento de Depósitos",
    page_icon="💰",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ============================================
# FUNCIONES AUXILIARES
# ============================================

@st.cache_resource
def cargar_modelos_y_transformadores():
    """Carga todos los modelos y transformadores desde archivos joblib"""
    
    # Directorio de archivos
    base_dir = Path(__file__).parent
    
    try:
        # Cargar modelo entrenado
        modelo = joblib.load(base_dir / 'archivos_modelo/lightgbm_modelo_filtrado.joblib')
        
        # Cargar listado de variables
        features_list = joblib.load(base_dir / 'archivos_modelo/features_list_lightgbm_filtrado.joblib')
        
        # Cargar transformadores
        encoders = joblib.load(base_dir / 'archivos_modelo/encoders_binarios.joblib')
        ohe = joblib.load(base_dir / 'archivos_modelo/one_hot_encoder.joblib')
        scaler_edad = joblib.load(base_dir / 'archivos_modelo/minmax_scaler_edad.joblib')
        scaler_saldo = joblib.load(base_dir / 'archivos_modelo/minmax_scaler_saldo.joblib')
        quintiles_generador = joblib.load(base_dir / 'archivos_modelo/quintiles_generador.joblib')
        ohe_quintiles = joblib.load(base_dir / 'archivos_modelo/one_hot_encoder_quintiles.joblib')
        config = joblib.load(base_dir / 'archivos_modelo/model_config.joblib')
        
        return {
            'modelo': modelo,
            'features_list': features_list,
            'encoders': encoders,
            'ohe': ohe,
            'scaler_edad': scaler_edad,
            'scaler_saldo': scaler_saldo,
            'quintiles_generador': quintiles_generador,
            'ohe_quintiles': ohe_quintiles,
            'config': config
        }
    except FileNotFoundError as e:
        st.error(f"❌ Error: No se encontraron algunos archivos requeridos. {str(e)}")
        st.stop()



def transformar_datos(df, transformadores):
    """Aplica todas las transformaciones al dataframe de entrada
    Genera exactamente las 18 variables que espera el modelo LightGBM filtrado
    """
    
    encoders = transformadores['encoders']
    ohe = transformadores['ohe']
    scaler_edad = transformadores['scaler_edad']
    scaler_saldo = transformadores['scaler_saldo']
    ohe_quintiles = transformadores['ohe_quintiles']
    features_list = transformadores['features_list']
    
    df_processed = df.copy()
    
    # 1. Aplicar LabelEncoder a variables binarias (tiene_vivienda, tiene_prestamo)
    variables_binarias = ['tiene_vivienda', 'tiene_prestamo']
    for var in variables_binarias:
        if var in df_processed.columns and var in encoders:
            df_processed[var] = encoders[var].transform(df_processed[var])
    
    # 2. Aplicar OneHotEncoder a variables categóricas (trabajo, estado_civil, educacion)
    variables_categoricas = ['trabajo', 'estado_civil', 'educacion']
    if all(var in df_processed.columns for var in variables_categoricas):
        ohe_encoded = ohe.transform(df_processed[variables_categoricas])
        ohe_feature_names = ohe.get_feature_names_out(variables_categoricas)
        df_ohe = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df_processed.index)
        df_processed = pd.concat([df_processed.drop(variables_categoricas, axis=1), df_ohe], axis=1)
    
    # 3. Escalar edad y saldo (reemplazar columnas originales)
    if 'edad' in df_processed.columns:
        df_processed['edad'] = scaler_edad.transform(df_processed[['edad']])
    
    if 'saldo' in df_processed.columns:
        df_processed['saldo'] = scaler_saldo.transform(df_processed[['saldo']])
    
    # 4. Crear variable derivada: norm_cant_productos = (tiene_vivienda + tiene_prestamo) / 2
    df_processed['norm_cant_productos'] = (df_processed['tiene_vivienda'] + df_processed['tiene_prestamo']) / 2
    
    # 5. Crear variable derivada: edad_saldo (producto de edad y saldo escalados)
    df_processed['edad_saldo'] = df_processed['edad'] * df_processed['saldo']
    
    # 6. Crear quintiles de edad y saldo usando pd.cut con bins: [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    # Ya que edad y saldo están escalados en [0,1], podemos usar bins uniformes
    bins_quintiles = [0, 0.2, 0.4, 0.6, 0.8, 1.0000001]  # 1.0000001 para incluir el 1.0
    labels_quintiles = ['Q1_Muy_Bajo', 'Q2_Bajo', 'Q3_Medio', 'Q4_Alto', 'Q5_Muy_Alto']
    
    quintil_edad = pd.cut(df_processed['edad'], bins=bins_quintiles, labels=labels_quintiles, include_lowest=True)
    quintil_saldo = pd.cut(df_processed['saldo'], bins=bins_quintiles, labels=labels_quintiles, include_lowest=True)
    
    # Renombrar etiquetas para que coincidan con el OneHotEncoder
    quintil_edad = quintil_edad.astype(str).str.replace('Q1_Muy_Bajo', 'Q1_Edad_Muy_Bajo')\
                                       .str.replace('Q2_Bajo', 'Q2_Edad_Bajo')\
                                       .str.replace('Q3_Medio', 'Q3_Edad_Medio')\
                                       .str.replace('Q4_Alto', 'Q4_Edad_Alto')\
                                       .str.replace('Q5_Muy_Alto', 'Q5_Edad_Muy_Alto')
    
    quintil_saldo = quintil_saldo.astype(str).str.replace('Q1_Muy_Bajo', 'Q1_Saldo_Muy_Bajo')\
                                        .str.replace('Q2_Bajo', 'Q2_Saldo_Bajo')\
                                        .str.replace('Q3_Medio', 'Q3_Saldo_Medio')\
                                        .str.replace('Q4_Alto', 'Q4_Saldo_Alto')\
                                        .str.replace('Q5_Muy_Alto', 'Q5_Saldo_Muy_Alto')
    
    # Crear dataframe con quintiles (nombres DEBEN coincidir con lo entrenado)
    quintiles_df = pd.DataFrame({
        'quintil_edad': quintil_edad,
        'quintil_saldo': quintil_saldo
    }, index=df_processed.index)
    
    # Aplicar OneHotEncoder a quintiles
    quintiles_encoded = ohe_quintiles.transform(quintiles_df)
    quintiles_feature_names = ohe_quintiles.get_feature_names_out(['quintil_edad', 'quintil_saldo'])
    df_quintiles = pd.DataFrame(quintiles_encoded, columns=quintiles_feature_names, index=df_processed.index)
    
    # Combinar todo
    df_processed = pd.concat([df_processed, df_quintiles], axis=1)
    
    # Eliminar columnas que no se usan en el modelo
    cols_a_mantener = set(features_list)
    cols_actuales = set(df_processed.columns)
    cols_a_eliminar = cols_actuales - cols_a_mantener
    
    if cols_a_eliminar:
        df_processed = df_processed.drop(columns=list(cols_a_eliminar))
    
    return df_processed


def decodificar_prediccion(prediccion_clase):
    """Decodifica la predicción binaria a texto descriptivo"""
    if prediccion_clase == 0:
        return ("✅ CUMPLIMIENTO", "Sí", "#2ecc71")
    else:
        return ("⚠️ INCUMPLIMIENTO", "No", "#e74c3c")


def hacer_prediccion(df_input, transformadores):
    """Realiza predicción con el modelo cargado"""
    
    try:
        # Transformar datos
        df_transformed = transformar_datos(df_input, transformadores)
        
        # Obtener solo las features necesarias
        features_list = transformadores['features_list']
        
        # Verificar que todas las features estén presentes
        features_faltantes = [f for f in features_list if f not in df_transformed.columns]
        if features_faltantes:
            st.error(f"❌ Variables faltantes después de la transformación: {features_faltantes}")
            return None
        
        # Preparar datos para predicción
        X_pred = df_transformed[features_list]
        
        # Hacer predicción
        modelo = transformadores['modelo']
        prediccion_proba = modelo.predict_proba(X_pred)[:, 1][0]
        prediccion_clase = modelo.predict(X_pred)[0]
        
        clase_texto, clase_binario, clase_color = decodificar_prediccion(int(prediccion_clase))
        
        return {
            'clase': int(prediccion_clase),
            'clase_texto': clase_texto,
            'clase_binario': clase_binario,
            'clase_color': clase_color,
            'probabilidad': prediccion_proba,
            'confianza': max(prediccion_proba, 1 - prediccion_proba)
        }
    
    except Exception as e:
        st.error(f"❌ Error en predicción: {str(e)}")
        return None


# ============================================
# INTERFAZ STREAMLIT
# ============================================

# Header
st.title("💰 Predictor de Depósitos Bancarios")
st.markdown("---")
st.markdown("""
Sistema inteligente de predicción basado en **LightGBM** para evaluar la probabilidad
de que un cliente contrate un depósito a plazo fijo.
""")

# Sidebar - Información del modelo
with st.sidebar:
    st.header("📊 Información del Modelo")
    
    st.subheader("Características")
    st.info("""
    - **Algoritmo**: LightGBM (Gradient Boosting)
    - **Variables**: 18 características seleccionadas
    - **Métrica**: ROC-AUC = 0.6680
    - **Validación**: 10-Fold Cross-Validation
    - **Objetivo**: Propensión a contratar depósitos
    """)
    
    st.subheader("Sobre el modelo")
    st.markdown("""
    Este modelo fue entrenado con datos históricos de clientes bancarios
    para predecir la probabilidad de que un cliente contrate un depósito
    a plazo fijo en base a sus características demográficas y bancarias.
    
    **Nota**: Las predicciones son estimaciones estadísticas basadas en
    patrones históricos y no deben considerarse como garantías.
    """)

# Cargar modelos
with st.spinner("⏳ Cargando modelo y transformadores..."):
    transformadores = cargar_modelos_y_transformadores()

st.success("✅ Modelo cargado exitosamente")

# Tabs
tab1, tab2, tab3 = st.tabs(["🎯 Predicción Individual", "📤 Predicción en Lote", "📋 Guía de Uso"])

# ============================================
# TAB 1: PREDICCIÓN INDIVIDUAL
# ============================================
with tab1:
    st.header("Predicción Individual")
    st.markdown("Ingresa los datos del cliente para predecir la propensión a contratar depósitos")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Datos Personales")
        
        trabajo = st.selectbox(
            "👨‍💼 Ocupación",
            options=["administrador", "autonomo", "desempleado", "empleada_hogar", 
                    "empresario", "estudiante", "gestion", "jubilado", "obrero", 
                    "servicios", "tecnico", "desconocido"],
            index=0,
            help="Tipo de ocupación del cliente"
        )
        
        estado_civil = st.selectbox(
            "💍 Estado Civil",
            options=["casado", "divorciado", "soltero"],
            index=2,
            help="Estado civil actual"
        )
        
        edad = st.number_input(
            "📅 Edad",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
            help="Edad del cliente en años"
        )
    
    with col2:
        st.subheader("🏦 Datos Bancarios")
        
        educacion = st.selectbox(
            "🎓 Educación",
            options=["primaria", "secundaria", "terciaria", "desconocida"],
            index=1,
            help="Nivel educativo"
        )
        
        tiene_vivienda = st.selectbox(
            "🏠 Tiene Vivienda",
            options=["no", "si"],
            index=0,
            help="¿Tiene vivienda?"
        )
        
        tiene_prestamo = st.selectbox(
            "💳 Tiene Préstamo",
            options=["no", "si"],
            index=0,
            help="¿Tiene préstamo activo?"
        )
    
    st.markdown("---")
    
    col3, col4 = st.columns(2)
    
    with col3:
        saldo = st.number_input(
            "💵 Saldo (euros)",
            min_value=0,
            max_value=300000,
            value=1500,
            step=100,
            help="Saldo en la cuenta (en euros)"
        )
    
    # Botón de predicción centrado
    st.markdown("---")
    col_btn_left, col_btn_center, col_btn_right = st.columns([1, 2, 1])
    
    with col_btn_center:
        predict_button = st.button(
            "🔮 REALIZAR PREDICCIÓN",
            use_container_width=True,
            type="primary"
        )
    
    if predict_button:
        # Crear DataFrame con los datos ingresados
        df_entrada = pd.DataFrame({
            'trabajo': [trabajo],
            'estado_civil': [estado_civil],
            'educacion': [educacion],
            'edad': [edad],
            'tiene_vivienda': [tiene_vivienda],
            'tiene_prestamo': [tiene_prestamo],
            'saldo': [saldo]
        })
        
        # Hacer predicción
        with st.spinner("🔄 Procesando predicción..."):
            resultado = hacer_prediccion(df_entrada, transformadores)
        
        if resultado:
            st.markdown("---")
            st.header("📌 Resultado de la Predicción")
            
            # Mostrar resultado principal con colores
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.markdown(f"""
                <div style="background-color: {resultado['clase_color']}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h3 style="color: white; margin: 0;">Predicción</h3>
                    <h2 style="color: white; margin: 5px 0;">{resultado['clase_texto']}</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.metric(
                    "Probabilidad de Depósito",
                    f"{resultado['probabilidad']:.2%}",
                    delta=f"Sin depósito: {(1-resultado['probabilidad']):.2%}"
                )
            
            with col_res3:
                st.metric(
                    "Confianza del Modelo",
                    f"{resultado['confianza']:.2%}"
                )
            
            # Gráfico de probabilidad
            st.markdown("---")
            st.subheader("📊 Distribución de Probabilidades")
            
            prob_cumplimiento = 1 - resultado['probabilidad']
            prob_incumplimiento = resultado['probabilidad']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colores = ['#2ecc71', '#e74c3c']
            valores = [prob_cumplimiento, prob_incumplimiento]
            labels = [
                f'Cumplimiento (Sí)\n{prob_cumplimiento:.1%}',
                f'Incumplimiento (No)\n{prob_incumplimiento:.1%}'
            ]
            
            wedges, texts, autotexts = ax.pie(
                valores,
                labels=labels,
                colors=colores,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            
            ax.set_title('Probabilidades Predichas', fontsize=14, weight='bold', pad=20)
            
            st.pyplot(fig)
            
            # Resumen de datos ingresados
            st.markdown("---")
            st.subheader("📋 Datos Ingresados")
            
            datos_resumen = pd.DataFrame({
                'Variable': [
                    'Ocupación',
                    'Estado Civil',
                    'Educación',
                    'Edad',
                    'Tiene Vivienda',
                    'Tiene Préstamo',
                    'Saldo'
                ],
                'Valor': [
                    trabajo,
                    estado_civil,
                    educacion,
                    f"{edad} años",
                    tiene_vivienda,
                    tiene_prestamo,
                    f"€{saldo:,}"
                ]
            })
            
            st.dataframe(datos_resumen, use_container_width=True, hide_index=True)
            
            # Gráfico de probabilidad
            st.markdown("---")
            st.subheader("📊 Distribución de Probabilidades")
            
            prob_deposito = resultado['probabilidad']
            prob_no_deposito = 1 - resultado['probabilidad']
            
            fig, ax = plt.subplots(figsize=(8, 6))
            
            colores = ['#e74c3c', '#2ecc71']
            valores = [prob_no_deposito, prob_deposito]
            labels = [
                f'Sin Depósito\n{prob_no_deposito:.1%}',
                f'Con Depósito\n{prob_deposito:.1%}'
            ]
            
            wedges, texts, autotexts = ax.pie(
                valores,
                labels=labels,
                colors=colores,
                autopct='%1.1f%%',
                startangle=90,
                textprops={'fontsize': 12, 'weight': 'bold'}
            )
            
            ax.set_title('Probabilidades Predichas', fontsize=14, weight='bold', pad=20)
            
            st.pyplot(fig)
            
            # Recomendación
            st.markdown("---")
            st.subheader("💡 Recomendación")
            
            if resultado['clase'] == 1:
                st.success(f"""
                ✅ **ALTA PROPENSIÓN - CONTRATARÁ DEPÓSITO (SÍ)**
                
                El cliente tiene un perfil con alta probabilidad de contratar depósito.
                Probabilidad: **{prob_deposito:.1%}**
                
                Se recomienda realizar seguimiento y oferta personalizada.
                """)
            else:
                st.warning(f"""
                ⚠️ **BAJA PROPENSIÓN - NO CONTRATARÁ DEPÓSITO (NO)**
                
                El cliente no muestra propensión a contratar depósito.
                Probabilidad: **{prob_deposito:.1%}**
                
                Se recomienda trabajo en relación con el cliente antes de ofrecer depósitos.
                """)


# ============================================
# TAB 2: PREDICCIÓN EN LOTE
# ============================================
with tab2:
    st.header("Predicción en Lote")
    
    st.markdown("""
    Carga un archivo CSV con múltiples registros para obtener predicciones en lote.
    El archivo debe contener las siguientes columnas:
    """)
    
    st.code("""trabajo, estado_civil, educacion, edad, tiene_vivienda, tiene_prestamo, saldo""")
    
    uploaded_file = st.file_uploader("Selecciona archivo CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df_carga = pd.read_csv(uploaded_file)
            
            st.write(f"✓ Archivo cargado: {uploaded_file.name}")
            st.write(f"✓ Registros encontrados: {len(df_carga)}")
            
            if st.button("🔮 Realizar Predicción en Lote", use_container_width=True, type="primary"):
                
                with st.spinner("⏳ Procesando predicciones..."):
                    resultados = []
                    
                    for idx, row in df_carga.iterrows():
                        df_una_fila = df_carga.iloc[[idx]]
                        resultado = hacer_prediccion(df_una_fila, transformadores)
                        
                        if resultado:
                            resultados.append({
                                'Registro': idx + 1,
                                'Predicción': resultado['clase_texto'],
                                'Respuesta': resultado['clase_binario'],
                                'Prob. Incumplimiento': f"{resultado['probabilidad']:.4f}",
                                'Confianza': f"{resultado['confianza']:.4f}"
                            })
                    
                    if resultados:
                        df_resultados = pd.DataFrame(resultados)
                        
                        st.markdown("---")
                        st.header("📊 Resultados")
                        st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                        
                        # Estadísticas
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        n_incumplimiento = (df_resultados['Predicción'].str.contains('INCUMPLIMIENTO')).sum()
                        n_cumplimiento = (df_resultados['Predicción'].str.contains('CUMPLIMIENTO')).sum()
                        
                        with col_stat1:
                            st.metric("Total Procesados", len(df_resultados))
                        
                        with col_stat2:
                            st.metric("Riesgo (No) ⚠️", n_incumplimiento)
                        
                        with col_stat3:
                            st.metric("Seguro (Sí) ✅", n_cumplimiento)
                        
                        # Descargar resultados
                        csv = df_resultados.to_csv(index=False)
                        st.download_button(
                            label="📥 Descargar Resultados CSV",
                            data=csv,
                            file_name="predicciones_lote.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
        
        except Exception as e:
            st.error(f"❌ Error procesando archivo: {str(e)}")


# ============================================
# TAB 3: GUÍA DE USO
# ============================================
with tab3:
    st.header("📋 Guía de Uso")
    
    st.subheader("1️⃣ Descripción del Modelo")
    st.markdown("""
    Este modelo utiliza **LightGBM**, un algoritmo de Gradient Boosting altamente eficiente,
    entrenado con 29,847 registros de clientes bancarios para predecir la probabilidad de 
    contratar un depósito a plazo fijo.
    
    **Características principales:**
    - 18 variables seleccionadas tras análisis de Information Value (IV)
    - Variables: edad, saldo, ocupación, estado civil, educación, vivienda, préstamo
    - Validación cruzada de 10 folds
    - Métrica de evaluación: ROC-AUC (0.6680)
    - Procesamiento automático de datos (encoding, scaling, feature engineering)
    """)
    
    st.subheader("2️⃣ Interpretación de Resultados")
    
    col_interp1, col_interp2 = st.columns(2)
    
    with col_interp1:
        st.success("""
        **✅ ALTA PROPENSIÓN - DEPÓSITO (SÍ)**
        
        Respuesta: **Sí**
        
        - Probabilidad > 50%
        - Cliente muy propenso a contratar
        - Recomendación: Ofrecer depósitos
        """)
    
    with col_interp2:
        st.warning("""
        **⚠️ BAJA PROPENSIÓN - NO DEPÓSITO (NO)**
        
        Respuesta: **No**
        
        - Probabilidad ≤ 50%
        - Cliente poco propenso a contratar
        - Recomendación: Trabajar relación primero
        """)
    
    st.markdown("---")
    
    st.subheader("3️⃣ Variables de Entrada")
    
    variables_info = pd.DataFrame({
        'Variable': [
            'Ocupación',
            'Estado Civil',
            'Educación',
            'Edad',
            'Tiene Vivienda',
            'Tiene Préstamo',
            'Saldo'
        ],
        'Tipo': [
            'Categórica',
            'Categórica',
            'Categórica',
            'Numérica',
            'Binaria',
            'Binaria',
            'Numérica'
        ],
        'Ejemplo/Rango': [
            'administrador, empresario, jubilado, etc.',
            'casado, divorciado, soltero',
            'primaria, secundaria, terciaria',
            '18-100 años',
            'sí, no',
            'sí, no',
            '0-300,000 euros'
        ]
    })
    
    st.dataframe(variables_info, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("4️⃣ Transformaciones de Datos")
    st.markdown("""
    El modelo aplica automáticamente las siguientes transformaciones:
    
    1. **Encoding Binario**: Conversión de variables Sí/No a 0/1
    2. **One-Hot Encoding**: Expansión de categorías (ocupación, estado civil, educación)
    3. **Scaling (MinMax)**: Normalización de edad y saldo a rango [0,1]
    4. **Feature Engineering**: Creación de variables derivadas:
       - norm_cant_productos = (tiene_vivienda + tiene_prestamo) / 2
       - edad_saldo = edad_normalizada * saldo_normalizado
    5. **Quintiles**: Discretización de edad y saldo en 5 categorías
    6. **Selección de Features**: Solo las 18 variables más relevantes
    """)
    
    st.markdown("---")
    
    st.subheader("5️⃣ Limitaciones")
    st.warning("""
    - Predicciones basadas en patrones históricos
    - No constituyen garantías ni recomendaciones finales
    - Requiere validación manual para decisiones críticas
    - Mejor desempeño con datos similares a datos de entrenamiento
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <small>Modelo de Predicción de Incumplimiento de Depósitos | LightGBM | 2026</small>
</div>
""", unsafe_allow_html=True)

