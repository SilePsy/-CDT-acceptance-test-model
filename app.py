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
        modelo = joblib.load(base_dir / 'lightgbm_modelo_filtrado.joblib')
        
        # Cargar listado de variables
        features_list = joblib.load(base_dir / 'features_list_lightgbm_filtrado.joblib')
        
        # Cargar transformadores
        encoders = joblib.load(base_dir / 'encoders_binarios.joblib')
        ohe = joblib.load(base_dir / 'one_hot_encoder.joblib')
        scaler_edad = joblib.load(base_dir / 'minmax_scaler_edad.joblib')
        scaler_saldo = joblib.load(base_dir / 'minmax_scaler_saldo.joblib')
        quintiles_generador = joblib.load(base_dir / 'quintiles_generador.joblib')
        ohe_quintiles = joblib.load(base_dir / 'one_hot_encoder_quintiles.joblib')
        config = joblib.load(base_dir / 'model_config.joblib')
        
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
    """Aplica todas las transformaciones al dataframe de entrada"""
    
    encoders = transformadores['encoders']
    ohe = transformadores['ohe']
    scaler_edad = transformadores['scaler_edad']
    scaler_saldo = transformadores['scaler_saldo']
    quintiles_generador = transformadores['quintiles_generador']
    ohe_quintiles = transformadores['ohe_quintiles']
    
    df_processed = df.copy()
    
    # 1. Aplicar LabelEncoder a variables binarias
    variables_binarias = ['profesi', 'estcivil']
    for var in variables_binarias:
        if var in df_processed.columns and var in encoders:
            df_processed[var] = encoders[var].transform(df_processed[var])
    
    # 2. Aplicar OneHotEncoder
    variables_categoricas = ['canalprim', 'rentero', 'nivelacd']
    if all(var in df_processed.columns for var in variables_categoricas):
        ohe_encoded = ohe.transform(df_processed[variables_categoricas])
        ohe_feature_names = ohe.get_feature_names_out(variables_categoricas)
        df_ohe = pd.DataFrame(ohe_encoded, columns=ohe_feature_names, index=df_processed.index)
        df_processed = pd.concat([df_processed.drop(variables_categoricas, axis=1), df_ohe], axis=1)
    
    # 3. Escalar edad y saldo
    if 'edad' in df_processed.columns:
        df_processed['edad_escalada'] = scaler_edad.transform(df_processed[['edad']])
    
    if 'saldo' in df_processed.columns:
        df_processed['saldo_escalado'] = scaler_saldo.transform(df_processed[['saldo']])
    
    # 4. Crear quintiles
    variables_quintiles = ['cant_productos', 'meses_cliente']
    if all(var in df_processed.columns for var in variables_quintiles):
        quintiles = pd.cut(df_processed['cant_productos'], bins=5, labels=False, duplicates='drop')
        quintiles_meses = pd.cut(df_processed['meses_cliente'], bins=5, labels=False, duplicates='drop')
        
        # OneHotEncode los quintiles
        quintiles_combined = pd.DataFrame({
            'cant_productos_quintil': quintiles,
            'meses_cliente_quintil': quintiles_meses
        })
        
        if len(quintiles_combined.columns) > 0:
            quintiles_encoded = ohe_quintiles.transform(quintiles_combined)
            quintiles_ohe_names = ohe_quintiles.get_feature_names_out(['cant_productos_quintil', 'meses_cliente_quintil'])
            df_quintiles = pd.DataFrame(quintiles_encoded, columns=quintiles_ohe_names, index=df_processed.index)
            df_processed = pd.concat([df_processed, df_quintiles], axis=1)
    
    # 5. Crear variables derivadas
    if 'cant_productos' in df_processed.columns:
        df_processed['norm_cant_productos'] = df_processed['cant_productos'] / df_processed['cant_productos'].max()
    
    if 'edad_escalada' in df_processed.columns and 'saldo_escalado' in df_processed.columns:
        df_processed['edad_saldo_interaction'] = df_processed['edad_escalada'] * df_processed['saldo_escalado']
    
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
st.title("💰 Predictor de Incumplimiento de Depósitos")
st.markdown("---")
st.markdown("""
Sistema inteligente de predicción basado en **LightGBM** para evaluar el riesgo 
de incumplimiento de depósitos bancarios.
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
    - **Clase**: Incumplimiento (Binario)
    """)
    
    st.subheader("Sobre el modelo")
    st.markdown("""
    Este modelo fue entrenado con datos históricos de clientes bancarios
    para predecir la probabilidad de incumplimiento en depósitos.
    
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
    st.markdown("Completa los datos del cliente y haz clic en el botón de predicción")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Datos Personales")
        
        profesi = st.selectbox(
            "👨‍💼 Profesión",
            options=["Empleado", "Empresa"],
            index=0,
            help="Tipo de ocupación del cliente"
        )
        
        estcivil = st.selectbox(
            "💍 Estado Civil",
            options=["Soltero", "Casado", "Divorciado", "Viudo"],
            index=0,
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
        
        canalprim = st.selectbox(
            "📱 Canal Principal",
            options=["Sucursal", "Internet", "Teléfono", "Móvil"],
            index=0,
            help="Canal principal de contacto"
        )
        
        rentero = st.selectbox(
            "🏠 Rentero",
            options=["No", "Sí"],
            index=0,
            help="¿Es rentero?"
        )
        
        nivelacd = st.selectbox(
            "🎖️ Nivel de Acceso",
            options=["Básico", "Premium", "Gold"],
            index=0,
            help="Nivel de acceso a servicios"
        )
    
    st.markdown("---")
    
    col3, col4, col5 = st.columns(3)
    
    with col3:
        cant_productos = st.number_input(
            "📦 Cantidad de Productos",
            min_value=1,
            max_value=20,
            value=3,
            step=1,
            help="Número de productos bancarios"
        )
    
    with col4:
        saldo = st.number_input(
            "💵 Saldo (miles)",
            min_value=0,
            max_value=1000000,
            value=50000,
            step=1000,
            help="Saldo en la cuenta (en miles)"
        )
    
    with col5:
        meses_cliente = st.number_input(
            "⏱️ Meses Cliente",
            min_value=1,
            max_value=600,
            value=24,
            step=1,
            help="Meses como cliente del banco"
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
            'profesi': [profesi],
            'estcivil': [estcivil],
            'edad': [edad],
            'canalprim': [canalprim],
            'rentero': [rentero],
            'nivelacd': [nivelacd],
            'cant_productos': [cant_productos],
            'saldo': [saldo],
            'meses_cliente': [meses_cliente]
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
                    "Probabilidad de Incumplimiento",
                    f"{resultado['probabilidad']:.2%}",
                    delta=f"Cumplimiento: {(1-resultado['probabilidad']):.2%}"
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
                    'Profesión',
                    'Estado Civil',
                    'Edad',
                    'Canal',
                    'Rentero',
                    'Nivel Acceso',
                    'Productos',
                    'Saldo',
                    'Meses Cliente'
                ],
                'Valor': [
                    profesi,
                    estcivil,
                    f"{edad} años",
                    canalprim,
                    rentero,
                    nivelacd,
                    cant_productos,
                    f"${saldo:,}",
                    f"{meses_cliente} meses"
                ]
            })
            
            st.dataframe(datos_resumen, use_container_width=True, hide_index=True)
            
            # Recomendación
            st.markdown("---")
            st.subheader("💡 Recomendación")
            
            if resultado['clase'] == 0:
                st.success(f"""
                ✅ **BAJO RIESGO - CUMPLIMIENTO (SÍ)**
                
                El cliente tiene un perfil de bajo riesgo según el modelo.
                Probabilidad de cumplimiento: **{prob_cumplimiento:.1%}**
                
                Se recomienda autorizar el depósito.
                """)
            else:
                st.error(f"""
                ⚠️ **ALTO RIESGO - INCUMPLIMIENTO (NO)**
                
                El cliente presenta señales de riesgo elevado.
                Probabilidad de incumplimiento: **{prob_incumplimiento:.1%}**
                
                Se recomienda revisar caso manualmente o aplicar condiciones especiales.
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
    
    st.code("""profesi, estcivil, edad, canalprim, rentero, nivelacd, 
cant_productos, saldo, meses_cliente""")
    
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
    entrenado con 29,847 registros de clientes para predecir el incumplimiento de depósitos.
    
    **Características principales:**
    - 18 variables seleccionadas tras análisis de importancia
    - Validación cruzada de 10 folds
    - Métrica de evaluación: ROC-AUC (0.6680)
    - Procesamiento automático de datos (encoding, scaling, feature engineering)
    """)
    
    st.subheader("2️⃣ Interpretación de Resultados")
    
    col_interp1, col_interp2 = st.columns(2)
    
    with col_interp1:
        st.success("""
        **✅ BAJO RIESGO - CUMPLIMIENTO**
        
        Respuesta: **Sí**
        
        - Probabilidad < 50%
        - Cliente es más probable que cumpla
        - Recomendación: Autorizar depósito
        """)
    
    with col_interp2:
        st.error("""
        **⚠️ ALTO RIESGO - INCUMPLIMIENTO**
        
        Respuesta: **No**
        
        - Probabilidad ≥ 50%
        - Cliente es más probable que incumpla
        - Recomendación: Revisar caso manualmente
        """)
    
    st.markdown("---")
    
    st.subheader("3️⃣ Variables de Entrada")
    
    variables_info = pd.DataFrame({
        'Variable': [
            'Profesión',
            'Estado Civil',
            'Edad',
            'Canal Principal',
            'Rentero',
            'Nivel de Acceso',
            'Cantidad de Productos',
            'Saldo',
            'Meses como Cliente'
        ],
        'Tipo': [
            'Categórica',
            'Categórica',
            'Numérica',
            'Categórica',
            'Categórica',
            'Categórica',
            'Numérica',
            'Numérica',
            'Numérica'
        ],
        'Opciones/Rango': [
            'Empleado, Empresa',
            'Soltero, Casado, Divorciado, Viudo',
            '18-100 años',
            'Sucursal, Internet, Teléfono, Móvil',
            'Sí, No',
            'Básico, Premium, Gold',
            '1-20',
            '0-1,000,000',
            '1-600'
        ]
    })
    
    st.dataframe(variables_info, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("4️⃣ Transformaciones de Datos")
    st.markdown("""
    El modelo aplica automáticamente las siguientes transformaciones:
    
    1. **Encoding Binario**: Conversión de variables categóricas a binarias
    2. **One-Hot Encoding**: Expansión de variables categóricas
    3. **Scaling (MinMax)**: Normalización de edad y saldo a rango [0,1]
    4. **Feature Engineering**: Creación de quintiles y variables derivadas
    5. **Selección de Features**: Solo 18 variables más relevantes
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
