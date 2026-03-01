import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path
from io import BytesIO

# Configuración de la página
st.set_page_config(
    page_title="Predicción de Aceptación de CDT",
    layout="wide"
)

st.title("Modelo de Predicción de Aceptación de CDT")
st.markdown("**Modelo: Random Forest** | **Umbral de decisión: 0.13**")
st.markdown("---")

# Directorio de archivos modelo
MODEL_DIR = Path(__file__).parent / "archivos_modelo"

# Constantes del modelo
UMBRAL_DECISION = 0.13

@st.cache_resource
def load_models():
    """Carga todos los archivos joblib necesarios para el modelo Random Forest."""
    try:
        encoders_binarios = joblib.load(MODEL_DIR / "encoders_binarios.joblib")
        minmax_scaler_edad = joblib.load(MODEL_DIR / "minmax_scaler_edad.joblib")
        minmax_scaler_saldo = joblib.load(MODEL_DIR / "minmax_scaler_saldo.joblib")
        one_hot_encoder = joblib.load(MODEL_DIR / "one_hot_encoder.joblib")
        quintiles_generador = joblib.load(MODEL_DIR / "quintiles_generador.joblib")
        one_hot_encoder_quintiles = joblib.load(MODEL_DIR / "one_hot_encoder_quintiles.joblib")
        model = joblib.load(MODEL_DIR / "modelo6_random_forest.joblib")
        
        return {
            'encoders_binarios': encoders_binarios,
            'minmax_scaler_edad': minmax_scaler_edad,
            'minmax_scaler_saldo': minmax_scaler_saldo,
            'one_hot_encoder': one_hot_encoder,
            'quintiles_generador': quintiles_generador,
            'one_hot_encoder_quintiles': one_hot_encoder_quintiles,
            'model': model
        }
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None

def preparar_datos(df_input, models):
    """
    Prepara los datos de entrada replicando el proceso del notebook de modelación.
    
    Pasos:
    1. Codificar variables binarias con LabelEncoder
    2. Normalizar edad y saldo con MinMaxScaler
    3. Aplicar OneHotEncoder a variables categóricas
    4. Crear variables derivadas
    5. Crear quintiles de edad
    6. Aplicar OneHotEncoder a quintiles
    
    Args:
        df_input: DataFrame con columnas originales
        models: Diccionario con los modelos cargados
    
    Returns:
        DataFrame procesado listo para predicción
    """
    df = df_input.copy()
    
    # PASO 1: Aplicar LabelEncoders binarios
    variables_binarias = ['incumplimiento', 'prestamo_vivienda', 'prestamo_consumo']
    for variable in variables_binarias:
        if variable in df.columns:
            encoder = models['encoders_binarios'][variable]
            df[variable] = encoder.transform(df[variable])
    
    # PASO 2: Normalizar edad con MinMaxScaler
    df['edad'] = models['minmax_scaler_edad'].transform(df[['edad']])
    
    # PASO 3: Normalizar saldo con MinMaxScaler
    df['saldo'] = models['minmax_scaler_saldo'].transform(df[['saldo']])
    
    # PASO 4: Crear variables derivadas
    df['norm_cant_productos'] = (df['prestamo_vivienda'] + df['prestamo_consumo']) / 2
    df['edad_saldo'] = (df['edad'] + df['saldo']) / 2
    
    # PASO 5: Aplicar One Hot Encoder a variables categóricas
    variables_categoricas = ['empleo', 'estado_civil', 'nivel_educativo']
    df_ohe = models['one_hot_encoder'].transform(df[variables_categoricas])
    feature_names_ohe = models['one_hot_encoder'].get_feature_names_out(variables_categoricas)
    df_ohe_encoded = pd.DataFrame(df_ohe, columns=feature_names_ohe, index=df.index)
    
    # Combinar con datos no categóricos
    df = pd.concat([df.drop(columns=variables_categoricas), df_ohe_encoded], axis=1)
    
    # PASO 6: Crear quintiles de edad
    quintiles_info = models['quintiles_generador']
    
    # Aplicar bins de quintiles guardados
    df['quintil_edad'] = pd.cut(
        df['edad'],
        bins=quintiles_info['edad_bins'],
        labels=quintiles_info['edad_labels'],
        include_lowest=True
    )
    
    # PASO 7: Aplicar One Hot Encoder a quintiles
    variables_quintiles = ['quintil_edad']
    quintiles_ohe = models['one_hot_encoder_quintiles'].transform(df[variables_quintiles])
    quintiles_feature_names = models['one_hot_encoder_quintiles'].get_feature_names_out(variables_quintiles)
    df_quintiles_encoded = pd.DataFrame(quintiles_ohe, columns=quintiles_feature_names, index=df.index)
    
    # Combinar datos finales
    df = pd.concat([df.drop(columns=variables_quintiles), df_quintiles_encoded], axis=1)
    
    # PASO 8: Obtener las características que el modelo espera
    expected_features = models['model'].feature_names_in_
    
    # Asegurar que todas las características esperadas existan
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0
    
    # Seleccionar solo las características necesarias en el orden correcto
    df_final = df[expected_features]
    
    return df_final

# Cargar modelos
models = load_models()
if models is None:
    st.stop()

# Crear tabs para las dos funcionalidades
tab1, tab2 = st.tabs(["Predicción Individual", "Predicción por Lote (Excel)"])

# ============================================================================
# TAB 1: PREDICCIÓN INDIVIDUAL
# ============================================================================
with tab1:
    # Información sobre el modelo
    with st.expander("Información del Modelo"):
        st.markdown("""
        Este modelo predice la probabilidad de que un cliente acepte un **CDT (Certificado de Depósito a Término)**.
        
        **Modelo utilizado:** Random Forest
        **Umbral de decisión:** 0.13 (optimizado para maximizar F1 Score)
        
        **Variables de entrada requeridas:**
        - **Edad:** Edad del cliente
        - **Saldo:** Saldo actual en la cuenta (EUR)
        - **Empleo:** Tipo de empleo del cliente
        - **Estado Civil:** Estado civil del cliente
        - **Nivel Educativo:** Nivel de educación del cliente
        - **Incumplimiento:** Si tiene historial de incumplimiento
        - **Préstamo Vivienda:** Si tiene préstamo de vivienda activo
        - **Préstamo Consumo:** Si tiene préstamo de consumo activo
        """)

    # Crear formulario
    with st.form("prediction_form"):
        st.header("Ingrese los Datos del Cliente")
        
        # Dividir en columnas
        col1, col2 = st.columns(2)
        
        with col1:
            edad = st.slider("Edad", min_value=18, max_value=100, value=40, step=1)
            saldo = st.slider("Saldo (EUR)", min_value=0, max_value=100000, value=1000, step=100)
            
            estado_civil = st.selectbox(
                "Estado Civil",
                options=['casado', 'divorciado', 'soltero']
            )
            
            nivel_educativo = st.selectbox(
                "Nivel Educativo",
                options=['basica', 'media', 'superior', 'desconocida']
            )
        
        with col2:
            empleo = st.selectbox(
                "Empleo",
                options=['administrador', 'obrero', 'empresario', 'empleada_hogar', 
                         'gestion', 'jubilado', 'autonomo', 'servicios', 
                         'estudiante', 'tecnico', 'desempleado', 'desconocido']
            )
            incumplimiento = st.selectbox("Tiene Incumplimiento", options=['no', 'si'])
            prestamo_vivienda = st.selectbox("Tiene Préstamo Vivienda", options=['no', 'si'])
            prestamo_consumo = st.selectbox("Tiene Préstamo Consumo", options=['no', 'si'])
        
        # Botón para enviar
        submit_button = st.form_submit_button("Hacer Predicción", use_container_width=True)

    # Procesar predicción individual
    if submit_button:
        st.markdown("---")
        st.header("Resultado de la Predicción")
        
        try:
            # Crear dataframe con los datos de entrada
            data = {
                'edad': [edad],
                'saldo': [saldo],
                'empleo': [empleo],
                'estado_civil': [estado_civil],
                'nivel_educativo': [nivel_educativo],
                'incumplimiento': [incumplimiento],
                'prestamo_vivienda': [prestamo_vivienda],
                'prestamo_consumo': [prestamo_consumo]
            }
            
            df_input = pd.DataFrame(data)
            
            # Guardar datos originales para mostrar
            datos_originales = df_input.copy()
            
            # Preparar datos para el modelo
            df_model_input = preparar_datos(df_input, models)
            
            # Hacer predicción
            probabilidad = models['model'].predict_proba(df_model_input)[0][1]
            prediccion = 1 if probabilidad >= UMBRAL_DECISION else 0
            
            # Mostrar resultados
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Probabilidad de Aceptación de CDT",
                    value=f"{probabilidad*100:.2f}%"
                )
            
            with col2:
                if prediccion == 1:
                    st.success(f"Predicción: **SÍ** aceptará el CDT")
                else:
                    st.info(f"Predicción: **NO** aceptará el CDT")
            
            # Tabla de resumen
            st.subheader("Resumen de Datos Ingresados")
            
            resumen_data = {
                'Variable': ['Edad', 'Saldo', 'Empleo', 'Estado Civil', 'Nivel Educativo',
                            'Incumplimiento', 'Préstamo Vivienda', 'Préstamo Consumo'],
                'Valor': [edad, f"€{saldo:,.2f}", empleo, estado_civil, nivel_educativo,
                         incumplimiento, prestamo_vivienda, prestamo_consumo]
            }
            
            df_resumen = pd.DataFrame(resumen_data)
            st.dataframe(df_resumen, use_container_width=True, hide_index=True)
            
            # Información adicional
            with st.expander("🔍 Detalles Técnicos"):
                st.markdown(f"""
                **Procesamiento aplicado:**
                - Variables binarias codificadas: ✓
                - Edad normalizada (MinMaxScaler): {df_model_input['edad'].values[0]:.4f}
                - Saldo normalizado (MinMaxScaler): {df_model_input['saldo'].values[0]:.4f}
                - Variables derivadas creadas: ✓ (norm_cant_productos, edad_saldo)
                - One Hot Encoding categóricas: ✓
                - Quintiles de edad generados: ✓
                - Características del modelo: {len(df_model_input.columns)}
                - Umbral de decisión: {UMBRAL_DECISION}
                - Probabilidad predicha: {probabilidad:.6f}
                """)
            
        except Exception as e:
            st.error(f"Error durante la predicción: {str(e)}")
            st.error("Por favor, revise los datos ingresados e intente nuevamente.")

# ============================================================================
# TAB 2: PREDICCIÓN POR LOTE (EXCEL)
# ============================================================================
with tab2:
    st.header("Predicción Masiva desde Archivo Excel")
    
    with st.expander("Instrucciones y Formato del Archivo"):
        st.markdown("""
        ### Formato requerido del archivo Excel
        
        El archivo Excel debe contener las siguientes columnas con los valores especificados:
        
        | Columna | Tipo | Valores Permitidos |
        |---------|------|-------------------|
        | `edad` | Numérico | 18-100 |
        | `saldo` | Numérico | Cualquier valor |
        | `empleo` | Texto | administrador, obrero, empresario, empleada_hogar, gestion, jubilado, autonomo, servicios, estudiante, tecnico, desempleado, desconocido |
        | `estado_civil` | Texto | casado, divorciado, soltero |
        | `nivel_educativo` | Texto | basica, media, superior, desconocida |
        | `incumplimiento` | Texto | si, no |
        | `prestamo_vivienda` | Texto | si, no |
        | `prestamo_consumo` | Texto | si, no |
        
        Descargar plantilla
        """)
        
        # Cargar archivo de test como plantilla
        plantilla_path = Path(__file__).parent / "test" / "nuevas_predicciones.xlsx"
        
        if plantilla_path.exists():
            with open(plantilla_path, 'rb') as f:
                plantilla_data = f.read()
            
            st.download_button(
                label="Descargar Plantilla Excel",
                data=plantilla_data,
                file_name="nuevas_predicciones.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        else:
            st.warning("⚠️ Archivo de plantilla no encontrado")

    
    # Widget para cargar archivo
    uploaded_file = st.file_uploader(
        "Seleccione un archivo Excel (.xlsx, .xls)",
        type=['xlsx', 'xls'],
        help="El archivo debe contener las columnas especificadas en las instrucciones"
    )
    
    if uploaded_file is not None:
        try:
            # Leer archivo Excel
            df_excel = pd.read_excel(uploaded_file)
            
            st.success(f"Archivo cargado exitosamente: {len(df_excel)} registros encontrados")
            
            # Mostrar preview de los datos cargados
            with st.expander("Vista previa de los datos cargados"):
                st.dataframe(df_excel.head(10), use_container_width=True)
            
            # Validar columnas requeridas
            columnas_requeridas = ['edad', 'saldo', 'empleo', 'estado_civil', 
                                   'nivel_educativo', 'incumplimiento', 
                                   'prestamo_vivienda', 'prestamo_consumo']
            
            columnas_faltantes = [col for col in columnas_requeridas if col not in df_excel.columns]
            
            if columnas_faltantes:
                st.error(f"Columnas faltantes en el archivo: {', '.join(columnas_faltantes)}")
            else:
                # Botón para procesar
                if st.button("Procesar Predicciones", use_container_width=True, type="primary"):
                    with st.spinner("Procesando predicciones..."):
                        try:
                            # Preparar datos
                            df_procesado = preparar_datos(df_excel[columnas_requeridas], models)
                            
                            # Obtener probabilidades
                            probabilidades = models['model'].predict_proba(df_procesado)[:, 1]
                            
                            # Aplicar umbral
                            predicciones = (probabilidades >= UMBRAL_DECISION).astype(int)
                            
                            # Crear DataFrame de resultados
                            df_resultados = df_excel.copy()
                            df_resultados['probabilidad_cdt'] = probabilidades
                            df_resultados['probabilidad_cdt_pct'] = (probabilidades * 100).round(2)
                            df_resultados['prediccion'] = predicciones
                            df_resultados['prediccion_texto'] = df_resultados['prediccion'].map({1: 'SÍ', 0: 'NO'})
                            
                            st.success("Predicciones completadas!")
                            
                            # Métricas resumen
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("Total Registros", len(df_resultados))
                            with col2:
                                st.metric("Predicción SÍ", (predicciones == 1).sum())
                            with col3:
                                st.metric("Predicción NO", (predicciones == 0).sum())
                            with col4:
                                st.metric("Prob. Promedio", f"{probabilidades.mean()*100:.2f}%")
                            
                            # Mostrar resultados
                            st.subheader("Resultados de Predicción")
                            
                            # Seleccionar columnas para mostrar
                            columnas_mostrar = columnas_requeridas + ['probabilidad_cdt_pct', 'prediccion_texto']
                            
                            # Estilizar el DataFrame
                            def colorear_prediccion(val):
                                if val == 'SÍ':
                                    return 'background-color: #d4edda; color: #155724'
                                else:
                                    return 'background-color: #f8d7da; color: #721c24'
                            
                            df_mostrar = df_resultados[columnas_mostrar].copy()
                            df_mostrar.columns = ['Edad', 'Saldo', 'Empleo', 'Estado Civil', 
                                                 'Nivel Educativo', 'Incumplimiento', 
                                                 'Préstamo Vivienda', 'Préstamo Consumo',
                                                 'Probabilidad (%)', 'Predicción']
                            
                            st.dataframe(
                                df_mostrar.style.applymap(
                                    colorear_prediccion, 
                                    subset=['Predicción']
                                ),
                                use_container_width=True,
                                hide_index=True
                            )
                            
                            # Botón para descargar resultados
                            st.subheader("💾 Descargar Resultados")
                            
                            # Preparar archivo para descarga
                            buffer_resultado = BytesIO()
                            df_resultados.to_excel(buffer_resultado, index=False, engine='openpyxl')
                            buffer_resultado.seek(0)
                            
                            st.download_button(
                                label="Descargar Resultados en Excel",
                                data=buffer_resultado,
                                file_name="resultados_prediccion_cdt.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                type="primary"
                            )
                            
                            # Estadísticas adicionales
                            with st.expander("Estadísticas Detalladas"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Distribución de Probabilidades:**")
                                    stats_df = pd.DataFrame({
                                        'Estadística': ['Mínimo', 'Máximo', 'Promedio', 'Mediana', 'Desv. Estándar'],
                                        'Valor': [
                                            f"{probabilidades.min()*100:.2f}%",
                                            f"{probabilidades.max()*100:.2f}%",
                                            f"{probabilidades.mean()*100:.2f}%",
                                            f"{np.median(probabilidades)*100:.2f}%",
                                            f"{probabilidades.std()*100:.2f}%"
                                        ]
                                    })
                                    st.dataframe(stats_df, hide_index=True)
                                
                                with col2:
                                    st.markdown("**Distribución por Rangos de Probabilidad:**")
                                    rangos = pd.cut(
                                        probabilidades, 
                                        bins=[0, 0.13, 0.25, 0.50, 0.75, 1.0],
                                        labels=['0-13% (No CDT)', '13-25%', '25-50%', '50-75%', '75-100%']
                                    )
                                    rangos_df = rangos.value_counts().reset_index()
                                    rangos_df.columns = ['Rango', 'Cantidad']
                                    st.dataframe(rangos_df, hide_index=True)
                            
                        except Exception as e:
                            st.error(f"Error al procesar las predicciones: {str(e)}")
                            st.exception(e)
        
        except Exception as e:
            st.error(f"Error al leer el archivo Excel: {str(e)}")
            st.info("Asegúrese de que el archivo esté en formato Excel válido (.xlsx o .xls)")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desarrollado con Streamlit • Modelo Random Forest</p>
</div>
""", unsafe_allow_html=True)
