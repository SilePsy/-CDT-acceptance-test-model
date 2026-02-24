import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Configurar la página
st.set_page_config(
    page_title="Predicción de Depósito a Término",
    page_icon="💰",
    layout="wide"
)

st.title("💰 Modelo de Predicción de Depósito a Término")
st.markdown("---")

# Directorio de archivos modelo
MODEL_DIR = Path(__file__).parent / "archivos_modelo"

@st.cache_resource
def load_models():
    """Carga todos los archivos joblib necesarios para el modelo."""
    try:
        encoders_binarios = joblib.load(MODEL_DIR / "encoders_binarios.joblib")
        minmax_scaler_edad = joblib.load(MODEL_DIR / "minmax_scaler_edad.joblib")
        minmax_scaler_saldo = joblib.load(MODEL_DIR / "minmax_scaler_saldo.joblib")
        one_hot_encoder = joblib.load(MODEL_DIR / "one_hot_encoder.joblib")
        quintiles_generador = joblib.load(MODEL_DIR / "quintiles_generador.joblib")
        one_hot_encoder_quintiles = joblib.load(MODEL_DIR / "one_hot_encoder_quintiles.joblib")
        features_list = joblib.load(MODEL_DIR / "features_list_lightgbm_filtrado.joblib")
        model = joblib.load(MODEL_DIR / "modelo_final.joblib")
        
        return {
            'encoders_binarios': encoders_binarios,
            'minmax_scaler_edad': minmax_scaler_edad,
            'minmax_scaler_saldo': minmax_scaler_saldo,
            'one_hot_encoder': one_hot_encoder,
            'quintiles_generador': quintiles_generador,
            'one_hot_encoder_quintiles': one_hot_encoder_quintiles,
            'features_list': features_list,
            'model': model
        }
    except Exception as e:
        st.error(f"Error al cargar los modelos: {e}")
        return None

# Cargar modelos
models = load_models()
if models is None:
    st.stop()

# Información sobre el modelo
with st.expander("ℹ️ Información del Modelo"):
    st.markdown("""
    Este modelo predice la probabilidad de que un cliente realice un depósito a término.
    
    **Variables de entrada requeridas:**
    - Edad
    - Saldo
    - Trabajo
    - Estado Civil
    - Educación
    - Incumplimiento
    - Tiene Vivienda
    - Tiene Préstamo
    """)

# Crear formulario
with st.form("prediction_form"):
    st.header("📋 Ingrese los Datos del Cliente")
    
    # Dividir en columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Datos Numéricos")
        edad = st.slider("Edad", min_value=18, max_value=100, value=40, step=1)
        saldo = st.number_input("Saldo (EUR)", value=1000.0, step=100.0)
    
    with col2:
        st.subheader("📋 Datos Categóricos")
        trabajo = st.selectbox(
            "Trabajo",
            options=['administrador', 'obrero', 'empresario', 'empleada_hogar', 
                     'gestion', 'jubilado', 'autonomo', 'servicios', 
                     'estudiante', 'tecnico', 'desempleado', 'desconocido']
        )
        
        estado_civil = st.selectbox(
            "Estado Civil",
            options=['casado', 'divorciado', 'soltero']
        )
        
        educacion = st.selectbox(
            "Educación",
            options=['primaria', 'secundaria', 'terciaria', 'desconocida']
        )
    
    col3, col4 = st.columns(2)
    
    with col3:
        incumplimiento = st.selectbox("Incumplimiento", options=['no', 'si'])
        tiene_vivienda = st.selectbox("Tiene Vivienda", options=['no', 'si'])
    
    with col4:
        tiene_prestamo = st.selectbox("Tiene Préstamo", options=['no', 'si'])
    
    # Botón para enviar
    submit_button = st.form_submit_button("🔮 Hacer Predicción", use_container_width=True)

# Procesar predicción
if submit_button:
    st.markdown("---")
    st.header("📈 Resultado de la Predicción")
    
    try:
        # Crear dataframe con los datos de entrada
        data = {
            'edad': [edad],
            'saldo': [saldo],
            'trabajo': [trabajo],
            'estado_civil': [estado_civil],
            'educacion': [educacion],
            'incumplimiento': [incumplimiento],
            'tiene_vivienda': [tiene_vivienda],
            'tiene_prestamo': [tiene_prestamo]
        }
        
        df_input = pd.DataFrame(data)
        
        # PASO 1: Aplicar LabelEncoders binarios
        # Codificar variables binarias
        for variable in ['incumplimiento', 'tiene_vivienda', 'tiene_prestamo']:
            encoder = models['encoders_binarios'][variable]
            df_input[variable] = encoder.transform(df_input[variable])
        
        # PASO 2: Normalizar edad con MinMaxScaler
        edad_normalizada = models['minmax_scaler_edad'].transform([[df_input['edad'].values[0]]])
        df_input['edad'] = edad_normalizada[0][0]
        
        # PASO 3: Normalizar saldo con MinMaxScaler
        saldo_normalizado = models['minmax_scaler_saldo'].transform([[df_input['saldo'].values[0]]])
        df_input['saldo'] = saldo_normalizado[0][0]
        
        # PASO 4: Crear variables derivadas
        df_input['norm_cant_productos'] = (df_input['tiene_vivienda'] + df_input['tiene_prestamo']) / 2
        df_input['edad_saldo'] = df_input['edad'] * df_input['saldo']
        
        # PASO 5: Aplicar One Hot Encoder a variables categóricas
        variables_categoricas = ['trabajo', 'estado_civil', 'educacion']
        df_ohe = models['one_hot_encoder'].transform(df_input[variables_categoricas])
        
        # Obtener nombres de características del One Hot Encoder
        feature_names_ohe = models['one_hot_encoder'].get_feature_names_out(variables_categoricas)
        df_ohe_encoded = pd.DataFrame(df_ohe, columns=feature_names_ohe)
        
        # Combinar con datos no categóricos (excepto trabajo, estado_civil, educacion)
        cols_to_drop = variables_categoricas
        df_combined = pd.concat([
            df_input.drop(columns=cols_to_drop),
            df_ohe_encoded
        ], axis=1)
        
        # PASO 6: Crear quintiles
        cantidad_quintiles = len(models['quintiles_generador']['edad_labels'])
        
        # Crear quintiles de edad
        df_combined['quintil_edad'] = pd.cut(
            df_combined['edad'],
            bins=models['quintiles_generador']['edad_bins'],
            labels=models['quintiles_generador']['edad_labels'],
            include_lowest=True
        )
        
        # Crear quintiles de saldo
        df_combined['quintil_saldo'] = pd.cut(
            df_combined['saldo'],
            bins=models['quintiles_generador']['saldo_bins'],
            labels=models['quintiles_generador']['saldo_labels'],
            include_lowest=True
        )
        
        # PASO 7: Aplicar One Hot Encoder a quintiles
        variables_quintiles = ['quintil_edad', 'quintil_saldo']
        quintiles_ohe = models['one_hot_encoder_quintiles'].transform(df_combined[variables_quintiles])
        
        # Obtener nombres de características del One Hot Encoder de quintiles
        quintiles_feature_names = models['one_hot_encoder_quintiles'].get_feature_names_out(variables_quintiles)
        df_quintiles_encoded = pd.DataFrame(quintiles_ohe, columns=quintiles_feature_names)
        
        # Combinar datos finales
        df_final = pd.concat([
            df_combined.drop(columns=variables_quintiles),
            df_quintiles_encoded
        ], axis=1)
        
        # PASO 8: Seleccionar solo las características que el modelo espera
        features_expected = models['features_list']
        
        # Asegurar que todas las características esperadas existan
        for feature in features_expected:
            if feature not in df_final.columns:
                df_final[feature] = 0
        
        # Seleccionar solo las características necesarias en el orden correcto
        df_model_input = df_final[features_expected]
        
        # PASO 9: Hacer predicción
        probabilidad = models['model'].predict_proba(df_model_input)[0][1]
        prediccion = models['model'].predict(df_model_input)[0]
        
        # Mostrar resultados
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric(
                "Probabilidad de Depósito a Término",
                f"{probabilidad:.2%}",
                delta=f"{(probabilidad - 0.5)*100:.1f}%" if probabilidad > 0.5 else f"{(probabilidad - 0.5)*100:.1f}%"
            )
        
        with col2:
            if prediccion == 1:
                st.success(f"✅ Predicción: **SÍ** realizará depósito a término")
            else:
                st.info(f"❌ Predicción: **NO** realizará depósito a término")
        
        # Tabla de resumen
        st.subheader("📊 Resumen de Datos Ingresados")
        
        resumen_data = {
            'Variable': ['Edad', 'Saldo', 'Trabajo', 'Estado Civil', 'Educación', 
                        'Incumplimiento', 'Tiene Vivienda', 'Tiene Préstamo'],
            'Valor': [edad, saldo, trabajo, estado_civil, educacion,
                     incumplimiento, tiene_vivienda, tiene_prestamo]
        }
        
        df_resumen = pd.DataFrame(resumen_data)
        st.dataframe(df_resumen, use_container_width=True, hide_index=True)
        
        # Información adicional
        with st.expander("🔍 Detalles Técnicos"):
            st.markdown(f"""
            **Procesamiento aplicado:**
            - Variables binarias codificadas: ✓
            - Edad normalizada (MinMaxScaler): {df_input['edad'].values[0]:.4f}
            - Saldo normalizado (MinMaxScaler): {df_input['saldo'].values[0]:.4f}
            - Variables derivadas creadas: ✓ (norm_cant_productos, edad_saldo)
            - One Hot Encoding categóricas: ✓
            - Quintiles generados: ✓
            - Características del modelo: {len(features_expected)}
            - Probabilidad predicha: {probabilidad:.6f}
            """)
        
    except Exception as e:
        st.error(f"Error durante la predicción: {str(e)}")
        st.error("Por favor, revise los datos ingresados e intente nuevamente.")

st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Desarrollado con Streamlit • Modelo LightGBM</p>
</div>
""", unsafe_allow_html=True)
