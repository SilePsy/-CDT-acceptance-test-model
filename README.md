# 💰 Predictor de Incumplimiento de Depósitos Bancarios

Sistema inteligente de predicción basado en **LightGBM** para evaluar el riesgo de incumplimiento de depósitos bancarios.

## 📋 Tabla de Contenidos

- [Descripción General](#descripción-general)
- [Características del Modelo](#características-del-modelo)
- [Requisitos](#requisitos)
- [Instalación](#instalación)
- [Uso](#uso)
- [Estructura de Archivos](#estructura-de-archivos)
- [API de Predicción](#api-de-predicción)
- [Transformaciones de Datos](#transformaciones-de-datos)
- [Resultados y Rendimiento](#resultados-y-rendimiento)
- [Guía de Troubleshooting](#guía-de-troubleshooting)

---

## 📊 Descripción General

Este proyecto implementa un **modelo de machine learning de clasificación binaria** para predecir la probabilidad de que un cliente incumpla en sus obligaciones de depósito bancario.

### Propósito
- Identificar clientes con alto riesgo de incumplimiento
- Facilitar toma de decisiones crediticias
- Automatizar evaluaciones de riesgo
- Reducir exposición a insolvencia

### Contexto del Desarrollo
El modelo fue desarrollado mediante un pipeline completo de ML que incluye:
- Exploración y limpieza de datos (42,639 registros iniciales)
- Ingeniería de características (36 características seleccionadas)
- Evaluación estadística de 3 modelos (Regresión Logística + 2 configuraciones de LightGBM)
- Análisis ANOVA para validación estadística
- Selección del modelo óptimo con validación cruzada de 10 folds

---

## ⚙️ Características del Modelo

### Especificaciones Técnicas

| Parámetro | Valor |
|-----------|-------|
| **Algoritmo** | Light Gradient Boosting Machine (LightGBM) |
| **Tipo de Problema** | Clasificación Binaria |
| **Métrica Principal** | ROC-AUC = 0.6680 |
| **Validación** | 10-Fold Stratified Cross-Validation |
| **Características** | 36 variables seleccionadas |
| **Tamaño de Entrenamiento** | 29,847 registros (70%) |
| **Tamaño de Prueba** | 12,792 registros (30%) |
| **Variable Objetivo** | Incumplimiento (Binaria: 0=Cumple, 1=Incumple) |

### Parámetros Óptimos del Modelo

```python
{
    'learning_rate': 0.025,
    'num_leaves': 10,
    'max_depth': 5,
    'n_estimators': 200,
    'boosting_type': 'gbdt',
    'reg_lambda': 2.0,
    'reg_alpha': 0.1,
    'objective': 'binary',
}
```

### Variables de Entrada (36 características)

#### Información Demográfica
- `profesi`: Profesión (Empleado/Empresa)
- `estcivil`: Estado Civil (Soltero/Casado/Divorciado/Viudo)
- `edad`: Edad del cliente
- `edad_escalada`: Edad normalizada [0,1]

#### Información Bancaria - Canales
- `canalprim`: Canal Principal de Contacto (Sucursal/Internet/Teléfono/Móvil)
- `canalprim_*`: Variables one-hot encoded

#### Información Bancaria - Productos
- `rentero`: Es rentero (Sí/No)
- `nivelacd`: Nivel de Acceso (Básico/Premium/Gold)
- `cant_productos`: Cantidad de Productos Bancarios
- `norm_cant_productos`: Cantidad de Productos Normalizada
- `meses_cliente`: Meses como Cliente

#### Variables Financieras
- `saldo`: Saldo de la Cuenta
- `saldo_escalado`: Saldo Escalado [0,1]
- `edad_saldo_interaction`: Interacción edad × saldo

#### Quintiles de Variables
- `cant_productos_quintil_*`: Variables quintil one-hot encoded (5 variables)
- `meses_cliente_quintil_*`: Variables quintil one-hot encoded (5 variables)

---

## 🔧 Requisitos

### Requisitos del Sistema
- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, macOS, Linux
- **RAM**: Mínimo 2GB
- **Espacio en Disco**: ~100MB (con modelos)

### Archivos Requeridos

El sistema requiere los siguientes archivos joblib en el directorio raíz:

```
Proyecto Final/
├── app.py                                    # Aplicación Streamlit
├── requirements.txt                         # Dependencias Python
├── README.md                               # Este archivo
│
├── lightgbm_modelo_filtrado.joblib          # Modelo entrenado
├── features_list_lightgbm_filtrado.joblib   # Listado de variables
├── model_config.joblib                      # Configuración del modelo
│
├── encoders_binarios.joblib                 # Encoders para variables binarias
├── one_hot_encoder.joblib                   # OneHotEncoder para categóricas
├── scaler_edad.joblib                       # MinMaxScaler para edad
├── scaler_saldo.joblib                      # MinMaxScaler para saldo
├── quintiles_generador.joblib               # Generador de quintiles
├── one_hot_encoder_quintiles.joblib         # OneHotEncoder para quintiles
│
└── sample_input.csv                         # Archivo de ejemplo (opcional)
```

---

## 📦 Instalación

### Paso 1: Clonar/Copiar el Repositorio
```bash
cd "d:\Proyecto Final"
```

### Paso 2: Crear Ambiente Virtual (Recomendado)

#### En Windows (PowerShell):
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

#### En macOS/Linux:
```bash
python3 -m venv venv
source venv/bin/activate
```

### Paso 3: Instalar Dependencias
```bash
pip install -r requirements.txt
```

**Nota**: Si encuentras problemas instalando LightGBM, intenta:
```bash
pip install --no-binary lightgbm lightgbm
```

### Paso 4: Verificar Instalación
```bash
python -c "import streamlit, lightgbm, joblib; print('✓ Librerías instaladas correctamente')"
```

---

## 🚀 Uso

### Ejecutar la Aplicación Streamlit

```bash
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador predeterminado en `http://localhost:8501`.

### Interfaz Web

#### 🎯 Tab 1: Predicción Individual
- Ingresa datos de un cliente
- Campos básicos: Profesión, Estado Civil, Edad, etc.
- Visualización instantánea de predicción
- Gráfico de distribución de probabilidades

#### 📤 Tab 2: Predicción en Lote
- Carga archivo CSV con múltiples registros
- Procesa todas las filas automáticamente
- Descarga resultados en CSV

#### 📋 Tab 3: Guía de Uso
- Descripción detallada del modelo
- Especificaciones de variables
- Interpretación de resultados
- Limitaciones del modelo

---

## 📂 Estructura de Archivos

```
d:\Proyecto Final\
│
├── 📄 app.py
│   └── Aplicación principal Streamlit
│       - Interfaz de usuario
│       - Funciones de predicción
│       - Carga de modelos
│
├── 📄 requirements.txt
│   └── Dependencias Python necesarias
│
├── 📄 README.md
│   └── Documentación del proyecto (este archivo)
│
├── 📊 Modelos y Transformadores
│   ├── lightgbm_modelo_filtrado.joblib
│   ├── features_list_lightgbm_filtrado.joblib
│   ├── model_config.joblib
│   ├── encoders_binarios.joblib
│   ├── one_hot_encoder.joblib
│   ├── scaler_edad.joblib
│   ├── scaler_saldo.joblib
│   ├── quintiles_generador.joblib
│   └── one_hot_encoder_quintiles.joblib
│
└── 📝 Documentación Adicional
    ├── cv_results_consolidados_all_folds.csv
    ├── cv_results_consolidados_all_folds.xlsx
    └── cv_results_summary_statistics.csv
```

---

## 🔮 API de Predicción

### Estructura de Datos de Entrada

```python
{
    'profesi': 'Empleado' | 'Empresa',
    'estcivil': 'Soltero' | 'Casado' | 'Divorciado' | 'Viudo',
    'edad': int (18-100),
    'canalprim': 'Sucursal' | 'Internet' | 'Teléfono' | 'Móvil',
    'rentero': 'Sí' | 'No',
    'nivelacd': 'Básico' | 'Premium' | 'Gold',
    'cant_productos': int (1-20),
    'saldo': float (0-1000000),
    'meses_cliente': int (1-600)
}
```

### Estructura de Datos de Salida

```python
{
    'clase': 'CUMPLIMIENTO' | 'INCUMPLIMIENTO',
    'probabilidad': float (0.0-1.0),  # Probabilidad de incumplimiento
    'confianza': float (0.0-1.0)      # Confianza de la predicción
}
```

### Interpretación de Resultados

| Probabilidad | Predicción | Acción Recomendada |
|--------------|-----------|-------------------|
| < 0.3 | ✅ BAJO RIESGO | Autorizar sin restricciones |
| 0.3 - 0.5 | ✅ MODERADO | Autorizar con revisión |
| 0.5 - 0.7 | ⚠️ ALTO RIESGO | Revisar manualmente |
| > 0.7 | ⚠️ MUY ALTO RIESGO | Rechazar o condiciones especiales |

---

## 🔄 Transformaciones de Datos

### Flujo de Procesamiento

```
Datos Crudos
    ↓
[1] Encoding Binario (profesi, estcivil)
    ↓
[2] One-Hot Encoding (canalprim, rentero, nivelacd)
    ↓
[3] Scaling MinMax (edad, saldo) → [0,1]
    ↓
[4] Feature Engineering
    - norm_cant_productos = cant_productos / max
    - edad_saldo_interaction = edad_scaled × saldo_scaled
    ↓
[5] Quintiles (cant_productos, meses_cliente)
    - División en 5 quintiles
    - One-Hot Encoding de quintiles
    ↓
[6] Selección de Features
    - Solo 36 características relevantes
    ↓
Características Procesadas
    ↓
Predicción de Modelo
```

### Ejemplo de Transformación

**Entrada Original:**
```csv
profesi,estcivil,edad,canalprim,rentero,nivelacd,cant_productos,saldo,meses_cliente
Empleado,Casado,35,Internet,Sí,Premium,3,50000,24
```

**Después de Transformaciones:**
```
profesi (LabelEncoder)           → 0.0
estcivil (LabelEncoder)          → 1.0
edad (MinMaxScaler)              → 0.382
edad_escalada                    → 0.382
canalprim_Internet               → 1.0
canalprim_Móvil                  → 0.0
canalprim_Sucursal               → 0.0
canalprim_Teléfono               → 0.0
rentero_No                       → 0.0
rentero_Sí                       → 1.0
nivelacd_Básico                  → 0.0
nivelacd_Gold                    → 0.0
nivelacd_Premium                 → 1.0
cant_productos                   → 3.0
norm_cant_productos              → 0.150
saldo (MinMaxScaler)             → 0.050
saldo_escalado                   → 0.050
edad_saldo_interaction           → 0.019
meses_cliente                    → 24.0
cant_productos_quintil_1         → 1.0
cant_productos_quintil_2         → 0.0
... (quintiles restantes)        → 0.0
meses_cliente_quintil_1          → 0.0
meses_cliente_quintil_2          → 1.0
... (quintiles restantes)        → 0.0
```

---

## 📈 Resultados y Rendimiento

### Validación Cruzada (10-Fold)

```
┌─────────────────────────┬──────────┬──────────┐
│ Modelo                  │ Media    │ Std Dev  │
├─────────────────────────┼──────────┼──────────┤
│ LightGBM (Sin Bajo IV)  │ 0.6680   │ 0.0114   │ ← SELECCIONADO
│ LightGBM (Todas)        │ 0.6692   │ 0.0125   │
│ Regresión Logística     │ 0.6580   │ 0.0118   │
└─────────────────────────┴──────────┴──────────┘

Análisis ANOVA:
- F(2,27) = 2.636, p = 0.0900
- Conclusión: NO hay diferencias estadísticamente significativas
- LightGBM (Sin Bajo IV) seleccionado por simplicidad (36 vs 37 variables)
```

### Análisis Detallado del Modelo Seleccionado

**LightGBM (Sin Bajo IV)**

```
Métrica              Valor    Interpretación
─────────────────────────────────────────
AUC-ROC (CV)         0.6680   Discriminación moderada
Sensibilidad (CV)    ~65%     Detecta 65% de incumplidores
Especificidad (CV)   ~65%     Identifica 65% de cumplidores
Precisión (CV)       ~63%     63% de predicciones positivas correctas
F1-Score (CV)        ~0.64    Balance razonable precisión-recall
```

### Comparación Entre Modelos

```
ANOVA TEST RESULTS:
┌──────────────────────────────────────────┐
│ H0: Las medias son iguales                │
│ H1: Al menos una media es diferente       │
│                                          │
│ T-Test: t(18) = 0.218, p = 0.8297      │
│ Resultado: NO RECHAZAR H0                │
│                                          │
│ Decisión: NO hay diferencias             │
│ significativas entre LightGBM models     │
│ (α=0.05)                                 │
└──────────────────────────────────────────┘
```

---

## 🐛 Guía de Troubleshooting

### Problema: Error "ModuleNotFoundError: No module named 'lightgbm'"

**Solución:**
```bash
# 1. Actualiza pip
pip install --upgrade pip

# 2. Reinstala lightgbm sin compilar
pip install --no-binary lightgbm lightgbm

# 3. Si sigue fallando, intenta con conda:
conda install -c conda-forge lightgbm
```

### Problema: Error "FileNotFoundError: features_list_lightgbm_filtrado.joblib"

**Solución:**
- Verifica que todos los archivos .joblib estén en el directorio correcto
- Asegúrate de que no hay caracteres especiales en la ruta
- En Windows, usa la ruta: `d:\Proyecto Final\`

### Problema: La predicción es lenta

**Solución:**
```bash
# 1. Verifica que estés usando la versión CPU de LightGBM
pip show lightgbm

# 2. Para mejorar rendimiento, descomenta GPU en app.py:
# 'device': 'gpu'  # Requiere CUDA instalado
```

### Problema: Error "CUDA not available" (GPU)

**Solución:**
- Usa la versión CPU (predeterminada)
- O instala CUDA + cuDNN para usar GPU

### Problema: App Streamlit no se abre

**Solución:**
```bash
# 1. Verifica que Streamlit está instalado
pip install --upgrade streamlit

# 2. Ejecuta con configuración mínima
streamlit run app.py --logger.level=debug

# 3. Abre manualmente: http://localhost:8501
```

### Problema: Datos de entrada inválidos

**Solución:**
- Verifica los tipos de datos (categóricos vs numéricos)
- Asegúrate que los valores estén dentro de los rangos permitidos
- Comprueba la ortografía exacta de opciones categóricas

---

## 📞 Contacto y Soporte

- **Proyecto**: Predictor de Incumplimiento de Depósitos
- **Versión**: 1.0
- **Fecha de Creación**: 2026
- **Framework**: Streamlit + LightGBM

---

## 📜 Licencia y Términos de Uso

Este modelo se proporciona "tal cual" para fines educacionales y de negocio. 

**Disclaimer:**
- Las predicciones son estimaciones estadísticas basadas en datos históricos
- No constituyen garantías ni asesoramiento financiero profesional
- Debe validarse manualmente antes de tomar decisiones críticas
- El autor no se responsabiliza por decisiones tomadas basadas en estas predicciones

---

## 🎯 Casos de Uso

### Caso 1: Análisis de Riesgo Individual

```python
Cliente: Juan García
- Profesión: Empleado
- Estado Civil: Casado
- Edad: 35 años
- Productos: 3
- Saldo: $50,000

Predicción: ✅ BAJO RIESGO (35%)
Acción: Autorizar depósito
```

### Caso 2: Evaluación en Lote de Cartera

```python
Entrada: 1,000 clientes
Procesamiento: ~2 segundos
Resultados:
- 650 bajo riesgo (65%)
- 250 moderado (25%)
- 100 alto riesgo (10%)

Acción: Revisión especial del 10% de riesgo alto
```

---

## 🔮 Mejoras Futuras

- [ ] Incorporar más variables (empleador, histórico de crédito)
- [ ] Calibración de probabilidades con Platt scaling
- [ ] API REST para integración con sistemas bancarios
- [ ] Dashboard de monitoreo de desempeño
- [ ] Explicabilidad SHAP para interpretaciones
- [ ] Versión móvil de la aplicación

---

**Última actualización**: 2026
**Estado**: Producción ✅
