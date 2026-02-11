# Predicción de Salarios con XGBoost

Proyecto de Machine Learning enfocado en la predicción de salarios utilizando el algoritmo de Gradient Boosting XGBoost, optimizado para obtener predicciones precisas basadas en características demográficas y profesionales.

## 📋 Descripción

Este proyecto implementa un modelo de XGBoost (Extreme Gradient Boosting) para predecir niveles de ingresos basándose en variables como educación, experiencia laboral, ocupación, edad, y otras características relevantes. El objetivo es proporcionar estimaciones precisas que puedan ser útiles para análisis de compensación, planificación de recursos humanos, y estudios socioeconómicos.

## 🎯 Objetivos del Proyecto

- Predecir rangos salariales con alta precisión
- Identificar las variables más importantes que influyen en el salario
- Implementar técnicas avanzadas de feature engineering
- Optimizar hiperparámetros para maximizar el rendimiento
- Proporcionar un modelo interpretable y escalable

## 🚀 Características

- **Modelo XGBoost Optimizado**: Implementación del algoritmo de gradient boosting más eficiente
- **Feature Engineering**: Creación y transformación de variables predictivas
- **Encoding Inteligente**: Manejo de variables categóricas mediante Label Encoding
- **Validación Robusta**: Evaluación mediante cross-validation y métricas múltiples
- **Análisis de Importancia**: Identificación de features más relevantes
- **Modelo Serializado**: Modelos guardados listos para producción (`xgb_income_model.pkl`, `label_encoder.pkl`)

## 📁 Estructura del Proyecto

```
Ejercicio-XGBOOST-Salario/
│
├── src/
│   ├── app.py              # Script principal del proyecto
│   ├── explore.ipynb       # Notebook de exploración y experimentación
│   └── utils.py            # Funciones auxiliares
│
├── data/
│   ├── raw/                # Datos originales sin procesar
│   ├── interim/            # Datos en transformación
│   └── processed/          # Datos preprocesados para el modelo
│
├── models/                 # Directorio para modelos adicionales
│
├── xgb_income_model.pkl    # Modelo XGBoost entrenado
├── label_encoder.pkl       # Encoder para variables categóricas
│
├── .devcontainer/          # Configuración de desarrollo
├── .vscode/                # Configuración de VS Code
├── requirements.txt        # Dependencias del proyecto
└── README.md              # Este archivo
```

## 🔧 Instalación

### Opción 1: GitHub Codespaces (Recomendado)

1. Abre el repositorio en GitHub Codespaces
2. El entorno se configurará automáticamente
3. Todas las dependencias se instalarán automáticamente
4. ¡Listo para usar!

### Opción 2: Instalación Local

**Requisitos Previos**
- Python 3.11 o superior
- pip (gestor de paquetes de Python)
- Git

**Pasos de Instalación**

1. Clona el repositorio:
```bash
git clone https://github.com/DannyAIX/Ejercicio-XGBOOST-Salario.git
cd Ejercicio-XGBOOST-Salario
```

2. Crea un entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. Instala las dependencias:
```bash
pip install -r requirements.txt
```

4. Configura variables de entorno (si es necesario):
```bash
cp .env.example .env
# Edita .env según necesites
```

## 💻 Uso

### Ejecutar el Modelo

```bash
python src/app.py
```

### Exploración y Experimentación

Para análisis exploratorio y pruebas:

```bash
jupyter notebook src/explore.ipynb
```

### Flujo de Trabajo Típico

1. **Carga de Datos**
   - Coloca tus datos en `data/raw/`
   - Formato recomendado: CSV con variables demográficas y salariales

2. **Preprocesamiento**
   - Limpieza de datos
   - Manejo de valores faltantes
   - Encoding de variables categóricas
   - Feature engineering

3. **Entrenamiento**
   - Configuración de hiperparámetros de XGBoost
   - Entrenamiento con validación cruzada
   - Optimización mediante GridSearch o RandomSearch

4. **Evaluación**
   - Métricas de clasificación (Accuracy, Precision, Recall, F1)
   - Análisis de importancia de features
   - Matriz de confusión

5. **Predicción**
   - Cargar modelos pre-entrenados
   - Realizar predicciones en nuevos datos

## 📊 Uso del Modelo Pre-entrenado

```python
import pickle
import pandas as pd
import numpy as np

# Cargar el modelo entrenado
with open('xgb_income_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Cargar el label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Preparar datos de ejemplo
new_data = pd.DataFrame({
    'age': [35],
    'education_num': [13],
    'hours_per_week': [40],
    'occupation_encoded': [5],
    # ... otras features necesarias
})

# Realizar predicción
prediction = model.predict(new_data)
prediction_label = label_encoder.inverse_transform(prediction)

print(f'Predicción de nivel de ingresos: {prediction_label[0]}')
```

## 🤖 Sobre XGBoost

XGBoost (Extreme Gradient Boosting) es un algoritmo de machine learning basado en árboles de decisión que utiliza gradient boosting. Es especialmente efectivo para problemas de clasificación y regresión.

### Ventajas de XGBoost

- **Alto Rendimiento**: Uno de los algoritmos más precisos disponibles
- **Velocidad**: Optimizado para ser extremadamente rápido
- **Regularización**: Previene el overfitting mediante L1 y L2 regularization
- **Paralelización**: Aprovecha múltiples núcleos del procesador
- **Manejo de Valores Faltantes**: Puede trabajar con datos incompletos
- **Feature Importance**: Identifica automáticamente las variables más importantes

### Hiperparámetros Clave

```python
from xgboost import XGBClassifier

model = XGBClassifier(
    max_depth=6,              # Profundidad máxima de los árboles
    learning_rate=0.1,        # Tasa de aprendizaje (eta)
    n_estimators=100,         # Número de árboles
    subsample=0.8,            # Fracción de muestras para entrenar cada árbol
    colsample_bytree=0.8,     # Fracción de features por árbol
    gamma=0,                  # Reducción de pérdida mínima para split
    reg_alpha=0,              # Regularización L1
    reg_lambda=1,             # Regularización L2
    random_state=42
)
```

## 📈 Ejemplo Completo de Entrenamiento

```python
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Cargar datos
df = pd.read_csv('data/raw/salary_data.csv')

# Preprocesamiento
# Separar features y target
X = df.drop('income', axis=1)
y = df['income']

# Encoding de variables categóricas
label_encoders = {}
categorical_cols = X.select_dtypes(include=['object']).columns

for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Encoding del target
target_encoder = LabelEncoder()
y_encoded = target_encoder.fit_transform(y)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Entrenar modelo
model = XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=100,
    objective='binary:logistic',
    random_state=42
)

model.fit(X_train, y_train)

# Evaluar
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.4f}')

# Reporte de clasificación
print(classification_report(y_test, y_pred, 
                          target_names=target_encoder.classes_))

# Importancia de features
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print('\nTop 10 Features más importantes:')
print(feature_importance.head(10))

# Guardar modelo
with open('xgb_income_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

print('\n¡Modelo guardado exitosamente!')
```

## 📊 Métricas de Evaluación

El proyecto utiliza las siguientes métricas:

- **Accuracy**: Proporción de predicciones correctas
- **Precision**: Precisión en predicciones positivas
- **Recall**: Capacidad de encontrar todos los casos positivos
- **F1-Score**: Media armónica entre Precision y Recall
- **ROC-AUC**: Área bajo la curva ROC
- **Matriz de Confusión**: Visualización de predicciones correctas e incorrectas

## 🛠️ Tecnologías Utilizadas

### Core
- **Python 3.11+**
- **XGBoost**: Algoritmo principal de ML
- **pandas**: Manipulación de datos
- **numpy**: Operaciones numéricas

### Machine Learning
- **scikit-learn**: Preprocesamiento y evaluación
- **matplotlib/seaborn**: Visualizaciones

### Utilidades
- **pickle**: Serialización de modelos
- **jupyter**: Notebooks interactivos

## 🎯 Variables Típicas del Modelo

El modelo puede trabajar con variables como:

- **Demográficas**: Edad, género, estado civil
- **Educación**: Nivel educativo, años de estudio
- **Laborales**: Ocupación, sector, horas trabajadas
- **Experiencia**: Años de experiencia
- **Ubicación**: País, región, área metropolitana
- **Otras**: Capital gain, capital loss, etc.

## 🔍 Optimización de Hiperparámetros

```python
from sklearn.model_selection import GridSearchCV

# Definir grid de parámetros
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Grid Search
grid_search = GridSearchCV(
    estimator=XGBClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y_train)

print(f'Mejores parámetros: {grid_search.best_params_}')
print(f'Mejor score: {grid_search.best_score_:.4f}')

# Usar mejor modelo
best_model = grid_search.best_estimator_
```

## 📈 Casos de Uso

- **Recursos Humanos**: Planificación de compensaciones y beneficios
- **Reclutamiento**: Establecimiento de rangos salariales competitivos
- **Análisis Económico**: Estudios de desigualdad salarial
- **Consultoría**: Asesoramiento en estructuras de compensación
- **Investigación**: Análisis de factores que influyen en ingresos

## 🔍 Próximos Pasos

- [ ] Implementar SHAP values para explicabilidad del modelo
- [ ] Agregar más features derivadas (feature engineering avanzado)
- [ ] Comparar con otros algoritmos (LightGBM, CatBoost)
- [ ] Implementar API REST para predicciones en tiempo real
- [ ] Crear dashboard interactivo con Streamlit
- [ ] Implementar reentrenamiento automático del modelo
- [ ] Agregar detección de drift en los datos

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Para cambios importantes:

1. Fork del proyecto
2. Crea una rama feature (`git checkout -b feature/MejoraPredictiva`)
3. Commit tus cambios (`git commit -m 'Add: mejora en accuracy'`)
4. Push a la rama (`git push origin feature/MejoraPredictiva`)
5. Abre un Pull Request

## 📚 Referencias

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [XGBoost Paper](https://arxiv.org/abs/1603.02754)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Feature Engineering Techniques](https://www.kaggle.com/learn/feature-engineering)

## 📄 Licencia

Este proyecto está basado en el template de [4Geeks Academy](https://4geeksacademy.com) para el bootcamp de Data Science y Machine Learning.

## 👤 Autor

**DannyAIX**

- GitHub: [@DannyAIX](https://github.com/DannyAIX)
- Proyecto: [Ejercicio-XGBOOST-Salario](https://github.com/DannyAIX/Ejercicio-XGBOOST-Salario)

## 🙏 Agradecimientos

- [4Geeks Academy](https://4geeksacademy.com) por el template base y formación
- Comunidad de XGBoost y scikit-learn
- Desarrolladores del ecosistema de Data Science en Python

---

⭐️ Si este proyecto te resultó útil, considera darle una estrella en GitHub

💼 **¿Necesitas predecir salarios?** Este modelo está listo para ser integrado en sistemas de RRHH y análisis de compensación.
