#librerias 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier, plot_importance
import matplotlib.pyplot as plt


# CRISP-DM CRISP-DM significa Cross Industry Standard Process for Data Mining, y es el estándar más usado en la industria para desarrollar proyectos de anális

# Paso 1: Obtener Datos de CSV

total_data = pd.read_csv("/workspaces/Ejercicio-XGBOOST-Salario/data/raw/adult.csv",sep=';')  # pasar la data a un data fram

# si la voy a traer de la pagina
#total_data = pd.read_csv("https://storage.googleapis.com/breathecode/project-files/bank-marketing-campaign-data.csv",sep=';')
#total_data.to_csv("/workspaces/regresion-logistica/data/raw/total_data.csv", sep=';', index = False)

#Paso 2 Entender o explorar la data
print(total_data.head()) #ver rapidamente si cargo la info

print(total_data.shape)  # 15 columnas o variables y 32,561 filas o cantidad de registros
print(total_data.columns) # ver las columnas
print(total_data.info()) # ver valores nulos y memoria usada, todo de un vistazo. int64(6), object(9)

print(total_data.describe()) # ESTADISTICAS de cada columna,

# edad promedio 38, desvicion de 13, mas joven 17, mas viejo 90 mediana 37
# fnlwgt el peso final 3.25 promedio
# education.num promedio 10 std, 2.5, min 1 mediana 10 y max 16
# cappital.gain 1,077 promedio
# capital.loss perdidas capital 87 promeidio
#hors.per.week, promedio 40, std 12 min 1 max 99 media 40

# LIMPIEZA Y PREPARACIÓN

#Eliminar duplicados
print(f"Duplicados encontrados: {total_data.duplicated().sum()}")
total_data = total_data.drop_duplicates()

# Reemplazar valores "?" por NaN y eliminar filas vacías
total_data = total_data.replace("?", np.nan)
total_data = total_data.dropna()

# Revisar nuevamente estructura
print("Datos limpios:", total_data.shape)

#Codificar variables categóricas con LabelEncoder, esto las transforma a numeros
le = LabelEncoder()
for col in total_data.select_dtypes(include=["object"]).columns:
    total_data[col] = le.fit_transform(total_data[col])

#  Separar variables predictoras (X) y objetivo (y)
X = total_data.drop("income", axis=1)
y = total_data["income"]

print(total_data.head()) #ver rapidamente como quedo la info

# MODELADO CON XGBOOST
# =============================

# Dividir dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Crear modelo
model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric='logloss',
    random_state=42
)

# Entrenar modelo
model.fit(X_train, y_train)

# Predicciones
y_pred = model.predict(X_test)

#EVALUACIÓN
# =============================

print("\n📊 Reporte de Clasificación:\n", classification_report(y_test, y_pred))     #Precisión (Accuracy): 86.96%
print("✅ Precisión (Accuracy):", round(accuracy_score(y_test, y_pred)*100, 2), "%")
print("📉 Matriz de Confusión:\n", confusion_matrix(y_test, y_pred))
#                 Predijo <=50K	Predijo >50K
# Real <=50K (0)	4226 ✅	      301 ❌
# Real >50K (1)	     485 ❌	     1016 ✅

#Clase	     Significado	            Precision	Recall	F1-score	Interpretación
#0 (<=50K)	Personas con ingresos bajos	 0.90	    0.93	 0.91	    El modelo identifica bien a quienes ganan menos.
#1 (>50K)	Personas con ingresos altos	 0.77	    0.68	 0.72	    Le cuesta un poco más detectar a los de alto ingreso.

# Precision 0.77 → De los que predice como >50K, el 77% realmente lo son.
# Recall 0.68 → Solo detecta 68% de los que realmente ganan >50K (pierde algunos).
# F1-score 0.72 → Buen equilibrio, pero podría mejorar en recall.


# Importancia de variables
plt.figure(figsize=(10,6))
plot_importance(model, max_num_features=10)
plt.title("Importancia de las 10 variables más influyentes")
plt.show()

#optimización con XGBoost, usando RandomizedSearchCV para ajustar los hiperparámetros clave

from sklearn.model_selection import RandomizedSearchCV
import shap

# =============================
# OPTIMIZACIÓN DEL MODELO
# =============================

param_dist = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0],
    'gamma': [0, 0.1, 0.2, 0.5],
    'min_child_weight': [1, 3, 5]
}

xgb = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    random_state=42
)

random_search = RandomizedSearchCV(
    estimator=xgb,
    param_distributions=param_dist,
    n_iter=20,
    scoring='accuracy',
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

print("\n🔍 Mejores hiperparámetros encontrados:")
print(random_search.best_params_)

best_model = random_search.best_estimator_

best_model.fit(X_train, y_train)

y_pred_best = best_model.predict(X_test)

print("\n📊 Reporte de Clasificación (modelo optimizado):\n", classification_report(y_test, y_pred_best))
print("✅ Precisión (Accuracy):", round(accuracy_score(y_test, y_pred_best)*100, 2), "%")
print("📉 Matriz de Confusión:\n", confusion_matrix(y_test, y_pred_best))

# modelo optimizado mejoró ligeramente respecto al anterior (de 86.9 % → 87.4 %)



#guardar modelo

import joblib

joblib.dump(best_model, "xgb_income_model.pkl")
joblib.dump(le, "label_encoder.pkl")