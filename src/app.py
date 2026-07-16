from utils import db_connect
engine = db_connect()

# your code here
"""
Proyecto: Predicción de Diabetes con Árbol de Decisión
Dataset: Pima Indians Diabetes (diabetes__2_.csv)
Autor: Gaspar Diaz - 4Geeks Academy

Pipeline: EDA -> deteccion de ceros invalidos -> split -> imputacion (sin fuga de datos)
          -> arbol baseline -> poda con GridSearchCV -> evaluacion -> conclusiones
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_auc_score, ConfusionMatrixDisplay
)

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Carga de datos
# ---------------------------------------------------------------------------
df = pd.read_csv("diabetes__2_.csv")

print("Forma del dataset:", df.shape)
print("\nTipos de datos:\n", df.dtypes)
print("\nResumen estadistico:\n", df.describe())


# ---------------------------------------------------------------------------
# 2. Deteccion de ceros invalidos (faltantes ocultos)
# ---------------------------------------------------------------------------
cols_sospechosas = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

zeros_resumen = pd.DataFrame({
    "n_ceros": (df[cols_sospechosas] == 0).sum(),
    "pct_ceros": (df[cols_sospechosas] == 0).mean() * 100
})
print("\nCeros invalidos por columna:\n", zeros_resumen)


# ---------------------------------------------------------------------------
# 3. Indicadores de faltante + reemplazo de ceros por NaN
#    (esto NO usa la variable Outcome, por lo que es seguro hacerlo
#    antes del split)
# ---------------------------------------------------------------------------
df["SkinThickness_missing"] = (df["SkinThickness"] == 0).astype(int)
df["Insulin_missing"] = (df["Insulin"] == 0).astype(int)

df[cols_sospechosas] = df[cols_sospechosas].replace(0, np.nan)


# ---------------------------------------------------------------------------
# 4. EDA visual (sobre el dataset completo, solo exploratorio)
# ---------------------------------------------------------------------------
# 4.1 Balance de clases
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(x="Outcome", data=df, ax=ax)
ax.set_title("Balance de clases (Outcome)")
ax.set_xlabel("Outcome (0 = No diabetes, 1 = Diabetes)")
plt.tight_layout()
plt.savefig("eda_balance_clases.png", dpi=120)
plt.close(fig)

print("\nProporcion de clases:\n", df["Outcome"].value_counts(normalize=True) * 100)

# 4.2 Distribucion de variables numericas por Outcome
num_cols = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]

fig, axes = plt.subplots(4, 2, figsize=(14, 16))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.kdeplot(data=df, x=col, hue="Outcome", fill=True, ax=axes[i], common_norm=False)
    axes[i].set_title(f"Distribucion de {col} por Outcome")
plt.tight_layout()
plt.savefig("eda_distribuciones.png", dpi=120)
plt.close(fig)

# 4.3 Matriz de correlacion
fig, ax = plt.subplots(figsize=(10, 8))
corr = df.drop(columns=["Outcome"]).corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
ax.set_title("Matriz de correlacion")
plt.tight_layout()
plt.savefig("eda_correlacion.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 5. Split train/test (ANTES de imputar, para evitar fuga de datos)
# ---------------------------------------------------------------------------
X = df.drop(columns=["Outcome"])
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

print("\nTrain:", X_train.shape, "Test:", X_test.shape)


# ---------------------------------------------------------------------------
# 6. Imputacion con medianas calculadas SOLO en train
#    (sin condicionar a Outcome, para no filtrar informacion de la etiqueta)
# ---------------------------------------------------------------------------
medianas_train = X_train[cols_sospechosas].median()
print("\nMedianas calculadas en train:\n", medianas_train)

X_train[cols_sospechosas] = X_train[cols_sospechosas].fillna(medianas_train)
X_test[cols_sospechosas] = X_test[cols_sospechosas].fillna(medianas_train)

assert X_train.isnull().sum().sum() == 0
assert X_test.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# 7. Modelo baseline: Arbol de Decision con hiperparametros por defecto
# ---------------------------------------------------------------------------
tree_baseline = DecisionTreeClassifier(random_state=RANDOM_STATE)
tree_baseline.fit(X_train, y_train)

print("\n--- BASELINE ---")
print("Profundidad:", tree_baseline.get_depth(), "| Hojas:", tree_baseline.get_n_leaves())

y_train_pred = tree_baseline.predict(X_train)
y_test_pred = tree_baseline.predict(X_test)
y_test_proba = tree_baseline.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (baseline) ===")
print(classification_report(y_train, y_train_pred))
print("\n=== TEST (baseline) ===")
print(classification_report(y_test, y_test_pred))
print("AUC test (baseline):", roc_auc_score(y_test, y_test_proba))

importancias_baseline = pd.Series(
    tree_baseline.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (baseline):\n", importancias_baseline)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Blues")
ax.set_title("Matriz de confusion - Baseline (test)")
plt.tight_layout()
plt.savefig("confusion_baseline.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 8. Poda de hiperparametros (grid pequeno + validacion cruzada)
# ---------------------------------------------------------------------------
param_grid = {
    "max_depth": [3, 4, 5, 6, 7],
    "min_samples_leaf": [5, 10, 20],
    "class_weight": [None, "balanced"]
}

grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print("\n--- GRID SEARCH ---")
print("Mejores hiperparametros:", grid_search.best_params_)
print("Mejor F1 (CV):", grid_search.best_score_)

tree_tuned = grid_search.best_estimator_
print("Profundidad podado:", tree_tuned.get_depth(), "| Hojas:", tree_tuned.get_n_leaves())


# ---------------------------------------------------------------------------
# 9. Evaluacion del modelo final (podado)
# ---------------------------------------------------------------------------
y_train_pred_tuned = tree_tuned.predict(X_train)
y_test_pred_tuned = tree_tuned.predict(X_test)
y_test_proba_tuned = tree_tuned.predict_proba(X_test)[:, 1]

print("\n=== TRAIN (podado) ===")
print(classification_report(y_train, y_train_pred_tuned))
print("\n=== TEST (podado) ===")
print(classification_report(y_test, y_test_pred_tuned))
print("AUC test (podado):", roc_auc_score(y_test, y_test_proba_tuned))

importancias_tuned = pd.Series(
    tree_tuned.feature_importances_, index=X_train.columns
).sort_values(ascending=False)
print("\nImportancia de variables (podado):\n", importancias_tuned)

fig, ax = plt.subplots(figsize=(5, 4))
ConfusionMatrixDisplay(
    confusion_matrix(y_test, y_test_pred_tuned),
    display_labels=["No diabetes", "Diabetes"]
).plot(ax=ax, cmap="Blues")
ax.set_title("Matriz de confusion - Podado (test)")
plt.tight_layout()
plt.savefig("confusion_podado.png", dpi=120)
plt.close(fig)

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(
    tree_tuned,
    feature_names=X_train.columns,
    class_names=["No diabetes", "Diabetes"],
    filled=True,
    rounded=True,
    fontsize=8,
    ax=ax
)
plt.tight_layout()
plt.savefig("arbol_podado.png", dpi=120)
plt.close(fig)


# ---------------------------------------------------------------------------
# 10. Resumen final en consola
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("RESUMEN FINAL")
print("=" * 60)
print(f"Modelo final: DecisionTreeClassifier{grid_search.best_params_}")
print(f"AUC test: {roc_auc_score(y_test, y_test_proba_tuned):.3f}")
print("Variables mas importantes:")
print(importancias_tuned.head(3))
print("""
Nota metodologica: la imputacion de valores faltantes se realizo usando
UNICAMENTE estadisticos (medianas) calculados sobre el conjunto de train,
sin condicionar a la variable Outcome, para evitar fuga de datos hacia
el modelo.
""")