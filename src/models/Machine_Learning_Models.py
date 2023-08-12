# Importar paquetes
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append("..")
from utility import plot_settings

import warnings

warnings.filterwarnings("ignore")

# Machine Learning
## Preprocesamiento
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

## Modelos de Clasificación
import lightgbm as lgb
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

## Métricas de Evaluación
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
    balanced_accuracy_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
)


# Carga de datos
data = pd.read_csv("../data/Customer_Churn.csv", sep=";")

# Eliminamos la variable que no aporta información relevante (identificador de cliente)
df = data.copy()
df.drop(columns="CustomerID", inplace=True)

data = data.dropna()

numeric_cols = [
    "Age",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

data_num = data[numeric_cols]
data_num.shape


# Crear instancia StandardScaler
scaler = StandardScaler()

# Estandarizar las variables
X_scaled = scaler.fit_transform(data_num)

# Dimensiones de los datos estandarizados
X_scaled.shape


cat_cols = ["Gender", "Subscription Type", "Contract Length"]

data_cat = data[cat_cols]
data_cat.shape


# Codificación
encoder = OneHotEncoder(sparse_output=False, drop="first")
X_encoded = encoder.fit_transform(data_cat)

# Cantidad de características generadas
X_encoded.shape


# Características categóricas codificadas
encoder.get_feature_names_out()


# Características a modelar
X = np.concatenate((X_scaled, X_encoded), axis=1)
X.shape


# Conservar nombres de las características
feature_names = numeric_cols + encoder.get_feature_names_out().tolist()

encoder_y = OneHotEncoder(sparse_output=False, drop="if_binary")
y = encoder_y.fit_transform(data["Churn"].values.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.8, random_state=31415
)

print("\nShape de los conjuntos de entrenamiento:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("\nShape de los conjuntos de prueba:")
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# Graficar antes
plt.figure(figsize=(6, 3))
sns.countplot(data=df, x="Churn", palette="viridis")
plt.title("Costumer Churn Count Distribution (before)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Balanceo con sobremuestreo SMOTE
smote = SMOTE(k_neighbors=5, random_state=42)
X_train_balance, y_train_balance = smote.fit_resample(X_train, y_train)


# Graficar luego
class_counts = pd.Series(y_train_balance).value_counts()

plt.figure(figsize=(6, 3))
plt.bar(class_counts.index, class_counts, color=sns.color_palette("viridis"))
plt.xticks(range(2))
plt.title("Costumer Churn Count Distribution (after)")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.xticks(rotation=45)
plt.show()


# Define classifiers
classifiers = {
    "XGBoost": XGBClassifier(
        learning_rate=0.01,
        colsample_bytree=0.4,
        subsample=0.8,
        objective="binary:logistic",
        n_estimators=1000,
        reg_alpha=0.3,
        max_depth=4,
        gamma=10,
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=1000,
        max_depth=4,
    ),
    "LightGBM": lgb.LGBMClassifier(
        learning_rate=0.01,
        n_estimators=100,
        max_depth=4,
    ),
}


roc_data = []

# Fit, predict and compute ROC curve for each classifier
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute ROC curve and ROC area
    fpr, tpr, _ = roc_curve(y_test, clf.predict_proba(X_test)[:, 1])
    roc_auc = auc(fpr, tpr)

    # Save ROC data
    roc_data.append((fpr, tpr, roc_auc, name))

    # Get the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Truth")
    plt.title(f"{name} Confusion Matrix")
    plt.show()


# Initialize the plot for the ROC curve
plt.figure(figsize=(10, 7))

# Plot ROC curve for each classifier
for fpr, tpr, roc_auc, name in roc_data:
    plt.plot(fpr, tpr, label="%s ROC curve (area = %0.2f)" % (name, roc_auc))

# Finalize the plot for the ROC curve
plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic")
plt.legend(loc="lower right")
plt.show()


plt.figure(figsize=(10, 7))

# Compute Precision-Recall and plot curve for each classifier
for name, clf in classifiers.items():
    y_score = clf.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, y_score)
    average_precision = average_precision_score(y_test, y_score)

    plt.plot(
        recall,
        precision,
        label="%s Precision-Recall curve (area = %0.2f)" % (name, average_precision),
    )

# Set labels and legend
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title("Precision-Recall curve")
plt.legend(loc="lower left")

plt.show()


## KNN
# Definir una malla de hiperparámetros
knn_grid = {"n_neighbors": range(4, 6)}

# Establecer el GS-CV
knn_gs_cv = GridSearchCV(
    KNeighborsClassifier(),
    param_grid=knn_grid,
    cv=5,
    verbose=1,
    scoring="recall",
    return_train_score=True,
)

knn_gs_cv.fit(X_train_balance, y_train_balance)

# Create a Logistic Regression classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)
# Prediction
y_pred = clf.predict(X_test)

print("Accurcy: ", accuracy_score(y_test, y_pred))
print(metrics.classification_report(y_test, y_pred))


knn_gs_cv.best_score_
knn_gs_csv.best_params_

y_pred_knn = knn_gs_cv.best_estimator_.predict(X_test)
print("REPORTES DE CLASIFICACIÓN (PRUEBA)")
print("2. KNN")
print(classification_report(y_test, y_pred_knn))
print(f'{"-"*60}\n')


# Create a Multi-layer Perceptron classifier
clf = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)


clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Classification Report:\n", report)
