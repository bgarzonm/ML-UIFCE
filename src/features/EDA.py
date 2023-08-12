# Importar paquetes
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

import sys

sys.path.append("..")
from utility import plot_settings


import warnings

warnings.filterwarnings("ignore")

# Carga de datos
data = pd.read_csv("../data/Customer_Churn.csv", sep=";")

# Eliminamos la variable que no aporta información relevante (identificador de cliente)
df = data.copy()
df.drop(columns="CustomerID", inplace=True)


# plot_cat: función para realizar análisis de frecuencias
def plot_cat(df, col):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    sns.countplot(data=df, x=col, palette="viridis")
    plt.title(f"Count of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.xticks(rotation=45)

    plt.subplot(1, 2, 2)
    counts = df[col].value_counts()
    plt.pie(
        counts,
        labels=counts.index,
        colors=sns.color_palette("viridis"),
        autopct="%1.1f%%",
        startangle=90,
    )
    plt.axis("equal")
    plt.title(f"Distribution of {col}")

    plt.tight_layout()
    plt.show()


cat_cols = ["Churn", "Gender", "Subscription Type", "Contract Length"]
for col in cat_cols:
    plot_cat(df, col)


# Variables numéricas
# plot_num: función para realizar análisis de distribución (histograma y boxplot)
def plot_num(df, col, bins):
    fig, axes = plt.subplots(figsize=(12, 4), ncols=2)

    sns.histplot(data=df, x=col, kde=True, color="#2E86C1", ax=axes[0], bins=bins)
    axes[0].set_title(f"Distribution of {col}")
    axes[0].set_xlabel(col)
    axes[0].set_ylabel("Count")

    sns.boxplot(data=df, x=col, color="#2E86C1", ax=axes[1])
    axes[1].set_title(f"Boxplot of {col}")
    axes[1].set_xlabel(col)

    plt.tight_layout()
    plt.show()


numeric_cols = [
    "Age",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]
n_bins = 10

for col in numeric_cols:
    plot_num(df, col, n_bins)


def describe(df, stats):
    d = df.describe()
    custom_stats = df.agg(stats)
    return pd.concat([d, custom_stats], axis=0)


# Resumen descriptivo para variables numéricas
describe(df[numeric_cols], ["median", "kurtosis", "skew"])


# plot_hist_churn: función para realizar un histograma en función de la variable objetivo
def plot_hist_churn(df, col, ax=None):
    sns.histplot(
        data=df, x=col, hue="Churn", ax=ax, bins=n_bins, palette="dark", legend=False
    )
    plt.legend(title="", labels=["Huyó", "No huyó"])
    plt.xlabel(col)
    plt.ylabel("Count")

    plt.tight_layout()


# Malla de gráficos: histogramas de distribución
## Configuración general del gráfico
f, axes = plt.subplots(figsize=(14, 12), nrows=3, ncols=2)
plt.suptitle("Distribution according to costumer churn", size=20)

## Sub-gráficos
for index, col in enumerate(numeric_cols):
    plot_hist_churn(df, col, axes[index // 2, index % 2])

plt.show()


def correlation_heatmap(df):
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5
    )
    plt.title("Correlation Matrix of Numeric Variables")
    plt.show()


correlation_heatmap(df[numeric_cols])
