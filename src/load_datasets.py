import pandas as pd
import numpy as np
import os

# ======================================
# Configuração do numpy
# ======================================

# remove notação científica ao imprimir números
np.set_printoptions(suppress=True)

# ======================================
# Estruturas principais
# ======================================

# dicionário com os dados (apenas atributos)
datasets = {}

# dicionário com os rótulos verdadeiros
ground_truth = {}

# dicionário com número de clusters verdadeiros
n_clusters = {}

# ======================================
# Função para transformar rótulos
# ======================================

def encode_labels(series):
    """
    Converte rótulos categóricos em inteiros.

    Exemplo: ['A','B','A'] -> [1,2,1]

    Retorna numpy array.
    """
    return pd.factorize(series)[0] + 1

# ======================================
# Função genérica de carregamento
# ======================================

def load_dataset(path, name, features, label, sep=",", header=None):
    """
    Carrega um dataset e separa atributos e rótulos.

    Parâmetros
    ----------
    path : caminho do arquivo
    name : nome usado no dicionário
    features : colunas dos atributos
    label : coluna do rótulo
    sep : separador do arquivo
    header : indica se existe cabeçalho
    """

    # leitura do arquivo
    df = pd.read_csv(path, sep=sep, header=header)

    # matriz de atributos (X)
    datasets[name] = df.iloc[:, features].to_numpy()

    # vetor de rótulos verdadeiros (y)
    if isinstance(label, int):
        ground_truth[name] = encode_labels(df.iloc[:, label])
    else:
        ground_truth[name] = np.array(label)

    # calcula número de clusters reais
    n_clusters[name] = len(np.unique(ground_truth[name]))

    # print de verificação
    print(f"{name}: X{datasets[name].shape} | clusters={n_clusters[name]}")

# ======================================
# Caminhos das pastas
# ======================================

shape_path = os.path.join("datasets", "Shape sets")
uci_path = os.path.join("datasets", "UCI")

# ======================================
# Shape datasets
# (2 atributos + 1 coluna de rótulo)
# ======================================

load_dataset(os.path.join(shape_path,"flame.txt"), "dataset1", slice(0,2), 2, sep=r"\s+", header=None)
load_dataset(os.path.join(shape_path,"pathbased.txt"), "dataset2", slice(0,2), 2, sep=r"\s+", header=None)
load_dataset(os.path.join(shape_path,"spiral.txt"), "dataset3", slice(0,2), 2, sep=r"\s+", header=None)
load_dataset(os.path.join(shape_path,"jain.txt"), "dataset4", slice(0,2), 2, sep=r"\s+", header=None)
load_dataset(os.path.join(shape_path,"Compound.txt"), "dataset5", slice(0,2), 2, sep=r"\s+", header=None)
load_dataset(os.path.join(shape_path,"R15.txt"), "dataset6", slice(0,2), 2, sep=r"\s+", header=None)
load_dataset(os.path.join(shape_path,"Aggregation.txt"), "dataset7", slice(0,2), 2, sep=r"\s+", header=None)

# ======================================
# UCI datasets
# ======================================

# Iris
load_dataset(os.path.join(uci_path,"iris.txt"), "dataset8", slice(0,4), 4, sep=",", header=None)

# Wine
load_dataset(os.path.join(uci_path,"wine.txt"), "dataset9", slice(1,14), 0, sep=",", header=None)

# Sonar
load_dataset(os.path.join(uci_path,"sonar.txt"), "dataset10", slice(0,60), 60, sep=",", header=None)

# Glass
load_dataset(os.path.join(uci_path,"glass.txt"), "dataset11", slice(1,10), 10, sep=",", header=None)

# New thyroid
load_dataset(os.path.join(uci_path,"new-thyroid.txt"), "dataset12", slice(1,6), 0, sep=",", header=None)

# Seeds
load_dataset(os.path.join(uci_path,"seeds_dataset.csv"), "dataset13", slice(0,7), 7, sep=";", header=None)

# Heart failure (possui header)
load_dataset(os.path.join(uci_path,"heart_failure_clinical_records_dataset.csv"), "dataset14", slice(0,12), 12, sep=",", header=0)

# Ecoli
load_dataset(os.path.join(uci_path,"ecoli.csv"), "dataset15", slice(1,8), 8, sep=";", header=None)

# Ionosphere
load_dataset(os.path.join(uci_path,"ionosphere.txt"), "dataset16", [0]+list(range(2,34)), 34, sep=",", header=None)

# Libras
load_dataset(os.path.join(uci_path,"libras.txt"), "dataset17", slice(0,90), 90, sep=",", header=None)

# WDBC
load_dataset(os.path.join(uci_path,"wdbc.txt"), "dataset18", slice(2,32), 1, sep=",", header=None)

# Synthetic control
load_dataset(
    os.path.join(uci_path,"synthetic_control.csv"),
    "dataset19",
    slice(None),
    [i for i in range(1,7) for _ in range(100)],
    sep=";",
    header=None
)

# Yeast
load_dataset(os.path.join(uci_path,"yeast.csv"), "dataset20", slice(1,9), 9, sep=";", header=None)