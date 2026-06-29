"""
utils.py
--------
Funções e constantes compartilhadas entre as fases do projeto.

Caso de uso: prever a falha de equipamentos críticos (motores, bombas,
thrusters) de embarcações offshore nas próximas 72h.
Tarefa.....: Classificação binária (vai falhar no horizonte? sim/não).

Pontos importantes de método:
  - Classes desbalanceadas (falhas são raras) => usamos class_weight e
    avaliamos com PR-AUC / recall / precisão, NÃO com acurácia.
  - Leituras consecutivas do mesmo equipamento são muito parecidas. Um
    split aleatório vazaria informação entre treino e teste. Por isso
    dividimos POR EQUIPAMENTO (GroupShuffleSplit / GroupKFold): o modelo
    é testado em equipamentos que nunca viu.
"""

from pathlib import Path
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

# ----------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "data"
OUTPUTS = RAIZ / "outputs"
OUTPUTS.mkdir(exist_ok=True)

# ----------------------------------------------------------------------
# Configurações globais
# ----------------------------------------------------------------------
SEED = 42
TEST_SIZE = 0.25            # 25% dos EQUIPAMENTOS para teste
HORIZONTE_H = 72           # prever falha nas próximas 72 horas

ALVO = "failure_within_72h"
GRUPO = "equipment_id"      # usado para dividir sem vazamento
FEATURES_NUM = [
    "vibration_rms", "vibration_peak", "bearing_temp", "oil_temp",
    "oil_pressure", "rpm", "load_pct", "operating_hours",
]
FEATURES_CAT = ["equipment_type"]


def carregar_dados(nome_arquivo: str = "sensores_historico.csv") -> pd.DataFrame:
    """Carrega o CSV de sensores."""
    caminho = DATA / nome_arquivo
    if not caminho.exists():
        raise FileNotFoundError(
            f"\n>>> Arquivo não encontrado: {caminho}\n"
            f">>> Gere os dados de exemplo antes: python src/00_gerar_dados.py\n"
        )
    return pd.read_csv(caminho, parse_dates=["timestamp"])


def carregar_split():
    """Divide os dados POR EQUIPAMENTO (sem vazamento).
    Retorna X_train, X_test, y_train, y_test e os grupos do treino
    (necessários para a validação cruzada por grupo)."""
    df = carregar_dados()
    X = df[FEATURES_NUM + FEATURES_CAT]
    y = df[ALVO]
    grupos = df[GRUPO]

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=SEED)
    idx_train, idx_test = next(gss.split(X, y, grupos))

    return (
        X.iloc[idx_train], X.iloc[idx_test],
        y.iloc[idx_train], y.iloc[idx_test],
        grupos.iloc[idx_train],
    )


def obter_preprocessador() -> ColumnTransformer:
    """Padroniza as variáveis numéricas e faz one-hot da categórica."""
    return ColumnTransformer([
        ("num", StandardScaler(), FEATURES_NUM),
        ("cat", OneHotEncoder(handle_unknown="ignore"), FEATURES_CAT),
    ])


def obter_modelos() -> dict:
    """Modelos candidatos (todos tratam o desbalanceamento via class_weight)."""
    prep = obter_preprocessador
    return {
        "Regressão Logística": Pipeline([
            ("prep", prep()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep()),
            ("clf", RandomForestClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=SEED, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", prep()),
            ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=SEED)),
        ]),
    }


# Grades de hiperparâmetros para o ajuste fino do modelo vencedor
GRADES = {
    "Random Forest": {
        "clf__n_estimators": [300, 500],
        "clf__max_depth": [None, 12, 20],
        "clf__min_samples_leaf": [1, 5],
    },
    "Gradient Boosting": {
        "clf__learning_rate": [0.05, 0.1],
        "clf__max_depth": [None, 6, 10],
        "clf__max_iter": [200, 400],
    },
    # Regressão Logística: sem ajuste relevante neste exemplo.
}
