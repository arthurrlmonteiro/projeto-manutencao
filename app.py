"""
app.py  —  Demonstração interativa (Streamlit)
==============================================
App de apresentação do protótipo de Manutenção Preditiva de equipamentos
críticos de embarcações offshore (Duke / AI Product Manager — Projeto 2).

Reaproveita TODA a lógica de dados e modelos de `src/utils.py` — este arquivo
apenas constrói a interface e coloca cache nas partes pesadas para rodar bem
no Streamlit Community Cloud (plano grátis, ~1 GB de RAM).

Observação: a interface (texto visível) está em inglês; os comentários e os
nomes de funções/variáveis seguem em português (reaproveitados de utils.py).

Como rodar localmente:
    streamlit run app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Coloca a pasta src/ no caminho de import (utils.py vive lá).
SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.model_selection import cross_val_score, GroupKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from utils import (  # noqa: E402  (import após ajustar sys.path)
    carregar_dados, carregar_split, obter_modelos,
    FEATURES_NUM, FEATURES_CAT, ALVO,
)

sns.set_theme(style="whitegrid")

MODELO_FINAL = "Regressão Logística"   # vencedor documentado no README
METRICA = "average_precision"          # PR-AUC
RECALL_ALVO = 0.80                     # capturar ao menos 80% das falhas

# Rótulos em inglês p/ EXIBIÇÃO (as chaves originais vêm de utils.py / dos dados).
MODEL_DISPLAY = {
    "Regressão Logística": "Logistic Regression",
    "Random Forest": "Random Forest",
    "Gradient Boosting": "Gradient Boosting",
}
EQUIP_DISPLAY = {"motor": "Motor", "bomba": "Pump", "thruster": "Thruster"}

st.set_page_config(
    page_title="Predictive Maintenance — Offshore",
    page_icon="⚙️",
    layout="wide",
)


# ----------------------------------------------------------------------
# Funções com cache (carregadas uma única vez por sessão/contêiner)
# ----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dados() -> pd.DataFrame:
    return carregar_dados()


@st.cache_data(show_spinner=False)
def get_split():
    return carregar_split()


@st.cache_resource(show_spinner=False)
def get_modelo_final():
    """Treina o modelo vencedor (Regressão Logística) no conjunto de treino."""
    X_train, _, y_train, _, _ = get_split()
    modelo = obter_modelos()[MODELO_FINAL]
    modelo.fit(X_train, y_train)
    return modelo


@st.cache_data(show_spinner=False)
def get_comparacao_modelos() -> pd.DataFrame:
    """Validação cruzada 5-fold POR EQUIPAMENTO dos 3 modelos (PR-AUC)."""
    X_train, _, y_train, _, grupos_train = get_split()
    cv = GroupKFold(n_splits=5)
    linhas = []
    for nome, modelo in obter_modelos().items():
        scores = cross_val_score(
            modelo, X_train, y_train, groups=grupos_train,
            scoring=METRICA, cv=cv, n_jobs=-1,
        )
        linhas.append({
            "Model": MODEL_DISPLAY.get(nome, nome),
            "Mean_PR_AUC": scores.mean(),
            "PR_AUC_std": scores.std(),
        })
    return (pd.DataFrame(linhas)
            .sort_values("Mean_PR_AUC", ascending=False)
            .reset_index(drop=True))


@st.cache_data(show_spinner=False)
def get_avaliacao():
    """Avalia o modelo final no TESTE e escolhe o limiar p/ recall >= 80%."""
    modelo = get_modelo_final()
    _, X_test, _, y_test, _ = get_split()
    y_prob = modelo.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    prec, rec, lim = precision_recall_curve(y_test, y_prob)
    candidatos = [(p, r, t) for p, r, t in zip(prec[:-1], rec[:-1], lim)
                  if r >= RECALL_ALVO]
    limiar = max(candidatos, key=lambda x: x[0])[2] if candidatos else 0.5

    y_pred = (y_prob >= limiar).astype(int)
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "limiar": float(limiar),
        "precisao": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "prec_curve": prec,
        "rec_curve": rec,
        "cm": confusion_matrix(y_test, y_pred),
    }


def importancias(modelo) -> pd.Series | None:
    """Extrai importâncias/coeficientes do classificador dentro do pipeline."""
    prep = modelo.named_steps["prep"]
    clf = modelo.named_steps["clf"]
    nomes = prep.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=nomes).sort_values()
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_[0]), index=nomes).sort_values()
    return None


# ----------------------------------------------------------------------
# Cabeçalho
# ----------------------------------------------------------------------
st.title("⚙️ Predictive Maintenance of Critical Equipment — Offshore")
st.caption(
    "Prototype for **Project 2** — *Managing Machine Learning Projects* "
    "(Duke / AI Product Management). Synthetic data for demonstration."
)

df = get_dados()

aba_visao, aba_eda, aba_modelagem, aba_avaliacao, aba_previsao = st.tabs([
    "📌 Overview",
    "📊 EDA",
    "🤖 Modeling",
    "✅ Evaluation",
    "🔮 Prediction",
])


# ----------------------------------------------------------------------
# Aba 1 — Overview
# ----------------------------------------------------------------------
with aba_visao:
    st.subheader("The problem")
    st.markdown(
        """
        Predict the **failure of critical rotating equipment** (engines, pumps,
        thrusters) within the next **72 hours**, from vibration, temperature and
        pressure sensors.

        - **ML task:** binary classification with imbalanced classes.
        - **User:** maintenance / reliability manager (onshore).
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Readings", f"{len(df):,}")
    c2.metric("Equipment units", df["equipment_id"].nunique())
    c3.metric("Failures (positive class)", f"{df[ALVO].mean():.1%}")

    st.divider()
    st.subheader("Method decisions (and why they matter)")
    st.markdown(
        """
        - **Imbalanced classes.** Failures are ~3% of the readings. We use
          `class_weight="balanced"` and evaluate with **PR-AUC, recall and
          precision** — never accuracy (predicting "never fails" would give ~97%
          and be useless).
        - **Split by equipment (no leakage).** Consecutive readings from the same
          unit are almost identical. We split **by equipment**
          (`GroupShuffleSplit` / `GroupKFold`): the model is tested on units it
          has never seen — like a new ship joining the fleet.
        - **Threshold chosen by recall.** Missing a failure costs more than a
          false alarm, so we fix the threshold to capture ≥ 80% of failures and
          report the resulting precision.
        - **Baseline first.** We compare Logistic Regression, Random Forest and
          Gradient Boosting. On the sample data, **Logistic Regression**
          generalized best to new equipment.
        """
    )

    st.divider()
    st.subheader("Data sample")
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------------------------------------------------
# Aba 2 — EDA
# ----------------------------------------------------------------------
with aba_eda:
    st.subheader("Data understanding (Phase 2 — CRISP-DM)")

    st.markdown("**Descriptive statistics of the sensors**")
    st.dataframe(df[FEATURES_NUM].describe().round(2), use_container_width=True)

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown("**Class balance (failures are rare)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x=ALVO, color="steelblue", ax=ax)
        ax.set_xlabel("0 = normal   |   1 = failure within next 72h")
        ax.set_ylabel("Number of readings")
        st.pyplot(fig)
        plt.close(fig)

    with col_dir:
        st.markdown("**Correlation between sensors and the target**")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df[FEATURES_NUM + [ALVO]].corr(), annot=True,
                    cmap="coolwarm", fmt=".2f", square=True,
                    linewidths=0.5, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Sensors: healthy equipment (0) vs. about to fail (1)**")
    principais = ["vibration_rms", "bearing_temp", "oil_temp", "oil_pressure"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feat in zip(axes.ravel(), principais):
        sns.kdeplot(data=df, x=feat, hue=ALVO, common_norm=False,
                    fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"{feat} by class")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Aba 3 — Modeling
# ----------------------------------------------------------------------
with aba_modelagem:
    st.subheader("Model comparison (Phases 3 and 4 — CRISP-DM)")
    st.markdown(
        """
        5-fold cross-validation **by equipment** (`GroupKFold`, no leakage).
        Metric: **PR-AUC** (*average precision*) — suited to imbalanced classes,
        unlike accuracy.

        > The comparison trains 3 models (incl. Random Forest with 300 trees) and
        > may take up to ~1 min on the first run. The result is cached.
        """
    )

    if st.button("▶️ Run comparison", type="primary"):
        with st.spinner("Running cross-validation of the 3 models…"):
            tabela = get_comparacao_modelos()
        st.session_state["comparacao"] = tabela

    tabela = st.session_state.get("comparacao")
    if tabela is not None:
        melhor = tabela.iloc[0]["Model"]
        st.success(f"Best model in validation: **{melhor}**")

        col_esq, col_dir = st.columns([1, 1])
        with col_esq:
            st.dataframe(
                tabela.style.format(
                    {"Mean_PR_AUC": "{:.3f}", "PR_AUC_std": "{:.3f}"}
                ),
                use_container_width=True,
            )
        with col_dir:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=tabela, x="Mean_PR_AUC", y="Model",
                        color="steelblue", ax=ax)
            ax.errorbar(tabela["Mean_PR_AUC"], range(len(tabela)),
                        xerr=tabela["PR_AUC_std"], fmt="none",
                        c="black", capsize=4)
            ax.set_xlabel("Mean PR-AUC (CV) — higher is better")
            ax.set_title("Model comparison")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Click **Run comparison** to train and compare the models.")


# ----------------------------------------------------------------------
# Aba 4 — Evaluation
# ----------------------------------------------------------------------
with aba_avaliacao:
    st.subheader("Final evaluation on the test set (Phase 5 — CRISP-DM)")
    st.markdown(
        f"Final model: **{MODEL_DISPLAY[MODELO_FINAL]}**, evaluated on "
        f"**never-seen** equipment. Threshold chosen for "
        f"**recall ≥ {RECALL_ALVO:.0%}**."
    )

    with st.spinner("Evaluating the model on the test set…"):
        av = get_avaliacao()

    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", f"{av['pr_auc']:.3f}")
    c2.metric("ROC-AUC", f"{av['roc_auc']:.3f}")
    c3.metric("Threshold", f"{av['limiar']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", f"{av['recall']:.1%}", help="% of real failures captured")
    c5.metric("Precision", f"{av['precisao']:.1%}", help="% of alerts that were real failures")
    c6.metric("F1", f"{av['f1']:.3f}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Precision-Recall curve**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(av["rec_curve"], av["prec_curve"], color="steelblue",
                label=f"PR-AUC = {av['pr_auc']:.3f}")
        ax.scatter([av["recall"]], [av["precisao"]], color="red", zorder=5,
                   label=f"Chosen threshold (recall {av['recall']:.0%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**Confusion matrix**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(av["cm"], annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Normal", "Failure"],
                    yticklabels=["Normal", "Failure"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Feature importance for failure prediction**")
    imp = importancias(get_modelo_final())
    if imp is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=imp.values, y=imp.index, color="seagreen", ax=ax)
        ax.set_xlabel("Relative importance (|coefficient|)")
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Aba 5 — Prediction
# ----------------------------------------------------------------------
with aba_previsao:
    st.subheader("Interactive failure prediction")
    st.markdown(
        "Adjust the sensor values and see the estimated probability of "
        "**failure within the next 72 hours**. Tip: higher **vibration** and "
        "**temperature**, and lower **oil pressure**, indicate degradation."
    )

    modelo = get_modelo_final()
    av = get_avaliacao()
    limiar = av["limiar"]
    desc = df[FEATURES_NUM].describe()

    tipo = st.selectbox(
        "Equipment type",
        sorted(df["equipment_type"].unique()),
        format_func=lambda x: EQUIP_DISPLAY.get(x, x),
    )

    valores = {}
    cols = st.columns(2)
    for i, feat in enumerate(FEATURES_NUM):
        lo = float(desc.loc["min", feat])
        hi = float(desc.loc["max", feat])
        med = float(desc.loc["50%", feat])
        passo = max((hi - lo) / 100, 0.01)
        with cols[i % 2]:
            valores[feat] = st.slider(
                feat, min_value=round(lo, 2), max_value=round(hi, 2),
                value=round(med, 2), step=round(passo, 2),
            )

    entrada = pd.DataFrame([{**valores, "equipment_type": tipo}])
    entrada = entrada[FEATURES_NUM + FEATURES_CAT]  # ordem esperada pelo pipeline

    prob = float(modelo.predict_proba(entrada)[0, 1])

    st.divider()
    col_p, col_d = st.columns([1, 2])
    col_p.metric("Failure probability (72h)", f"{prob:.1%}")
    if prob >= limiar:
        col_d.error(
            f"🔴 **ALERT** — probability ({prob:.1%}) ≥ decision threshold "
            f"({limiar:.1%}). Recommend inspection/maintenance."
        )
    else:
        col_d.success(
            f"🟢 **OK** — probability ({prob:.1%}) < decision threshold "
            f"({limiar:.1%}). No immediate action."
        )
    st.progress(min(prob, 1.0))
    st.caption(
        f"The {limiar:.1%} threshold was chosen during evaluation to capture "
        f"≥ {RECALL_ALVO:.0%} of real failures."
    )
