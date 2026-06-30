"""
app.py  —  Demonstração interativa (Streamlit)
==============================================
App de apresentação do protótipo de Manutenção Preditiva de equipamentos
críticos de embarcações offshore (Duke / AI Product Manager — Projeto 2).

Reaproveita TODA a lógica de dados e modelos de `src/utils.py` — este arquivo
apenas constrói a interface e coloca cache nas partes pesadas para rodar bem
no Streamlit Community Cloud (plano grátis, ~1 GB de RAM).

A navegação é por barra lateral (não por abas) de propósito: assim apenas a
seção selecionada é executada a cada recarregamento, mantendo o uso de memória
baixo o suficiente para o plano grátis.

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
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from utils import (  # noqa: E402  (import após ajustar sys.path)
    carregar_dados, carregar_split, obter_modelos, obter_preprocessador,
    FEATURES_NUM, FEATURES_CAT, ALVO, SEED,
)

sns.set_theme(style="whitegrid")

MODELO_FINAL = "Regressão Logística"   # vencedor documentado no README
METRICA = "average_precision"          # PR-AUC
RECALL_ALVO = 0.80                     # capturar ao menos 80% das falhas
N_AMOSTRA_KDE = 8000                   # amostra p/ os KDEs (alivia memória)
N_AMOSTRA_CMP = 12000                  # amostra de treino p/ a comparação no app

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


def _modelos_leves() -> dict:
    """Versões enxutas dos 3 modelos, só para a comparação DENTRO do app —
    cabem na memória do plano grátis. Reaproveitam o mesmo pré-processamento
    de utils.py. A comparação completa (RF 300 árvores) está em
    src/02_modelagem.py."""
    prep = obter_preprocessador
    return {
        "Regressão Logística": Pipeline([
            ("prep", prep()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
        "Random Forest": Pipeline([
            ("prep", prep()),
            ("clf", RandomForestClassifier(
                n_estimators=120, max_depth=16, class_weight="balanced",
                random_state=SEED, n_jobs=1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("prep", prep()),
            ("clf", HistGradientBoostingClassifier(
                class_weight="balanced", random_state=SEED, max_iter=150)),
        ]),
    }


@st.cache_data(show_spinner=False)
def get_comparacao_modelos() -> pd.DataFrame:
    """Validação cruzada 5-fold POR EQUIPAMENTO dos 3 modelos (PR-AUC).
    Usa uma amostra do treino e modelos enxutos, com folds em sequência
    (n_jobs=1), para não estourar a memória do plano grátis."""
    X_train, _, y_train, _, grupos_train = get_split()

    # Amostra o treino (mantém os grupos/equipamentos) p/ aliviar a memória.
    if len(X_train) > N_AMOSTRA_CMP:
        idx = X_train.sample(n=N_AMOSTRA_CMP, random_state=SEED).index
        X_train = X_train.loc[idx]
        y_train = y_train.loc[idx]
        grupos_train = grupos_train.loc[idx]

    cv = GroupKFold(n_splits=5)
    linhas = []
    for nome, modelo in _modelos_leves().items():
        scores = cross_val_score(
            modelo, X_train, y_train, groups=grupos_train,
            scoring=METRICA, cv=cv, n_jobs=1,
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


@st.cache_data(show_spinner=False)
def get_predicoes_teste() -> pd.DataFrame:
    """Probabilidades do modelo final no TESTE, com tipo de equipamento e
    embarcação — base para a análise de subgrupo (viés) e a calibração.
    vessel_id é recuperado do df original pelo índice (carregar_split usa
    .iloc, então o índice de X_test alinha com o df)."""
    modelo = get_modelo_final()
    df_full = get_dados()
    _, X_test, _, y_test, _ = get_split()
    y_prob = modelo.predict_proba(X_test)[:, 1]
    return pd.DataFrame({
        "y_true": y_test.to_numpy(),
        "y_prob": y_prob,
        "equipment_type": X_test["equipment_type"].to_numpy(),
        "vessel_id": df_full.loc[X_test.index, "vessel_id"].to_numpy(),
    })


@st.cache_data(show_spinner=False)
def get_calibracao(n_bins: int = 10):
    """Reliability diagram: fração observada de falhas vs. probabilidade média
    prevista, por faixa (estratégia por quantis devido ao desbalanceamento)."""
    pred = get_predicoes_teste()
    frac_pos, mean_pred = calibration_curve(
        pred["y_true"], pred["y_prob"], n_bins=n_bins, strategy="quantile")
    return mean_pred, frac_pos


def _limpar_nomes(nomes) -> list:
    """Remove os prefixos do ColumnTransformer (num__/cat__) para exibição."""
    return [n.split("__", 1)[-1] for n in nomes]


def explicar_previsao(modelo, entrada: pd.DataFrame) -> pd.Series:
    """Contribuição de cada variável ao log-odds DESTA previsão (modelo linear):
    coef_j * valor_transformado_j. Positivo empurra para 'falha'; negativo,
    para 'normal'."""
    prep = modelo.named_steps["prep"]
    clf = modelo.named_steps["clf"]
    x = prep.transform(entrada)
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).ravel()
    contrib = clf.coef_[0] * x
    return pd.Series(contrib, index=_limpar_nomes(prep.get_feature_names_out()))


def metricas_por_subgrupo(pred: pd.DataFrame, coluna: str, limiar: float) -> pd.DataFrame:
    """Recall/Precision/PR-AUC por subgrupo no teste (para análise de viés).
    Métricas que exigem as duas classes ficam NaN se o subgrupo não as tiver."""
    linhas = []
    for valor, g in pred.groupby(coluna):
        y = g["y_true"].to_numpy()
        p = g["y_prob"].to_numpy()
        yhat = (p >= limiar).astype(int)
        n_falhas = int(y.sum())
        tem_2_classes = 0 < n_falhas < len(g)
        linhas.append({
            coluna: EQUIP_DISPLAY.get(valor, valor),
            "n": len(g),
            "Failures": n_falhas,
            "Recall": recall_score(y, yhat, zero_division=0) if n_falhas else np.nan,
            "Precision": precision_score(y, yhat, zero_division=0),
            "PR_AUC": average_precision_score(y, p) if tem_2_classes else np.nan,
        })
    return pd.DataFrame(linhas).sort_values("Recall").reset_index(drop=True)


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
# Cabeçalho + navegação (barra lateral)
# ----------------------------------------------------------------------
st.title("⚙️ Predictive Maintenance of Critical Equipment — Offshore")
st.caption(
    "Prototype for **Project 2** — *Managing Machine Learning Projects* "
    "(Duke / AI Product Management). Synthetic data for demonstration."
)

df = get_dados()

PAGINAS = [
    "📌 Overview", "📊 EDA", "🤖 Modeling", "✅ Evaluation",
    "🔮 Prediction", "⚖️ Fairness", "🛡️ Responsible AI",
]
st.sidebar.header("Navigation")
pagina = st.sidebar.radio("Section", PAGINAS, label_visibility="collapsed")
st.sidebar.caption("Predictive maintenance — 72h failure horizon.")


# ----------------------------------------------------------------------
# Overview
# ----------------------------------------------------------------------
if pagina == "📌 Overview":
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
    st.subheader("Task analysis")
    st.markdown(
        """
        **User & goal.** An onshore reliability manager oversees many units
        across a fleet and must decide **which to inspect next**, before a
        failure causes downtime or a safety incident at sea.

        **Current task flow (without ML).** Watch raw sensor dashboards →
        eyeball thresholds per gauge → react once a reading already looks bad →
        schedule maintenance. It is **manual, reactive, and easy to miss** weak
        early signals spread across several sensors.

        **How ML reshapes the flow.** The model fuses all sensors into a single
        **72-hour failure risk**, turning the task from *"watch every gauge"*
        into *"triage a ranked early-warning list."* Insights driving the design:

        - **Prioritize, don't replace.** The output is a **risk score + alert**,
          not an auto-shutdown — the manager stays the decision-maker.
        - **Recall-first.** Missing a failure is costlier than a false alarm, so
          the task is tuned to **catch ≥ 80% of failures** and surface them early.
        - **Explain to earn trust.** Each alert ships with **why** (which sensors
          drove it) and **how confident** it is, so the human can sanity-check it.
        - **Close the loop.** The manager confirms whether an alert was right,
          feeding monitoring and retraining (see *Prediction* and *Responsible AI*).
        """
    )

    st.divider()
    st.subheader("Data sample")
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------------------------------------------------
# EDA
# ----------------------------------------------------------------------
elif pagina == "📊 EDA":
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
    # Amostra para o KDE — mantém o gráfico representativo sem pesar na memória.
    amostra = df.sample(min(len(df), N_AMOSTRA_KDE), random_state=SEED)
    principais = ["vibration_rms", "bearing_temp", "oil_temp", "oil_pressure"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feat in zip(axes.ravel(), principais):
        sns.kdeplot(data=amostra, x=feat, hue=ALVO, common_norm=False,
                    fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"{feat} by class")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Modeling
# ----------------------------------------------------------------------
elif pagina == "🤖 Modeling":
    st.subheader("Model comparison (Phases 3 and 4 — CRISP-DM)")
    st.markdown(
        """
        5-fold cross-validation **by equipment** (`GroupKFold`, no leakage).
        Metric: **PR-AUC** (*average precision*) — suited to imbalanced classes,
        unlike accuracy.

        > Lightweight in-app comparison (sampled data, smaller models) so it runs
        > within the free tier's memory. The full-strength comparison (Random
        > Forest with 300 trees) lives in `src/02_modelagem.py`. Result is cached.
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
# Evaluation
# ----------------------------------------------------------------------
elif pagina == "✅ Evaluation":
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

    st.divider()
    st.markdown("**Calibration (are the probabilities trustworthy?)**")
    st.markdown(
        "A reliability diagram compares the **predicted** probability with the "
        "**observed** failure frequency. Points near the diagonal mean the "
        "probabilities can be read at face value — important when the number is "
        "shown to a human who must act on it."
    )
    mean_pred, frac_pos = get_calibracao()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed failure frequency")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Prediction
# ----------------------------------------------------------------------
elif pagina == "🔮 Prediction":
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

    # ---- Communicating uncertainty -----------------------------------
    st.divider()
    st.markdown("#### How confident is this estimate?")
    faixa = df[FEATURES_NUM].quantile([0.05, 0.95])
    fora = [f for f in FEATURES_NUM
            if not faixa.loc[0.05, f] <= valores[f] <= faixa.loc[0.95, f]]
    if fora:
        st.warning(
            "🟡 **Lower confidence** — these inputs are outside the typical "
            "training range (5th–95th percentile), so the model is "
            "extrapolating: **" + ", ".join(fora) + "**."
        )
    elif abs(prob - limiar) < 0.05:
        st.info(
            "🟠 **Borderline** — the probability sits close to the decision "
            "threshold; a small change in the sensors could flip the alert."
        )
    else:
        st.success(
            "🟢 **Higher confidence** — all inputs fall within the typical "
            "training range and the probability is clear of the threshold."
        )
    st.caption(
        "The probability is a model **estimate**, not a guarantee. Calibration "
        "is shown in the *Evaluation* section."
    )

    # ---- Transparency: why this prediction? --------------------------
    st.divider()
    st.markdown("#### Why this prediction? (per-sensor contribution)")
    st.markdown(
        "Because the model is **linear**, each sensor's push on the risk is "
        "exact: bars to the **right raise** the failure risk, bars to the "
        "**left lower** it (log-odds contribution for this specific input)."
    )
    contrib = explicar_previsao(modelo, entrada)
    top = (contrib.reindex(contrib.abs().sort_values(ascending=False).index)
           .head(8).iloc[::-1])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    cores = ["#d62728" if v > 0 else "#2ca02c" for v in top.values]
    ax.barh(top.index, top.values, color=cores)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Contribution to failure risk (log-odds) — right ↑ risk, left ↓ risk")
    st.pyplot(fig)
    plt.close(fig)

    # ---- Feedback loop (human-in-the-loop) ---------------------------
    st.divider()
    st.markdown("#### Feedback loop")
    st.markdown(
        "Was this alert useful? Your answer is the **feedback loop** that keeps "
        "the model honest over time."
    )
    st.session_state.setdefault("feedback", [])
    col_ok, col_no, col_cnt = st.columns([1, 1, 2])
    if col_ok.button("👍 Alert was correct"):
        st.session_state["feedback"].append(("correct", round(prob, 4)))
    if col_no.button("👎 Alert was wrong"):
        st.session_state["feedback"].append(("wrong", round(prob, 4)))
    col_cnt.metric("Feedback collected (this session)",
                   len(st.session_state["feedback"]))
    st.caption(
        "Demo only: feedback lives in this browser session. In production it "
        "would be logged to a database and feed **model monitoring and "
        "retraining** — with a human always confirming the decision."
    )


# ----------------------------------------------------------------------
# Fairness (bias by subgroup)
# ----------------------------------------------------------------------
elif pagina == "⚖️ Fairness":
    st.subheader("Fairness — performance across subgroups")
    st.markdown(
        "A single overall score can hide **blind spots**. Here we recompute the "
        "test-set metrics **per equipment type** and **per vessel**. A much "
        "lower **recall** in any group means the model misses more failures "
        "there — a fairness/safety risk to monitor and mitigate."
    )

    av = get_avaliacao()
    pred = get_predicoes_teste()
    limiar = av["limiar"]
    st.caption(
        f"Metrics at the global decision threshold ({limiar:.1%}). "
        f"`NaN` = the subgroup lacks both classes in the test split."
    )

    fmt = {"Recall": "{:.1%}", "Precision": "{:.1%}", "PR_AUC": "{:.3f}"}

    st.markdown("**By equipment type**")
    tab_tipo = metricas_por_subgrupo(pred, "equipment_type", limiar)
    col_t1, col_t2 = st.columns([1.1, 1])
    with col_t1:
        st.dataframe(tab_tipo.style.format(fmt, na_rep="—"), use_container_width=True)
    with col_t2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=tab_tipo, x="Recall", y="equipment_type",
                    color="steelblue", ax=ax)
        ax.axvline(RECALL_ALVO, color="red", ls="--",
                   label=f"Target recall {RECALL_ALVO:.0%}")
        ax.set_xlabel("Recall (failures caught)")
        ax.set_ylabel("")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.divider()
    st.markdown("**By vessel**")
    tab_nav = metricas_por_subgrupo(pred, "vessel_id", limiar)
    col_n1, col_n2 = st.columns([1.1, 1])
    with col_n1:
        st.dataframe(tab_nav.style.format(fmt, na_rep="—"), use_container_width=True)
    with col_n2:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        sns.barplot(data=tab_nav, x="Recall", y="vessel_id",
                    color="seagreen", ax=ax)
        ax.axvline(RECALL_ALVO, color="red", ls="--",
                   label=f"Target recall {RECALL_ALVO:.0%}")
        ax.set_xlabel("Recall (failures caught)")
        ax.set_ylabel("")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    st.info(
        "**How we act on this:** track recall per subgroup over time, raise the "
        "alarm when any group drifts below target, and rebalance / recalibrate / "
        "collect more data for the weakest groups before they cause a missed "
        "failure. See *Responsible AI* for the full mitigation plan."
    )


# ----------------------------------------------------------------------
# Responsible AI (privacy + ethics)
# ----------------------------------------------------------------------
elif pagina == "🛡️ Responsible AI":
    st.subheader("Responsible AI — privacy & ethics")

    st.markdown("### 🔒 Privacy")
    st.markdown(
        """
        **What we collect.** Sensor telemetry (vibration, temperature, oil
        pressure, RPM, load), equipment and vessel identifiers, and operating
        hours/timestamps.

        **Why it can be sensitive.** The readings describe *machines*, but they
        become **personal data** the moment they can be linked to **who was on
        shift** — turning the system into potential **worker monitoring**.
        Vessel operating patterns are also **commercially confidential**.

        **Applicable laws & obligations.**
        - **LGPD (Brazil)** and **GDPR (EU operations/crew):** require a **lawful
          basis**, **purpose limitation** (only for predicting maintenance, not
          for evaluating workers), **data minimization**, **limited retention**,
          and respect for **data-subject rights** (access, correction, deletion).
        - **DPIA / transparency:** because monitoring is involved, run a **Data
          Protection Impact Assessment** and **inform the crew** clearly.
        - **Security:** encryption in transit/at rest, **role-based access**, and
          audit logs of who sees what.

        **Our approach.** Pseudonymize equipment/crew IDs, **decouple** sensor
        data from individual identities by default, aggregate where possible, set
        explicit retention windows, and keep the model's purpose strictly
        maintenance — never silent performance scoring of people.
        """
    )

    st.divider()
    st.markdown("### ⚖️ Ethics — bias sources & mitigations")
    st.markdown(
        """
        | Potential bias source | Why it happens | Mitigation |
        |---|---|---|
        | **Subgroup imbalance** | Some equipment types / vessels have more data | Monitor recall per subgroup (*Fairness* tab); rebalance & collect more data |
        | **New-equipment gap** | A brand-new unit/vessel is under-represented | Group-aware split & validation on **unseen units**; flag low-confidence extrapolation |
        | **Sensor drift / calibration** | Sensors age and read differently across units | Periodic recalibration, drift monitoring, retraining cadence |
        | **Label noise** | "Failure" depends on inconsistent maintenance records | Standardize failure definitions; audit labels |
        | **Operating-condition confounding** | Harsh routes look "always risky" | Include context features; review per-condition performance |
        """
    )

    st.divider()
    st.markdown("### 🎯 Ethical AI goals")
    st.markdown(
        """
        - **Fairness.** Equitable **recall across equipment types and vessels**,
          not just a good overall number — measured live in the *Fairness* tab,
          with alerts when any group drifts below target.
        - **Accountability.** The model **recommends, a human decides**. Every
          alert and override is **logged and auditable**, the model is
          **versioned with a model card**, and ownership of decisions is explicit.
        - **Transparency.** Each prediction ships with **per-sensor explanations**,
          global **feature importance**, a **calibration** check, an **uncertainty
          flag** for out-of-range inputs, and clearly documented **limitations**
          (synthetic data; prototype, not production-certified).
        """
    )
