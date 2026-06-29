"""
app.py  —  Demonstração interativa (Streamlit)
==============================================
App de apresentação do protótipo de Manutenção Preditiva de equipamentos
críticos de embarcações offshore (Duke / AI Product Manager — Projeto 2).

Reaproveita TODA a lógica de dados e modelos de `src/utils.py` — este arquivo
apenas constrói a interface e coloca cache nas partes pesadas para rodar bem
no Streamlit Community Cloud (plano grátis, ~1 GB de RAM).

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

st.set_page_config(
    page_title="Manutenção Preditiva — Offshore",
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
            "Modelo": nome,
            "PR_AUC_medio": scores.mean(),
            "PR_AUC_desvio": scores.std(),
        })
    return (pd.DataFrame(linhas)
            .sort_values("PR_AUC_medio", ascending=False)
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
st.title("⚙️ Manutenção Preditiva de Equipamentos Críticos — Offshore")
st.caption(
    "Protótipo do **Projeto 2** — *Managing Machine Learning Projects* "
    "(Duke / AI Product Management). Dados sintéticos para demonstração."
)

df = get_dados()

aba_visao, aba_eda, aba_modelagem, aba_avaliacao, aba_previsao = st.tabs([
    "📌 Visão geral",
    "📊 EDA",
    "🤖 Modelagem",
    "✅ Avaliação",
    "🔮 Previsão",
])


# ----------------------------------------------------------------------
# Aba 1 — Visão geral
# ----------------------------------------------------------------------
with aba_visao:
    st.subheader("O problema")
    st.markdown(
        """
        Prever a **falha de equipamentos rotativos críticos** (motores, bombas,
        thrusters) nas próximas **72 horas**, a partir de sensores de vibração,
        temperatura e pressão.

        - **Tarefa de ML:** classificação binária com classes desbalanceadas.
        - **Usuário:** gestor de manutenção / confiabilidade (onshore).
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Leituras", f"{len(df):,}".replace(",", "."))
    c2.metric("Equipamentos", df["equipment_id"].nunique())
    c3.metric("Falhas (classe positiva)", f"{df[ALVO].mean():.1%}")

    st.divider()
    st.subheader("Decisões de método (e por que importam)")
    st.markdown(
        """
        - **Classes desbalanceadas.** Falhas são ~3% das leituras. Usamos
          `class_weight="balanced"` e avaliamos com **PR-AUC, recall e precisão**
          — nunca acurácia (prever "nunca falha" daria ~97% e seria inútil).
        - **Split por equipamento (sem vazamento).** Leituras consecutivas do
          mesmo equipamento são quase idênticas. Dividimos **por equipamento**
          (`GroupShuffleSplit` / `GroupKFold`): o modelo é testado em
          equipamentos que nunca viu — como um navio novo na frota.
        - **Limiar escolhido pelo recall.** Deixar passar uma falha custa mais
          que um falso alarme, então fixamos o limiar para capturar ≥ 80% das
          falhas e reportamos a precisão resultante.
        - **Baseline primeiro.** Comparamos Logística, Random Forest e Gradient
          Boosting. Nos dados de exemplo, a **Logística** generalizou melhor para
          equipamentos novos.
        """
    )

    st.divider()
    st.subheader("Amostra dos dados")
    st.dataframe(df.head(20), use_container_width=True)


# ----------------------------------------------------------------------
# Aba 2 — EDA
# ----------------------------------------------------------------------
with aba_eda:
    st.subheader("Entendimento dos dados (Fase 2 — CRISP-DM)")

    st.markdown("**Estatísticas descritivas dos sensores**")
    st.dataframe(df[FEATURES_NUM].describe().round(2), use_container_width=True)

    col_esq, col_dir = st.columns(2)

    with col_esq:
        st.markdown("**Balanceamento das classes (falhas são raras)**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.countplot(data=df, x=ALVO, color="steelblue", ax=ax)
        ax.set_xlabel("0 = normal   |   1 = falha nas próximas 72h")
        ax.set_ylabel("Nº de leituras")
        st.pyplot(fig)
        plt.close(fig)

    with col_dir:
        st.markdown("**Correlação entre sensores e o alvo**")
        fig, ax = plt.subplots(figsize=(7, 6))
        sns.heatmap(df[FEATURES_NUM + [ALVO]].corr(), annot=True,
                    cmap="coolwarm", fmt=".2f", square=True,
                    linewidths=0.5, ax=ax)
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Sensores: equipamentos saudáveis (0) vs. prestes a falhar (1)**")
    principais = ["vibration_rms", "bearing_temp", "oil_temp", "oil_pressure"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, feat in zip(axes.ravel(), principais):
        sns.kdeplot(data=df, x=feat, hue=ALVO, common_norm=False,
                    fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"{feat} por classe")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ----------------------------------------------------------------------
# Aba 3 — Modelagem
# ----------------------------------------------------------------------
with aba_modelagem:
    st.subheader("Comparação de modelos (Fases 3 e 4 — CRISP-DM)")
    st.markdown(
        """
        Validação cruzada **5-fold por equipamento** (`GroupKFold`, sem
        vazamento). Métrica: **PR-AUC** (*average precision*) — adequada a
        classes desbalanceadas, ao contrário da acurácia.

        > A comparação treina 3 modelos (incl. Random Forest com 300 árvores) e
        > pode levar até ~1 min na primeira execução. O resultado fica em cache.
        """
    )

    if st.button("▶️ Rodar comparação", type="primary"):
        with st.spinner("Rodando validação cruzada dos 3 modelos…"):
            tabela = get_comparacao_modelos()
        st.session_state["comparacao"] = tabela

    tabela = st.session_state.get("comparacao")
    if tabela is not None:
        melhor = tabela.iloc[0]["Modelo"]
        st.success(f"Melhor modelo na validação: **{melhor}**")

        col_esq, col_dir = st.columns([1, 1])
        with col_esq:
            st.dataframe(
                tabela.style.format(
                    {"PR_AUC_medio": "{:.3f}", "PR_AUC_desvio": "{:.3f}"}
                ),
                use_container_width=True,
            )
        with col_dir:
            fig, ax = plt.subplots(figsize=(7, 4))
            sns.barplot(data=tabela, x="PR_AUC_medio", y="Modelo",
                        color="steelblue", ax=ax)
            ax.errorbar(tabela["PR_AUC_medio"], range(len(tabela)),
                        xerr=tabela["PR_AUC_desvio"], fmt="none",
                        c="black", capsize=4)
            ax.set_xlabel("PR-AUC médio (CV) — maior é melhor")
            ax.set_title("Comparação de modelos")
            st.pyplot(fig)
            plt.close(fig)
    else:
        st.info("Clique em **Rodar comparação** para treinar e comparar os modelos.")


# ----------------------------------------------------------------------
# Aba 4 — Avaliação
# ----------------------------------------------------------------------
with aba_avaliacao:
    st.subheader("Avaliação final no conjunto de teste (Fase 5 — CRISP-DM)")
    st.markdown(
        f"Modelo final: **{MODELO_FINAL}**, avaliado em equipamentos **nunca "
        f"vistos**. Limiar escolhido para **recall ≥ {RECALL_ALVO:.0%}**."
    )

    with st.spinner("Avaliando o modelo no conjunto de teste…"):
        av = get_avaliacao()

    c1, c2, c3 = st.columns(3)
    c1.metric("PR-AUC", f"{av['pr_auc']:.3f}")
    c2.metric("ROC-AUC", f"{av['roc_auc']:.3f}")
    c3.metric("Limiar", f"{av['limiar']:.3f}")

    c4, c5, c6 = st.columns(3)
    c4.metric("Recall", f"{av['recall']:.1%}", help="% das falhas reais capturadas")
    c5.metric("Precisão", f"{av['precisao']:.1%}", help="% dos alertas que eram falhas reais")
    c6.metric("F1", f"{av['f1']:.3f}")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Curva Precisão-Recall**")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(av["rec_curve"], av["prec_curve"], color="steelblue",
                label=f"PR-AUC = {av['pr_auc']:.3f}")
        ax.scatter([av["recall"]], [av["precisao"]], color="red", zorder=5,
                   label=f"Limiar escolhido (recall {av['recall']:.0%})")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precisão")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        st.markdown("**Matriz de confusão**")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(av["cm"], annot=True, fmt=",d", cmap="Blues",
                    xticklabels=["Normal", "Falha"],
                    yticklabels=["Normal", "Falha"], ax=ax)
        ax.set_xlabel("Previsto")
        ax.set_ylabel("Real")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("**Importância das variáveis na previsão de falha**")
    imp = importancias(get_modelo_final())
    if imp is not None:
        fig, ax = plt.subplots(figsize=(7, 5))
        sns.barplot(x=imp.values, y=imp.index, color="seagreen", ax=ax)
        ax.set_xlabel("Importância relativa (|coeficiente|)")
        st.pyplot(fig)
        plt.close(fig)


# ----------------------------------------------------------------------
# Aba 5 — Previsão interativa
# ----------------------------------------------------------------------
with aba_previsao:
    st.subheader("Previsão interativa de falha")
    st.markdown(
        "Ajuste os valores dos sensores e veja a probabilidade estimada de "
        "**falha nas próximas 72 horas**. Dica: vibração e temperatura mais "
        "**altas** e pressão de óleo mais **baixa** indicam degradação."
    )

    modelo = get_modelo_final()
    av = get_avaliacao()
    limiar = av["limiar"]
    desc = df[FEATURES_NUM].describe()

    tipo = st.selectbox("Tipo de equipamento", sorted(df["equipment_type"].unique()))

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
    col_p.metric("Probabilidade de falha (72h)", f"{prob:.1%}")
    if prob >= limiar:
        col_d.error(
            f"🔴 **ALERTA** — probabilidade ({prob:.1%}) ≥ limiar de decisão "
            f"({limiar:.1%}). Recomendar inspeção/manutenção."
        )
    else:
        col_d.success(
            f"🟢 **OK** — probabilidade ({prob:.1%}) < limiar de decisão "
            f"({limiar:.1%}). Sem ação imediata."
        )
    st.progress(min(prob, 1.0))
    st.caption(
        f"O limiar de {limiar:.1%} foi escolhido na avaliação para capturar "
        f"≥ {RECALL_ALVO:.0%} das falhas reais."
    )
