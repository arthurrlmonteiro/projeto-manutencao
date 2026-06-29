"""
03_avaliacao.py  —  FASE 5 do CRISP-DM: Avaliação do modelo
===========================================================
Avalia o modelo final no conjunto de TESTE (equipamentos nunca vistos).

Como o problema é desbalanceado e o custo de NÃO detectar uma falha é
alto, priorizamos o RECALL. Escolhemos o limiar (threshold) que atinge
um recall alvo (padrão: 80%) e reportamos a precisão resultante.

Métricas:
  - PR-AUC e ROC-AUC (independentes de limiar)
  - Precisão, Recall e F1 no limiar escolhido
  - Matriz de confusão

Saídas:
  - 05_curva_precisao_recall.png
  - 06_matriz_confusao.png
  - 07_importancia_features.png
  - avaliacao_final.txt

Como rodar:
    python src/03_avaliacao.py
"""

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix,
)

from utils import carregar_split, OUTPUTS

sns.set_theme(style="whitegrid")
RECALL_ALVO = 0.80   # queremos capturar ao menos 80% das falhas reais


def importancias(modelo):
    """Extrai importâncias do classificador dentro do pipeline."""
    prep = modelo.named_steps["prep"]
    clf = modelo.named_steps["clf"]
    nomes = prep.get_feature_names_out()
    if hasattr(clf, "feature_importances_"):
        return pd.Series(clf.feature_importances_, index=nomes).sort_values()
    if hasattr(clf, "coef_"):
        return pd.Series(np.abs(clf.coef_[0]), index=nomes).sort_values()
    return None


def main():
    caminho = OUTPUTS / "modelo_final.joblib"
    if not caminho.exists():
        raise FileNotFoundError("Rode antes: python src/02_modelagem.py")
    modelo = joblib.load(caminho)

    X_train, X_test, y_train, y_test, _ = carregar_split()
    y_prob = modelo.predict_proba(X_test)[:, 1]

    pr_auc = average_precision_score(y_test, y_prob)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Escolhe o limiar que atinge o recall alvo
    prec, rec, lim = precision_recall_curve(y_test, y_prob)
    # prec/rec têm tamanho len(lim)+1; alinhamos com os limiares
    candidatos = [(p, r, t) for p, r, t in zip(prec[:-1], rec[:-1], lim) if r >= RECALL_ALVO]
    if candidatos:
        # entre os que batem o recall alvo, pega o de maior precisão
        p_sel, r_sel, limiar = max(candidatos, key=lambda x: x[0])
    else:
        limiar = 0.5

    y_pred = (y_prob >= limiar).astype(int)
    precisao = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("=" * 60)
    print("FASE 5 — AVALIAÇÃO FINAL (equipamentos de teste)")
    print("=" * 60)
    print(f"\nPR-AUC : {pr_auc:.3f}   (qualidade geral na classe rara)")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"\nNo limiar = {limiar:.3f} (escolhido para recall >= {RECALL_ALVO:.0%}):")
    print(f"  Recall  : {recall:.3f}  (% das falhas reais capturadas)")
    print(f"  Precisão: {precisao:.3f}  (% dos alertas que eram falhas reais)")
    print(f"  F1      : {f1:.3f}")

    resumo = (
        "AVALIACAO FINAL (conjunto de teste)\n"
        "===================================\n"
        f"PR-AUC : {pr_auc:.3f}\n"
        f"ROC-AUC: {roc_auc:.3f}\n"
        f"Limiar : {limiar:.3f} (alvo de recall {RECALL_ALVO:.0%})\n"
        f"Recall : {recall:.3f}\n"
        f"Precisao: {precisao:.3f}\n"
        f"F1     : {f1:.3f}\n"
    )
    (OUTPUTS / "avaliacao_final.txt").write_text(resumo, encoding="utf-8")

    # Curva precisão-recall
    plt.figure(figsize=(6, 5))
    plt.plot(rec, prec, color="steelblue", label=f"PR-AUC = {pr_auc:.3f}")
    plt.scatter([recall], [precisao], color="red", zorder=5,
                label=f"Limiar escolhido (recall {recall:.0%})")
    plt.xlabel("Recall")
    plt.ylabel("Precisão")
    plt.title("Curva Precisão-Recall")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS / "05_curva_precisao_recall.png", dpi=150)
    plt.close()

    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Normal", "Falha"],
                yticklabels=["Normal", "Falha"])
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.title("Matriz de confusão")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "06_matriz_confusao.png", dpi=150)
    plt.close()

    # Importância das features
    imp = importancias(modelo)
    if imp is not None:
        plt.figure(figsize=(7, 5))
        sns.barplot(x=imp.values, y=imp.index, color="seagreen")
        plt.xlabel("Importância relativa")
        plt.title("Importância das variáveis na previsão de falha")
        plt.tight_layout()
        plt.savefig(OUTPUTS / "07_importancia_features.png", dpi=150)
        plt.close()

    print("\nArquivos salvos em outputs/. Projeto concluído!")


if __name__ == "__main__":
    main()
