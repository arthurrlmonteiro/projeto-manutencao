"""
01_eda.py  —  FASE 2 do CRISP-DM: Entendimento dos dados
========================================================
Explora os dados de sensores antes de modelar.

Gera:
  - 01_balanceamento_classes.png  (quão raras são as falhas)
  - 02_correlacao.png             (correlação entre variáveis numéricas)
  - 03_sensores_por_classe.png    (sensores em equipamentos saudáveis vs. em falha)

Como rodar:
    python src/01_eda.py
"""

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

from utils import carregar_dados, FEATURES_NUM, ALVO, OUTPUTS

sns.set_theme(style="whitegrid")


def main():
    df = carregar_dados()

    print("=" * 60)
    print("FASE 2 — ENTENDIMENTO DOS DADOS")
    print("=" * 60)
    print(f"\nFormato (linhas, colunas): {df.shape}")
    print(f"Equipamentos distintos: {df['equipment_id'].nunique()}")
    print(f"Período: {df['timestamp'].min()} a {df['timestamp'].max()}")
    print("\nEstatísticas descritivas:")
    print(df[FEATURES_NUM].describe().round(2))

    print("\nValores faltantes por coluna:")
    print(df.isnull().sum())

    # Balanceamento das classes
    dist = df[ALVO].value_counts(normalize=True).sort_index()
    print(f"\nDistribuição do alvo (0 = normal, 1 = falha em 72h):")
    print((dist * 100).round(2).astype(str) + " %")

    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x=ALVO, color="steelblue")
    plt.title("Balanceamento das classes (falhas são raras)")
    plt.xlabel("0 = normal   |   1 = falha nas próximas 72h")
    plt.ylabel("Nº de leituras")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "01_balanceamento_classes.png", dpi=150)
    plt.close()

    # Correlação entre variáveis numéricas + alvo
    plt.figure(figsize=(9, 7))
    sns.heatmap(df[FEATURES_NUM + [ALVO]].corr(), annot=True,
                cmap="coolwarm", fmt=".2f", square=True, linewidths=0.5)
    plt.title("Correlação entre sensores e o alvo")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "02_correlacao.png", dpi=150)
    plt.close()

    # Distribuição dos principais sensores por classe
    principais = ["vibration_rms", "bearing_temp", "oil_temp", "oil_pressure"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for ax, feat in zip(axes.ravel(), principais):
        sns.kdeplot(data=df, x=feat, hue=ALVO, common_norm=False,
                    fill=True, alpha=0.4, ax=ax)
        ax.set_title(f"{feat} por classe")
    fig.suptitle("Sensores: equipamentos saudáveis (0) vs. prestes a falhar (1)")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "03_sensores_por_classe.png", dpi=150)
    plt.close()

    print("\nGráficos salvos em outputs/. Próximo: python src/02_modelagem.py")


if __name__ == "__main__":
    main()
