"""
02_modelagem.py  —  FASES 3 e 4 do CRISP-DM
===========================================
FASE 3 (Preparação): separa o conjunto de TESTE por equipamento (sem
                     vazamento) e define a validação cruzada por grupo.
FASE 4 (Modelagem):  compara 3 modelos por validação cruzada, escolhe o
                     melhor e ajusta seus hiperparâmetros.

Métrica de comparação: PR-AUC (average precision). É a métrica adequada
para classes desbalanceadas — foca na qualidade das previsões da classe
rara (falha), diferente da acurácia, que seria enganosa aqui.

Saídas:
  - outputs/04_comparacao_modelos.png
  - outputs/comparacao_modelos.csv
  - outputs/modelo_final.joblib

Como rodar:
    python src/02_modelagem.py
"""

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import cross_val_score, GridSearchCV, GroupKFold

from utils import carregar_split, obter_modelos, GRADES, OUTPUTS

sns.set_theme(style="whitegrid")
METRICA = "average_precision"   # PR-AUC


def main():
    # FASE 3 — Preparação
    X_train, X_test, y_train, y_test, grupos_train = carregar_split()

    print("=" * 60)
    print("FASES 3 e 4 — PREPARAÇÃO E MODELAGEM")
    print("=" * 60)
    print(f"\nTreino: {len(X_train):,} leituras  |  Teste: {len(X_test):,} leituras")
    print(f"Falhas no treino: {y_train.mean():.1%}  |  no teste: {y_test.mean():.1%}")
    print("Validação: validação cruzada 5-fold POR EQUIPAMENTO (sem vazamento)")
    print(f"Métrica de comparação: PR-AUC (adequada a classes desbalanceadas)\n")

    cv = GroupKFold(n_splits=5)

    # FASE 4 — Comparação por validação cruzada
    resultados = []
    for nome, modelo in obter_modelos().items():
        scores = cross_val_score(
            modelo, X_train, y_train, groups=grupos_train,
            scoring=METRICA, cv=cv, n_jobs=-1,
        )
        resultados.append({
            "Modelo": nome,
            "PR_AUC_medio": scores.mean(),
            "PR_AUC_desvio": scores.std(),
        })
        print(f"{nome:<22}  PR-AUC = {scores.mean():.3f} (+/- {scores.std():.3f})")

    tabela = (pd.DataFrame(resultados)
              .sort_values("PR_AUC_medio", ascending=False)
              .reset_index(drop=True))
    tabela.to_csv(OUTPUTS / "comparacao_modelos.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=tabela, x="PR_AUC_medio", y="Modelo", color="steelblue")
    plt.errorbar(tabela["PR_AUC_medio"], range(len(tabela)),
                 xerr=tabela["PR_AUC_desvio"], fmt="none", c="black", capsize=4)
    plt.xlabel("PR-AUC médio (validação cruzada) — maior é melhor")
    plt.title("Comparação de modelos")
    plt.tight_layout()
    plt.savefig(OUTPUTS / "04_comparacao_modelos.png", dpi=150)
    plt.close()

    # Seleção + ajuste fino do vencedor
    melhor_nome = tabela.iloc[0]["Modelo"]
    print(f"\nMelhor modelo na validação: {melhor_nome}")
    melhor_modelo = obter_modelos()[melhor_nome]

    if melhor_nome in GRADES:
        print("Ajustando hiperparâmetros com GridSearchCV...")
        busca = GridSearchCV(
            melhor_modelo, GRADES[melhor_nome],
            scoring=METRICA, cv=cv, n_jobs=-1,
        )
        busca.fit(X_train, y_train, groups=grupos_train)
        modelo_final = busca.best_estimator_
        print(f"Melhores hiperparâmetros: {busca.best_params_}")
        print(f"PR-AUC da CV após ajuste: {busca.best_score_:.3f}")
    else:
        modelo_final = melhor_modelo.fit(X_train, y_train)

    modelo_final.fit(X_train, y_train)
    joblib.dump(modelo_final, OUTPUTS / "modelo_final.joblib")

    print("\nArquivos salvos em outputs/. Próximo: python src/03_avaliacao.py")


if __name__ == "__main__":
    main()
