# Manutenção Preditiva de Equipamentos Críticos — Embarcações Offshore

Protótipo (prova de conceito) que acompanha o **Projeto 2** do curso
*Managing Machine Learning Projects* (Duke / AI Product Management).

**Problema:** prever a falha de equipamentos rotativos críticos (motores,
bombas, thrusters) nas próximas **72 horas**, a partir de sensores de
vibração, temperatura e pressão.
**Tarefa de ML:** classificação binária com classes desbalanceadas.
**Usuário:** gestor de manutenção / confiabilidade (onshore).

> Este código existe para gerar uma **demonstração** para o vídeo do projeto.
> Os dados são **sintéticos** (criados pelo script `00_gerar_dados.py`); em
> produção, você usaria os dados reais dos sensores da frota.

---

## Estrutura

```
projeto-manutencao-preditiva/
├── app.py                       <- app interativo (Streamlit)
├── data/
│   └── sensores_historico.csv   <- dados de exemplo (já incluídos)
├── src/
│   ├── utils.py                 <- dados, split por grupo, modelos
│   ├── 00_gerar_dados.py        <- gera o CSV sintético de sensores
│   ├── 01_eda.py                <- Fase 2: entendimento dos dados
│   ├── 02_modelagem.py          <- Fases 3 e 4: validação e comparação
│   └── 03_avaliacao.py          <- Fase 5: avaliação no teste
├── outputs/                     <- gráficos e resultados (prints p/ o vídeo)
├── .streamlit/config.toml       <- tema do app
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Como rodar

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate   |   Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt

# O CSV de exemplo já vem pronto. Para regenerá-lo (opcional):
python src/00_gerar_dados.py

python src/01_eda.py
python src/02_modelagem.py
python src/03_avaliacao.py
```

---

## App interativo (Streamlit)

Além dos scripts, o projeto tem um **app interativo** que reúne tudo numa única
demonstração — ideal para o vídeo. A navegação fica na barra lateral:

- **Overview** — problema, decisões de método e *task analysis*.
- **EDA** — balanceamento, correlação e distribuições por classe.
- **Modeling** — comparação de modelos (versão leve, sob demanda).
- **Evaluation** — métricas no teste, curva precisão-recall, matriz de confusão,
  importância das variáveis e **calibração**.
- **Prediction** — previsão interativa com **explicação por previsão**,
  **comunicação de incerteza** e **loop de feedback** (human-in-the-loop).
- **Fairness** — desempenho por subgrupo (tipo de equipamento e embarcação) para
  expor possíveis vieses.
- **Responsible AI** — privacidade (LGPD/GDPR) e ética (fontes de viés; justiça,
  responsabilidade e transparência).

> As seções *Overview (task analysis)*, *Prediction*, *Fairness* e
> *Responsible AI* cobrem os quatro eixos do Projeto do Curso 3 (AI Product
> Management): análise da tarefa, UX de IA, privacidade e ética.

### Rodar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

O app abre em `http://localhost:8501`. Ele reaproveita `src/utils.py` e treina o
modelo em tempo de execução (com cache) — não precisa rodar os scripts antes.

### Publicar no Streamlit Community Cloud (grátis)

1. Suba este projeto para o GitHub:
   `https://github.com/arthurrlmonteiro/projeto-manutencao`
2. Acesse [share.streamlit.io](https://share.streamlit.io) e faça login com o GitHub.
3. Clique em **New app** e selecione:
   - **Repository:** `arthurrlmonteiro/projeto-manutencao`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Clique em **Deploy**. O Streamlit instala o `requirements.txt` e publica o app
   numa URL pública.

> O `data/sensores_historico.csv` está versionado no repositório, então o app
> funciona no Cloud sem nenhuma etapa extra.

---

## Decisões de método (e por que importam no vídeo)

**Classes desbalanceadas.** Falhas são ~3% das leituras. Por isso usamos
`class_weight="balanced"` e avaliamos com **PR-AUC, recall e precisão** —
nunca acurácia, que seria enganosa (prever "nunca falha" daria ~97% de
acurácia e seria inútil).

**Split por equipamento (sem vazamento).** Leituras consecutivas do mesmo
equipamento são quase idênticas. Um split aleatório colocaria leituras
quase iguais no treino e no teste, inflando o resultado. Dividimos **por
equipamento** (`GroupShuffleSplit`/`GroupKFold`): o modelo é testado em
equipamentos que nunca viu — como aconteceria com um navio novo na frota.

**Escolha do limiar pelo recall.** Como deixar passar uma falha é muito
mais caro do que um falso alarme, o `03_avaliacao.py` escolhe o limiar que
captura ao menos 80% das falhas e então reporta a precisão resultante.

**Baseline primeiro.** Comparamos Regressão Logística, Random Forest e
Gradient Boosting. Nos dados de exemplo, o baseline (logística) generalizou
melhor para equipamentos novos — um bom lembrete de que o modelo mais
complexo nem sempre vence.

---

## Como isso conecta com o documento conceitual

Cada fase deste código materializa uma parte do projeto conceitual:

| Código | Fase CRISP-DM | Seção do projeto |
|---|---|---|
| `01_eda.py` | Entendimento dos dados | Fatores relevantes (seção 2) |
| `02_modelagem.py` | Preparação + Modelagem | Plano de validação (seção 3) |
| `03_avaliacao.py` | Avaliação | Definição de sucesso / métricas (seção 2) |

Use os gráficos de `outputs/` como a demonstração rápida que a rubrica do
projeto pede no vídeo.
