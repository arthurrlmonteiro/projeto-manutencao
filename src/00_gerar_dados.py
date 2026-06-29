"""
00_gerar_dados.py  —  Gera o CSV de exemplo de sensores históricos
==================================================================
Cria dados SINTÉTICOS realistas de monitoramento de condição de
equipamentos críticos (motores, bombas, thrusters) de embarcações
offshore, para TESTAR o pipeline de modelagem.

Como funciona a simulação:
  - Cada equipamento passa por vários "ciclos" de vida (entre manutenções).
  - Dentro de um ciclo, a "saúde" do equipamento degrada de 1.0 até a falha,
    acelerando perto do fim.
  - Os sensores refletem a degradação:
        vibração e temperatura SOBEM, pressão de óleo CAI.
  - Rótulo (alvo): 1 se o equipamento vai falhar nas próximas HORIZONTE_H
    horas; 0 caso contrário. Falhas são raras => classes desbalanceadas
    (como na realidade).

IMPORTANTE: estes dados são apenas para validar o código e gerar uma
demonstração. Em produção, você usaria os dados reais dos sensores.

Como rodar:
    python src/00_gerar_dados.py
"""

import numpy as np
import pandas as pd

from utils import DATA, SEED, HORIZONTE_H

# Parâmetros da simulação
N_NAVIOS = 4
POR_TIPO_POR_NAVIO = 2          # equipamentos de cada tipo, em cada navio
TIPOS = ["motor", "bomba", "thruster"]
SPAN_H = 24 * 90               # 90 dias de leituras horárias por equipamento
VIDA_MEDIA_H = 1500            # vida média de um ciclo (horas)
VIDA_DESVIO_H = 500

# Valores de base dos sensores por tipo de equipamento
BASE = {
    "motor":    dict(vib=2.0, temp=65, oleo=72, pres=4.5, rpm=750),
    "bomba":    dict(vib=1.5, temp=55, oleo=60, pres=5.0, rpm=1450),
    "thruster": dict(vib=2.5, temp=60, oleo=68, pres=4.0, rpm=300),
}


def gerar():
    rng = np.random.default_rng(SEED)

    # Cria a lista de equipamentos
    equipamentos = []
    aid = 0
    for nav in range(1, N_NAVIOS + 1):
        for tipo in TIPOS:
            for _ in range(POR_TIPO_POR_NAVIO):
                aid += 1
                equipamentos.append({
                    "equipment_id": f"EQ{aid:03d}",
                    "vessel_id": f"NAV{nav}",
                    "equipment_type": tipo,
                })

    inicio = pd.Timestamp("2024-01-01")
    linhas = []

    for eq in equipamentos:
        b = BASE[eq["equipment_type"]]
        t = 0
        while t < SPAN_H:
            # Vida deste ciclo
            L = int(max(rng.normal(VIDA_MEDIA_H, VIDA_DESVIO_H), 300))
            fim = min(t + L, SPAN_H)
            houve_falha = (t + L) <= SPAN_H   # se o ciclo cabe no período, falha

            for h in range(t, fim):
                idade = h - t                  # horas desde a última manutenção
                frac = idade / L
                saude = 1.0 - frac ** 2        # degrada acelerando no fim
                degrad = 1.0 - saude           # 0 (novo) -> 1 (prestes a falhar)

                load = rng.uniform(30, 100)    # % de carga
                rpm = b["rpm"] * rng.uniform(0.85, 1.05)

                vib_rms = b["vib"] + 6.0 * degrad + 0.01 * load + rng.normal(0, 0.4)
                vib_peak = vib_rms * rng.uniform(1.6, 2.2) + rng.normal(0, 0.5)
                temp_mancal = b["temp"] + 30 * degrad + 0.10 * load + rng.normal(0, 1.5)
                temp_oleo = b["oleo"] + 20 * degrad + 0.08 * load + rng.normal(0, 1.5)
                pres_oleo = b["pres"] - 2.0 * degrad + rng.normal(0, 0.15)

                # Rótulo: falha dentro do horizonte de previsão?
                if houve_falha and (L - idade) <= HORIZONTE_H:
                    rotulo = 1
                else:
                    rotulo = 0

                linhas.append({
                    "timestamp": inicio + pd.Timedelta(hours=h),
                    "vessel_id": eq["vessel_id"],
                    "equipment_id": eq["equipment_id"],
                    "equipment_type": eq["equipment_type"],
                    "vibration_rms": round(vib_rms, 3),
                    "vibration_peak": round(vib_peak, 3),
                    "bearing_temp": round(temp_mancal, 2),
                    "oil_temp": round(temp_oleo, 2),
                    "oil_pressure": round(max(pres_oleo, 0.5), 3),
                    "rpm": round(rpm, 1),
                    "load_pct": round(load, 1),
                    "operating_hours": idade,
                    "failure_within_72h": rotulo,
                })

            # Próximo ciclo começa após uma pequena parada de manutenção
            t = fim + int(rng.integers(2, 12))

    df = pd.DataFrame(linhas)
    DATA.mkdir(exist_ok=True)
    caminho = DATA / "sensores_historico.csv"
    df.to_csv(caminho, index=False)

    # Resumo
    pos = df["failure_within_72h"].mean()
    print(f"CSV gerado em: {caminho}")
    print(f"Linhas: {len(df):,}  |  Equipamentos: {df['equipment_id'].nunique()}")
    print(f"Proporção de falhas (classe positiva): {pos:.1%}")
    print("\nPrimeiras linhas:")
    print(df.head())


if __name__ == "__main__":
    gerar()
