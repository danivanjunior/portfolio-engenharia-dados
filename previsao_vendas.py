import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# -------------------------------------------------------------
# 1. Preparação dos Dados
# -------------------------------------------------------------
meses_historico = np.array(range(1, 13)).reshape(-1, 1) # Meses de 1 a 12
vendas_historico = np.array([120, 135, 148, 160, 175, 190, 205, 210, 225, 240, 255, 270])

# -------------------------------------------------------------
# 2. Treinamento do Modelo de Regressão Linear
# -------------------------------------------------------------
modelo = LinearRegression()
modelo.fit(meses_historico, vendas_historico)

# -------------------------------------------------------------
# 3. Previsão dos Próximos 3 Meses (Mês 13, 14 e 15)
# -------------------------------------------------------------
meses_futuros = np.array([13, 14, 15]).reshape(-1, 1)
previsoes = modelo.predict(meses_futuros)

# Exibição dos resultados previstos
print("=== PREVISÃO DE VENDAS PARA OS PRÓXIMOS 3 MESES ===")
for mes, pred in zip([13, 14, 15], previsoes):
    print(f"Mês {mes}: {pred:.0f} unidades previstass")

# -------------------------------------------------------------
# 4. Visualização dos Dados (Gráfico)
# -------------------------------------------------------------
plt.figure(figsize=(10, 5))

# Histórico real
plt.plot(range(1, 13), vendas_historico, marker='o', label='Vendas Reais (Histórico)', color='blue')

# Linha de tendência + Previsão
todos_meses = np.array(range(1, 16)).reshape(-1, 1)
linha_tendencia = modelo.predict(todos_meses)
plt.plot(todos_meses, linha_tendencia, linestyle='--', color='red', label='Linha de Tendência / Previsão')

# Destaque para a área prevista
plt.scatter(meses_futuros, previsoes, color='orange', zorder=5, s=100, label='Projeção (Meses 13, 14 e 15)')

# Estilização do gráfico
plt.title('Previsibilidade de Vendas — Modelo de Regressão Linear', fontsize=14)
plt.xlabel('Mês')
plt.ylabel('Unidades Vendidas')
plt.xticks(range(1, 16), ['Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun', 'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez', 'Jan (Prev)', 'Fev (Prev)', 'Mar (Prev)'])
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
