import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 1. Dados simulados de uma característica de qualidade (ex: diâmetro de um componente em mm)
amostras = np.array(
    [10.2, 9.8, 10.1, 10.5, 9.9, 10.0, 10.3, 9.7, 11.2, 10.4, 10.2, 9.9]
)

# 2. Cálculo dos parâmetros estatísticos
media = np.mean(amostras)
desvio_padrao = np.std(amostras, ddof=1)  # Desvio padrão amostral

# Limites de Controle (3 Sigma)
ucl = media + 3 * (desvio_padrao / np.sqrt(1))  # Limite Superior de Controle
lcl = media - 3 * (desvio_padrao / np.sqrt(1))  # Limite Inferior de Controle

print("=== PARÂMETROS DO CEP ===")
print(f"Média do Processo: {media:.3f}")
print(f"Limite Superior de Controle (UCL): {ucl:.3f}")
print(f"Limite Inferior de Controle (LCL): {lcl:.3f}\n")

# 3. Detecção de Alertas (Regras de Controle)
anomalias = np.where((amostras > ucl) | (amostras < lcl))[0]
if len(anomalias) > 0:
    print(f"⚠️ ALERTA: Pontos fora de controle detectados nas amostras: {anomalias + 1}")
else:
    print("✅ Processo sob controle estatístico.")

# 4. Visualização Gráfica
plt.figure(figsize=(10, 5))
plt.plot(
    range(1, len(amostras) + 1),
    amostras,
    marker='o',
    color='blue',
    label='Valores Coletados',
)
plt.axhline(
    media, color='green', linestyle='-', linewidth=1.5, label=f'Média ({media:.2f})'
)
plt.axhline(
    ucl, color='red', linestyle='--', linewidth=1.5, label=f'UCL ({ucl:.2f})'
)
plt.axhline(
    lcl, color='red', linestyle='--', linewidth=1.5, label=f'LCL ({lcl:.2f})'
)

# Destacar pontos fora de controle no gráfico
if len(anomalias) > 0:
    plt.scatter(
        anomalias + 1,
        amostras[anomalias],
        color='orange',
        s=150,
        zorder=5,
        label='Causa Especial (Fora de Limite)',
    )

plt.title('Gráfico de Controle Estatístico de Processo (CEP/SPC)')
plt.xlabel('Amostras')
plt.ylabel('Valor Medido')
plt.grid(True, alpha=0.3)
plt.legend(loc='upper left')
plt.show()
