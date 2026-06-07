import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

resumen = pd.read_csv('results/resumen_por_estrategia.csv', index_col='tipo')
metricas = ['% item', '% resultado', '% referencia', '% flag', '% completo']

x = np.arange(len(metricas))
ancho = 0.3

fig, ax = plt.subplots(figsize=(10, 5))
barras1 = ax.bar(x - ancho/2, resumen.loc['pdfplumber', metricas], ancho, label='pdfplumber', color='#2ecc71')
barras2 = ax.bar(x + ancho/2, resumen.loc['easyocr', metricas], ancho, label='EasyOCR', color='#e74c3c')

for barra in barras1:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1,
            f'{barra.get_height():.0f}%', ha='center', va='bottom', fontweight='bold')
for barra in barras2:
    ax.text(barra.get_x() + barra.get_width()/2, barra.get_height() + 1,
            f'{barra.get_height():.0f}%', ha='center', va='bottom', fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(metricas)
ax.set_ylabel('% de acierto')
ax.set_title('Comparación: pdfplumber vs EasyOCR')
ax.set_ylim(0, 115)
ax.legend()
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('results/comparacion_metricas.png', dpi=150)
print('Grafico guardado en results/comparacion_metricas.png')
