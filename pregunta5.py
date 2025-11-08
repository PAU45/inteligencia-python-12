import glob, sys, os
import pandas as pd
paths = glob.glob('aids_.csv')
if not paths:
    paths = glob.glob('*aids*.csv')
if not paths:
    print('Archivo aids_clinical*.csv no encontrado')
    sys.exit(1)
df = pd.read_csv(paths[0], sep=';')
print(f"Archivo usado (relativo): {paths[0]}")
print(f"Archivo usado (absoluto): {os.path.abspath(paths[0])}")
if 'preanti' not in df.columns or 'wtkg' not in df.columns:
    print('Columnas preanti y/o wtkg no encontradas en el dataset')
    sys.exit(1)

pearson = df['preanti'].corr(df['wtkg'])
spearman = df['preanti'].corr(df['wtkg'], method='spearman')
zeros_preanti = (df['preanti'] == 0).sum()
print(f'Coeficiente de correlación de Pearson entre preanti y wtkg: {pearson:.4f}')
print(f'Coeficiente de correlación de Spearman entre preanti y wtkg: {spearman:.4f}')
print(f'Cantidad de observaciones con preanti==0: {zeros_preanti} / {len(df)}')
if pearson>0:
    print('La correlación lineal (Pearson) es positiva: en promedio, preanti y wtkg aumentan juntos.')
elif pearson<0:
    print('La correlación lineal (Pearson) es negativa: en promedio, cuando preanti aumenta, wtkg disminuye.')
else:
    print('La correlación lineal (Pearson) es cercana a 0.')

print('\nInterpretación adicional:')
print('preanti tiene muchos ceros y una distribución muy sesgada (outliers altos).')
print('Recomiendo usar Spearman o transformar preanti (por ejemplo log1p) antes de medir relaciones lineales o ajustar modelos.')
print('Recordatorio: correlación no implica causalidad; esto sólo describe asociación en este dataset.')
