import glob, sys, subprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
paths = glob.glob('breast_wisconsin-5.csv')
if not paths:
    paths = glob.glob('*breast*.csv')
if not paths:
    print('Archivo de datos no encontrado: buscar breast_wisconsin*.csv')
    sys.exit(1)
df = pd.read_csv(paths[0], sep=';')
if 'COD_identificacion_dni' in df.columns:
    df = df.drop(columns=['COD_identificacion_dni'])
if 'fractal_dimension3' not in df.columns:
    print('No se encontró la columna fractal_dimension3 en el dataset')
    sys.exit(1)
X = df.drop(columns=['fractal_dimension3'])
X = X.select_dtypes(include=[np.number])
y = df['fractal_dimension3']
print('1) Partición 70% train / 30% test')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print('2) Entrenar Random Forest Regressor')
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)
print('3) Evaluación en data test: calcular MSE y R2')
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'MSE: {mse:.4f}')
print(f'R2: {r2:.4f}')
print('Interpretación:')
print(f'MSE (error cuadrático medio): {mse:.4f} — valor más bajo indica mejores predicciones promedio en las mismas unidades que la variable objetivo.')
print(f'R2 (coeficiente de determinación): {r2:.4f} — proporción de varianza explicada por el modelo; cercano a 1 indica buen ajuste.')
