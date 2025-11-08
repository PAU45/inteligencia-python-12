import glob, sys, subprocess
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
paths = glob.glob('glioma_grading*.csv')
if not paths:
    paths = glob.glob('*glioma*.csv')
if not paths:
    print('Archivo de datos no encontrado: glioma_grading*.csv')
    sys.exit(1)
df = pd.read_csv(paths[0], sep=';')
if 'Grade' not in df.columns:
    print('No se encontró la columna Grade en el dataset')
    sys.exit(1)
X = df.drop(columns=['Grade'])
# Seleccionar solo columnas numéricas para entrenar (evitar Race u otras no numéricas)
X = X.select_dtypes(include=['number'])
y = df['Grade']
print('1) Partición 80% train / 20% test')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print('2) Primer algoritmo: Random Forest')
clf1 = RandomForestClassifier(random_state=42)
clf1.fit(X_train, y_train)
y_pred1 = clf1.predict(X_test)
acc1 = accuracy_score(y_test, y_pred1)
f11 = f1_score(y_test, y_pred1, average='weighted')
try:
    import xgboost as xgb
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xgboost"]) 
    import xgboost as xgb
print('3) Segundo algoritmo: XGBoost')
model2 = xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
acc2 = accuracy_score(y_test, y_pred2)
f12 = f1_score(y_test, y_pred2, average='weighted')
print('\nResultados RandomForest:')
print(f'Accuracy: {acc1:.4f}')
print(f'F1 (weighted): {f11:.4f}')
print('\nResultados XGBoost:')
print(f'Accuracy: {acc2:.4f}')
print(f'F1 (weighted): {f12:.4f}')
print('\nInterpretación:')
if (acc1>acc2 and f11>=f12) or (acc1>=acc2 and f11>f12):
    print('RandomForest tuvo mejor desempeño según las métricas (Accuracy y F1).')
else:
    print('XGBoost tuvo mejor desempeño según las métricas (Accuracy y F1).')
