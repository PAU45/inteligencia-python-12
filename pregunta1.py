import subprocess, sys
try:
    from ucimlrepo import fetch_ucirepo
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ucimlrepo"])
    from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
wine = fetch_ucirepo(id=109)
X = wine.data.features[["Alcohol","Alcalinity_of_ash","Nonflavanoid_phenols"]]
if hasattr(wine.data.targets, 'columns') and 'class' in wine.data.targets.columns:
    y = wine.data.targets['class']
else:
    y = wine.data.targets
print('1) Partición 70% train / 30% test')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print('2) Entrenar Random Forest con las variables seleccionadas')
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
print('3) Evaluación en data test: calcular Accuracy y F1 (ponderado)')
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f"Accuracy: {acc:.4f}")
print(f"F1 (weighted): {f1:.4f}")
print('Interpretación:')
print(f"Accuracy: el modelo acierta el {acc*100:.2f}% de las observaciones de prueba.")
print(f"F1 ponderado: {f1:.4f} — combina precisión y recall considerando el tamaño de cada clase; valores más altos indican mejor equilibrio entre precisión y exhaustividad por clase.")
