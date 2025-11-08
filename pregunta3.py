import glob, sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
paths = glob.glob('aids_clinical-5.csv')
if not paths:
    paths = glob.glob('*aids*.csv')
if not paths:
    print('Archivo de datos no encontrado: aids_clinical*.csv')
    sys.exit(1)
df = pd.read_csv(paths[0], sep=';')
if 'str2' not in df.columns:
    print('No se encontró la columna str2 en el dataset')
    sys.exit(1)
X = df.drop(columns=['str2'])

X = X.select_dtypes(include=['number'])
y = df['str2']
le = LabelEncoder()
y_enc = le.fit_transform(y.astype(str))
print('1) Partición 80% train / 20% test')
X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)
print('2) Entrenar algoritmo más conveniente (Random Forest)')
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
print('3) Evaluación en data test: Accuracy y F1 (weighted)')
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='weighted')
print(f'Accuracy: {acc:.4f}')
print(f'F1 (weighted): {f1:.4f}')
print('Interpretación:')
print(f'Accuracy: el modelo acierta el {acc*100:.2f}% en el conjunto de prueba.')
print(f'F1 ponderado: {f1:.4f} — indica el equilibrio entre precisión y recall teniendo en cuenta el soporte de las clases.')
