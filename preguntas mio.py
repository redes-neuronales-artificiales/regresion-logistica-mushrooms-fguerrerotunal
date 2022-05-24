
import pandas as pd

# Lea el archivo `mushrooms.csv` y asignelo al DataFrame `df`
df = pd.read_csv("mushrooms.csv")

# Remueva la columna `veil-type` del DataFrame `df`.
# Esta columna tiene un valor constante y no sirve para la detección de hongos.
df.drop(["veil_type"],axis=1, inplace=True)

# Asigne la columna `type` a la variable `y`.
y = df["type"]

# Asigne una copia del dataframe `df` a la variable `X`.
X = df.copy()

# Remueva la columna `type` del DataFrame `X`.
X.drop(["type"],axis=1, inplace=True)

print(X.shape)
print(y.shape)

# Importe train_test_split
from sklearn.model_selection import train_test_split

# Divida los datos de entrenamiento y prueba. La semilla del generador de números
# aleatorios es 123. Use 50 patrones para la muestra de prueba.
(X_train, X_test, y_train, y_test,) = train_test_split(
    X,
    y,
    test_size = 50,
    random_state = 123,
)

print(y_train.value_counts().to_dict())
print(y_test.value_counts().to_dict())
print(X_train.iloc[:, 0].value_counts().to_dict())
print(X_test.iloc[:, 1].value_counts().to_dict())

# Importe LogisticRegressionCV
# Importe OneHotEncoder
# Importe Pipeline
from sklearn.linear_model import LogisticRegressionCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Cree un pipeline que contenga un estimador OneHotEncoder y un estimador
# LogisticRegression con una regularización Cs=10
pipeline = Pipeline(
    steps=[
        ("oneHotEncoder", OneHotEncoder()),
        ("logisticRegression", LogisticRegressionCV(Cs=10)),
    ],
)

# Entrene el pipeline con los datos de entrenamiento.
pipeline.fit(X_train, y_train)

print(pipeline.score(X_train, y_train).round(6))
print(pipeline.score(X_test, y_test).round(6))

# Importe confusion_matrix
from sklearn.metrics import confusion_matrix

# Evalúe el pipeline con los datos de entrenamiento usando la matriz de confusion.
cfm_train = confusion_matrix(
    y_true=y_train,
    y_pred=pipeline.predict(X_train),
)

cfm_test = confusion_matrix(
    y_true=y_test,
    y_pred=pipeline.predict(X_test),
)

print(cfm_train.tolist())
print(cfm_test.tolist())