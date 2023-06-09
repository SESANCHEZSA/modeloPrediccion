# Importamos las librerías necesarias
import numpy as np # Para operaciones matriciales
import pandas as pd # Para manipulación de datos
from sklearn.model_selection import train_test_split # Para dividir los datos en entrenamiento y prueba
from sklearn.preprocessing import StandardScaler # Para escalar los datos
from keras.models import Sequential # Para crear el modelo de red neuronal
from keras.layers import Dense # Para añadir capas densas al modelo

# Leemos el archivo csv con los datos y mostramos sus columnas
df = pd.read_csv("datos.csv", encoding="latin-1")
r= df.columns
print(r)

# Separar las variables explicativas (X) y la variable objetivo (y)
X = df[["cantidad de fallos corregidos", "porcentaje por cada modulo"]] # Estas son las variables que usamos para predecir
y = df["posibilidad por cada modulo"] # Esta es la variable que queremos predecir

# Dividir los datos en dos partes: una para entrenar el modelo y otra para evaluarlo
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0) # Usamos el 20% de los datos para prueba y fijamos una semilla aleatoria para reproducibilidad

# Escalar los datos para que tengan una media de 0 y una desviación estándar de 1
scaler = StandardScaler() # Creamos un objeto de la clase StandardScaler
X_train = scaler.fit_transform(X_train) # Ajustamos el escalador con los datos de entrenamiento y los transformamos
X_test = scaler.transform(X_test) # Transformamos los datos de prueba con el mismo escalador

# Crear el modelo de red neuronal con dos capas ocultas y una capa de salida
model = Sequential() # Inicializamos el modelo como una secuencia de capas
model.add(Dense(units = 8, activation = "relu", input_dim = X.shape[1])) # Añadimos la primera capa oculta con 8 neuronas y función de activación ReLU. Especificamos que el número de entradas es igual al número de columnas de X.  La función ReLU permite que el modelo aprenda características no lineales y complejas de los datos.
model.add(Dense(units = 4, activation = "relu")) # Añadimos la segunda capa oculta con 4 neuronas y función de activación ReLU.
model.add(Dense(units = 1, activation = "sigmoid")) # Añadimos la capa de salida con una neurona y función de activación sigmoide. Esta función devuelve un valor entre 0 y 1 que representa la probabilidad de fallo.

# Compilar el modelo con el optimizador Adam y la función de pérdida binary_crossentropy
model.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"]) # El optimizador Adam es un algoritmo que ajusta los parámetros del modelo para minimizar la pérdida. La función de pérdida binary_crossentropy mide el error entre la predicción y la realidad para un problema de clasificación binaria. La métrica accuracy mide el porcentaje de aciertos del modelo.

# Entrenar el modelo con los datos de entrenamiento
model.fit(X_train, y_train, batch_size = 32, epochs = 100) # Usamos un tamaño de lote de 32 muestras y entrenamos el modelo durante 100 épocas. Una época es una pasada completa por todos los datos de entrenamiento.

# Evaluar el modelo con los datos de prueba
resultado = model.evaluate(X_test, y_test) # Calculamos la pérdida y la precisión del modelo en los datos de prueba.
print(resultado)#Mostramos el resultado en consola los datos de salida, que son la perdida y la porcentaje de precision para predecir los fallos. En el contexto del aprendizaje automático, una función de pérdida es una medida de cuánto se equivoca un modelo al predecir los datos. Se usa para ajustar los parámetros del modelo y minimizar el error. Existen diferentes tipos de funciones de pérdida según el tipo de problema y de modelo. Por ejemplo, la función de pérdida binary_crossentropy se usa para problemas de clasificación binaria, donde el modelo debe predecir si una muestra pertenece a una clase u otra. Esta función compara la probabilidad que el modelo asigna a cada clase con la realidad, y penaliza las predicciones incorrectas o inciertas.