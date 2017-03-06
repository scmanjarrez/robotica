import numpy as np
from sklearn.metrics.pairwise import euclidean_distances


class Classifier:
        def __init__(self):
                pass

        def fit(self):
                pass

        def predict(self):
                pass


class clasifEuclid(Classifier):
        def __init__(self):
                pass

        def fit(self, DataMatrix, TagsArray):
                """Entrena el clasificador
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                TagsArray: vector de etiquetas, tantos elementos como filas en X
                retorna objeto clasificador"""
                # Obtenemos una lista con las clases
                self.classes = list(set(TagsArray))
                # Obtenemos los centroides, para ello hallamos la media de columnas entre todos los elementos de la misma clase
                self.centroide = [np.mean(DataMatrix[[pos for pos, elem in enumerate(TagsArray) if elem == name]], axis=0) for name in self.classes]
                return self.centroide, self.classes

        def predict(self, DataMatrix):
                """Estima el grado de pertenencia de cada dato a las clases
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                retorna una matriz, cada fila almacena los valores pertenencia"""
                n_rows, n_columns = DataMatrix.shape
                # Realizamos el calculo de la distancia euclidea, aplicando la ecuacion
                # self.data = [[self.calcula(DataMatrix[i], self.centroide[j])for j in range(len(self.centroide))]for i in range(n_rows)]
                self.data = [euclidean_distances(DataMatrix, c) for c in self.centroids]

        def predLabel(self, DataMatrix):
                """Estima la etiqueta de cada dato
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                retorna un vector con las etiquetas de cada dato"""
                self.predict(DataMatrix)
                # Calculamos el valor mas alto, y a partir de este obtenemos el nombre de la etiqueta
                tags = [self.classes[np.argmax(elem)] for elem in self.data]
                return tags

        def calcula(self, Punto, Centroide):
                """Funcion auxiliar encargada de realizar el calculo de la distancia
                euclidea, de acuerdo a: (Z1,...Zn, -1/2(Zt*Z)) * (X1,...Xn,1)"""
                Zmatrix = np.append(Centroide, -1.0/2*np.dot(Centroide.T, Centroide))
                Xmatrix = np.append(Punto, 1)
                resul = np.dot(Zmatrix, Xmatrix)
                return resul
