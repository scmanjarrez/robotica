# -*- coding: utf-8 -*-
import numpy as np
import scipy.stats as stats
import scipy.linalg as la
from sklearn.metrics.pairwise import euclidean_distances
import sys


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

        def fit(self, listaClases):
                """Entrena el clasificador
                listaClases: lista de clases, cada elemento tiene Y columnas (medidas)
                retorna objeto clasificador"""
                # Lista de clases
                self.classes = range(len(listaClases))
                # Obtenemos los centroides, para ello hallamos la media de columnas entre todos los elementos de la misma clase
                self.centroide = [np.mean(listaClases[i], axis=0) for i in self.classes]
                return self.centroide, self.classes

        def predict(self, DataMatrix):
                """Estima el grado de pertenencia de cada dato a las clases
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                retorna una matriz, cada fila almacena los valores pertenencia"""
                # Realizamos el calculo de la distancia euclidea, aplicando la ecuacion
                self.data = [euclidean_distances(row, self.centroide) for row in DataMatrix]

        def predLabel(self, DataMatrix):
                """Estima la etiqueta de cada dato
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                retorna un vector con las etiquetas de cada dato"""
                self.predict(DataMatrix)
                # Calculamos el valor mas alto, y a partir de este obtenemos el nombre de la etiqueta
                tags = [[self.classes[np.argmax(subrow)] for subrow in row] for row in self.data]
                return tags


class clasifCuadrad(Classifier):
        def __init__(self):
                pass

        def fit(self, listaClases):
                """Entrena el clasificador
                listaClases: lista de clases, cada elemento tiene Y columnas (medidas)
                retorna objeto clasificador"""
                # Obtenemos una lista con las clases
                self.classes = range(len(listaClases))
                # Obtenemos las medias de cada clase, seran n medias por clase
                self.medias = [np.mean(listaClases[i], axis=0) for i in self.classes]
                # Obtenemos la matriz de covarianzas
                self.covmatrix = [np.cov(clase.T) for clase in listaClases]
                return self.classes, self.medias, self.covmatrix

        def predict(self, DataMatrix):
                """Estima el grado de pertenencia de cada dato a las clases
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                retorna una matriz, cada fila almacena los valores pertenencia"""
                # Calculamos la funcion de distribucion gaussiana haciendo uso de la libreria
                # de scipy.stats.multivariate_normal.pdf
                aux = [stats.multivariate_normal(self.medias[i], self.covmatrix[i], allow_singular=True) for i in range(len(self.classes))]
                self.data = [[[elem.pdf(subrow) for elem in aux] for subrow in row] for row in DataMatrix]

        def predLabel(self, DataMatrix):
                """Estima la etiqueta de cada dato
                DataMatrix: matriz numpy cada fila es un dato, cada columna una medida
                retorna un vector con las etiquetas de cada dato"""
                self.predict(DataMatrix)
                # Calculamos el valor mas alto, y a partir de este obtenemos el nombre de la etiqueta
                tags = [self.classes[np.argmax(elem)] for elem in self.data]
                return tags


def myError(Clasif, DataMatrixTrain, TagsArrayTrain, DataMatrixTest, TagsArrayTest):
        """ Funcion que cuenta los errores cometidos por un clasificador.
        clasif: clasificador que se desea evaluar
        Xtrain: matriz numpy que almacena los datos de entrenamiento
        cada fila es un dato, cada columna una medida
        ytrain: vector que almacena las etiquetas de entrenamiento,
        tiene tantos elementos como filas hay en X
        Xtest: matriz numpy que almacena los datos de test
        cada fila es un dato, cada columna una medida
        ytest: vector que almacena las etiquetas de test,
        tiene tantos elementos como filas hay en X"""
        # Entrenamos, predecimos y calculamos los errores entre la prediccion
        # y el etiquetado que teniamos anteriormente
        Clasif.fit(DataMatrixTrain, TagsArrayTrain)
        predEtiq = Clasif.predLabel(DataMatrixTest)
        return len([i for i in range(len(TagsArrayTest)) if TagsArrayTest[i] != predEtiq[i]])
