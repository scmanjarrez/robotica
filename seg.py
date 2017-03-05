import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
class Classifier:
    def __init__(self):
        pass

    def fit(self):
        pass

    def predLabel(self):
        pass


class segEuclid(Classifier):
    def __init__(self,X):
        	"""Entrena el clasificador
       		X: matriz numpy cada fila es un dato, cada columna una medida
        	y: vector de etiquetas, tantos elementos como filas en X
           	retorna objeto clasificador"""
        	#print y
		"""se obtienen las distintas clases"""
		#print p
		"""se obtienen los indices de los datos que pertenecen a cada clase"""
		#print t[0]
		#o=[sum(X[i,:]) for i in t]
                #print o 
		#o=[i.tolist() for i in o] 
		#print o
		#o=np.asarray(o)
		#u=[[o[j,i]/len(t[j]) for i in range(0,len(o[j]))] for j in range(0,len(t))]
		#print u
		"""se calcula la media de cada clase en base a los datos pertenecientes a las mismas"""
		self.c=[np.mean(X[i],axis=0)  for i in range(0,len(X))]
		#print "opcional"
		#print self.c
		"""guardo las clases y retorno el propio objeto"""
    def predict(self,X):
        	"""Estima el grado de pertenencia de cada dato a las clases 
        	X: matriz numpy cada fila es un dato, cada columna una medida
           	retorna una matriz, cada fila almacena los valores pertenencia"""
        	#print np.asarray(self.c)
                #print X
		"""guardo las meidas en una variable"""
		centr=np.asarray(self.c)
		"""hago el calculo de la distancia de cada dato a cada media de cada clase"""
		#v= [euclidean_distances(X[i],centr)for i in range(0,len(X))]
		centrd=[(0.5*np.dot(centr[k],np.transpose(centr[k]))) for k in range(0,len(centr))]
		centr2=np.transpose(centr)
                v=[np.dot(X[i],centr2)-centrd for i in range(0,len(X))]
		#r=np.asarray(r)
		
		#print v
		#print centrd
		#print r[0]-centrd
		
  		
		#print v
		#print 'CAMBIO'
		#print "intermedio"
		#print v
		#print np.dot(np.transpose(centr[0,:]),centr[0,:])
		"""devuelvo la matriz"""
		return v 

    def segmenta(self,X):
        	"""Estima la etiqueta de cada dato
                X: matriz numpy cada fila es un dato, cada columna una medida
                retorna un vector con las etiquetas de cada dato"""
                r2=self.predict(X)
                r=np.asarray(r2)
		""" se obtienen los valores maximos y se obtiene la etiqueta asociada al mayor valor"""
                re=[np.argmax(r[i],axis=1)for i in range(0,len(r))]
		#print re
		"""se retornan esaos valores maximos"""
		return re

