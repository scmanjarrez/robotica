# -*- coding: utf-8 -*-
import signal
from multiprocessing import Process
import numpy as np
import clasifScKt as cl
from sklearn import cross_validation as cv
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from scipy.misc import imread
import sys


def main(DB):
    if DB == 'iris':
        # DataMatrix = np.genfromtxt('irisData.txt', dtype=float,
        #                            delimiter=None, usecols=np.arange(0, 4))
        # TagsArray = np.genfromtxt('irisData.txt', dtype=str, usecols=-1)
        origImg = imread('OriginalImg1512.png')
        markImg = imread('TrainingImg1512.png')

    elif DB == 'cancer':
        DataMatrix = np.genfromtxt('wdbc.data', dtype=float, delimiter=",", usecols=np.arange(2, 32))
        TagsArray = np.genfromtxt('wdbc.data', dtype=str, delimiter=",", usecols=1)

    elif DB == 'wine':
        DataMatrix = np.genfromtxt('wine.data', dtype=float, delimiter=",", usecols=np.arange(1, 14))
        TagsArray = np.genfromtxt('wine.data', dtype=str, delimiter=",", usecols=0)

    elif DB == 'isolet':
        DataMatrix = np.genfromtxt('isolet1234.data', dtype=float, delimiter=",", usecols=np.arange(0, 617))
        TagsArray = np.genfromtxt('isolet1234.data', dtype=str, delimiter=",", usecols=-1)

    # Para que no se bloquee el proceso en la grafica, se hace uso de multiproceso
    # para evitar este comportamiento
    # dim = nonBlockingRawInput("Choose dimensions to be represented: 2, 3\n", True)
    # plotOtherProcess = Process(target=printSamples, args=(DB, DataMatrix, TagsArray, dim))
    # plotOtherProcess.start()

    ImgNorm = np.rollaxis((np.rollaxis(origImg, 2)+0.0)/np.sum(origImg, 2), 0, 3)
    data_redN = ImgNorm[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
    data_greenN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
    data_blueN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

    # Creo el clasificador
    clasifDE = cl.clasifEuclid()
    centroides, classes = clasifDE.fit([data_redN, data_greenN, data_blueN])

    predict = clasifDE.predLabel(origImg)
    print predict[0]
    sys.exit()
    # clasifEB = cl.clasifCuadrad()

    # Generamos los indices de validacion cruzada con 10 folds (grupos)
    # skf = cv.StratifiedKFold(TagsArray, 10)

    # Resustitucion
    # erroresDEre = cl.myError(clasifDE, DataMatrix, TagsArray, DataMatrix, TagsArray)
    # erroresEBre = cl.myError(clasifEB, DataMatrix, TagsArray, DataMatrix, TagsArray)

    # print '\n\nRESUSTITUCION'

    # print 'Distancia Euclidea:'
    # print ' *Errores: ', erroresDEre
    # print ' *Tasa de error: ', erroresDEre*1.0/DataMatrix.shape[0]
    # print ' *Tasa de acierto: ', (DataMatrix.shape[0]-erroresDEre)*1.0/DataMatrix.shape[0]

    # print 'Estadistico Bayesiano:'
    # print ' *Errores: ', erroresEBre
    # print ' *Tasa de error: ', erroresEBre*1.0/DataMatrix.shape[0]
    # print ' *Tasa de acierto: ', (DataMatrix.shape[0]-erroresEBre)*1.0/DataMatrix.shape[0]

    # # Validacion cruzada
    # erroresDEcv = sum([cl.myError(clasifDE, DataMatrix[train], TagsArray[train], DataMatrix[test], TagsArray[test]) for train, test in skf])
    # erroresEBcv = sum([cl.myError(clasifEB,  DataMatrix[train], TagsArray[train], DataMatrix[test], TagsArray[test]) for train, test in skf])

    # print '\nVALIDACION CRUZADA'

    # print 'Distancia Euclidea:'
    # print ' *Errores: ', erroresDEcv
    # print ' *Tasa de error: ', erroresDEcv*1.0/DataMatrix.shape[0]
    # print ' *Tasa de acierto: ', (DataMatrix.shape[0]-erroresDEcv)*1.0/DataMatrix.shape[0]

    # print 'Estadistico Bayesiano:'
    # print ' *Errores: ', erroresEBcv
    # print ' *Tasa de error: ', erroresEBcv*1.0/DataMatrix.shape[0]
    # print ' *Tasa de acierto: ', (DataMatrix.shape[0]-erroresEBcv)*1.0/DataMatrix.shape[0]

    # print '\n\nClose plot window to finish...'
    # plotOtherProcess.join()


def printSamples(DB, DataMatrix, TagsArray, dim):
    """ Funcion para visualizar los datos en una grafica haciendo
    uso de la libreria matplotlib.
    Se redimensionan las caracteristicas de todos los datos a dim y
    se imprime cada una en un eje """
    transf = pca(DataMatrix, dim)

    classes = list(set(TagsArray))
    indexes = [[pos for pos, elem in enumerate(TagsArray) if elem == name] for name in classes]
    filled_markers = ('o', 'v', 'd', '*', '>', '8', '<', 's', '^', 'h', 'H', 'D', 'p')
    colors_markers = ('blue', 'red', 'magenta', 'green', 'cyan', 'yellow', 'black', 'white')

    if dim == "2":
        for pos, name in enumerate(classes):
            auxM = pos % len(filled_markers)
            auxC = pos % len(colors_markers)
            plt.plot(transf[indexes[pos], 0], transf[indexes[pos], 1],
                     filled_markers[auxM], markersize=5, color=colors_markers[auxC], alpha=0.5, label=name)

    elif dim == "3":
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')
        plt.rcParams['legend.fontsize'] = 10

        for pos, name in enumerate(classes):
            auxM = pos % len(filled_markers)
            auxC = pos % len(colors_markers)
            ax.plot(transf[indexes[pos], 0], transf[indexes[pos], 1], transf[indexes[pos], 2],
                    filled_markers[auxM], markersize=5, color=colors_markers[auxC], alpha=0.5, label=name)

    plt.legend(loc='upper left')
    plt.title('Base de Datos '+DB)
    plt.show()


def pca(DataMatrix, dim):
    """ Funcion que realiza la transformacion de datos a un nuevo
    subespacio de dimension dim """
    mean = np.mean(DataMatrix)
    diff = DataMatrix-mean
    covsMatrix = np.cov(diff.T)
    eigVal, eigVec = np.linalg.eig(covsMatrix)
    revSort = np.argsort(eigVal)[::-1]
    eigVal = eigVal[revSort]
    eigVec = eigVec[:, revSort]
    projMatrix = eigVec[:, 0:int(dim)]
    return np.asarray([np.dot(projMatrix.T, diff[i]) for i in range(len(diff))])


class AlarmException(Exception):
    pass


def alarmHandler(signum, frame):
    raise AlarmException


def nonBlockingRawInput(prompt='', dim=False, timeout=2):
    """ Funcion que lee por stdin, intentando imitar una lectura
    no bloqueante, ya que tras cierto periodo, se usara una opcion por defecto """

    signal.signal(signal.SIGALRM, alarmHandler)
    signal.alarm(timeout)
    try:
        text = raw_input(prompt)
        signal.alarm(0)
        return text
    except AlarmException:
        if not dim:
            print '\nTimeout: Choosing default DB - iris.'
        else:
            print '\nTimeout: Choosing default dimension - 3.'

    signal.signal(signal.SIGALRM, signal.SIG_IGN)
    if not dim:
        return 'iris'
    else:
        return '3'


if __name__ == '__main__':
    # db = nonBlockingRawInput("Choose database to be used: iris, wine, cancer, isolet\n")
    main('iris')
