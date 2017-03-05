#!/usr/bin/python
import numpy as np
import re
import sys
from os import listdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid

import cv2
import seg
import select_pixels as sel


def marking(video):
    capture = cv2.VideoCapture(video)
    count = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Frame', frame)

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(1000)

            #    key = 1010 0000 0000 1011 0110 1110
            #     &
            #   0xFF =                     1111 1111
            # n==110 =                     0110 1110

            # (n)ext image
            if key & 0xFF == ord('n'):
                count += 1
                continue

            # mark image, (s)top
            if key & 0xFF == ord('s'):
                # change from BGR to RGB format
                imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mark training pixels
                markImg = sel.select_fg_bg(imRGB)

                imsave('OriginalImg'+str(count)+'.png', imRGB)
                imsave('TrainingImg'+str(count)+'.png', markImg)

            # (q)uit program
            if key & 0xFF == ord('q'):
                break

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


def training():
    # Height x Width x channel
    origImg = imread('OriginalImg1512.png')
    markImg = imread('TrainingImg1512.png')

    # Normalization: all = R+G+B, R = R/all, G = G/all, B = B/all
    # [[[1,2,3],	         [[[1,4],                                   [[[0.1666,0.2666],                      [[[0.1666,0.3333,0.5000],
    #   [4,5,6]],       	   [6,8],                                     [0.4000,0.3809],                        [0.2666,0.3333,0.4000]],
    #                              [5,8]],                                    [0.2777,0.4705]],
    #
    #
    #  [[6,5,4],   rollaxis(x,2)  [[2,5],    np.sum(x,2)  [[ 6,15],    R/S   [[0.3333,0.3333],     rollaxis(D,0,3)   [[0.4000,0.3333,0.2666],
    #   [8,7,6]],  ------------>   [5,7],    ---------->   [15,21],   ---->   [0.3333,0.3333],     -------------->    [0.3809,0.3333,0.2857]],
    #                    R         [6,9]],        S        [18,17]]     D     [0.3333,0.5294]],
    #
    #  [[5,6,7],                  [[3,6],                                    [[0.5000,0.4000],                       [[0.2777,0.3333,0.3888],
    #   [8,9,0]]]          	   [4,6],                                     [0.2666,0.2857],                        [0.4705,0.5294,0.0000]]]
    #                       	   [7,0]]]                                    [0.3888,0.0000]]]
    ImgNorm = np.rollaxis((np.rollaxis(origImg, 2)+0.0)/np.sum(origImg, 2), 0, 3)

    # Get marked points from original image
    # np.equal(markImg, (255, 0, 0) --> X*Y*3
    # Matrix of X rows, each row have Y rows with 3 columns of booleans
    # np.all(np.equal..., 2) --> X*Y
    # Matrix of X rows with Y columns, True if pixel has red mark
    # np.where(np.all...) --> X*Y
    # Matrix of indices with red marked pixels

    # data_red = origImg[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
    # data_green = origImg[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
    # data_blue = origImg[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

    data_redN = ImgNorm[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
    data_greenN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
    data_blueN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

    # data = np.concatenate([data_red, data_green, data_blue])
    dataN = np.concatenate([data_redN, data_greenN, data_blueN])

    # target = np.concatenate([np.zeros(len(data_red[:]), dtype=int),
    #                          np.ones(len(data_green[:]), dtype=int),
    #                          np.full(len(data_blue[:]), 2, dtype=int)])

    targetN = np.concatenate([np.zeros(len(data_redN[:]), dtype=int),
                              np.ones(len(data_greenN[:]), dtype=int),
                              np.full(len(data_blueN[:]), 2, dtype=int)])

    # clf = NearestCentroid()
    # clf.fit(data, target)

    clfN = NearestCentroid()
    clfN.fit(dataN, targetN)

    return clfN


def segmentation(clfN, video):
    capture = cv2.VideoCapture(video)
    count = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Original', frame)
            shape = frame.shape
            ImgNorm = np.rollaxis((np.rollaxis(frame, 2)+0.0)/np.sum(frame, 2), 0, 3)

            # Reshape in order to reduce the 3-dimensional array to 2-dimensional (needed for predict)
            # labels = clf.predict(frame.reshape(shape[0]*shape[1], 3))
            labelsN = clfN.predict(ImgNorm.reshape(shape[0]*shape[1], 3))

            # labels = clf.segmenta(frame)
            paleta = np.array([[0, 0, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)

            # Reshape back, from 2-dimensional to 3-dimensional
            # aux = paleta[labels]
            auxN = paleta[labelsN]
            # segm = aux.reshape(shape[0], shape[1], 3)
            segmN = auxN.reshape(shape[0], shape[1], 3)

            # segmImg = cv2.cvtColor(segm, cv2.COLOR_RGB2BGR)
            # cv2.imshow("Segmentation", segmImg)

            segmImgN = cv2.cvtColor(segmN, cv2.COLOR_RGB2BGR)
            cv2.imshow("SegmNormalized", segmImgN)

            # cv2.imwrite('SegmImg'+str(count)+'.png', segmImgN)
            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(100)

            # (q)uit program
            if key & 0xFF == ord('q'):
                break

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


def genVideo():
    direct = "segmentation"
    images = [f for f in listdir(direct) if isfile(join(direct, f))]
    images = natural_sort(images)
    img1 = cv2.imread(direct+"/"+images[0])

    height, width, layers = img1.shape

    video = cv2.VideoWriter('segmentation.avi', cv2.cv.CV_FOURCC('M', 'P', '4', '2'), 5.0, (width, height))

    for img in images:
        video.write(cv2.imread(direct+"/"+img))

    cv2.destroyAllWindows()
    video.release()


def natural_sort(images_list):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(images_list, key=alphanum_key)


def main(video):
    # marking(video)
    clfN = training()
    segmentation(clfN, video)
    # genVideo()

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        print "Expected one argument, choosing default video."
        main("video2017-3.avi")
    else:
        main(sys.argv[1])
