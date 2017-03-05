#!/usr/bin/python
import sys
from scipy.misc import imsave, imread
from sklearn.neighbors.nearest_centroid import NearestCentroid

import select_pixels as sel
import numpy as np
import cv2


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
    origImg = imread('OriginalImg1272.png')
    markImg = imread('TrainingImg1272.png')

    # Get marked points from original image
    # np.equal(markImg, (255, 0, 0) --> X*Y*3
    # Matrix of X rows, each row have Y rows with 3 columns of booleans
    # np.all(np.equal..., 2) --> X*Y
    # Matrix of X rows with Y columns, True if pixel has red mark
    # np.where(np.all...) --> X*Y
    # Matrix of indices with red marked pixels
    data_red = origImg[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
    data_green = origImg[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
    data_blue = origImg[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

    data = np.concatenate([data_red, data_green, data_blue])
    target = np.concatenate([np.zeros(len(data_red[:]), dtype=int),
                             np.ones(len(data_green[:]), dtype=int),
                             np.full(len(data_blue[:]), 2, dtype=int)])

    clf = NearestCentroid()
    clf.fit(data, target)
    # ImgNorm = np.rollaxis((np.rollaxis(origImg, 2)+0.0)/np.sum(origImg, 2), 0, 3)[:, :, :2]
    return clf


def segmentation(clf, video):
    capture = cv2.VideoCapture(video)
    count = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Frame', frame)
            shape = frame.shape
            # Reshape in order to reduce the 3-dimensional array to 2-dimensional (needed for predict)
            labels = clf.predict(frame.reshape(shape[0]*shape[1], 3))
            paleta = np.array([[0, 0, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)
            # Reshape back, from 2-dimensional to 3-dimensional
            aux = paleta[labels]
            segm = aux.reshape(shape[0], shape[1], 3)
            cv2.imshow("Segmentation", cv2.cvtColor(segm, cv2.COLOR_RGB2BGR))


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


def main(video):
    # marking(video)
    clf = training()
    segmentation(clf, video)

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        print "Expected one argument, choosing default video."
        main("video2017-3.avi")
    else:
        main(sys.argv[1])
