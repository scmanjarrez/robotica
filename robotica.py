#!/usr/bin/python
import numpy as np
import re
import sys
from os import listdir, mkdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid
import cv2
import select_pixels as sel
import argparse


video = 'video2017-3.avi'
trainImg = '1536'
trainDir = 'TrainFrames'
segmDir = 'SegmFrames'


def marking():
    capture = cv2.VideoCapture(video)
    count = 0

    try:
        mkdir(trainDir)
    except OSError:
        # print "Directory already created."
        pass

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

                imsave(join(trainDir, 'OriginalImg'+str(count)+'.png'), imRGB)
                imsave(join(trainDir, 'TrainingImg'+str(count)+'.png'), markImg)

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
    origImg = imread(join(trainDir, 'OriginalImg'+trainImg+'.png'))
    markImg = imread(join(trainDir, 'TrainingImg'+trainImg+'.png'))

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

    data_redN = ImgNorm[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
    data_greenN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
    data_blueN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

    dataN = np.concatenate([data_redN, data_greenN, data_blueN])

    targetN = np.concatenate([np.zeros(len(data_redN[:]), dtype=int),
                              np.ones(len(data_greenN[:]), dtype=int),
                              np.full(len(data_blueN[:]), 2, dtype=int)])

    # Train the system with +20 images
    # dataN, targetN = training_multiple_images()

    clfN = NearestCentroid()
    clfN.fit(dataN, targetN)

    return clfN


def segmentation(clfN, args):
    capture = cv2.VideoCapture(video)
    count = 0

    if args.genVideo:
        try:
            mkdir(segmDir)
        except OSError:
            # print "Directory already created."
            pass

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Original', frame)
            shape = frame.shape

            ImgNorm = np.rollaxis((np.rollaxis(frame, 2)+0.0)/np.sum(frame, 2), 0, 3)

            # Reshape in order to reduce the 3-dimensional array to 2-dimensional (needed for predict)
            labelsN = clfN.predict(ImgNorm.reshape(shape[0]*shape[1], 3))

            paleta = np.array([[0, 0, 255], [0, 0, 0], [255, 0, 0]], dtype=np.uint8)

            # Reshape back, from 2-dimensional to 3-dimensional
            auxN = paleta[labelsN]
            segmN = auxN.reshape(shape[0], shape[1], 3)

            segmImgN = cv2.cvtColor(segmN, cv2.COLOR_RGB2BGR)

            cv2.imshow("SegmNormalized", segmImgN)

            if args.genVideo:
                cv2.imwrite(join(segmDir, 'SegmImg'+str(count)+'.png'), segmImgN)

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


def gen_video(name):
    images = [f for f in listdir(segmDir) if isfile(join(segmDir, f))]

    if not len(images):
        print "No images to create the video."
        sys.exit()

    images = natural_sort(images)
    aux = cv2.imread(join(segmDir, images[0]))

    height, width, layers = aux.shape

    video = cv2.VideoWriter(name+'.avi', cv2.cv.CV_FOURCC('M', 'P', '4', '2'), 1.0, (width, height))

    for img in images:
        video.write(cv2.imread(join(segmDir, img)))

    cv2.destroyAllWindows()
    video.release()


def natural_sort(images_list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(images_list, key=alphanum_key)


def training_multiple_images():
    train = ['240', '504', '576', '600', '1272',
             '1488', '1512', '1536', '1776',
             '1944', '2160', '2304', '2784']

    origImg = imread(join(trainDir, 'OriginalImg'+train[0]+'.png'))
    markImg = imread(join(trainDir, 'TrainingImg'+train[0]+'.png'))
    ImgNorm = np.rollaxis((np.rollaxis(origImg, 2)+0.0)/np.sum(origImg, 2), 0, 3)

    data_redN = ImgNorm[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
    data_greenN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
    data_blueN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

    dataN = np.concatenate([data_redN, data_greenN, data_blueN])

    targetN = np.concatenate([np.zeros(len(data_redN[:]), dtype=int),
                              np.ones(len(data_greenN[:]), dtype=int),
                              np.full(len(data_blueN[:]), 2, dtype=int)])

    for elem in train[1:]:
        origImg = imread(join(trainDir, 'OriginalImg'+elem+'.png'))
        markImg = imread(join(trainDir, 'TrainingImg'+elem+'.png'))
        ImgNorm = np.rollaxis((np.rollaxis(origImg, 2)+0.0)/np.sum(origImg, 2), 0, 3)

        data_redN = ImgNorm[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
        data_greenN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
        data_blueN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

        dataN = np.concatenate([dataN, data_redN, data_greenN, data_blueN])

        targetN = np.concatenate([targetN,
                                  np.zeros(len(data_redN[:]), dtype=int),
                                  np.ones(len(data_greenN[:]), dtype=int),
                                  np.full(len(data_blueN[:]), 2, dtype=int)])

    return dataN, targetN

def main(parser, args):
    global video, trainImg

    if args.video:
        video = args.video

    if args.trainImg:
        trainImg = args.trainImg

    # We want to mark lots of images,
    # then choose the ones in the training process
    if args.mark:
        marking()

    elif args.seg:
        clfN = training()
        segmentation(clfN, args)

    if args.genVideo:
        gen_video(args.genVideo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PROG')

    parser.add_argument('-v', '--video',
                        help='Select a different video.')

    parser.add_argument('-ti', '--trainImg',
                        help='Select a different trainingImg.')

    group = parser.add_argument_group('Commands')

    group.add_argument('-m', '--mark',
                       action='store_true',
                       help='Start marking process.')

    group.add_argument('-s', '--seg',
                       action='store_true', default='True',
                       help='Start segmentation process.')

    group.add_argument('-gv', '--genVideo',
                       nargs='?', const='segmentation',
                       help='Generate segmentation video.')

    args = parser.parse_args()

    main(parser, args)
