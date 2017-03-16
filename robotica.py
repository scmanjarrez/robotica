#!/usr/bin/python
import argparse
import numpy as np
import re
import sys
from os import listdir, mkdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid

import cv2
import select_pixels as sel


# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d


video = 'video2017-3.avi'
trainImg = '1512'
trainDir = 'TrainFrames'
segmDir = 'SegmFrames'
normDir = 'NormFrames'


def marking():
    capture = cv2.VideoCapture(video)
    count = 0

    check_dir(trainDir)

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
            if (key & 0xFF) == ord('n'):
                count += 1
                continue

            # mark image, (s)top
            if (key & 0xFF) == ord('s'):
                # change from BGR to RGB format
                imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mark training pixels
                markImg = sel.select_fg_bg(imRGB)

                imsave(join(trainDir, 'OriginalImg'+str(count)+'.png'), imRGB)
                imsave(join(trainDir, 'TrainingImg'+str(count)+'.png'), markImg)

            # (q)uit program
            if (key & 0xFF) == ord('q'):
                break

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


def training(args):
    check_dir(trainDir)
    check_dir(normDir)

    if not args.multiTrain:
        # Height x Width x channel
        origImg = imread(join(trainDir, 'OriginalImg'+trainImg+'.png'))
        # origImg = imread(join(trainDir, 'frame.png'))

        if not args.gimpImg:
            markImg = imread(join(trainDir, 'TrainingImg'+trainImg+'.png'))
            # markImg = imread(join(trainDir, 'frame_painted.png'))
        else:
            markImg = imread(join(trainDir, 'GimpTrain'+trainImg+'.png'))

        # Normalization: all = R+G+B, R = R/all, G = G/all, B = B/all
        # [[[1, 2, 3],                 [[[1, 4],                                     [[[1/6 , 4/15],                      [[[1/6 , 2/6 , 3/6 ],
        #   [4, 5, 6]],                  [6, 8],                                       [6/15, 8/21],                        [4/15, 5/15, 6/15]],
        #                                [5, 8]],                                      [5/18, 8/17]],
        #
        #
        #  [[6, 5, 4],   rollaxis(x,2)  [[2, 5],   np.sum(x,2)   [[ 6,15],     R/S    [[2/6 , 5/15],     rollaxis(D,0,3)   [[6/15, 5/15, 4/15],
        #   [8, 7, 6]],  ------------>   [5, 7],   ---------->    [15,21],   ------>   [5/15, 7/21],     -------------->    [8/21, 7/21, 6/21]],
        #                      R         [6, 9]],       S         [18,17]]      D      [6/18, 9/17]],
        #
        #  [[5, 6, 7],                  [[3, 6],                                      [[3/6 , 6/15],                       [[5/18, 6/18, 7/18],
        #   [8, 9, 0]]]                  [4, 6],                                       [4/15, 6/21],                        [8/17, 9/17, 0/17]]]
        #                                [7, 0]]]                                      [7/18, 0/17]]]
        ImgNorm = np.rollaxis((np.rollaxis(origImg, 2)+0.0)/np.sum(origImg, 2), 0, 3)

        if args.normImg:
            imsave(join(normDir, 'Norm'+trainImg+'.png'), ImgNorm*255)

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

        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # plt.rcParams['legend.fontsize'] = 10

        # data = [data_redN, data_greenN, data_blueN]
        # colors_markers = ('red', 'green', 'blue')
        # for pos in range(3):
        #     ax.plot(data[pos][:, 0], data[pos][:, 1], data[pos][:, 2], '*',
        #             markersize=5, color=colors_markers[pos], alpha=0.5, label=colors_markers[pos])

        # plt.legend(loc='upper left')
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 1)
        # ax.set_zlim(0, 1)
        # plt.show()
    else:
        # Train the system with +20 images
        dataN, targetN = training_multiple_images()

    clfN = NearestCentroid()
    clfN.fit(dataN, targetN)

    return clfN


def segmentation(clfN, args):
    capture = cv2.VideoCapture(video)
    count = 0

    if args.genVideo:
        check_dir(segmDir)

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret and not count % 24:
            rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow('Original', frame)

            shape = frame.shape

            ImgNorm = np.rollaxis((np.rollaxis(rgbFrame, 2)+0.0)/np.sum(rgbFrame, 2), 0, 3)

            # Reshape in order to reduce the 3-dimensional array to 2-dimensional (needed for predict)
            labelsN = clfN.predict(ImgNorm.reshape(shape[0]*shape[1], 3))

            paleta = np.array([[255, 0, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8)

            # Reshape back, from 2-dimensional to 3-dimensional
            auxN = paleta[labelsN]
            segmN = auxN.reshape(shape[0], shape[1], 3)

            segmImgN = cv2.cvtColor(segmN, cv2.COLOR_RGB2BGR)

            gray = cv2.cvtColor(segmN, cv2.COLOR_RGB2GRAY)

            gauss_blur = cv2.GaussianBlur(gray, (5, 5), 2)

            retval, thresh = cv2.threshold(gauss_blur, 10, 255, cv2.THRESH_BINARY)

            cnts, hier = cv2.findContours(thresh.copy()[90:], cv2.RETR_EXTERNAL,
                                          cv2.CHAIN_APPROX_SIMPLE)

            # drawContours is destructive
            copy = frame.copy()[90:]

            # cnts, hier = cv2.findContours(gauss_blur.copy()[90:], cv2.RETR_EXTERNAL,
            #                               cv2.CHAIN_APPROX_SIMPLE)

            # # Return list of indices of points in contour
            # chullList = [cv2.convexHull(cont, returnPoints=False) for cont in cnts]

            # # Return convexity defects from previous contours, each contour must have at least 3 points
            # # Convexity Defect = [start_point, end_point, farthest_point, distance_to_farthest_point]
            # convDefs = [cv2.convexityDefects(cont, chull) for (cont, chull) in
            #             zip(cnts, chullList) if len(cont) > 3 and len(chull) > 3]

            # for pos, cont in enumerate(cnts):
            #     cnvDef = convDefs[pos]
            #     # Saco la lista de agujeros del contorno
            #     listConvDefs = cnvDef[:, 0, :].tolist()
            #     # Devuelvo la lista de agujeros mayores de 4 pixeles, aproximadamente
            #     bigger_4_px = [[init, end, mid, length] for init, end, mid, length in listConvDefs if length > 1000]

            #     if len(bigger_4_px):
            #         print bigger_4_px[0]
            #         cv2.line(copy, bigger_4_px[0][0], bigger_4_px[0][1], [255, 255, 255], 2)
            #         cv2.circle(copy, bigger_4_px[0][2], 5, [0, 0, 255], -1)

            cv2.drawContours(copy, cnts, -1, (0, 255, 0), 1)

            cv2.imshow("Contours", copy)

            cv2.imshow("SegmNormalized", segmImgN)

            if args.genVideo:
                cv2.imwrite(join(segmDir, 'SegmImg'+str(count)+'.png'), segmImgN)

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(1000)

            # (n)ext image
            if (key & 0xFF) == ord('n'):
                count += 1
                continue

            # (q)uit program
            if (key & 0xFF) == ord('q'):
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

    if args.normImg:
        imsave(join(normDir, 'Norm'+train[0]+'.png'), ImgNorm*255)

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

        if args.normImg:
            imsave(join(normDir, 'Norm'+elem+'.png'), ImgNorm*255)

        data_redN = ImgNorm[np.where(np.all(np.equal(markImg, (255, 0, 0)), 2))]
        data_greenN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 255, 0)), 2))]
        data_blueN = ImgNorm[np.where(np.all(np.equal(markImg, (0, 0, 255)), 2))]

        dataN = np.concatenate([dataN, data_redN, data_greenN, data_blueN])

        targetN = np.concatenate([targetN,
                                  np.zeros(len(data_redN[:]), dtype=int),
                                  np.ones(len(data_greenN[:]), dtype=int),
                                  np.full(len(data_blueN[:]), 2, dtype=int)])

    return dataN, targetN


def check_dir(dirName):
    try:
        mkdir(dirName)
    except OSError:
        # print "Directory already created."
        pass


def main(parser, args):
    global video, trainImg

    if args.video:
        video = args.video

    if args.trainImg:
        trainImg = args.trainImg

    # Mark lots of images
    if args.mark:
        marking()
    # Select the ones you want to train
    elif args.seg:
        clfN = training(args)
        segmentation(clfN, args)

    if args.genVideo:
        gen_video(args.genVideo)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='PROG')

    parser.add_argument('-v', '--video',
                        help='Select a different video.')

    parser.add_argument('-ti', '--trainImg',
                        help='Select a different trainingImg.')

    parser.add_argument('-mt', '--multiTrain',
                        action='store_true',
                        help='Train the system with multiple images.')

    parser.add_argument('-g', '--gimpImg',
                        action='store_true',
                        help='Train the system with a fully colored image.')

    parser.add_argument('-n', '--normImg',
                        action='store_true',
                        help='Save normalized traning images.')

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
