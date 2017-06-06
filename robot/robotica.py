#!/usr/bin/python
import argparse
import numpy as np
import re
import sys
import time # noqa, disable flycheck warning

from matplotlib import pyplot as plt # noqa, disable flycheck warning
from os import listdir, mkdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import LeaveOneOut # noqa, disable flycheck warning
from sklearn.decomposition import PCA # noqa, disable flycheck warning

import cv2
import select_pixels as sel
import math


VIDEO = 'video2017-3.avi'
TRAIN_IMG = '576'

TRAIN_DIR = 'TrainFrames'
SEGM_DIR = 'SegmFrames'
NORM_DIR = 'NormFrames'
ANALY_DIR = 'AnalyFrames'
CHULL_DIR = 'ChullFrames'
VID_DIR = 'OutputVideos'
MARK_DIR = 'TrainMark'

MARKS = ['Cruz', 'Escalera', 'Persona', 'Telefono']
COLORS = ['red', 'blue', 'green', 'black']

FONT = cv2.FONT_HERSHEY_SIMPLEX

neigh_clf = None
pca = None
col = None
marks = None
plx = None
ply = None


class PIDController:
    def __init__(self):
        self.error = 0
        self.last_error = 0
        self.derivative = 0
        self.integral = 0
        self.latest_point = 0
        self.Kp = 1
        self.Kd = 0.1
        self.Ki = 0.2

    def getForwardTurnVelocity(self, frame):
        fringe = frame[110:130, :].copy()
        fringe_cp = fringe.copy()
        fringe_cnt, hier = cv2.findContours(fringe_cp, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        fringe_cnt = [cnt for cnt in fringe_cnt if cv2.contourArea(cnt) > 30]
        mid_point = [160, 10]
        all_mp_fringe = []
        if fringe_cnt:
            for m in fringe_cnt:
                mp = np.mean(m[:, :, :], axis=0, dtype=np.int32)
                all_mp_fringe.extend(mp.tolist())
        if all_mp_fringe:
            line_point = all_mp_fringe[np.argmin([abs(160 - mp[0]) for mp in all_mp_fringe])]
            self.error = (mid_point[0] - line_point[0])/160.0
            self.derivative = self.error - self.last_error
            self.integral = self.integral + self.error
            self.last_error = self.error

        turn = self.Kp*self.error + self.Kd*self.derivative  # + self.Ki*self.integral
        forward = max(0, 1 - abs(turn*1.5))
        return forward, turn


class TypeObjAutomaton:
    # recta==giro_izq==giro_dcha, cruce_2_vias==cruce_3_vias
    # States are -, 0 , +
    # On success increase state
    # On failure decrease state
    def __init__(self):
        self.state = 0

    def _state(self):
        return self.state

    def _reset(self):
        if self.state:
            self.state = 0

    # Return 0==marca if recta/giro, 1==flecha if cruce
    def __decrease(self):
        self.state -= 1
        return 0 if (self.state < 0) else 1

    def __increase(self):
        self.state += 1
        return 0 if (self.state < 0) else 1

    def getType(self, state):
        if state:
            return self.__decrease()
        else:
            return self.__increase()


class MarkAutomaton:
    # States are 0, 1, 2, 3
    # Get the maximum state
    def __init__(self):
        self.state = [0, 0, 0, 0]

    def _state(self):
        return self.state

    def _reset(self):
        if not all(self.state):
            self.state = [0, 0, 0, 0]

    def getType(self, pred):
        if 5 not in self.state:
            self.state[pred] += 1
            return MARKS[np.argmax(self.state)]
        else:
            return self.state.index(5)


def marking():
    capture = cv2.VideoCapture(VIDEO)
    count = 0

    make_dir(TRAIN_DIR)

    pause = False
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Image', frame)

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
                im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mark training pixels
                mark_img = sel.select_fg_bg(im_rgb)

                imsave(join(TRAIN_DIR, 'OriginalImg'+str(count)+'.png'), im_rgb)
                imsave(join(TRAIN_DIR, 'TrainingImg'+str(count)+'.png'), mark_img)

            # (q)uit program
            if (key & 0xFF) == ord('q'):
                break

            # (p)ause program
            if (key & 0xFF) == ord('p'):
                pause = not pause

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


# mark and train_img_m params in case of training knn classifier (marks)
# train with a different training image
def training(mark=False, train_img_m=''):
    make_dir(NORM_DIR)

    # Height x Width x channel
    if mark:
        orig_img = imread(join(TRAIN_DIR, 'OriginalImg'+train_img_m+'.png'))
        mark_img = imread(join(TRAIN_DIR, 'TrainingImg'+train_img_m+'.png'))
    else:
        orig_img = imread(join(TRAIN_DIR, 'OriginalImg'+TRAIN_IMG+'.png'))
        mark_img = imread(join(TRAIN_DIR, 'TrainingImg'+TRAIN_IMG+'.png'))

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
    img_norm = np.rollaxis((np.rollaxis(orig_img, 2)+0.1)/(np.sum(orig_img, 2)+0.1), 0, 3)

    # Get marked points from original image
    # np.equal(markImg, (255, 0, 0) --> X*Y*3
    # Matrix of X rows, each row have Y rows with 3 columns of booleans
    # np.all(np.equal..., 2) --> X*Y
    # Matrix of X rows with Y columns, True if pixel has red mark
    # np.where(np.all...) --> X*Y
    # Matrix of indices with red marked pixels

    data_red = img_norm[np.where(np.all(np.equal(mark_img, (255, 0, 0)), 2))]
    data_green = img_norm[np.where(np.all(np.equal(mark_img, (0, 255, 0)), 2))]
    data_blue = img_norm[np.where(np.all(np.equal(mark_img, (0, 0, 255)), 2))]

    data = np.concatenate([data_red, data_green, data_blue])

    target = np.concatenate([np.zeros(len(data_red[:]), dtype=int),
                             np.ones(len(data_green[:]), dtype=int),
                             np.full(len(data_blue[:]), 2, dtype=int)])

    # data = np.load('training.data')
    # target = np.load('training.target')
    clf = NearestCentroid()
    clf.fit(data, target)
    return clf


# mark param to segmentate all the image, not just the 90: pixels
# segm param to show a frame with the segmentated image
def segmentation(clf, frame, count, args, segm, mark=False):
    if not mark:
        shape = frame[90:, :].shape
        frame_rgb = cv2.cvtColor(frame[90:, :], cv2.COLOR_BGR2RGB)
    else:
        shape = frame.shape  # Segm all
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Segm all

    shape = frame[50:, :].shape  # Segm all
    frame_rgb = cv2.cvtColor(frame[50:, :], cv2.COLOR_BGR2RGB)  # Segm all

    img_norm = np.rollaxis((np.rollaxis(frame_rgb, 2)+0.1)/(np.sum(frame_rgb, 2)+0.1), 0, 3)

    if args.genVideo and args.genVideo == 'norm':
        imsave(join(NORM_DIR, 'Norm'+str(count)+'.png'), img_norm*255)

    # Reshape in order to reduce the 3-dimensional array to 1-dimensional (needed for predict)
    reshape = img_norm.reshape(shape[0]*shape[1], 3)
    labels = clf.predict(reshape)

    # Reshape back, from 1-dimensional to 2-dimensional
    reshape_back = labels.reshape(shape[0], shape[1])

    paleta = np.array([[255, 0, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8)

    # Automatic reshape is being done here, from 2-dimensional to 3-dimensional array [[1, 1, ...]] -> [[[0,0,0], ....]]
    aux = paleta[reshape_back]

    segm_img = cv2.cvtColor(aux, cv2.COLOR_RGB2BGR)

    if segm:
        cv2.imshow('Segm', segm_img)

    if args.genVideo:
        if args.genVideo == 'segm':
            cv2.imwrite(join(SEGM_DIR, 'SegmImg'+str(count)+'.png'), segm_img)

    # Image with the line in white
    line_img = (reshape_back == 2).astype(np.uint8)*255

    # Image with the arrow/mark in white
    arrow_mark_img = (reshape_back == 0).astype(np.uint8)*255

    return line_img, arrow_mark_img


def analysis(clf, args, segm=False):
    if VIDEO == '0':
        capture = cv2.VideoCapture(0)
    else:
        capture = cv2.VideoCapture(VIDEO)
    count = 0
    latest_org = 0
    latest_dst = 0

    if args.genVideo:
        if args.genVideo == 'segm':
            make_dir(SEGM_DIR)
        elif args.genVideo == 'norm':
            make_dir(NORM_DIR)
        elif args.genVideo == 'analy':
            make_dir(ANALY_DIR)
        elif args.genVideo == 'chull':
            make_dir(CHULL_DIR)

    pause = False
    type_aut = TypeObjAutomaton()
    PID = PIDController()
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        # if ret and not count % 24:
        if ret:
            cv2.imshow('Original', frame)

            line_img, arrow_mark_img = segmentation(clf, frame, count, args, segm)

            # FindContours is destructive, so we copy make a copy
            line_img_cp = line_img.copy()
            line_fringe = line_img.copy()
            forward, turn = PID.getForwardTurnVelocity(line_fringe)
            print "forward: ", forward, " turn: ", turn

            ignore_mark_arrow = False
            ignore_frame = arrow_mark_img[:30, :].copy()
            ignore_frame2 = arrow_mark_img[:30, :].copy()
            cv2.imshow("ign", ignore_frame2)
            ig_cnts_am, hier = cv2.findContours(ignore_frame, cv2.RETR_LIST,
                                                cv2.CHAIN_APPROX_NONE)
            if ig_cnts_am:
                ignore_mark_arrow = True

            # FindContours is destructive, so we copy make a copy
            arrow_mark_img_cp = arrow_mark_img.copy()

            # Should we use cv2.CHAIN_APPROX_NONE? or cv2.CHAIN_APPROX_SIMPLE? the former stores all points, the latter stores the basic ones
            # Find contours of line
            cnts_l, hier = cv2.findContours(line_img, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)

            if not ignore_mark_arrow:
                # Find contours of arror/mark
                cnts_am, hier = cv2.findContours(arrow_mark_img, cv2.RETR_LIST,
                                                 cv2.CHAIN_APPROX_NONE)

            # Removes small contours, i.e: small squares
            newcnts_l = [cnt for cnt in cnts_l if len(cnt) > 100]
            if not ignore_mark_arrow:
                newcnts_am = [cnt for cnt in cnts_am if len(cnt) > 75]

            # DrawContours is destructive
            analy = frame[50:, :].copy()

            # Return list of indices of points in contour
            chull_list_l = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_l]
            if not ignore_mark_arrow:
                chull_list_am = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_am]

            # Return convexity defects from previous contours, each contour must have at least 3 points
            # Convexity Defect -> [start_point, end_point, farthest_point, distance_to_farthest_point]
            conv_defs_l = [(cv2.convexityDefects(cont, chull), pos) for pos, (cont, chull) in
                           enumerate(zip(newcnts_l, chull_list_l)) if len(cont) > 3 and len(chull) > 3]

            if not ignore_mark_arrow:
                conv_defs_am = [(cv2.convexityDefects(cont, chull), pos) for pos, (cont, chull) in
                                enumerate(zip(newcnts_am, chull_list_am)) if len(cont) > 3 and len(chull) > 3]

            list_conv_defs_l = []
            list_cont_l = []
            list_conv_defs_am = []
            list_cont_am = []
            # Only save the convexity defects whose hole is larger than ~4 pixels (1000/256).
            for el in conv_defs_l:
                if el is not None:
                    aux = el[0][:, :, 3] > 1000
                    if any(aux):
                        list_conv_defs_l.append(el[0][aux])  # el = (convDefs, position)
                        list_cont_l.append(newcnts_l[el[1]])

            if not ignore_mark_arrow:
                for el in conv_defs_am:
                    if el is not None:
                        aux = el[0][:, :, 3] > 1000
                        if any(aux):
                            list_conv_defs_am.append(el[0][aux])
                            list_cont_am.append(newcnts_am[el[1]])

            left_border = line_img_cp[:, :10].copy()
            right_border = line_img_cp[:, 310:].copy()
            top_border = line_img_cp[:10, 10:310].copy()
            bot_border = line_img_cp[180:, 10:310].copy()

            all_mlc = []
            all_mrc = []
            all_mtc = []
            all_mbc = []

            left_cnt, hier = cv2.findContours(left_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            left_cnt = [cnt for cnt in left_cnt if cv2.contourArea(cnt) > 30]
            if left_cnt:
                for l in left_cnt:
                    mlc = np.mean(l[:, :, :], axis=0, dtype=np.int32)
                    all_mlc.extend(mlc.tolist())
                    # cv2.circle(analy, (mlc[0, 0], mlc[0, 1]), 3, (0, 255, 0), -1)

            right_cnt, hier = cv2.findContours(right_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            right_cnt = [cnt for cnt in right_cnt if cv2.contourArea(cnt) > 30]
            if right_cnt:
                for r in right_cnt:
                    r[:, :, 0] = r[:, :, 0] + 310
                    mrc = np.mean(r[:, :, :], axis=0, dtype=np.int32)
                    all_mrc.extend(mrc.tolist())
                    # cv2.circle(analy, (mrc[0, 0], mrc[0, 1]), 3, (0, 255, 0), -1)

            top_cnt, hier = cv2.findContours(top_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            top_cnt = [cnt for cnt in top_cnt if cv2.contourArea(cnt) > 30]
            if top_cnt:
                for t in top_cnt:
                    t[:, :, 0] = t[:, :, 0] + 10
                    mtc = np.mean(t[:, :, :], axis=0, dtype=np.int32)
                    all_mtc.extend(mtc.tolist())
                    # cv2.circle(analy, (mtc[0, 0], mtc[0, 1]), 3, (0, 255, 0), -1)

            bot_cnt, hier = cv2.findContours(bot_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            bot_cnt = [cnt for cnt in bot_cnt if cv2.contourArea(cnt) > 30]
            if bot_cnt:
                for b in bot_cnt:
                    b[:, :, 0] = b[:, :, 0] + 10
                    b[:, :, 1] = b[:, :, 1] + 180
                    mbc = np.mean(b[:, :, :], axis=0, dtype=np.int32)
                    all_mbc.extend(mbc.tolist())
                    # cv2.circle(analy, (mbc[0, 0], mbc[0, 1]), 3, (255, 0, 255), -1)

            mark = True

            if not list_conv_defs_l:
                cv2.putText(analy, "Linea recta", (0, 140),
                            FONT, 0.5, (0, 0, 0), 1)
                degiro, org, dst = giroLinea(all_mbc, all_mlc, all_mtc, all_mrc)
                cv2.circle(analy, (org[0], org[1]), 3, (0, 0, 255), -1)
                cv2.circle(analy, (dst[0], dst[1]), 3, (0, 0, 255), -1)

            for pos, el in enumerate(list_conv_defs_l):
                for i in range(el.shape[0]):
                    if el.shape[0] == 1:
                        # [NormX, NormY, PointX, PointY]
                        [vx, vy, x, y] = cv2.fitLine(list_cont_l[pos], cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
                        slope = vy/vx
                        if slope > 0:
                            cv2.putText(analy, "Giro izq", (0, 140),
                                        FONT, 0.5, (0, 0, 0), 1)
                        else:
                            cv2.putText(analy, "Giro dcha", (0, 140),
                                        FONT, 0.5, (0, 0, 0), 1)
                        degiro, org, dst = giroLinea(all_mbc, all_mlc, all_mtc, all_mrc)

                        cv2.circle(analy, (org[0], org[1]), 3, (0, 0, 255), -1)
                        cv2.circle(analy, (dst[0], dst[1]), 3, (0, 0, 255), -1)

                    elif el.shape[0] == 2 or el.shape[0] == 3:
                        cv2.putText(analy, "Cruce 2 salidas", (0, 140),
                                    FONT, 0.5, (0, 0, 0), 1)
                        mark = False
                    elif el.shape[0] == 4:
                        cv2.putText(analy, "Cruce 3 salidas", (0, 140),
                                    FONT, 0.5, (0, 0, 0), 1)
                        mark = False

                    if args.genVideo and args.genVideo == 'chull':
                        # Draw convex hull and hole
                        s, e, f, d = el[i]
                        start = tuple(list_cont_l[pos][s][0])
                        end = tuple(list_cont_l[pos][e][0])
                        far = tuple(list_cont_l[pos][f][0])
                        cv2.line(analy, start, end, [0, 255, 0], 2)
                        cv2.circle(analy, far, 3, [0, 0, 255], -1)

            if args.genVideo and args.genVideo == 'chull':
                for pos, el in enumerate(list_conv_defs_am):
                    for i in range(el.shape[0]):
                        # Draw convex hull and hole
                        s, e, f, d = el[i]
                        start = tuple(list_cont_am[pos][s][0])
                        end = tuple(list_cont_am[pos][e][0])
                        far = tuple(list_cont_am[pos][f][0])
                        cv2.line(analy, start, end, [0, 255, 0], 2)
                        cv2.circle(analy, far, 3, [0, 0, 255], -1)

            if not ignore_mark_arrow:
                if not newcnts_am:
                    type_aut._reset()
                    # mark_aut._reset()
                    latest_dst = 0
                    latest_org = 0
                    # print "inicianlizando latest_dst a 0"
                else:
                    if not type_aut.getType(mark):
                        if len(newcnts_am) == 1:
                            hu_mom = cv2.HuMoments(cv2.moments(newcnts_am[0])).flatten()
                            # hu_mom2 = -np.sign(hu_mom)*np.log10(np.abs(hu_mom))
                            pred = neigh_clf.predict([hu_mom])
                            if pred == 0:
                                cv2.putText(analy, "Cruz", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            elif pred == 1:
                                cv2.putText(analy, "Escalera", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            elif pred == 2:
                                cv2.putText(analy, "Persona", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                            elif pred == 3:
                                cv2.putText(analy, "Telefono", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    else:
                        cv2.putText(analy, "Flecha", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                        for c in newcnts_am:
                            ellipse = cv2.fitEllipse(c)
                            center, axis, angle = ellipse

                            # Axis angles, major, minor
                            maj_ang = np.deg2rad(angle)
                            min_ang = maj_ang + np.pi/2

                            # Axis lenghts
                            major_axis = axis[1]
                            minor_axis = axis[0]

                            # Lines of axis, first line and his complementary
                            lineX1 = int(center[0]) + int(np.sin(maj_ang)*(major_axis/2))
                            lineY1 = int(center[1]) - int(np.cos(maj_ang)*(major_axis/2))
                            lineX2 = int(center[0]) - int(np.sin(maj_ang)*(major_axis/2))
                            lineY2 = int(center[1]) + int(np.cos(maj_ang)*(major_axis/2))

                            if args.genVideo and args.genVideo == 'chull':
                                linex1 = int(center[0]) + int(np.sin(min_ang)*(minor_axis/2))
                                liney1 = int(center[1]) - int(np.cos(min_ang)*(minor_axis/2))
                                cv2.line(analy, (int(center[0]), int(center[1])), (lineX1, lineY1), (0, 0, 255), 2)
                                cv2.line(analy, (int(center[0]), int(center[1])), (linex1, liney1), (255, 0, 0), 2)
                                cv2.circle(analy, (int(center[0]), int(center[1])), 3, (0, 0, 0), -1)
                                cv2.ellipse(analy, ellipse, (0, 255, 0), 2)
                                cv2.putText(analy, "Ang. elipse: "+str(angle), (0, 110),
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                            # Get coordinates of arrow pixels
                            # arrow = cv2.findNonZero(arrow_mark_img_cp)[:, 0, :]
                            idx = np.where(arrow_mark_img_cp != 0)
                            size_idx = len(idx[0])
                            arrow = np.array([[idx[1][idy], idx[0][idy]] for idy in range(size_idx)])
                            angle360 = angle        # Initial angle in [0,180)
                            if 45 <= angle <= 135:  # Arrow kind of horizontal -> cut in vertical
                                # Divide arrow in two lists depending on X coordinate of the center
                                left = [1 for px in arrow if px[0] < center[0]]
                                right = [1 for px in arrow if px[0] > center[0]]
                                if len(right) >= len(left):
                                    peak = (lineX1, lineY1)  # Arrow peak is the point in major axis 1
                                else:
                                    peak = (lineX2, lineY2)  # Arrow peak is the point in major axis 2
                                    angle360 += 180          # Real angle in [0,360)
                            else:  # Arrow kind of vertical -> cut in horizontal
                                # Divide arrow in two lists depending on Y coordinate of the center
                                up = [1 for px in arrow if px[1] < center[1]]
                                down = [1 for px in arrow if px[1] > center[1]]
                                if (len(up) >= len(down) and angle < 45) or (len(down) >= len(up) and angle > 135):
                                    peak = (lineX1, lineY1)  # Arrow peak is the point in major axis 1
                                else:
                                    peak = (lineX2, lineY2)  # Arrow peak is the point in major axis 2
                                    angle360 += 180

                            angle360 = int(angle360)
                            hasLine = 1
                            if angle360 >= 337.5 or angle360 < 22.5:
                                cv2.putText(analy, "Norte (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = 0
                            elif angle360 < 67.5:
                                cv2.putText(analy, "Noreste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = -0.25
                            elif angle360 < 112.5:
                                cv2.putText(analy, "Este (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = -0.5
                            elif angle360 < 157.5:
                                cv2.putText(analy, "Sureste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = -0.75
                            elif angle360 < 202.5:
                                cv2.putText(analy, "Sur (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = 1
                            elif angle360 < 247.5:
                                cv2.putText(analy, "Suroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = 0.75
                            elif angle360 < 292.5:
                                cv2.putText(analy, "Oeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = 0.5
                            elif angle360 < 337.5:
                                cv2.putText(analy, "Noroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                                lineDistance = 0.25

                            cv2.line(analy, (int(peak[0]), int(peak[1])), (int(center[0]), int(center[1])), (0, 0, 255), 2)
                            cv2.circle(analy, (int(peak[0]), int(peak[1])), 3, (0, 255, 0), -1)

                            # deg_mlc = []
                            # deg_mrc = []
                            # deg_mtc = []
                            # pt_org = all_mbc[0]
                            # for i, mlc in enumerate(all_mlc):
                            #     if pt_org[1]-mlc[1] == 0:
                            #         deg_mlc[i] = 90
                            #     else:
                            #         deg_mlc[i] = math.degrees( np.arctan((pt_org[0]-mlc[0])/(pt_org[1]-mlc[1])) )
                            # for i, mrc in enumerate(all_mlc):
                            #     if pt_org[1]-mlc[1] == 0:
                            #         deg_mlc[i] = -90
                            #     else:
                            #         deg_mlc[i] = math.degrees( np.arctan((pt_org[0]-mrc[0])/(pt_org[1]-mrc[1])) )
                            # for i, mtc in enumerate(all_mtc):
                            #     deg_mtc[i] = math.degrees( np.arctan((pt_org[0]-mtc[0])/(pt_org[1]-mtc[1])) )
                            #


                            ## Nuevo Anais, funciona 10/10
                            if all_mbc:
                                org = all_mbc[np.argmin([abs(160 - mbc[0]) for mbc in all_mbc])]  # compare bottom points with the center of the image - horizontally
                                if not latest_org: latest_org = org
                                mysqrt = np.sqrt((org[0] - latest_org[0]) ** 2 + (org[1] - latest_org[1]) ** 2)
                                if mysqrt > 20: org = latest_org
                                else: latest_org = org
                            else:
                                org = latest_org

                            if not mark:
                                # Sentido de centro a pico CP-> sentido flecha -> [xp-xc,yp-yc]
                                vec_arrow = [peak[0]-center[0],peak[1]-center[1]]
                                mod_arrow = np.sqrt(vec_arrow[0]**2+vec_arrow[1]**2)
                                # Si sabemos que es algo del Oeste o algo del Este, no mirar a Dcha o a Izqd
                                if lineDistance < 0:  # Seguro que podemos no mirar contornos de la izquierda
                                    all_mlc = []
                                if lineDistance > 0:  # Seguro que podemos no mirar contornos de la derecha
                                    all_mrc = []

                                salidas = []
                                salidas.extend(all_mlc)
                                salidas.extend(all_mtc)
                                salidas.extend(all_mrc)
                                simil = []

                                for sal in salidas:
                                    vec_sal = [sal[0]-center[0], sal[1]-center[1]]
                                    mod_sal = np.sqrt(sal[0]**2+sal[1]**2)
                                    simil.append( (vec_arrow[0]*vec_sal[0]+vec_arrow[1]*vec_sal[1])/(mod_arrow*mod_sal) )

                                if salidas:
                                    # El punto destino sera el que mas se parezca a la flecha
                                    dst = salidas[simil.index(min(simil, key=lambda z: abs(z - 1)))]
                                    if not latest_dst: latest_dst = dst
                                    mysqrt = np.sqrt((dst[0] - latest_dst[0]) ** 2 + (dst[1] - latest_dst[1]) ** 2)
                                    if mysqrt > 50: dst = latest_dst
                                    else: latest_dst = dst
                                else: dst = latest_dst

                            else:
                                if latest_dst:
                                    dst = latest_dst  # reconoce marca tras reconocer flecha
                                else:
                                    if len(all_mtc) != 0:
                                        print "primero"
                                        dst = all_mtc[0]
                                    elif len(all_mlc) != 0:
                                        print "segundo"
                                        dst = all_mlc[0]
                                    elif len(all_mrc) != 0:
                                        print "tercero"
                                        dst = all_mrc[0]
                                    elif len(all_mbc) > 1:
                                        print "cuarto"
                                        dst = all_mbc[np.argmax([abs(160 - mbc[0, 0]) for mbc in all_mbc])]
                                    print "Entro! dst: ", dst
                            # print "mtc: ", all_mtc, " mlc: ", all_mlc, " mrc: ", all_mrc, " mbc: ", all_mbc
                            # print "org:",org
                            # print "dst:",dst
                            org_dst = np.array([org, dst])  # tam 2 (1 es punto origen, 2 punto salida)
                            # print "org_dst: ", org_dst
                            cv2.circle(analy, (org_dst[0, 0], org_dst[0, 1]), 3, (229, 9, 127), -1)
                            cv2.circle(analy, (org_dst[1, 0], org_dst[1, 1]), 3, (229, 9, 127), -1)

                            # Aqui deberias tener ya el punto origen org y destino dst mas probable
                            if org_dst[0][1] - org_dst[1][1] == 0:
                                if org_dst[0][0] < org_dst[1][0]:
                                    degiro = -90
                                else:
                                    degiro = 90
                            else:
                                degiro = math.degrees(np.arctan((org_dst[0][0] - org_dst[1][0]+0.0) / (org_dst[0][1] - org_dst[1][1])))
                            lineDistance = degiro
                            ## Fin codigo Anais, 10/10

            if args.genVideo and args.genVideo == 'chull':
                cv2.drawContours(analy, left_cnt, -1, (255, 0, 0), 2)
                cv2.drawContours(analy, right_cnt, -1, (0, 255, 0), 2)
                cv2.drawContours(analy, top_cnt, -1, (255, 0, 255), 2)
                cv2.drawContours(analy, bot_cnt, -1, (0, 0, 255), 2)

            cv2.drawContours(analy, newcnts_l, -1, (255, 0, 0), 1)
            if not ignore_mark_arrow:
                cv2.drawContours(analy, newcnts_am, -1, (0, 0, 255), 1)
            cv2.imshow("Contours", analy)
            print "degiro: ", degiro

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(10)

            # (n)ext image
            if (key & 0xFF) == ord('n'):
                count += 1
                continue

            # (q)uit program
            if (key & 0xFF) == ord('q'):
                break

            # (p)ause program
            if (key & 0xFF) == ord('p'):
                pause = not pause

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


def mark_train(args):
    all_hu = np.load('moments.hu')
    labels = np.load('moments.labels')

    q_n = 1
    cov_list = np.cov(all_hu.reshape(400, 7).T)
    neigh = KNeighborsClassifier(n_neighbors=q_n, weights='distance',
                                 metric='mahalanobis', metric_params={'V': cov_list})

    n_images = 4*100
    neigh.fit(all_hu.reshape((n_images, 7)), labels.reshape((n_images,)))
    global neigh_clf
    neigh_clf = neigh
    return neigh


def gen_video(name, procedure):

    make_dir(VID_DIR)

    aux_dir = None
    if procedure == 'segm':
        aux_dir = SEGM_DIR
    elif procedure == 'norm':
        aux_dir = NORM_DIR
    elif procedure == 'analy':
        aux_dir = ANALY_DIR
    elif procedure == 'chull':
        aux_dir = CHULL_DIR

    images = [f for f in listdir(aux_dir) if isfile(join(aux_dir, f))]

    if not len(images):
        print "No images to create the video."
        sys.exit()

    images = natural_sort(images)
    aux = cv2.imread(join(aux_dir, images[0]))

    height, width, layers = aux.shape

    video = cv2.VideoWriter(join(VID_DIR, name+'.avi'), cv2.cv.CV_FOURCC('M', 'P', '4', '2'), 1.0, (width, height))

    for img in images:
        video.write(cv2.imread(join(aux_dir, img)))

    cv2.destroyAllWindows()
    video.release()


def natural_sort(images_list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(images_list, key=alphanum_key)


def make_dir(dirName):
    try:
        mkdir(dirName)
    except OSError:
        pass

def giroLinea(all_mbc, all_mlc, all_mtc, all_mrc):
    # Devuelve un angulo de giro del origen al destino cuando no hay bifurcacion
    # Encontrar el origen
    if all_mbc:
        org = all_mbc[np.argmin([abs(160 - mbc[0]) for mbc in all_mbc])]
        salidas = []; salidas.extend(all_mlc); salidas.extend(all_mtc); salidas.extend(all_mrc)
        dst = salidas[salidas.index(max(salidas, key=lambda z: z[1]))]  # Destino el de mayor Y (evitar mierdecillas)
    else:
        bordes = []; bordes.extend(all_mlc); bordes.extend(all_mrc)
        org = bordes[bordes.index(max(bordes, key=lambda z: z[1]))]  # Origen el de mayor Y
        bordes.extend(all_mtc)
        dst = bordes[bordes.index(min(bordes, key=lambda z: z[1]))]  # Destino el de menor Y

    if org[1] - dst[1] == 0:  # Horizontal
        org[1] += 1  # Mover un pixel para evitar division por cero

    print "org: ", org, " dst: ", dst
    degiro = math.degrees(np.arctan((org[0] - dst[0]+0.0) / (org[1] - dst[1])))
    return degiro, org, dst


def main(parser, args):
    global VIDEO, TRAIN_IMG

    if args.video:
        VIDEO = args.video

    if args.trainImg:
        TRAIN_IMG = args.trainImg

    # Mark lots of images
    if args.mark:
        marking()
    elif args.segm:
        print "Training q-NN classifier..."
        mark_train(args)
        print "Starting video analysis..."
        clf = training()
        analysis(clf, args, segm=True)
    elif args.analy:
        print "Training q-NN classifier..."
        mark_train(args)
        print "Starting video analysis..."
        clf = training()
        analysis(clf, args)

    if args.genVideo:
        gen_video(args.output, args.genVideo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='robotica.py')

    parser.add_argument('-v', '--video',
                        help='Select a different video.')

    parser.add_argument('-t', '--trainImg',
                        help='Select a different trainingImg.')

    parser.add_argument('-o', '--output',
                        default='video_output',
                        help='Choose the output video name.')

    group = parser.add_argument_group('Commands')

    group.add_argument('-m', '--mark',
                       action='store_true',
                       help='Start marking process.')

    group.add_argument('-s', '--segm',
                       action='store_true',
                       help='Start segmentation process.')

    group.add_argument('-a', '--analy',
                       action='store_true', default='True',
                       help='Start analysis process.')

    group.add_argument('-g', '--genVideo',
                       choices=['segm', 'norm', 'analy', 'chull'],
                       nargs='?', const='analy',
                       help='Generate choosen procedure video.')

    args = parser.parse_args()

    main(parser, args)
