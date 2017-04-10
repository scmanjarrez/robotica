#!/usr/bin/python
import argparse
import numpy as np
import re
import sys
from os import listdir, mkdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

import cv2
import select_pixels as sel
import time # noqa, disable flycheck warning

# from matplotlib import pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from mpl_toolkits.mplot3d import proj3d

video = 'video2017-3.avi'
train_img = '576'
train_dir = 'TrainFrames'
segm_dir = 'SegmFrames'
norm_dir = 'NormFrames'
analy_dir = 'AnalyFrames'
chull_dir = 'ChullFrames'
vid_dir = 'OutputVideos'

font = cv2.FONT_HERSHEY_SIMPLEX

# class StateAutomata:
#     # mark, ~mark, ~arrow, arrow
#     # States are 00, 01, 10, 11
#     # On success increase state
#     # On failure decrease state
#     def __init__(self):
#         self.states = ["-3", "-2", "-1", "0", "+1", "+2", "+3"]
#         self.state = 3

#     def state(self):
#         return self.state

#     def __decrease(self):
#         if self.state > 0:
#             self.state -= 1
#         return 0 if (self.state < 3) else 1

#     def __increase(self):
#         if self.state < 6:
#             self.state += 1
#         return 0 if (self.state < 3) else 1

#     def getState(self, state):
#         if state == "mark":
#             return self.__decrease()
#         elif state == "arrow":
#             return self.__increase()


def marking():
    capture = cv2.VideoCapture(video)
    count = 0

    make_dir(train_dir)

    pause = False
    while(capture.isOpened()):
        if not pause:
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
                im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mark training pixels
                mark_img = sel.select_fg_bg(im_rgb)

                imsave(join(train_dir, 'OriginalImg'+str(count)+'.png'), im_rgb)
                imsave(join(train_dir, 'TrainingImg'+str(count)+'.png'), mark_img)

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


def training(args):
    make_dir(norm_dir)

    # Height x Width x channel
    orig_img = imread(join(train_dir, 'OriginalImg'+train_img+'.png'))
    mark_img = imread(join(train_dir, 'TrainingImg'+train_img+'.png'))

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
    img_norm = np.rollaxis((np.rollaxis(orig_img, 2)+0.0)/np.sum(orig_img, 2), 0, 3)

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

    # fig = plt.figure(figsize=(8, 8))
    # ax = fig.add_subplot(111, projection='3d')
    # plt.rcParams['legend.fontsize'] = 10

    # data = [data_red, data_green, data_blue]
    # colors_markers = ('red', 'green', 'blue')
    # for pos in range(3):
    #     ax.plot(data[pos][:, 0], data[pos][:, 1], data[pos][:, 2], '*',
    #             markersize=5, color=colors_markers[pos], alpha=0.5, label=colors_markers[pos])

    # plt.legend(loc='upper left')
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 1)
    # ax.set_zlim(0, 1)
    # plt.show()

    clf = NearestCentroid()
    clf.fit(data, target)

    return clf


def segmentation(clf, args):
    capture = cv2.VideoCapture(video)
    count = 0

    if args.genVideo:
        if args.genVideo == 'segm':
            make_dir(segm_dir)
        elif args.genVideo == 'norm':
            make_dir(norm_dir)
        elif args.genVideo == 'analy':
            make_dir(analy_dir)
        elif args.genVideo == 'chull':
            make_dir(chull_dir)

    pause = False
    # ultEnt = (0, 0)
    # i = 0
    # state_automata = StateAutomata()
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        if ret and not count % 24:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            cv2.imshow('Original', frame)

            shape = frame.shape

            img_norm = np.rollaxis((np.rollaxis(rgb_frame, 2)+0.0)/np.sum(rgb_frame, 2), 0, 3)

            if args.genVideo and args.genVideo == 'norm':
                imsave(join(norm_dir, 'Norm'+str(count)+'.png'), img_norm*255)

            # Reshape in order to reduce the 3-dimensional array to 1-dimensional (needed for predict)
            reshape = img_norm.reshape(shape[0]*shape[1], 3)
            labels = clf.predict(reshape)

            # Reshape back, from 1-dimensional to 2-dimensional
            reshape_back = labels.reshape(shape[0], shape[1])

            # Image with the line in white
            line_px = (reshape_back == 2).astype(np.uint8)[90:, :]*255

            # FindContours is destructive, so we copy this image
            line_px_aux = line_px.copy()

            # Image with the arrow/mark in white
            arrow_mark_px = (reshape_back == 0).astype(np.uint8)[90:, :]*255

            # FindContours is destructive, so we copy this image
            arrow_mark_px_aux = arrow_mark_px.copy()

            paleta = np.array([[255, 0, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8)

            # Automatic reshape is being done here, from 2-dimensional to 3-dimensional array [[1, 1, ...]] -> [[[0,0,0], ....]]
            aux = paleta[reshape_back]

            segm_img = cv2.cvtColor(aux, cv2.COLOR_RGB2BGR)

            # Should we use cv2.CHAIN_APPROX_NONE? or cv2.CHAIN_APPROX_SIMPLE? the former stores all points, the latter stores the basic ones
            # Find contours of line
            cnts_l, hier = cv2.findContours(line_px, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)

            # Find contours of arror/mark
            cnts_am, hier = cv2.findContours(arrow_mark_px, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_NONE)

            # Removes small contours, i.e: small squares
            newcnts_l = [cnt for cnt in cnts_l if len(cnt) > 100]
            newcnts_am = [cnt for cnt in cnts_am if len(cnt) > 50]

            # DrawContours is destructive
            analy = frame.copy()[90:]

            # Return list of indices of points in contour
            chull_list_l = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_l]
            chull_list_am = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_am]

            # Return convexity defects from previous contours, each contour must have at least 3 points
            # Convexity Defect -> [start_point, end_point, farthest_point, distance_to_farthest_point]
            conv_defs_l = [cv2.convexityDefects(cont, chull) for (cont, chull) in
                           zip(newcnts_l, chull_list_l) if len(cont) > 3 and len(chull) > 3]

            conv_defs_am = [cv2.convexityDefects(cont, chull) for (cont, chull) in
                            zip(newcnts_am, chull_list_am) if len(cont) > 3 and len(chull) > 3]

            list_conv_defs_l = []
            list_cont_l = []
            list_conv_defs_am = []
            list_cont_am = []
            # Only save the convexity defects whose hole is larger than ~4 pixels (1000/256).
            for pos, el in enumerate(conv_defs_l):
                if el is not None:
                    aux = el[:, :, 3] > 1000
                    if any(aux):
                        list_conv_defs_l.append(el[aux])
                        list_cont_l.append(newcnts_l[pos])

            for pos, el in enumerate(conv_defs_am):
                if el is not None:
                    aux = el[:, :, 3] > 1000
                    if any(aux):
                        list_conv_defs_am.append(el[aux])
                        list_cont_am.append(newcnts_am[pos])

            # obj = None
            mark = True

            if not list_conv_defs_l:
                cv2.putText(analy, "Linea recta", (0, 140),
                            font, 0.5, (0, 0, 0), 1)
                # obj = "mark"

            for pos, el in enumerate(list_conv_defs_l):
                for i in range(el.shape[0]):
                    if el.shape[0] == 1:
                        # [NormX, NormY, PointX, PointY]
                        [vx, vy, x, y] = cv2.fitLine(list_cont_l[pos], cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
                        slope = vy/vx
                        if slope > 0:
                            cv2.putText(analy, "Giro izq", (0, 140),
                                        font, 0.5, (0, 0, 0), 1)
                        else:
                            cv2.putText(analy, "Giro dcha", (0, 140),
                                        font, 0.5, (0, 0, 0), 1)
                        # obj = "mark"

                    elif el.shape[0] == 2 or el.shape[0] == 3:
                        cv2.putText(analy, "Cruce 2 salidas", (0, 140),
                                    font, 0.5, (0, 0, 0), 1)
                        mark = False
                        # obj = "arrow"
                    elif el.shape[0] == 4:
                        cv2.putText(analy, "Cruce 3 salidas", (0, 140),
                                    font, 0.5, (0, 0, 0), 1)
                        mark = False
                        # obj = "arrow"

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

            # print state_automata.state
            # if newcnts_am and state_automata.getState(obj):
            if not mark:
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
                    arrow = []
                    for y, row in enumerate(arrow_mark_px_aux):
                        for x, col in enumerate(row):
                            if col == 255:
                                arrow.append([x, y])
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

                    if angle360 >= 337.5 or angle360 < 22.5:
                        cv2.putText(analy, "Norte (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 67.5:
                        cv2.putText(analy, "Noreste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 112.5:
                        cv2.putText(analy, "Este (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 157.5:
                        cv2.putText(analy, "Sureste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 202.5:
                        cv2.putText(analy, "Sur (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 247.5:
                        cv2.putText(analy, "Suroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 292.5:
                        cv2.putText(analy, "Oeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    elif angle360 < 337.5:
                        cv2.putText(analy, "Noroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

                    cv2.line(analy, (int(peak[0]), int(peak[1])), (int(center[0]), int(center[1])), (0, 0, 255), 2)
                    cv2.circle(analy, (int(peak[0]), int(peak[1])), 3, (0, 255, 0), -1)

            left_border = line_px_aux[:, :10].copy()
            right_border = line_px_aux[:, 310:].copy()
            top_border = line_px_aux[:10, 10:310].copy()
            bot_border = line_px_aux[140:, 10:310].copy()

            left_cnt, hier = cv2.findContours(left_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            left_cnt = [cnt for cnt in left_cnt if cv2.contourArea(cnt) > 75]
            if left_cnt:
                for l in left_cnt:
                    mlc = np.mean(l[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mlc[0, 0], mlc[0, 1]), 3, (0, 255, 0), -1)

            right_cnt, hier = cv2.findContours(right_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            right_cnt = [cnt for cnt in right_cnt if cv2.contourArea(cnt) > 75]
            if right_cnt:
                for r in right_cnt:
                    r[:, :, 0] = r[:, :, 0] + 310
                    mrc = np.mean(r[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mrc[0, 0], mrc[0, 1]), 3, (0, 255, 0), -1)

            top_cnt, hier = cv2.findContours(top_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            top_cnt = [cnt for cnt in top_cnt if cv2.contourArea(cnt) > 75]
            if top_cnt:
                for t in top_cnt:
                    t[:, :, 0] = t[:, :, 0] + 10
                    mtc = np.mean(t[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mtc[0, 0], mtc[0, 1]), 3, (0, 255, 0), -1)

            bot_cnt, hier = cv2.findContours(bot_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            bot_cnt = [cnt for cnt in bot_cnt if cv2.contourArea(cnt) > 75]
            if bot_cnt:
                for b in bot_cnt:
                    b[:, :, 0] = b[:, :, 0] + 10
                    b[:, :, 1] = b[:, :, 1] + 140
                    mbc = np.mean(b[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mbc[0, 0], mbc[0, 1]), 3, (255, 0, 255), -1)

            if args.genVideo and args.genVideo == 'chull':
                cv2.drawContours(analy, left_cnt, -1, (255, 0, 0), 2)
                cv2.drawContours(analy, right_cnt, -1, (0, 255, 0), 2)
                cv2.drawContours(analy, top_cnt, -1, (255, 0, 255), 2)
                cv2.drawContours(analy, bot_cnt, -1, (0, 0, 255), 2)

            cv2.drawContours(analy, newcnts_l, -1, (255, 0, 0), 1)
            cv2.drawContours(analy, newcnts_am, -1, (0, 0, 255), 1)
            cv2.imshow("Contours", analy)

            if args.genVideo:
                if args.genVideo == 'segm':
                    cv2.imwrite(join(segm_dir, 'SegmImg'+str(count)+'.png'), segm_img)
                elif args.genVideo == 'analy':
                    cv2.imwrite(join(analy_dir, 'AnalyImg'+str(count)+'.png'), analy)
                elif args.genVideo == 'chull':
                    cv2.imwrite(join(chull_dir, 'ChullImg'+str(count)+'.png'), analy)

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(1000)

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


def mark_train():
    marks = ['Cruz', 'Escalera', 'Persona', 'Telefono']
    aux_dir = 1
    # neigh = KNeighborsClassifier(n_neighbors=1)
    pass


def gen_video(name, procedure):
    make_dir(vid_dir)

    aux_dir = None
    if procedure == 'segm':
        aux_dir = segm_dir
    elif procedure == 'norm':
        aux_dir = norm_dir
    elif procedure == 'analy':
        aux_dir = analy_dir
    elif procedure == 'chull':
        aux_dir = chull_dir

    images = [f for f in listdir(aux_dir) if isfile(join(aux_dir, f))]

    if not len(images):
        print "No images to create the video."
        sys.exit()

    images = natural_sort(images)
    aux = cv2.imread(join(aux_dir, images[0]))

    height, width, layers = aux.shape

    video = cv2.VideoWriter(join(vid_dir, name+'.avi'), cv2.cv.CV_FOURCC('M', 'P', '4', '2'), 1.0, (width, height))

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
        # print "Directory already created."
        pass


def main(parser, args):
    global video, train_img

    if args.video:
        video = args.video

    if args.trainImg:
        train_img = args.trainImg

    # Mark lots of images
    if args.mark:
        marking()
    # Select the ones you want to train
    elif args.seg:
        clfN = training(args)
        segmentation(clfN, args)

    if args.genVideo:
        gen_video(args.output, args.genVideo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='robotica.py')

    parser.add_argument('-v', '--video',
                        help='Select a different video.')

    parser.add_argument('-ti', '--trainImg',
                        help='Select a different trainingImg.')

    parser.add_argument('-o', '--output',
                        default='video_output',
                        help='Choose the output video name.')

    group = parser.add_argument_group('Commands')

    group.add_argument('-m', '--mark',
                       action='store_true',
                       help='Start marking process.')

    group.add_argument('-s', '--seg',
                       action='store_true', default='True',
                       help='Start segmentation process.')

    group.add_argument('-gv', '--genVideo',
                       choices=['segm', 'norm', 'analy', 'chull'],
                       nargs='?', const='analy',
                       help='Generate choosen procedure video.')

    args = parser.parse_args()

    main(parser, args)
