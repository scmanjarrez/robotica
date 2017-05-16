#!/usr/bin/python
import argparse
import numpy as np
import re
import sys
import time # noqa, disable flycheck warning
import cv2

from matplotlib import pyplot as plt # noqa, disable flycheck warning
from os import listdir, mkdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import LeaveOneOut # noqa, disable flycheck warning
from sklearn.decomposition import PCA # noqa, disable flycheck warning


VIDEO = 'resized.avi'
TRAIN_IMG = '9999'

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

    img_norm = np.rollaxis((np.rollaxis(orig_img, 2)+0.1)/(np.sum(orig_img, 2)+0.1), 0, 3)

    data_red = img_norm[np.where(np.all(np.equal(mark_img, (255, 0, 0)), 2))]
    data_green = img_norm[np.where(np.all(np.equal(mark_img, (0, 255, 0)), 2))]
    data_blue = img_norm[np.where(np.all(np.equal(mark_img, (0, 0, 255)), 2))]

    data = np.concatenate([data_red, data_green, data_blue])

    target = np.concatenate([np.zeros(len(data_red[:]), dtype=int),
                             np.ones(len(data_green[:]), dtype=int),
                             np.full(len(data_blue[:]), 2, dtype=int)])

    clf = NearestCentroid()
    clf.fit(data, target)
    return clf


def segmentation(clf, frame, count, args, segm, mark=False):
    shape = frame.shape  # Segm all
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Segm all

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
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        if ret and not count % 8:
        # if ret:
            # if video == '0':
            #     ret = capture.set(3, 340)
            #     ret = capture.set(240)
            cv2.imshow('Original', frame)

            line_img, arrow_mark_img = segmentation(clf, frame, count, args, segm)

            # FindContours is destructive, so we copy make a copy
            line_img_cp = line_img.copy()

            # FindContours is destructive, so we copy make a copy
            arrow_mark_img_cp = arrow_mark_img.copy()

            # Should we use cv2.CHAIN_APPROX_NONE? or cv2.CHAIN_APPROX_SIMPLE? the former stores all points, the latter stores the basic ones
            # Find contours of line
            cnts_l, hier = cv2.findContours(line_img, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)

            # Find contours of arror/mark
            cnts_am, hier = cv2.findContours(arrow_mark_img, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_NONE)

            # Removes small contours, i.e: small squares
            newcnts_l = [cnt for cnt in cnts_l if len(cnt) > 100]
            newcnts_am = [cnt for cnt in cnts_am if len(cnt) > 75]

            # DrawContours is destructive
            analy = frame.copy()

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

            mark = True

            if not list_conv_defs_l:
                cv2.putText(analy, "Linea recta", (0, 140),
                            FONT, 0.5, (0, 0, 0), 1)

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

                    elif el.shape[0] == 2 or el.shape[0] == 3:
                        cv2.putText(analy, "Cruce 2 salidas", (0, 140),
                                    FONT, 0.5, (0, 0, 0), 1)
                        mark = False
                    elif el.shape[0] == 4:
                        cv2.putText(analy, "Cruce 3 salidas", (0, 140),
                                    FONT, 0.5, (0, 0, 0), 1)
                        mark = False

            if mark:
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

                    # pr_pca = pca.transform(hu_mom.reshape(1, -1))
                    # for pos, p in enumerate(plx):
                    #     plt.scatter(p, ply[pos], label=MARKS[pos], color=COLORS[pos])
                    #     plt.scatter(pr_pca[0, 0], pr_pca[0, 1], label="ToPredict", color="cyan")
                    # plt.legend()
                    # plt.show()
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
                    arrow = cv2.findNonZero(arrow_mark_img_cp)[:, 0, :]
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

            border_size = 10
            height, width = line_img_cp.shape
            left_border = line_img_cp[:, :border_size].copy()
            right_border = line_img_cp[:, -border_size:].copy()
            top_border = line_img_cp[:border_size, border_size:-border_size].copy()
            bot_border = line_img_cp[-border_size:, border_size:-border_size].copy()

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
                    r[:, :, 0] = r[:, :, 0] + (width - border_size)
                    mrc = np.mean(r[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mrc[0, 0], mrc[0, 1]), 3, (0, 255, 0), -1)

            top_cnt, hier = cv2.findContours(top_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            top_cnt = [cnt for cnt in top_cnt if cv2.contourArea(cnt) > 75]
            if top_cnt:
                for t in top_cnt:
                    t[:, :, 0] = t[:, :, 0] + border_size
                    mtc = np.mean(t[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mtc[0, 0], mtc[0, 1]), 3, (0, 255, 0), -1)

            bot_cnt, hier = cv2.findContours(bot_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            bot_cnt = [cnt for cnt in bot_cnt if cv2.contourArea(cnt) > 75]
            if bot_cnt:
                for b in bot_cnt:
                    b[:, :, 0] = b[:, :, 0] + border_size
                    b[:, :, 1] = b[:, :, 1] + (height - border_size)
                    mbc = np.mean(b[:, :, :], axis=0, dtype=np.int32)
                    cv2.circle(analy, (mbc[0, 0], mbc[0, 1]), 3, (255, 0, 255), -1)

            cv2.drawContours(analy, newcnts_l, -1, (255, 0, 0), 1)
            cv2.drawContours(analy, newcnts_am, -1, (0, 0, 255), 1)
            cv2.imshow("Contours", analy)

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
    global plx, ply, pca, neigh_clf, MARK_DIR

    clf = training(mark=True, train_img_m='9999')

    all_hu = []
    labels = []
    if args.dir:
        MARK_DIR = args.dir

    for idx, m in enumerate(MARKS):
        files = [join(MARK_DIR, m, 'chosen', f)
                 for f in listdir(join(MARK_DIR, m, 'chosen'))]
        h = []
        l = []
        for i in files:
            frame = imread(i)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            _, arrow_mark_img = segmentation(clf, frame_bgr, 0, args, segm=False, mark=True)
            cnts, hier = cv2.findContours(arrow_mark_img, cv2.RETR_LIST,
                                          cv2.CHAIN_APPROX_NONE)
            cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 75]
            hu_mom = cv2.HuMoments(cv2.moments(cnts[0])).flatten()
            h.append(hu_mom)
            l.append(idx)
        all_hu.append(h)
        labels.append(l)

    all_hu = np.array(all_hu)
    labels = np.array(labels)

    q_n = 1
    if args.img200:
        cov_list = np.cov(all_hu.reshape(800, 7).T)
    else:
        cov_list = np.cov(all_hu.reshape(400, 7).T)

    neigh = KNeighborsClassifier(n_neighbors=q_n, weights='distance',
                                 metric='mahalanobis', metric_params={'V': cov_list})

    if args.img200:
        loo = LeaveOneOut(200)
        s = 4*199
    else:
        loo = LeaveOneOut(100)
        s = 4*99

    fallo_cruz = 0
    fallo_escalera = 0
    fallo_persona = 0
    fallo_telefono = 0
    for train_idx, test_idx in loo:
        neigh.fit(all_hu[:, train_idx].reshape((s, 7)), labels[:, train_idx].reshape((s,)))
        res = neigh.predict(all_hu[:, test_idx].reshape(4, 7))
        if res[0] != 0:
            fallo_cruz += 1
        if res[1] != 1:
            fallo_escalera += 1
        if res[2] != 2:
            fallo_persona += 1
        if res[3] != 3:
            fallo_telefono += 1

    print "q-NN: ", q_n
    print "Acierto Cruz     (%): ", 100-fallo_cruz
    print "Acierto Escalera (%): ", 100-fallo_escalera
    print "Acierto Persona  (%): ", 100-fallo_persona
    print "Acierto Telefono (%): ", 100-fallo_telefono

    # if args.img200:
    #     s = 4*200
    # else:
    #     s = 4*100
    # fallo_cruz = 0
    # fallo_escalera = 0
    # fallo_persona = 0
    # fallo_telefono = 0

    # pca = PCA(n_components=2)
    # tr_data = pca.fit_transform(all_hu.reshape((s, 7)))
    # tr_label = labels.reshape((s,))
    # plx = [[], [], [], []]
    # ply = [[], [], [], []]
    # for idx, el in enumerate(tr_data):
    #     ps = tr_label[idx]
    #     plx[ps].append(el[0])
    #     ply[ps].append(el[1])
    # for pos, p in enumerate(plx):
    #     plt.scatter(p, ply[pos], label=MARKS[pos], color=COLORS[pos])
    # plt.legend()
    # plt.show()
    # sys.exit()

    # neigh.fit(all_hu.reshape((s, 7)), labels.reshape((s,)))
    # for idx in range(100):
    #     res = neigh.predict(all_hu[:, idx].reshape(4, 7))
    #     if res[0] != 0:
    #         fallo_cruz += 1
    #     if res[1] != 1:
    #         fallo_escalera += 1
    #     if res[2] != 2:
    #         fallo_persona += 1
    #     if res[3] != 3:
    #         fallo_telefono += 1

    # print "K-neighbors: ", k_n
    # print "% Acierto Cruz: ", 100-fallo_cruz
    # print "% Acierto Escalera: ", 100-fallo_escalera
    # print "% Acierto Persona: ", 100-fallo_persona
    # print "% Acierto Telefono: ", 100-fallo_telefono

    neigh_clf = neigh


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


def main(parser, args):
    global VIDEO, TRAIN_IMG

    if args.video:
        VIDEO = args.video

    if args.trainImg:
        TRAIN_IMG = args.trainImg

    if args.segm:
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

    parser.add_argument('-d', '--dir',
                        help='Select a different mark training directory.')


    parser.add_argument('-i', '--img200',
                        action='store_true',
                        help='Use 200 images to train the mark classifier.')

    parser.add_argument('-t', '--trainImg',
                        help='Select a different trainingImg.')

    parser.add_argument('-o', '--output',
                        default='video_output',
                        help='Choose the output video name.')

    group = parser.add_argument_group('Commands')

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
