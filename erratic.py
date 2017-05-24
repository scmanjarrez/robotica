#!/usr/bin/python
import numpy as np
import re
import sys

from matplotlib import pyplot as plt # noqa, disable flycheck warning
from os import listdir, mkdir
from os.path import isfile, join
from scipy.misc import imread, imsave
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX

neigh_clf = None


class TypeObjAutomaton:
    def __init__(self):
        self.state = 0

    def _state(self):
        return self.state

    def _reset(self):
        if self.state:
            self.state = 0

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


def training():
    data = np.load('training.data')
    target = np.load('training.target')

    clf = NearestCentroid()
    clf.fit(data, target)
    return clf


def segmentation(clf, frame):
    shape = frame[90:, :].shape
    frame_rgb = cv2.cvtColor(frame[90:, :], cv2.COLOR_BGR2RGB)

    img_norm = np.rollaxis((np.rollaxis(frame_rgb, 2)+0.1)/(np.sum(frame_rgb, 2)+0.1), 0, 3)

    reshape = img_norm.reshape(shape[0]*shape[1], 3)
    labels = clf.predict(reshape)

    reshape_back = labels.reshape(shape[0], shape[1])

    paleta = np.array([[255, 0, 0], [0, 0, 0], [0, 0, 255]], dtype=np.uint8)

    aux = paleta[reshape_back]

    segm_img = cv2.cvtColor(aux, cv2.COLOR_RGB2BGR)

    cv2.imshow('Segm', segm_img)

    line_img = (reshape_back == 2).astype(np.uint8)*255

    arrow_mark_img = (reshape_back == 0).astype(np.uint8)*255

    return line_img, arrow_mark_img


def analysis(clf):
    capture = cv2.VideoCapture(0)
    count = 0

    pause = False
    type_aut = TypeObjAutomaton()
    while(capture.isOpened()):
        if not pause:
            ret, frame = capture.read()
        # if ret and not count % 24:
        if ret:
            cv2.imshow('Original', frame)

            line_img, arrow_mark_img = segmentation(clf, frame)

            line_img_cp = line_img.copy()

            arrow_mark_img_cp = arrow_mark_img.copy()

            cnts_l, hier = cv2.findContours(line_img, cv2.RETR_LIST,
                                            cv2.CHAIN_APPROX_NONE)

            cnts_am, hier = cv2.findContours(arrow_mark_img, cv2.RETR_LIST,
                                             cv2.CHAIN_APPROX_NONE)

            newcnts_l = [cnt for cnt in cnts_l if len(cnt) > 100]
            newcnts_am = [cnt for cnt in cnts_am if len(cnt) > 75]

            analy = frame.copy()[90:]

            chull_list_l = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_l]
            chull_list_am = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_am]

            conv_defs_l = [cv2.convexityDefects(cont, chull) for (cont, chull) in
                           zip(newcnts_l, chull_list_l) if len(cont) > 3 and len(chull) > 3]

            conv_defs_am = [cv2.convexityDefects(cont, chull) for (cont, chull) in
                            zip(newcnts_am, chull_list_am) if len(cont) > 3 and len(chull) > 3]

            list_conv_defs_l = []
            list_cont_l = []
            list_conv_defs_am = []
            list_cont_am = []

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

            if not newcnts_am:
                type_aut._reset()
            else:
                if not type_aut.getType(mark):
                    if len(newcnts_am) == 1:
                        hu_mom = cv2.HuMoments(cv2.moments(newcnts_am[0])).flatten()
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

                        maj_ang = np.deg2rad(angle)

                        major_axis = axis[1]

                        lineX1 = int(center[0]) + int(np.sin(maj_ang)*(major_axis/2))
                        lineY1 = int(center[1]) - int(np.cos(maj_ang)*(major_axis/2))
                        lineX2 = int(center[0]) - int(np.sin(maj_ang)*(major_axis/2))
                        lineY2 = int(center[1]) + int(np.cos(maj_ang)*(major_axis/2))

                        arrow = cv2.findNonZero(arrow_mark_img_cp)[:, 0, :]
                        angle360 = angle
                        if 45 <= angle <= 135:
                            left = [1 for px in arrow if px[0] < center[0]]
                            right = [1 for px in arrow if px[0] > center[0]]
                            if len(right) >= len(left):
                                peak = (lineX1, lineY1)
                            else:
                                peak = (lineX2, lineY2)
                                angle360 += 180
                        else:
                            up = [1 for px in arrow if px[1] < center[1]]
                            down = [1 for px in arrow if px[1] > center[1]]
                            if (len(up) >= len(down) and angle < 45) or (len(down) >= len(up) and angle > 135):
                                peak = (lineX1, lineY1)
                            else:
                                peak = (lineX2, lineY2)
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

            left_border = line_img_cp[:, :10].copy()
            right_border = line_img_cp[:, 310:].copy()
            top_border = line_img_cp[:10, 10:310].copy()
            bot_border = line_img_cp[140:, 10:310].copy()

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

            cv2.drawContours(analy, newcnts_l, -1, (255, 0, 0), 1)
            cv2.drawContours(analy, newcnts_am, -1, (0, 0, 255), 1)
            cv2.imshow("Contours", analy)

            key = cv2.waitKey(10)

            if (key & 0xFF) == ord('n'):
                count += 1
                continue

            if (key & 0xFF) == ord('q'):
                break

            if (key & 0xFF) == ord('p'):
                pause = not pause

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()


def mark_train():
    global neigh_clf

    all_hu = np.load('moments.hu')
    labels = np.load('moments.labels')

    q_n = 1
    cov_list = np.cov(all_hu.T)
    neigh = KNeighborsClassifier(n_neighbors=q_n, weights='distance',
                                 metric='mahalanobis', metric_params={'V': cov_list})

    n_images = 4*100
    neigh.fit(all_hu.reshape((n_images, 7)), labels.reshape((n_images,)))
    neigh_clf = neigh


def natural_sort(images_list):
    def convert(text):
        return int(text) if text.isdigit() else text.lower()

    def alphanum_key(key):
        return [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(images_list, key=alphanum_key)


if __name__ == "__main__":
    mark_train()
    clf = training()
    analysis(clf)
