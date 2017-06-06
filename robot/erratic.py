#!/usr/bin/python
import numpy as np
from scipy.misc import imread
from os import mkdir
from os.path import join
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier

import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX
ANALY_DIR = "AnalyFrames"

latest_dst = 0
latest_org = 0
pid = None
clf = None


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

    def _reset(self):
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
            line_point = all_mp_fringe[np.argmin([abs(160 - mpp[0]) for mpp in all_mp_fringe])]
            self.error = (mid_point[0] - line_point[0])/160.0
            self.derivative = self.error - self.last_error
            self.last_error = self.error

        turn = self.Kp*self.error + self.Kd*self.derivative
        forward = max(0, 1 - abs(turn*1.5))
        return forward, turn


def segmentation(clf, frame):
    shape = frame[60:, :].shape
    frame_rgb = cv2.cvtColor(frame[60:, :], cv2.COLOR_BGR2RGB)

    img_norm = np.rollaxis((np.rollaxis(frame_rgb, 2)+0.1)/(np.sum(frame_rgb, 2)+0.1), 0, 3)

    reshape = img_norm.reshape(shape[0]*shape[1], 3)
    labels = clf.predict(reshape)

    reshape_back = labels.reshape(shape[0], shape[1])

    line_img = (reshape_back == 2).astype(np.uint8)*255

    arrow_mark_img = (reshape_back == 0).astype(np.uint8)*255

    return line_img, arrow_mark_img


def make_dir(dirName):
    try:
        mkdir(dirName)
    except OSError:
        pass


def analysis(frame, neigh_clf, count):
    global pid, clf
    if not clf:
        clf = training()
    if not pid:
        pid = PIDController()

    make_dir(ANALY_DIR)
    lineDistance = 0
    global latest_dst, latest_org
    hasLine = 0  # 0 = noLine | 1 = Line

    forward = 0
    turn = 0

    type_aut = TypeObjAutomaton()

    line_img, arrow_mark_img = segmentation(clf, frame)

    line_img_cp = line_img.copy()

    arrow_mark_img_cp = arrow_mark_img.copy()

    cnts_l, hier = cv2.findContours(line_img, cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_NONE)
    if len(cnts_l) == 0:
        return 0, 0, 0, False
    cnts_am, hier = cv2.findContours(arrow_mark_img, cv2.RETR_LIST,
                                     cv2.CHAIN_APPROX_NONE)

    newcnts_l = [cnt for cnt in cnts_l if len(cnt) > 100]
    if not newcnts_l:
        return 0, 0, 0, 0
    newcnts_am = [cnt for cnt in cnts_am if len(cnt) > 75]

    analy = frame[60:, :].copy()

    chull_list_l = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_l]
    chull_list_am = [cv2.convexHull(cont, returnPoints=False) for cont in newcnts_am]

    conv_defs_l = [(cv2.convexityDefects(cont, chull), pos) for pos, (cont, chull) in
                   enumerate(zip(newcnts_l, chull_list_l)) if len(cont) > 3 and len(chull) > 3]

    conv_defs_am = [(cv2.convexityDefects(cont, chull), pos) for pos, (cont, chull) in
                    enumerate(zip(newcnts_am, chull_list_am)) if len(cont) > 3 and len(chull) > 3]

    left_border = line_img_cp[:, :20].copy()
    right_border = line_img_cp[:, 300:].copy()
    top_border = line_img_cp[:20, 20:300].copy()
    bot_border = line_img_cp[160:, 20:300].copy()

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
            r[:, :, 0] = r[:, :, 0] + 300
            mrc = np.mean(r[:, :, :], axis=0, dtype=np.int32)
            all_mrc.extend(mrc.tolist())
            # cv2.circle(analy, (mrc[0, 0], mrc[0, 1]), 3, (0, 255, 0), -1)

    top_cnt, hier = cv2.findContours(top_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    top_cnt = [cnt for cnt in top_cnt if cv2.contourArea(cnt) > 30]
    if top_cnt:
        for t in top_cnt:
            t[:, :, 0] = t[:, :, 0] + 20
            mtc = np.mean(t[:, :, :], axis=0, dtype=np.int32)
            all_mtc.extend(mtc.tolist())
            # cv2.circle(analy, (mtc[0, 0], mtc[0, 1]), 3, (0, 255, 0), -1)

    bot_cnt, hier = cv2.findContours(bot_border, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    bot_cnt = [cnt for cnt in bot_cnt if cv2.contourArea(cnt) > 30]
    if bot_cnt:
        for b in bot_cnt:
            b[:, :, 0] = b[:, :, 0] + 20
            b[:, :, 1] = b[:, :, 1] + 160
            mbc = np.mean(b[:, :, :], axis=0, dtype=np.int32)
            all_mbc.extend(mbc.tolist())
            # cv2.circle(analy, (mbc[0, 0], mbc[0, 1]), 3, (255, 0, 255), -1)

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

    for el in conv_defs_am:
        if el is not None:
            aux = el[0][:, :, 3] > 1000
            if any(aux):
                list_conv_defs_am.append(el[0][aux])
                list_cont_am.append(newcnts_am[el[1]])

    mark = True

    if not list_conv_defs_l:
        cv2.putText(analy, "Linea recta", (0, 140),
                    FONT, 0.5, (0, 0, 0), 1)
        hasLine = 1
        forward, turn = pid.getForwardTurnVelocity(line_img.copy())
        print "Estoy viendo linea recta"

    for pos, el in enumerate(list_conv_defs_l):
        for i in range(el.shape[0]):
            if el.shape[0] == 1:  # Giro izquierda o derecha me da igual
                hasLine = 1
                forward, turn = pid.getForwardTurnVelocity(line_img.copy())

            elif el.shape[0] == 2 or el.shape[0] == 3:
                print "Estoy viendo cruce 2"
                cv2.putText(analy, "Cruce 2 salidas", (0, 140),
                            FONT, 0.5, (0, 0, 0), 1)
                mark = False
                hasLine = 1
            elif el.shape[0] == 4:
                print "Estoy viendo cruce 3"
                cv2.putText(analy, "Cruce 3 salidas", (0, 140),
                            FONT, 0.5, (0, 0, 0), 1)
                mark = False
                hasLine = 1

    if not newcnts_am:
        type_aut._reset()
    else:
        if not type_aut.getType(mark):  # Mark
            if len(newcnts_am) == 1:
                hu_mom = cv2.HuMoments(cv2.moments(newcnts_am[0])).flatten()
                pred = neigh_clf.predict([hu_mom])
                if pred == 0:
                    cv2.putText(analy, "Cruz", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    print "Cruz"
                elif pred == 1:
                    cv2.putText(analy, "Escalera", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    print "Escalera"
                elif pred == 2:
                    cv2.putText(analy, "Persona", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    print "Persona"
                elif pred == 3:
                    cv2.putText(analy, "Telefono", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    print "Telefono"
        else:
            # cv2.putText(analy, "Flecha", (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for c in newcnts_am:
                ellipse = cv2.fitEllipse(c)
                center, axis, angle = ellipse

                maj_ang = np.deg2rad(angle)

                major_axis = axis[1]

                lineX1 = int(center[0]) + int(np.sin(maj_ang)*(major_axis/2))
                lineY1 = int(center[1]) - int(np.cos(maj_ang)*(major_axis/2))
                lineX2 = int(center[0]) - int(np.sin(maj_ang)*(major_axis/2))
                lineY2 = int(center[1]) + int(np.cos(maj_ang)*(major_axis/2))

                idx = np.where(arrow_mark_img_cp != 0)
                size_idx = len(idx[0])
                arrow = np.array([[idx[1][idy], idx[0][idy]] for idy in range(size_idx)])
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
                    return 0.5, 0, -1, 1
                    hasLine = 1
                    lineDistance = 0
                    print "Norte"
                elif angle360 < 67.5:
                    cv2.putText(analy, "Noreste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0.1, -0.6, -1, 1
                    hasLine = 1
                    lineDistance = -0.25
                    print "Noreste"
                elif angle360 < 112.5:
                    cv2.putText(analy, "Este (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0.1, -1, -1, 1
                    hasLine = 1
                    lineDistance = -0.5
                    print "Este"
                elif angle360 < 157.5:
                    cv2.putText(analy, "Sureste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0, -1, -1, 1
                    hasLine = 1
                    lineDistance = -0.8
                    print "Sureste"
                elif angle360 < 202.5:
                    cv2.putText(analy, "Sur (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0, -1, -1, 1
                    hasLine = 1
                    lineDistance = 1
                    print "Sur"
                elif angle360 < 247.5:
                    cv2.putText(analy, "Suroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0, 1, -1, 1
                    hasLine = 1
                    lineDistance = 0.8
                    print "Suroeste"
                elif angle360 < 292.5:
                    cv2.putText(analy, "Oeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0.1, 1, -1, 1
                    hasLine  = 1
                    lineDistance = 0.5
                    print "Oeste"
                elif angle360 < 337.5:
                    cv2.putText(analy, "Noroeste (ang: "+str(angle360)+")", (0, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    return 0.1, 0.6, -1, 1
                    hasLine = 1
                    lineDistance = 0.25
                    print "Noroeste"

            pid._reset()  # end for newcnts_am, que sabe que aqui es flecha

    cv2.drawContours(analy, newcnts_l, -1, (255, 0, 0), 1)
    cv2.drawContours(analy, newcnts_am, -1, (0, 0, 255), 1)
    cv2.imwrite(join(ANALY_DIR, 'AnalyImg' + str(count) + '.png'), analy)
    return forward, turn, hasLine, not mark


def mark_train():
    all_hu = np.load('moments.hu')
    labels = np.load('moments.labels')

    q_n = 1
    cov_list = np.cov(all_hu.reshape(400, 7).T)
    neigh = KNeighborsClassifier(n_neighbors=q_n, weights='distance',
                                 metric='mahalanobis', metric_params={'V': cov_list})

    n_images = 4*100
    neigh.fit(all_hu.reshape((n_images, 7)), labels.reshape((n_images,)))
    return neigh


def training():
    orig_img = imread('OriginalImg576.png')
    mark_img = imread('TrainingImg576.png')

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
