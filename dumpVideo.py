#!/usr/bin/python
import sys
from os import mkdir
from os.path import join
from scipy.misc import imsave

import cv2


def main(name):
    count = 0
    try:
        mkdir("imagenes")
    except:
        pass
    capture = cv2.VideoCapture(name)
    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret:
            imsave(join("imagenes", "imagen"+str(count)+".png"),
                   cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        elif not ret:
            break
        count += 1


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "Debes pasar un nombre de video como parametro."
        sys.exit()
    main(sys.argv[1])
