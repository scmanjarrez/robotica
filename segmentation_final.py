#!/usr/bin/python
import sys
from scipy.misc import imsave

import select_pixels as sel
import cv2


def main(video):
    capture = cv2.VideoCapture(video)
    count = 0

    while(capture.isOpened()):
        ret, frame = capture.read()
        if ret and not count % 24:
            cv2.imshow('Frame', frame)

            # compare key pressed with the ascii code of the character
            key = cv2.waitKey(1000)

            #  key = 1010 0000 0000 1011 0110 1110
            #   &
            # 0xFF =                     1111 1111
            #  110 =                     0110 1110

            # next image
            if key & 0xFF == ord('n'):
                count += 1
                continue

            # mark image
            if key & 0xFF == ord('m'):
                # change from BGR to RGB format
                imRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # mark training pixels
                markImg = sel.select_fg_bg(imRGB)

                imsave('TrainingImg'+str(count)+'.png', markImg)

            # quit program
            if key & 0xFF == ord('q'):
                break

        elif not ret:
            print "End of video"
            break

        count += 1

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    if len(sys.argv[1:]) != 1:
        print "Expected one argument, choosing default video."
        main("video2017-3.avi")
    else:
        main(sys.argv[1])
