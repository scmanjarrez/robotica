from pyrobot.brain import Brain
import erratic as err
from WebcamVideoStream import WebcamVideoStream
import os

neigh_clf = None
ws = None


class BrainTestNavigator(Brain):
    NO_FORWARD = 0.05
    SLOW_FORWARD = 0.3
    MED_FORWARD = 0.4
    FULL_FORWARD = 0.5

    NO_TURN = 0
    # LEFT
    HARD_LEFT = 0.4
    MED_LEFT1 = 0.2
    MED_LEFT2 = 0.05
    # RIGHT
    MED_RIGHT2 = -0.05
    MED_RIGHT1 = -0.2
    HARD_RIGHT = -0.4

    NO_ERROR = 0

    NOLINE = False
    count_elipse = 0
    count_l = 0
    count_r = 0
    count_e = 0
    speed_t = 1
    speed_f = 0.5
    MAX = 28

    cc = 0

    pos = [None, None]
    OBS = False

    def setup(self):
        pass

    def step(self):
        frame = ws.read()
        forward, turn, hasLine, arrow = err.analysis(frame, neigh_clf, self.cc)

        if hasLine == 1:
            self.count_elipse = 0
            self.count_l = 0
            self.count_r = 0
            self.count_e = 0
            if not arrow:
                self.move(forward*0.5, turn*0.5)

        elif hasLine == -1:  # Arrow
            self.count_elipse = 0
            self.count_l = 0
            self.count_r = 0
            self.count_e = 0
            self.move(forward, turn)
        else:  # noline
            if self.count_l < 14:
                self.move(0, 0.5)
                self.count_l += 1
            elif self.count_r < 28:
                self.move(0, -0.5)
                self.count_r += 1
            elif self.count_e < 14:
                self.move(0, 0.5)
                self.count_e += 1
            else:
                if self.count_elipse < self.MAX:
                    self.move(self.speed_f, self.speed_t)
                    self.count_elipse += 1
                else:
                    self.MAX = (1/self.speed_t)**2 * 14
                    self.speed_t -= 0.1
                    self.count_elipse = 0

        self.cc += 1


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass

    filelist = [f for f in os.listdir("AnalyFrames") if f.endswith(".png") ]
    for f in filelist:
        os.remove('AnalyFrames/'+f)

    global ws, neigh_clf
    ws = WebcamVideoStream()
    ws.start()

    neigh_clf = err.mark_train()

    return BrainTestNavigator('BrainTestNavigator', engine)
