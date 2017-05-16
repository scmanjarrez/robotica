from pyrobot.brain import Brain


class BrainTestNavigator(Brain):
    NO_FORWARD = 0.2
    SLOW_FORWARD = 0.45
    MED_FORWARD = 0.5
    FULL_FORWARD = 0.6

    NO_TURN = 0
    # LEFT:
    MED_LEFT1 = 0.4
    MED_LEFT2 = 0.1
    HARD_LEFT = 0.7
    # RIGHT
    MED_RIGHT2 = -0.1
    MED_RIGHT1 = -0.4
    HARD_RIGHT = -0.7

    NO_ERROR = 0

    NOLINE = False
    count_l = 0
    count_r = 0
    count_e = 0

    pos = [None, None]
    OBS = False

    def setup(self):
        pass

    def step(self):
        hasLine, lineDistance, searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
        print "I got from the simulation", hasLine, lineDistance, searchRange
        #front = [2, 3, 4, 5]
        res = 0
        div = 0
        frontAve = [s.value for s in self.robot.sonar[0]['front-all'] if s.value < 5]
        if len(frontAve):
            res = sum(frontAve)/len(frontAve)
            # for f in front:
            #     val = self.robot.sonar[0][f].value
            #     if val < 5:
            #         res += val
            #         div += 1
            # if div:
            #     res = res / div
            #     # frontAve = sum(s.value for s in self.robot.sonar[0]['front-all']) / 6.0
            print "Frente: ", res
            if res < 2:
                print "****Obstacle found****"
                self.OBS = True
                self.move(0, 0.5)

        elif hasLine:
            self.OBS = False
            self.count_l = 0
            self.count_r = 0
            self.count_e = 0
            if (1 >= lineDistance / 10 > 0.6):
                self.move(self.NO_FORWARD, self.HARD_LEFT)
            elif (0.6 >= lineDistance / 10 > 0.3):
                self.move(self.SLOW_FORWARD, self.MED_LEFT1)
            elif (0.3 >= lineDistance / 10 > 0.2):
                self.move(self.MED_FORWARD, self.MED_LEFT2)
            elif (0.2 >= lineDistance / 10 > -0.2):
                self.move(self.FULL_FORWARD, self.NO_TURN)
            elif (-0.2 >= lineDistance / 10 > -0.3):
                self.move(self.MED_FORWARD, self.MED_RIGHT2)
            elif (-0.3 >= lineDistance / 10 > -0.6):
                self.move(self.SLOW_FORWARD, self.MED_RIGHT1)
            elif (-0.6 >= lineDistance / 10 >= -1):
                self.move(self.NO_FORWARD, self.HARD_RIGHT)

        else:
            if self.OBS:
                print "****Obstacle found****, no line"
                right = min([s.distance() for s in self.robot.range["right"]])
                print "Right: ", right
                if right < 2:
                    self.move(0.5, 0)
                else:
                    self.move(0, -1)

            else:
                print "No line found, count(l, r): ", self.count_l, self.count_r
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
                    self.move(1, 0)
                    self.count_l = 0
                    self.count_r = 0
                    self.count_e = 0


def INIT(engine):
    assert (engine.robot.requires("range-sensor") and
            engine.robot.requires("continuous-movement"))

    # If we are allowed (for example you can't in a simulation), enable
    # the motors.
    try:
        engine.robot.position[0]._dev.enable(1)
    except AttributeError:
        pass

    return BrainTestNavigator('BrainTestNavigator', engine)