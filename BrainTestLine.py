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
    count_elipse = 0
    count_l = 0
    count_r = 0
    count_e = 0
    speed_t = 1
    speed_f = 1
    MAX = 28

    pos = [None, None]
    OBS = False

    def setup(self):
        pass

    def step(self):
        hasLine, lineDistance, searchRange = eval(self.robot.simulation[0].eval("self.getLineProperties()"))
        print "I got from the simulation", hasLine, lineDistance, searchRange
        frontAve = sum(s.value for s in self.robot.sonar[0]['front-all']) / 6.0
        # print "Frente: ", frontAve
        if frontAve < 7:
            print "****Obstacle found, front sensors****"
            self.OBS = True
            self.move(0, 0.5)

        elif hasLine:
            self.OBS = False
            self.count_l = 0
            self.count_r = 0
            self.count_e = 0
            self.count_elipse = 0
            self.speed_t = 1
            self.speed_f = 0.5
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
                print "****Obstacle found, right sensors****"
                right = sum([s.distance() for s in self.robot.range[5, 6, 7]]) / 3.0
                if right < 4:
                    self.move(0.3, 0)
                else:
                    self.move(0, -0.7)
            else:
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
