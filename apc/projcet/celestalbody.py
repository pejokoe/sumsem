import scipy.constants as constants
import numpy as np
# scipy constant implies use of meters as distance metric
print(constants.G)
class CelestalBody():
    def __init__(self, radius, mass, x, y, v, a):
        self.mass = mass
        self.x = x
        self.y = y
        self.v = v
        self.a = a
    

    def acceleration(allBodies):
        pass

class Planet(CelestalBody):
    def __init__(self, mass, x, y):
        super().__init__(mass, x, y)
        moons = []

    def acceleration(self, allBodies):
        return acceleration(self, allBodies)


def acceleration(caller, allBodies):
    Force = np.zeros(2)
    for body in allBodies:
        if body is not caller:
            distance = np.sqrt((caller.x - body.x)**2 + (caller.y - body.y)**2)
            FAbs = constants.G * caller.mass * body.mass / (distance**2)
            unitVectorX = (body.x - caller.x) / distance
            unitVectorY = (body.y - caller.y) / distance
            Force[0] += unitVectorX
            Force[1] += unitVectorY
    Force = Force / caller.mass
    return Force

