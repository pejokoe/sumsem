import scipy.constants as constants
import numpy as np
MASS_SUN = 1.989e30
# scipy constant implies use of meters as distance metric
print(constants.G)
class CelestalBody():
    def __init__(self, name, mass, x, y):
        self.name = name
        self.mass = mass
        self.x = x
        self.y = y
        self.v = initVelocity(self)
        self.a = 0
    

    def acceleration(allBodies):
        pass

class Planet(CelestalBody):
    def __init__(self, name, mass, x, y):
        super().__init__(mass, x, y)
        moons = []

    def acceleration(self, allBodies):
        return acceleration(self, allBodies)
    
    def velocity(self):
        pass


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
    caller.a = Force / caller.mass

def initVelocity(Planet):
    v0 = np.sqrt(2*constants.G*MASS_SUN*1/abs(Planet.x))
    return v0