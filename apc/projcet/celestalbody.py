import scipy.constants as constants
import numpy as np
# scipy constant implies use of meters as distance metric
print(constants.G)
class CelestalBody():
    def __init__(self, radius, mass, x, y):
        self.mass = mass
        self.x = x
        self.y = y
    

    def acceleration(allBodies):
        pass

class Planet(CelestalBody):
    def __init__(self, mass, x, y):
        super().__init__(mass, x, y)

    def acceleration(self, allBodies):
        Force = np.zeros(2)
        for body in allBodies:
            distance = np.sqrt((self.x - body.x)**2 + (self.y - body.y)**2)
            FAbs = constants.G * self.mass * body.mass / distance
            unitVectorX = (body.x - self.x) / distance
            unitVectorY = (body.y - self.y) / distance
            Force[0] += unitVectorX
            Force[1] += unitVectorY
        Force = Force / self.mass



