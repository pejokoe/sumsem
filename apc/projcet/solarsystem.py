from celestalbody import *
import time

class SolarSystem():
    def __init__(self):
        planets = []
        initPlanets(planets)
        while(True):
            time.wait(1)
            for planet in planets:
                planet.acceleration()
                planet.velocity()
                p0s = planet.position()



def initPlanets(planets):
    planets.append(Planet("earth,", 5.972e24, 149.597e6, 0))