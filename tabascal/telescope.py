from tabascal.coordinates import ENU_to_GEO
import numpy as np

class Telescope(object):
    def __init__(self, latitude, longitude, elevation, ENU_array=None, ENU_path=None, name=None):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.GEO = np.array([latitude, longitude, elevation])
        self.ENU_path = None
        self.createArrayENU(ENU_array=ENU_array, ENU_path=ENU_path)
        self.n_ants = len(self.ENU)

    def __str__(self):
        return f"""Telescope location
                   \nLatitude : {self.latitude}\nLongitude : {self.longitude}\nElevation : {self.elevation}"""

    def createArrayENU(self, ENU_array=None, ENU_path=None):
        if ENU_array is not None:
            self.ENU = ENU_array
        elif ENU_path is not None:
            self.ENU = np.loadtxt(ENU_path)
        else:
            self.ENU = None
            print('Error : East-North-Up coordinates are needed either in an array or as a csv like file.')
            return

        self.ENU_path = ENU_path
        self.GEO_ants = ENU_to_GEO(self.GEO, self.ENU)
