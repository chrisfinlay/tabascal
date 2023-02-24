from tabascal.coordinates import ENU_to_GEO
import numpy as np


class Telescope(object):
    def __init__(
        self,
        latitude: float,
        longitude: float,
        elevation: float,
        ENU_array=None,
        ENU_path=None,
        name=None,
    ):
        self.name = name
        self.latitude = latitude
        self.longitude = longitude
        self.elevation = elevation
        self.GEO = np.array([latitude, longitude, elevation])
        self.ENU_path = None
        self.createArrayENU(ENU_array=ENU_array, ENU_path=ENU_path)
        self.n_ants = len(self.ENU)

    def __str__(self):
        return f"""Telescope Location
                   ------------------
                   Latitude : {self.latitude}
                   Longitude : {self.longitude}
                   Elevation : {self.elevation}"""

    def createArrayENU(self, ENU_array=None, ENU_path=None):
        if ENU_array is not None:
            self.ENU = ENU_array
        elif ENU_path is not None:
            self.ENU = np.loadtxt(ENU_path)
        else:
            self.ENU = None
            msg = """Error : East-North-Up coordinates are needed either in an 
                     array or as a csv like file."""
            print(msg)
            return

        self.ENU_path = ENU_path
        self.GEO_ants = ENU_to_GEO(self.GEO, self.ENU)
