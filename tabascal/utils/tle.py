from astropy.time import Time
from datetime import datetime
from skyfield.api import EarthSatellite, load
import requests
import numpy as np
from numpy.typing import ArrayLike

import os
import json

class SpaceTrackClient:
    """
    Client for fetching TLE data from Space-Track.org
    """
    
    def __init__(self, username: str, password: str):
        self.auth = {
            'identity': username,
            'password': password
        }
        self.base_url = 'https://www.space-track.org'
        self.session = requests.Session()
        self._authenticate()

    def _authenticate(self) -> None:
        """Authenticate with Space-Track.org"""
        auth_url = f"{self.base_url}/ajaxauth/login"
        response = self.session.post(auth_url, data=self.auth)
        response.raise_for_status()

    def get_tles_by_id(self, norad_ids: list[int], epoch_jd: float, limit: int) -> list[dict]:
        
        if len(norad_ids)>500:
            ValueError("Can only request 500 NORAD IDs at a time.")

        # Calculate the date threshold
        start_time = Time(epoch_jd-1, format="jd", scale="ut1").strftime('%Y-%m-%d')
        end_time = Time(epoch_jd+1, format="jd", scale="ut1").strftime('%Y-%m-%d')
        date_str = f"%3E{start_time}%2C%3C{end_time}"
        id_str = "".join([str(i)+"%2C" for i in norad_ids])[:-3]
        
        try:
            # Build query URL for Starlink satellites
            # This query:
            # 1. Gets latest TLEs for objects with OBJECT_NAME starting with STARLINK
            # 2. Only includes TLEs updated after the specified date
            # 3. Orders by NORAD_CAT_ID for consistency
            query_url = (
                f"{self.base_url}/basicspacedata/query/class/tle_latest/"
                f"NORAD_CAT_ID/{id_str}/EPOCH/{date_str}/"
                f"orderby/NORAD_CAT_ID%20asc/limit/{limit}/format/json"
            )

            # Make request
            response = self.session.get(query_url)
            response.raise_for_status()
            
            # Parse response
            data = response.json()
            
            # Add a timestamp for when this data was fetched
            timestamp = Time.now().fits
            for entry in data:
                entry['_fetch_timestamp'] = timestamp
            
            return data
            
        except Exception as e:
            print(f"Error fetching TLEs: {str(e)}")
            raise

    # def best_tles_by_id(self, norad_ids, epoch_jd, limit):

    #     tles = self.get_tles_by_id(norad_ids, epoch_jd, limit)

    #     return tles

    def save_tles(self, filename: str, norad_ids: list[int], epoch_jd: float, limit: int) -> None:
        
        data = self.get_tles_by_id(norad_ids, epoch_jd, limit)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump({
                'fetch_timestamp': Time.now().fits,
                'total_satellites': len(data),
                'tle_data': data
            }, f, indent=2)
        
        print(f"Saved {len(data)} TLEs to {filename}")

        return data

    def close(self) -> None:
        """Close the session and logout"""
        logout_url = f"{self.base_url}/auth/logout"
        try:
            self.session.get(logout_url)
        finally:
            self.session.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def spacetrack_time_to_isot(spacetrack_time: str) -> str:
    
    from datetime import datetime
    
    dt = datetime.strptime(spacetrack_time, "%Y-%m-%d %H:%M:%S")
    isot = dt.strftime("%Y-%m-%dT%H:%M:%S.000")
    
    return isot


def get_tle_data_st(username: str, password: str, norad_ids: list[int], epoch_jd: float, limit: int, save_path: str=None) -> dict:

    with SpaceTrackClient(username, password) as client:
        data = []
        for i in range(int(len(norad_ids)/500)):

            if save_path is not None:
                name, ext = os.path.splitext(save_path)
                save_name = name + "_" + Time(epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d") + f"_{i:03}" + ext
                data += client.save_tles(save_name, norad_ids[500*i:500*(i+1)], epoch_jd, limit)
            else:
                data += client.get_tles_by_id(norad_ids[500*i:500*(i+1)], epoch_jd, limit)

    ids = np.array(list(map(lambda x: int(x["NORAD_CAT_ID"]), data)))
    epochs = np.array(list(map(lambda x: Time(spacetrack_time_to_isot(x["EPOCH"])).jd, data)))

    tles = {}
    for n_id in norad_ids:
        id_idx = np.where(epochs[ids==n_id])[0]
        if len(id_idx)>0:
            min_idx = np.argmin(epochs[ids==n_id]-np.mean(epoch_jd))
            tles[n_id] = np.array(data)[id_idx][min_idx]

    found = list(tles.keys())
    tles = {
        "missing": list(set(norad_ids)-set(found)), 
        "found": tles
    }

    return tles


def load_tle_data_st(tle_json_path: str, norad_ids: list[int], epoch_jd: float) -> dict:
    
    data = json.load(open(tle_json_path))["tle_data"]

    ids = np.array(list(map(lambda x: int(x["NORAD_CAT_ID"]), data)))
    epochs = np.array(list(map(lambda x: Time(spacetrack_time_to_isot(x["EPOCH"])).jd, data)))

    tles = {}
    for n_id in norad_ids:
        id_idx = np.where(epochs[ids==n_id])[0]
        if len(id_idx)>0:
            min_idx = np.argmin(epochs[ids==n_id]-np.mean(epoch_jd))
            tles[n_id] = np.array(data)[id_idx][min_idx]

    found = list(tles.keys())
    tles = {
        "missing": list(set(norad_ids)-set(found)), 
        "found": tles
    }

    return tles


def sathub_time_to_isot(sathub_time: str) -> str:
    """Convert the epoch time return by a call to the SatChecker from SatHub to isot.

    Parameters
    ----------
    sathub_time : str
        Time format returned by the SatChecker api.

    Returns
    -------
    str
        Time format in isot, easily ingested by astropy.time.Time.
    """
    
    dt = datetime.strptime(sathub_time, "%Y-%m-%d %H:%M:%S UTC")
    isot = dt.strftime("%Y-%m-%dT%H:%M:%S.000")
    
    return isot


def get_tle_data(norad_id: int|str, jd: float) -> dict:
    """Request the TLE data using the SatChecker API with an epoch closest to the given Julian date.

    Parameters
    ----------
    norad_id : int | str
        NORAD ID of the satellite.
    jd : float
        Julian date to search around.

    Returns
    -------
    dict
        SatChecker response with closest epoch.
    """

    url = 'https://satchecker.cps.iau.org/tools/get-tle-data/'
    params = {
        'id': str(norad_id),
        'id_type': 'catalog',
        'start_date_jd': str(jd-1),
        'end_date_jd': str(jd+1),
    }
    response = requests.get(url, params=params).json()

    if len(response)>0:
        delta_epoch = [np.abs(Time(sathub_time_to_isot(resp["epoch"]), format="isot").jd - jd) for resp in response]
    else:
        ValueError("No TLEs found within 1 day of the given date.")
        
    return response[np.argmin(delta_epoch)]


def get_sat_pos_tle(tle_line1: str, tle_line2: str, sat_name: str, times_jd: float) -> ArrayLike:
    """Calculate the satellite position in GCRS (ECI) frame at the given Julian dates. 

    Parameters
    ----------
    tle_line1 : str
        First line of the TLE.
    tle_line2 : str
        Second line fo the TLE
    sat_name : str
        Satellite name. This is often given in the line above the TLE.
    times_jd : float
        Julian dates at which to evaluate the satellite position.

    Returns
    -------
    ArrayLike
        Satellite positions in metres in the GCRS (ECI) frame.
    """
    
    ts = load.timescale()
    sat = EarthSatellite(tle_line1, tle_line2, sat_name, ts)
    t_s = ts.ut1_jd(times_jd)
    sat_pos = sat.at(t_s).position.m

    return sat_pos


def get_sat_pos(norad_id: int|str, times_jd: float) -> ArrayLike:
    
    sathub_resp = get_tle_data(norad_id, np.mean(times_jd))
    sat_pos = get_sat_pos_tle(sathub_resp["tle_line1"], sathub_resp["tle_line2"], sathub_resp["satellite_name"], times_jd)

    return sat_pos.T 