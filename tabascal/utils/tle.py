from astropy.time import Time
from datetime import datetime
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.positionlib import position_of_radec
import requests
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from tqdm import tqdm

import os
import json

class SpaceTrackClient:
    """
    Client for fetching TLE data from Space-Track.org
    """
    
    def __init__(self, username: str, password: str):
        self.auth = {
            "identity": username,
            "password": password
        }
        self.base_url = "https://www.space-track.org"
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
        start_time = Time(epoch_jd-1, format="jd", scale="ut1").strftime("%Y-%m-%d")
        end_time = Time(epoch_jd+1, format="jd", scale="ut1").strftime("%Y-%m-%d")
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
                entry["_fetch_timestamp"] = timestamp
            
            return data
            
        except Exception as e:
            print(f"Error fetching TLEs: {str(e)}")
            raise

    def save_tles(self, filename: str, norad_ids: list[int], epoch_jd: float, limit: int) -> None:
        
        data = self.get_tles_by_id(norad_ids, epoch_jd, limit)
        
        # Save to file
        with open(filename, "w") as f:
            json.dump({
                "fetch_timestamp": Time.now().fits,
                "total_satellites": len(data),
                "tle_data": data
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
    """Convert times returned by a SpaceTrack API call to ISOT.

    Parameters
    ----------
    spacetrack_time : str
        SpaceTrack formatted time.

    Returns
    -------
    str
        ISOT formatted time.
    """
    
    from datetime import datetime
    
    dt = datetime.strptime(spacetrack_time, "%Y-%m-%d %H:%M:%S")
    isot = dt.strftime("%Y-%m-%dT%H:%M:%S.000")
    
    return isot

def get_closest_times(df: pd.DataFrame, target_time_jd: float, id_col: str="NORAD_CAT_ID", time_jd_col: str="EPOCH_JD") -> pd.DataFrame:
    """
    For each unique item in the DataFrame, find the instance with time closest to target_time.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing items and their time instances
    target_time : datetime or timestamp
        The reference time to compare against
    id_col : str, default="NORAD_CAT_ID"
        Name of the column containing norad_ids.
    time_jd_col : str, default="EPOCH_JD"
        Name of the column containing time values in Julian date.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame containing one row per unique item, with the instance closest to target_time
    """
    # Calculate absolute time difference for each row
    df = df.copy()
    df["time_diff"] = df[time_jd_col] - target_time_jd
    df["time_diff_abs"] = np.abs(df[time_jd_col] - target_time_jd)
    
    # Group by item and get the row with minimum time difference
    closest_instances = df.loc[df.groupby(id_col)["time_diff_abs"].idxmin()]
    
    return closest_instances


def get_tle_data_st(username: str, password: str, norad_ids: list[int], epoch_jd: float, limit: int=3000, save_path: str=None) -> pd.DataFrame:
    """Download TLE data from SpaceTrack for the given NORAD IDs.

    Parameters
    ----------
    username : str
        Username for SpaceTrack login.
    password : str
        Password for SpaceTrack login.
    norad_ids : list[int]
        NORAD IDs.
    epoch_jd : float
        Epoch in Julian date to search around.
    limit : int, default=3000
        Number of results to limit to.
    save_path : str, optional
        Path to save downloaded TLEs to.

    Returns
    -------
    pd.DataFrame
        Dataframe of TLE data for each NORAD ID.
    """
    max_query_id = 500
    
    with SpaceTrackClient(username, password) as client:
        data = []
        n_iter = int(np.ceil(len(norad_ids)/max_query_id))
        for i in range(n_iter):
            slc = slice(max_query_id*i, max_query_id*(i+1))
    
            if save_path is not None:
                name, ext = os.path.splitext(save_path)
                save_name = name + "_" + Time(epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d") + f"_{i:03}" + ext
                data += client.save_tles(save_name, norad_ids[slc], epoch_jd, limit)
            else:
                data += client.get_tles_by_id(norad_ids[slc], epoch_jd, limit)
    
    df = pd.DataFrame(data)
    if len(df)>0:
        df["EPOCH_JD"] = df["EPOCH"].apply(lambda x: Time(spacetrack_time_to_isot(x)).jd)
        df = get_closest_times(df, epoch_jd, "NORAD_CAT_ID", "EPOCH_JD")
        numeric_cols = [
            "ORDINAL", "NORAD_CAT_ID", "EPOCH_MICROSECONDS", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
            "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE", "ELEMENT_SET_NO", 
            "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", "FILE", "OBJECT_NUMBER", 
            "SEMIMAJOR_AXIS", "PERIOD", "APOGEE", "PERIGEE"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        df["DECAYED"] = pd.to_numeric(df["DECAYED"]).astype(bool)

    return df


def load_tle_data_st(tle_json_path: str, epoch_jd: float, norad_ids: list[int]=None) -> pd.DataFrame:
    """Load TLE data from JSON files downloaded using the get_tle_data_st function.

    Parameters
    ----------
    tle_json_path : str
        TLE JSON path.
    epoch_jd : float
        Epoch to search for TLE in Julian date.
    norad_ids : list[int], optional
        NORAD IDs to search for., by default None

    Returns
    -------
    pd.DataFrame
        Dataframe of TLE data for given NORAD IDs. If none are given, all are returned.
    """
    
    data = json.load(open(tle_json_path))["tle_data"]

    df = pd.DataFrame(data)
    if len(df)>0:
        df["EPOCH_JD"] = df["EPOCH"].apply(lambda x: Time(spacetrack_time_to_isot(x)).jd)
        df = get_closest_times(df, epoch_jd, "NORAD_CAT_ID", "EPOCH_JD")
        numeric_cols = [
            "ORDINAL", "NORAD_CAT_ID", "EPOCH_MICROSECONDS", "MEAN_MOTION", "ECCENTRICITY", "INCLINATION",
            "RA_OF_ASC_NODE", "ARG_OF_PERICENTER", "MEAN_ANOMALY", "EPHEMERIS_TYPE", "ELEMENT_SET_NO", 
            "REV_AT_EPOCH", "BSTAR", "MEAN_MOTION_DOT", "MEAN_MOTION_DDOT", "FILE", "OBJECT_NUMBER", 
            "SEMIMAJOR_AXIS", "PERIOD", "APOGEE", "PERIGEE"
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col])
        df["DECAYED"] = pd.to_numeric(df["DECAYED"]).astype(bool)
        if norad_ids is not None:
            df = df[df["NORAD_CAT_ID"].isin(norad_ids)]

    return df


def load_tle_data_json(json_paths: list[str], epoch_jd: float, norad_ids: list[int]=None) -> pd.DataFrame:
    """Load TLE data from JSON files downloaded using the get_tle_data_st function.

    Parameters
    ----------
    json_paths : list[str]
        TLE JSON path.
    epoch_jd : float
        Epoch to search for TLE in Julian date.
    norad_ids : list[int], optional
        NORAD IDs to search for., by default None

    Returns
    -------
    pd.DataFrame
        Dataframe of TLE data for given NORAD IDs. If none are given, all are returned.
    """

    tles = pd.concat([load_tle_data_st(path, epoch_jd, norad_ids) for path in json_paths])
    
    return tles


def make_window(times: ArrayLike[Time], alt: ArrayLike, angular_sep: ArrayLike) -> dict:
    """Make a dictionary containing the start and end times of a satellite pass including some stats.

    Parameters
    ----------
    times : ArrayLike[Time]
        Times of the satellite pass.
    alt : ArrayLike
        Altitude of the satellite during pass.
    angular_sep : ArrayLike
        Angular separation of the satellite during pass from target.

    Returns
    -------
    dict
        Dictionary of stats.
    """

    window = {
        "start_time": times[0].datetime.strftime(f"%Y-%m-%d-%H:%M:%S {times.scale.upper()}"),
        "end_time": times[-1].datetime.strftime(f"%Y-%m-%d-%H:%M:%S {times.scale.upper()}"),
        "min_ang_sep": np.min(angular_sep),
        "max_elevation": np.max(alt),
    }

    return window

def check_visibility(tle_line1: str, tle_line2: str, times: ArrayLike[Time], observer_lat: float, observer_lon: float, observer_elevation: float, target_ra: float, target_dec: float, max_ang_sep: float, min_elev: float) -> list:
    """Calculate visibility windows for a satellite when observing a celestial target.

    This function determines time windows when a satellite will pass a celestial
    target based on the satellite's orbital parameters (TLE), observer location,
    target coordinates, and visibility constraints.

    Parameters
    ----------
    tle_line1 : str
        First line of the satellite's Two-Line Element set (TLE).
    tle_line2 : str
        Second line of the satellite's Two-Line Element set (TLE).
    times : ArrayLike[Time]
        Array of observation times as Astropy Time objects.
    observer_lat : float
        Observer's latitude in degrees.
    observer_lon : float
        Observer's longitude in degrees.
    observer_elevation : float
        Observer's elevation above sea level in meters.
    target_ra : float
        Right Ascension of the target in degrees.
    target_dec : float
        Declination of the target in degrees.
    max_ang_sep : float
        Maximum allowed angular separation between satellite and target in degrees.
    min_elev : float
        Minimum required elevation of the satellite above horizon in degrees.

    Returns
    -------
    list
        List of visibility windows, where each window is a dictionary containing:
        - 'start_time': Start time of the visibility window
        - 'end_time': End time of the visibility window
        - 'max_elevation': Maximum elevation during the window
        - 'min_angular_separation': Minimum angular separation during the window

    Notes
    -----
    The function uses the WGS84 Earth model and converts the satellite's position
    to topocentric coordinates for elevation calculations. Visibility windows are
    determined based on both elevation constraints and angular separation from the
    target.

    The function requires the Skyfield library for satellite calculations and
    assumes the existence of a `make_window` helper function to format the output
    windows.
    """

    ts = load.timescale()
    sf_times = ts.ut1_jd(times.jd)
    
    # Set up observer location
    observer_location = wgs84.latlon(observer_lat, observer_lon, observer_elevation)
    
    # Create satellite object
    satellite = EarthSatellite(tle_line1, tle_line2, ts=ts)
    
    # Create celestial target position
    target = position_of_radec(ra_hours=target_ra/15, dec_degrees=target_dec)  # Convert RA to hours
    
    satellite_position = satellite.at(sf_times)
    
    topocentric = satellite_position - observer_location.at(sf_times)
    alt, az, distance = topocentric.altaz()
    
    angular_sep = satellite_position.separation_from(target).degrees

    vis_idx = np.where((alt.degrees>min_elev) & (angular_sep<max_ang_sep))[0]
    break_idx = np.where(np.diff(vis_idx)>1)[0]
    if len(break_idx)>0 or len(vis_idx)>0:
        break_idx = np.concatenate([[0], break_idx, [len(times)]])
        windows = [make_window(times[vis_idx[break_idx[i]:break_idx[i+1]]], alt.degrees[vis_idx[break_idx[i]:break_idx[i+1]]], angular_sep[vis_idx[break_idx[i]:break_idx[i+1]]]) for i in range(len(break_idx)-1)]
    else:
        windows = []
    
    return windows

def check_satellite_visibilibities(norad_ids: ArrayLike[int], tles_line1: ArrayLike[str], tles_line2: ArrayLike[str], times: ArrayLike[Time], observer_lat: float, observer_lon: float, observer_elevation: float, target_ra: float, target_dec: float, max_ang_sep: float, min_elev: float) -> dict:
    """Calculate visibility windows for a satellite when observing a celestial target.

    This function determines time windows when a satellite will pass a celestial
    target based on the satellite's orbital parameters (TLE), observer location,
    target coordinates, and visibility constraints.

    Parameters
    ----------
    norad_ids: ArrayLike[int]
        NORAD IDs to calculate for.
    tles_line1 : ArrayLike[str]
        First line of the satellites' Two-Line Element set (TLE).
    tles_line2 : ArrayLike[str]
        Second line of the satellites' Two-Line Element set (TLE).
    times : ArrayLike[Time]
        Array of observation times as Astropy Time objects.
    observer_lat : float
        Observer's latitude in degrees.
    observer_lon : float
        Observer's longitude in degrees.
    observer_elevation : float
        Observer's elevation above sea level in meters.
    target_ra : float
        Right Ascension of the target in degrees.
    target_dec : float
        Declination of the target in degrees.
    max_ang_sep : float
        Maximum allowed angular separation between satellite and target in degrees.
    min_elev : float
        Minimum required elevation of the satellite above horizon in degrees.

    Returns
    -------
    dict
        Dict of list of visibility windows for each NORAD ID, where each window is a dictionary containing:
        - 'start_time': Start time of the visibility window
        - 'end_time': End time of the visibility window
        - 'max_elevation': Maximum elevation during the window
        - 'min_angular_separation': Minimum angular separation during the window

    Notes
    -----
    The function uses the WGS84 Earth model and converts the satellite's position
    to topocentric coordinates for elevation calculations. Visibility windows are
    determined based on both elevation constraints and angular separation from the
    target.

    The function requires the Skyfield library for satellite calculations and
    assumes the existence of a `make_window` helper function to format the output
    windows.
    """

    all_windows = {}
    for i in tqdm(range(len(norad_ids))):
        windows = check_visibility(
            tles_line1[i], tles_line2[i], times, 
            observer_lat, observer_lon, observer_elevation,
            target_ra, target_dec,
            max_ang_sep, min_elev
        )
        if len(windows)>0:
            all_windows[norad_ids[i]] = windows
    return all_windows


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