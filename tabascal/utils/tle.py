from astropy.time import Time
from astropy.coordinates import EarthLocation
from datetime import datetime
from skyfield.api import load, wgs84, EarthSatellite
from skyfield.positionlib import position_of_radec
import requests
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from spacetrack import SpaceTrackClient
import spacetrack.operators as op

from tqdm import tqdm

import os
import ast

from glob import glob

import string
import random

from spacetrack import SpaceTrackClient
from astropy.time import Time
import pandas as pd
from spacetrack import operators as op
import ast


def get_space_track_client(username, password):
    return SpaceTrackClient(identity=username, password=password)


def fetch_tle_data(
    st_client: SpaceTrackClient,
    norad_ids: list[int],
    epoch_jd: float,
    window_days: float = 1.0,
    limit: int = 2000,
):
    """
    Fetch TLE data for given NORAD IDs around a specific epoch.

    Parameters
    ----------
    st_client : SpaceTrackClient
        SpaceTrackClient instance
    norad_ids : list[int]
        List of NORAD IDs
    epoch_jd : float
        Julian date for the epoch
    window_days : int
        Window size in days around the epoch
    limit : int
        Maximum number of results to return

    Returns :
        pandas.DataFrame containing TLE data
    """
    start_time = Time(epoch_jd - window_days, format="jd", scale="ut1").datetime
    end_time = Time(epoch_jd + window_days, format="jd", scale="ut1").datetime
    date_range = op.inclusive_range(start_time, end_time)

    try:
        raw_data = st_client.tle(
            norad_cat_id=norad_ids, epoch=date_range, limit=limit, format="json"
        )
        return pd.DataFrame(ast.literal_eval(raw_data))
    except Exception as e:
        print(f"Error fetching TLE data: {str(e)}")
        raise


def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return "".join(random.choice(chars) for _ in range(size))


def get_tles_by_id(
    username: str,
    password: str,
    norad_ids: list[int],
    epoch_jd: float,
    window_days: float = 1.0,
    limit: int = 2000,
    tle_dir: str = None,
) -> pd.DataFrame:

    norad_ids = list(np.array(list(set(norad_ids))).astype(int))
    n_ids_start = len(norad_ids)
    epoch_str = Time(epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d")

    tles_local = pd.DataFrame()
    if tle_dir:
        tle_dir = os.path.abspath(tle_dir)
        tle_paths = glob(os.path.join(tle_dir, f"{epoch_str}-*.json"))
        local_ids = []
        if len(tle_paths) > 0:
            tles_local = pd.concat([pd.read_json(tle_path) for tle_path in tle_paths])
            tles_local = tles_local[tles_local["NORAD_CAT_ID"].isin(norad_ids)]
            local_ids = tles_local["NORAD_CAT_ID"].unique()
            norad_ids = list(set(norad_ids) - set(local_ids))
        print(f"Local TLEs loaded  : {len(local_ids)}")
    else:
        local_ids = []

    max_ids = 500
    n_ids = len(norad_ids)

    n_req = n_ids // max_ids + 1 if n_ids % max_ids > 0 else n_ids // max_ids

    remote_ids = []
    tles = pd.DataFrame()
    if len(norad_ids) > 0:
        client = get_space_track_client(username, password)
        tles = [0] * n_req
        for i in range(n_req):
            tles[i] = fetch_tle_data(client, norad_ids, epoch_jd, window_days, limit)
        if sum([len(tle) for tle in tles]) > 0:
            tles = pd.concat(tles)
            tles["Fetch_Timestamp"] = Time.now().fits
            remote_ids = tles["NORAD_CAT_ID"].unique()
        else:
            tles = pd.DataFrame()

    print(f"Remote TLEs loaded : {len(remote_ids)}")
    print(f"TLEs not found     : {n_ids_start - len(remote_ids) - len(local_ids)}")

    save_name = id_generator()

    if tle_dir and len(tles) > 0:
        save_path = os.path.join(tle_dir, f"{epoch_str}-{save_name}.json")
        tles.to_json(save_path)
        print(f"Saving remotely obtained TLEs to {save_path}")
    elif len(tles) > 0:
        save_path = os.path.join("./", f"{epoch_str}-{save_name}.json")
        tles.to_json(save_path)
        print(f"Saving remotely obtained TLEs to {save_path}")

    if tle_dir:
        tles = pd.concat([tles_local, tles])

    if len(tles) > 0:
        tles.reset_index(drop=True, inplace=True)
        tles["EPOCH_JD"] = tles["EPOCH"].apply(
            lambda x: Time(spacetrack_time_to_isot(x)).jd
        )
        tles = type_cast_tles(tles)
        tles = get_closest_times(tles, epoch_jd)

    return tles


def get_tles_by_name(
    username: str,
    password: str,
    names: list[str],
    epoch_jd: float,
    window_days: float = 1.0,
    limit: int = 10000,
    tle_dir: str = "./tles",
) -> pd.DataFrame:

    os.makedirs(tle_dir, exist_ok=True)

    # Calculate the date threshold
    epoch_str = Time(epoch_jd, format="jd", scale="ut1").strftime("%Y-%m-%d")
    start_time = Time(epoch_jd - window_days, format="jd", scale="ut1").datetime
    end_time = Time(epoch_jd + window_days, format="jd", scale="ut1").datetime
    drange = op.inclusive_range(start_time, end_time)

    names_op = [op.like(name.upper()) for name in names]

    st = SpaceTrackClient(identity=username, password=password)

    local_ids = 0
    remote_ids = 0
    tles = [0] * len(names)
    for i, name in enumerate(names):
        tle_path = os.path.join(tle_dir, f"{epoch_str}-{name}.json")
        if os.path.isfile(tle_path):
            tle = pd.read_json(tle_path)
            tles[i] = tle
            local_ids += len(tle["NORAD_CAT_ID"].unique())
        else:
            tle = pd.DataFrame(
                ast.literal_eval(
                    st.tle(
                        object_name=names_op[i],
                        epoch=drange,
                        limit=limit,
                        format="json",
                    )
                )
            )
            tle["Fetch_Timestamp"] = Time.now().strftime("%Y-%m-%d %H:%M:%S")
            tles[i] = tle
            if len(tle) > 0:
                remote_ids += len(tle["NORAD_CAT_ID"].unique())
                tles[i].to_json(tle_path)

    print(f"Local TLEs loaded   : {local_ids}")
    print(f"Remote TLEs loaded  : {remote_ids}")

    tles = pd.concat(tles)
    if len(tles) > 0:
        tles.reset_index(drop=True, inplace=True)
        tles["EPOCH_JD"] = tles["EPOCH"].apply(
            lambda x: Time(spacetrack_time_to_isot(x)).jd
        )
        tles = type_cast_tles(tles)
        tles = get_closest_times(tles, epoch_jd)

    return tles


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

    dt = datetime.strptime(spacetrack_time, "%Y-%m-%d %H:%M:%S")
    isot = dt.strftime("%Y-%m-%dT%H:%M:%S.000")

    return isot


def get_closest_times(
    df: pd.DataFrame,
    target_time_jd: float,
    id_col: str = "NORAD_CAT_ID",
    time_jd_col: str = "EPOCH_JD",
) -> pd.DataFrame:
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


def get_visible_satellite_tles(
    username: str,
    password: str,
    times: ArrayLike,
    observer_lat: float,
    observer_lon: float,
    observer_elevation: float,
    target_ra: float,
    target_dec: float,
    max_angular_separation: float,
    min_elevation: float,
    names: ArrayLike = [],
    norad_ids: ArrayLike = [],
    tle_dir: str = "./tles",
) -> tuple:
    """Get the TLEs corresponding to satellites that satisfy the conditions given.

    Parameters
    ----------
    username : str
        SpaceTrack username
    password : str
        SpaceTrack password.
    times : ArrayLike
        Times to condsider in Astropy.time.Time format.
    observer_lat : float
        Observer latitude in degrees.
    observer_lon : float
        Observer longitude in degrees.
    observer_elevation : float
        Observer elevation in metres above sea level.
    target_ra : float
        Right ascension of the target direction.
    target_dec : float
        Declination of the target direction.
    max_angular_separation : float
        Maximum angular separation, in degrees, to accept a satellite pass.
    min_elevation : float
        Minimum elevation, in degrees, above the horizon to accept the satellite pass.
    norad_ids : ArrayLike
        NORAD IDs to consider.
    names: list
        Satellite names to consider. An approximate search is done.
    tle_dir: str
        Directory path where TLEs should be / are cached.

    Returns
    -------
    tuple
        - NORAD IDs that pass the criteria.
        - TLEs for the satellites corresponding to the returned NORAD IDs.
    """

    tles = pd.DataFrame()
    if len(norad_ids) > 0:
        tles = get_tles_by_id(
            username, password, norad_ids, np.mean(times.jd), tle_dir=tle_dir
        )
    if len(names) > 0:
        tles = pd.concat(
            [
                tles,
                get_tles_by_name(
                    username, password, names, np.mean(times.jd), tle_dir=tle_dir
                ),
            ]
        )

    if len(tles) > 0:
        windows = check_satellite_visibilibities(
            tles["NORAD_CAT_ID"].values,
            tles["TLE_LINE1"].values,
            tles["TLE_LINE2"].values,
            times,
            observer_lat,
            observer_lon,
            observer_elevation,
            target_ra,
            target_dec,
            max_angular_separation,
            min_elevation,
        )

        if len(windows) > 0:
            tles_ = tles[tles["NORAD_CAT_ID"].isin(windows["norad_id"])][
                ["NORAD_CAT_ID", "TLE_LINE1", "TLE_LINE2"]
            ].values
            return tles_[:, 0], tles_[:, 1:]
        else:
            return [], None
    else:
        return [], None


def type_cast_tles(tles: pd.DataFrame) -> pd.DataFrame:

    numeric_cols = [
        "NORAD_CAT_ID",
        "EPOCH_MICROSECONDS",
        "MEAN_MOTION",
        "ECCENTRICITY",
        "INCLINATION",
        "RA_OF_ASC_NODE",
        "ARG_OF_PERICENTER",
        "MEAN_ANOMALY",
        "EPHEMERIS_TYPE",
        "ELEMENT_SET_NO",
        "REV_AT_EPOCH",
        "BSTAR",
        "MEAN_MOTION_DOT",
        "MEAN_MOTION_DDOT",
        "FILE",
        "OBJECT_NUMBER",
        "SEMIMAJOR_AXIS",
        "PERIOD",
        "APOGEE",
        "PERIGEE",
    ]

    for col in numeric_cols:
        tles[col] = pd.to_numeric(tles[col])
    tles["DECAYED"] = pd.to_numeric(tles["DECAYED"]).astype(bool)

    return tles


def make_window(
    times: ArrayLike, alt: ArrayLike, angular_sep: ArrayLike, idx: ArrayLike
) -> dict:
    """Make a dictionary containing the start and end times of a satellite pass including some stats.

    Parameters
    ----------
    times : ArrayLike[Time]
        Times of the satellite pass.
    alt : ArrayLike
        Altitude of the satellite during pass.
    angular_sep : ArrayLike
        Angular separation of the satellite during pass from target.
    idx: ArrayLike
        Index locations of the window.
    Returns
    -------
    dict
        Dictionary of stats.
    """

    window = {
        "start_time": times[idx][0].datetime.strftime(
            f"%Y-%m-%d-%H:%M:%S {times.scale.upper()}"
        ),
        "end_time": times[idx][-1].datetime.strftime(
            f"%Y-%m-%d-%H:%M:%S {times.scale.upper()}"
        ),
        "min_ang_sep": np.min(angular_sep[idx]),
        "max_elevation": np.max(alt[idx]),
    }

    return window


def check_visibility(
    tle_line1: str,
    tle_line2: str,
    times: list[Time],
    observer_lat: float,
    observer_lon: float,
    observer_elevation: float,
    target_ra: float,
    target_dec: float,
    max_ang_sep: float,
    min_elev: float,
) -> list:
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
    times : list[Time]
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
    target = position_of_radec(
        ra_hours=target_ra / 15, dec_degrees=target_dec
    )  # Convert RA to hours

    satellite_position = satellite.at(sf_times)

    topocentric = satellite_position - observer_location.at(sf_times)
    alt, az, distance = topocentric.altaz()

    angular_sep = topocentric.separation_from(target).degrees

    vis_idx = np.where((alt.degrees > min_elev) & (angular_sep < max_ang_sep))[0]
    break_idx = np.where(np.diff(vis_idx) > 1)[0]
    if len(break_idx) > 0 or len(vis_idx) > 0:
        break_idx = np.concatenate([[0], break_idx, [len(times)]])
        windows = [
            make_window(
                times,
                alt.degrees,
                angular_sep,
                vis_idx[break_idx[i] : break_idx[i + 1]],
            )
            for i in range(len(break_idx) - 1)
        ]
    else:
        windows = []

    return windows


def check_satellite_visibilibities(
    norad_ids: list[int],
    tles_line1: list[str],
    tles_line2: list[str],
    times: list[Time],
    observer_lat: float,
    observer_lon: float,
    observer_elevation: float,
    target_ra: float,
    target_dec: float,
    max_ang_sep: float,
    min_elev: float,
) -> dict:
    """Calculate visibility windows for a satellite when observing a celestial target.

    This function determines time windows when a satellite will pass a celestial
    target based on the satellite's orbital parameters (TLE), observer location,
    target coordinates, and visibility constraints.

    Parameters
    ----------
    norad_ids: list[int]
        NORAD IDs to calculate for.
    tles_line1 : list[str]
        First line of the satellites' Two-Line Element set (TLE).
    tles_line2 : list[str]
        Second line of the satellites' Two-Line Element set (TLE).
    times : list[Time]
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

    print()
    print(
        f"Searching which satellites satisfy max_ang_sep: {max_ang_sep:.0f} and min_elev: {min_elev:.0f}"
    )
    all_windows = []
    for i in tqdm(range(len(norad_ids))):
        windows = check_visibility(
            tles_line1[i],
            tles_line2[i],
            times,
            observer_lat,
            observer_lon,
            observer_elevation,
            target_ra,
            target_dec,
            max_ang_sep,
            min_elev,
        )
        if len(windows) > 0:
            all_windows += [{"norad_id": norad_ids[i], **window} for window in windows]

    print(f"Found {len(all_windows)} matching satellites")
    return pd.DataFrame(all_windows)


def get_satellite_positions(tles: list, times_jd: list) -> ArrayLike:
    """Calculate the ICRS positions of satellites by propagating their TLEs over the given times.

    Parameters
    ----------
    tles : Array (n_sat, 2)
        TLEs usind to propagate positions.
    times : Array (n_time,)
        Times to calculate positions at in Julian date.

    Returns
    -------
    Array (n_sat, n_time, 3)
        Satellite positions over time
    """

    ts = load.timescale()
    sf_times = ts.ut1_jd(times_jd)

    sat_pos = np.array(
        [
            EarthSatellite(tle_line1, tle_line2, ts=ts).at(sf_times).position.km.T * 1e3
            for tle_line1, tle_line2 in tles
        ]
    )

    return sat_pos


def ant_pos(ant_itrf: ArrayLike, times_jd: ArrayLike) -> ArrayLike:

    ts = load.timescale()
    t = ts.ut1_jd(times_jd)

    location = EarthLocation(x=ant_itrf[0], y=ant_itrf[1], z=ant_itrf[2], unit="m")
    observer = wgs84.latlon(
        location.lat.degree, location.lon.degree, location.height.value
    )

    return (observer.at(t).position.km * 1e3).T


def ants_pos(ants_itrf: ArrayLike, times_jd: ArrayLike) -> ArrayLike:

    return np.transpose(
        np.array([ant_pos(ant_itrf, times_jd) for ant_itrf in ants_itrf]),
        axes=(1, 0, 2),
    )


def sat_distance(tle: list[str], times_jd: ArrayLike, obs_itrf: ArrayLike) -> ArrayLike:

    ts = load.timescale()

    t = ts.ut1_jd(times_jd)

    satellite = EarthSatellite(tle[0], tle[1], ts=ts)

    location = EarthLocation(x=obs_itrf[0], y=obs_itrf[1], z=obs_itrf[2], unit="m")

    observer = wgs84.latlon(location.lat.degree, location.lon.degree, location.height)

    topo = (satellite - observer).at(t)

    return topo.distance().m


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


def get_tle_data(norad_id: int | str, jd: float) -> dict:
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

    url = "https://satchecker.cps.iau.org/tools/get-tle-data/"
    params = {
        "id": str(norad_id),
        "id_type": "catalog",
        "start_date_jd": str(jd - 1),
        "end_date_jd": str(jd + 1),
    }
    response = requests.get(url, params=params).json()

    if len(response) > 0:
        delta_epoch = [
            np.abs(Time(sathub_time_to_isot(resp["epoch"]), format="isot").jd - jd)
            for resp in response
        ]
    else:
        ValueError("No TLEs found within 1 day of the given date.")

    return response[np.argmin(delta_epoch)]


def get_sat_pos_tle(
    tle_line1: str, tle_line2: str, sat_name: str, times_jd: float
) -> ArrayLike:
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


def get_sat_pos(norad_id: int | str, times_jd: float) -> ArrayLike:

    sathub_resp = get_tle_data(norad_id, np.mean(times_jd))
    sat_pos = get_sat_pos_tle(
        sathub_resp["tle_line1"],
        sathub_resp["tle_line2"],
        sathub_resp["satellite_name"],
        times_jd,
    )

    return sat_pos.T


def get_visible_sats(
    norad_id: int, times_jd, latitude, longitude, elevation: float = 0.0
):

    jd_step = np.diff(times_jd)[0]

    url = "https://satchecker.cps.iau.org/ephemeris/catalog-number-jdstep/"
    params = {
        "catalog": str(norad_id),
        "latitude": latitude,
        "longitude": longitude,
        "elevation": elevation,
        "startjd": np.min(times_jd),
        "stopjd": np.max(times_jd),
        "stepjd": jd_step,
        "min_altitude": 0,
    }
    r = requests.get(url, params=params).json()
    if "count" in r.keys():
        return pd.DataFrame(
            [
                {key: val for key, val in zip(r["fields"], r["data"][i])}
                for i in range(r["count"])
            ]
        )
    else:
        return pd.DataFrame()
