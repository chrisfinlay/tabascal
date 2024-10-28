import jax.numpy as jnp
from jax import jit, vmap, jacrev, config, Array
from tabascal.utils.jax_extras import jit_with_doc

config.update("jax_enable_x64", True)

# Constants
G = 6.67408e-11  # Gravitational constant in m^3/kg/s^2
M_e = 5.9722e24  # Mass of the Earth in kilograms
R_e = 6.371e6  # Average radius of the Earth in metres
T_s = 86164.0905  # Sidereal day in seconds
Omega_e = 2 * jnp.pi / T_s # Earth rotation rate in rad/s
C = 299792458.0 # Speed of light in m/s

@jit_with_doc
def radec_to_lmn(
    ra: Array, dec: Array, phase_centre: Array
) -> Array:
    """
    Convert right-ascension and declination positions of a set of sources to
    direction cosines.

    Parameters
    ----------
    ra : ndarray (n_src,)
        Right-ascension in degrees.
    dec : ndarray (n_src,)
        Declination in degrees.
    phase_centre : ndarray (2,)
        The ra and dec coordinates of the phase centre in degrees.

    Returns
    -------
    lmn : ndarray (n_src, 3)
        The direction cosines, (l,m,n), coordinates of each source.
    """
    ra = jnp.asarray(ra)
    dec = jnp.asarray(dec)
    phase_centre = jnp.asarray(phase_centre)
    ra, dec = jnp.deg2rad(jnp.array([ra, dec]))
    phase_centre = jnp.deg2rad(phase_centre)

    delta_ra = ra - phase_centre[0]
    dec_0 = phase_centre[1]

    l = jnp.cos(dec) * jnp.sin(delta_ra)
    m = jnp.sin(dec) * jnp.cos(dec_0) - jnp.cos(dec) * jnp.sin(dec_0) * jnp.cos(
        delta_ra
    )
    n = jnp.sqrt(1 - l**2 - m**2)

    return jnp.array([l, m, n]).T


@jit_with_doc
def radec_to_XYZ(ra: Array, dec: Array) -> Array:
    """
    Convert Right ascension and Declination to unit vector in ECI coordinates.

    Parameters
    ----------
    ra : ndarray (n_src,)
        Right-ascension in degrees.
    dec : ndarray (n_src,)
        Declination in degrees.

    Returns
    -------
    xyz: ndarray (n_src, 3) or (3,)
        The ECI coordinate unit vector of each source.
    """
    ra = jnp.asarray(ra)
    dec = jnp.asarray(dec)
    ra, dec = jnp.deg2rad(jnp.array([ra, dec]))
    x = jnp.cos(ra) * jnp.cos(dec)
    y = jnp.sin(ra) * jnp.cos(dec)
    z = jnp.sin(dec)

    return jnp.array([x, y, z]).T


# @jit_with_doc
# def radec_to_altaz(ra: float, dec: float, latitude: float, longitude: float, times: Array) -> Array:
#     """
#     !! Do not use this function - Needs to be checked !!
#     Convert Right ascension and Declination to unit vector in ECI coordinates.

#     Parameters
#     ----------
#     ra : float
#         Right-ascension in degrees.
#     dec : float
#         Declination in degrees.
#     latitude: float
#         The latitude of the observer in degrees.
#     times: ndarray (n_time,)
#         The time of each position in seconds.

#     Returns
#     -------
#     altaz: ndarray (n_time, 2)
#         The altiude and azimuth of the source at each time.
#     """
#     ra, dec = jnp.deg2rad(jnp.asarray([ra, dec]))
#     lat, lon = jnp.deg2rad(jnp.asarray([latitude, longitude]))
#     times = jnp.asarray(times)
#     gmst = 2.0 * jnp.pi * times / T_s
#     ha = (gmst + lon - ra) % 2 * jnp.pi

#     alt = jnp.arcsin(
#         jnp.sin(dec) * jnp.sin(lat) + jnp.cos(dec) * jnp.cos(lat) * jnp.cos(ha)
#     )

#     az = jnp.arctan2(
#         -jnp.sin(ha), jnp.cos(ha) * jnp.sin(lat) - jnp.tan(dec) * jnp.cos(lat)
#     )

#     # az = jnp.arccos(
#     #     (jnp.sin(dec) - jnp.sin(alt) * jnp.sin(lat)) / (jnp.cos(alt) * jnp.cos(lat))
#     # )
#     # az = jnp.where(ha < 0, 2 * jnp.pi - az, az)

#     # az = jnp.arcsin(
#     #     jnp.sin(dec) * jnp.sin(lat) + jnp.cos(dec) * jnp.cos(lat) * jnp.cos(ha)
#     # )
#     # az = jnp.where(ha < jnp.pi, 2 * jnp.pi - az, az)

#     return jnp.rad2deg(jnp.array([alt, az]).T)


@jit_with_doc
def ENU_to_GEO(geo_ref: Array, enu: Array) -> Array:
    """
    Convert a set of points in ENU co-ordinates to geographic coordinates i.e.
    (latitude, longitude, elevation).

    Parameters
    ----------
    geo_ref: ndarray (3,)
        The latitude, longitude and elevation, (lat,lon,elevation), of the
        reference position i.e. ENU = (0,0,0).
    enu: ndarray (n_ants, 3)
        The ENU coordinates of each antenna. (East, North, Up).

    Returns
    -------
    geo_ants: ndarray (n_ant, 3)
        The geographic coordinates, (lat,lon,elevation), of each antenna.
    """
    geo_ref = jnp.asarray(geo_ref)
    enu = jnp.asarray(enu)
    R_e = earth_radius(geo_ref[0])
    d_lon = jnp.rad2deg(
        jnp.arcsin(enu[:, 1] / (R_e * jnp.cos(jnp.deg2rad(geo_ref[0]))))
    )
    d_lat = jnp.rad2deg(jnp.arcsin(enu[:, 0] / R_e))
    geo_ants = jnp.array(geo_ref)[None, :] + jnp.array([d_lat, d_lon, enu[:, -1]]).T

    return geo_ants


@jit_with_doc
def GEO_to_XYZ(geo: Array, times: Array) -> Array:
    """
    Convert geographic coordinates to an Earth Centred Inertial (ECI)
    coordinate frame. This is different to ECEF as ECI remains fixed with the
    celestial sphere whereas ECEF coordinates rotate w.r.t. the celestial
    sphere. (0,0,0) is the Earth's centre of mass, +z points to the North Pole
    and +x is in the plane of the Equator passing through the Meridian at t = 0
    and +y is also in the plane of the Equator and passes through 90 degrees
    East at t = 0. ECEF and ECI are aligned when t % T_s = 0.

    Parameters
    ----------
    geo: ndarray (n_time, 3)
        The geographic coordinates, (lat,lon,elevation), at each point in time.
    times: ndarray (n_time,)
        The time of each position in seconds.

    Returns
    -------
    xyz: ndarray (n_time, 3)
        The ECI coordinates at each time, (lat,lon,elevation), of each antenna.
    """
    times = jnp.asarray(times)
    geo = jnp.asarray(geo) * jnp.ones((len(times), 3))
    lat, lon, elevation = geo.T
    R_e = earth_radius(lat)
    r = R_e + elevation
    lat = jnp.deg2rad(lat)
    lon = jnp.deg2rad(lon)
    omega = 2.0 * jnp.pi / T_s

    x = r * jnp.cos(lon + (omega * times)) * jnp.cos(lat)
    y = r * jnp.sin(lon + (omega * times)) * jnp.cos(lat)
    z = r * jnp.sin(lat)

    return jnp.array([x, y, z]).T


@jit_with_doc
def GEO_to_XYZ_vmap0(geo: Array, times: Array) -> Array:
    """
    Convert geographic coordinates to an Earth Centred Inertial (ECI)
    coordinate frame. This is different to ECEF as ECI remains fixed with the
    celestial sphere whereas ECEF coordinates rotate w.r.t. the celestial
    sphere. (0,0,0) is the Earth's centre of mass, +z points to the North Pole
    and +x is in the plane of the Equator passing through the Meridian at t = 0
    and +y is also in the plane of the Equator and passes through 90 degrees
    East at t = 0. ECEF and ECI are aligned when t % T_s = 0.

    Parameters
    ----------
    geo: ndarray (n_src, n_time, 3)
        The geographic coordinates, (lat,lon,elevation), at each point in time.
    times: ndarray (n_time,)
        The time of each position in seconds.

    Returns
    -------
    xyz: ndarray (n_src, n_time, 3)
        The ECI coordinates at each time, (lat,lon,elevation), of each antenna.
    """
    return vmap(GEO_to_XYZ, in_axes=(0, None), out_axes=0)(geo, times)


@jit_with_doc
def GEO_to_XYZ_vmap1(geo: Array, times: Array) -> Array:
    """
    Convert geographic coordinates to an Earth Centred Inertial (ECI)
    coordinate frame. This is different to ECEF as ECI remains fixed with the
    celestial sphere whereas ECEF coordinates rotate w.r.t. the celestial
    sphere. (0,0,0) is the Earth's centre of mass, +z points to the North Pole
    and +x is in the plane of the Equator passing through the Meridian at t = 0
    and +y is also in the plane of the Equator and passes through 90 degrees
    East at t = 0. ECEF and ECI are aligned when t % T_s = 0.

    Parameters
    ----------
    geo: ndarray (n_time, n_ant, 3)
        The geographic coordinates, (lat,lon,elevation), at each point in time.
    times: ndarray (n_time,)
        The time of each position in seconds.

    Returns
    -------
    xyz: ndarray (n_time, n_ant, 3)
        The ECI coordinates at each time, (lat,lon,elevation), of each antenna.
    """
    return vmap(GEO_to_XYZ, in_axes=(1, None), out_axes=1)(geo, times)


@jit_with_doc
def alt_az_of_source(lst: Array, lat: float, ra: float, dec: float) -> Array:
    """Calculate the altitude and azimuth of a given source direction over time. 
    Taken from https://astronomy.stackexchange.com/questions/14492/need-simple-equation-for-rise-transit-and-set-time

    Parameters
    ----------
    lst : Array (n_time,)
        Local sidereal time in degrees.
    lat : float
        Latitude of the observer in degrees.
    ra : float
        Right ascension of the source in degrees.
    dec : float
        DEclination of the source in degrees.

    Returns
    -------
    Array (n_time, 2)
        Altitude and azimuth of the source in degrees relative to the observer.
    """

    h0 = jnp.atleast_1d(lst) - ra 
    ones = jnp.ones_like(h0)
    lat, h0, dec = jnp.deg2rad(jnp.array([lat*ones, h0, dec*ones]))

    a = jnp.cos(lat)*jnp.sin(dec) - jnp.cos(dec)*jnp.sin(lat)*jnp.cos(h0)
    b = jnp.cos(dec)*jnp.sin(h0)
    c = jnp.sin(dec)*jnp.sin(lat) + jnp.cos(dec)*jnp.cos(lat)*jnp.cos(h0)

    alt = jnp.rad2deg(jnp.arctan2(c, jnp.sqrt( a**2 + b**2 )))
    az = jnp.rad2deg(jnp.arctan2(b, a))

    return jnp.array([alt, az]).T


@jit_with_doc
def rise_and_set_of_source(lat: float, ra: float, dec: float) -> Array:

    lat, dec = jnp.deg2rad(jnp.array([lat, dec]))
    
    a = jnp.rad2deg(jnp.arccos(-jnp.tan(dec)*jnp.tan(lat)))

    return jnp.array([ra-a, ra+a])


@jit_with_doc
def lst_deg2sec(lst: Array) -> Array:
    """Convert a sidereal time in degrees to seconds.

    Parameters
    ----------
    lst : Array
        Sidereal time in degrees to convert.

    Returns
    -------
    Array
        Sidereal time in seconds.
    """
    
    return lst / 360 * T_s


@jit_with_doc
def lst_sec2deg(lst: Array) -> Array:
    """Convert a sidereal time from seconds to degrees.

    Parameters
    ----------
    lst : Array
        Sidereal time in seconds to convert.

    Returns
    -------
    Array
        Sidereal time in degrees.
    """

    return lst / T_s * 360


@jit_with_doc
def gmst_to_lst(gmst: Array, lon: float) -> Array:
    """Calculate the local sidereal time at a given longitude in degrees.

    Parameters
    ----------
    gmst : Array
        Greenwich Mean Sidereal Time in seconds.
    lon : float
        Longitude of the location in degrees

    Returns
    -------
    Array
        Local Sidereal Time at the location in degrees.
    """

    gmst = jnp.asarray(gmst)
    lon = jnp.asarray(lon)

    lst = lst_sec2deg(gmst) + lon

    return lst

def gmst_from_jd(jd: float) -> float:
    """Get the Greenwich Mean Sidereal Time in seconds from the Julian Day (UT1).
    Calculated using https://aa.usno.navy.mil/faq/GAST

    Parameters
    ----------
    jd : float
        Julian Day (UT1).

    Returns
    -------
    float
        Greenwich Mean Sidereal Time in seconds. 
    """

    gmst_hours = 18.697375 + 24.065709824279 * (jd - 2451545.0)

    gmst = gmst_hours * 3600

    return gmst


def jd_from_gmst(gmst: float) -> float:
    """Get the Julian Date (UT1) from the Greenwich Mean Sidereal Time in seconds.

    Parameters
    ----------
    gmst : float
        Greenwich Mean Sidereal Time in seconds.

    Returns
    -------
    float
        Julian Date (UT1).
    """

    gmst_hours = gmst / 3600
    
    jd = (gmst_hours - 18.697375) / 24.065709824279 + 2451545.0

    return jd

def time_above_horizon(lat: float, dec: float) -> Array:
    """
    The number of degrees an object is above the horizon in a given day.
    """
    
    lat, dec = jnp.deg2rad(jnp.array([lat, dec]))
    
    if jnp.cos(dec)==0 or jnp.cos(lat)==0:
        return jnp.inf
        
    H = 2*jnp.rad2deg(jnp.arccos( -jnp.tan(lat)*jnp.tan(dec) ))
                   
    return H


@jit_with_doc
def transit_altitude(lat: float, dec: float) -> float:
    """Calculate the altitude of a source at transit given an observer's latitude.

    Parameters
    ----------
    lat : float
        Latitude of the observer in degrees.
    dec : float
        Declination of the source in degrees.

    Returns
    -------
    float
        Altitude of the source at transit in degrees.
    """

    alt = 90 - jnp.abs(dec-lat)

    return alt


@jit_with_doc
def earth_radius(lat: float) -> float:
    """Calculate the earth radius according to the ellipsoidal model at a given latitude.

    Parameters
    ----------
    lat : float
        Latitude of the location in degrees.

    Returns
    -------
    float
        Earth radius in metres at the given latitude.
    """
    a = 6378137.0 # equitorial radius
    b = 6356752.3 # polar radius
    lat = jnp.deg2rad(lat)
    cos = jnp.cos(lat)
    sin = jnp.sin(lat)
    r = jnp.sqrt(( (a**2*cos)**2 + (b**2*sin)**2 ) / ( (a*cos)**2 + (b*sin)**2 ) )

    return r


@jit_with_doc
def enu_to_itrf(enu: Array, lat: float, lon: float, el: float) -> Array:
    """
    Calculate ITRF coordinates from ENU coordinates of antennas given the
    latitude and longitude of the antenna array centre.

    Paramters
    ---------
    enu: Array (n_ant, 3)
        The East, North, Up coordinates of each antenna.
    lat: float
        The latitude of the observer/telescope.
    lon: float
        The longitude of the observer/telescope.
    el: float
        The elevation of the observer/telescope.

    Returns
    -------
    itrf: Array (n_ant, 3)
        The ITRF coordinates of the antennas.
    """

    enu = jnp.atleast_2d(enu)
    R = earth_radius(lat) + el
    lat, lon = jnp.deg2rad(jnp.array([lat, lon]))
    
    r0 = R*jnp.array([
        jnp.cos(lat)*jnp.cos(lon),
        jnp.cos(lat)*jnp.sin(lon),
        jnp.sin(lat)
    ])

    R = jnp.array([
        [-jnp.sin(lon), jnp.cos(lon), 0],
        [-jnp.cos(lon)*jnp.sin(lat), -jnp.sin(lon)*jnp.sin(lat), jnp.cos(lat)],
        [jnp.cos(lat)*jnp.cos(lon), jnp.cos(lat)*jnp.sin(lon), jnp.sin(lat)],
    ])

    return r0[None,:] + jnp.dot(enu, R)

def enu_to_xyz_local(enu, lat):
    """
    https://web.njit.edu/~gary/728/Lecture6.html
    """
    enu = jnp.atleast_2d(enu)
    lat = jnp.deg2rad(lat)

    R = jnp.array([
        [0, -jnp.sin(lat), jnp.cos(lat)],
        [1, 0, 0],
        [0, jnp.cos(lat), jnp.sin(lat)],
    ])

    return jnp.dot(enu, R.T)


@jit_with_doc
def itrf_to_geo(itrf: Array) ->  Array:
    """Convert ITRF coordinates to geodetic coordinates.

    Parameters
    ----------
    itrf : Array (n_ant, 3)
        ITRF coordinates.

    Returns
    -------
    Array (n_ant, 3)
        Geodetic coordinates (latitude, longitude, elevation)
    """
    x, y, z = jnp.atleast_2d(itrf).T
    xy = jnp.sqrt(x**2 + y**2)

    lat = jnp.rad2deg(jnp.arcsin(z/xy))
    lon = jnp.rad2deg(jnp.arctan2(y, x))
    el = jnp.linalg.norm(itrf, axis=-1) - earth_radius(lat)

    return jnp.array([lat, lon, el]).T


@jit_with_doc
def itrf_to_xyz(itrf: Array, gsa: Array) -> Array:
    """Transform coordinates from the ITRF (ECEF) frame to an ECI frame that aligns with the celestial sphere.

    Parameters
    ----------
    itrf : Array (n_ant, 3)
        ITRF coordinates in metres.
    gsa : Array (n_time,)
        Greenwich sidereal time in degrees

    Returns
    -------
    Array (n_time, n_ant, 3)
        ECI coordinates in metres.
    """
    
    itrf = jnp.atleast_2d(itrf)
    gsa = jnp.atleast_1d(gsa)
    ecef_to_eci = lambda ecef, gsa: jnp.einsum("ij,aj->ai", Rotz(gsa), ecef)
    xyz = vmap(ecef_to_eci, in_axes=(None,0))(itrf, gsa)

    return xyz


@jit_with_doc
def xyz_to_itrf(xyz: Array, gsa: Array) -> Array:
    """Transform coordinates from the ECI frame to the ITRF (ECEF) frame that is fixed with the Earth.

    Parameters
    ----------
    xyz : Array (n_time, 3)
        ECI coordinates in metres.
    gsa : Array (n_time,)
        Greenwich sidereal time in degrees.

    Returns
    -------
    Array (n_time, 3)
        ITRF (ECEF) coordinates in metres.
    """
    
    xyz = jnp.atleast_2d(xyz)
    gsa = jnp.atleast_1d(gsa)
    eci_to_ecef = lambda eci, gsa: Rotz(-gsa) @ eci
    itrf = vmap(eci_to_ecef, in_axes=(0,0))(xyz, gsa)

    return itrf

@jit_with_doc
def itrf_to_uvw(itrf: Array, h0: Array, dec: float) -> Array:
    """
    Calculate uvw coordinates from ITRF/ECEF coordinates,
    source hour angle and declination. Use the Greenwich hour 
    angle when using true ITRF coordinates such as those produced 
    with 'enu_to_itrf'. Use local hour angle when using local 'xyz' 
    coordinates as defined in most radio interferometry textbooks 
    or those produced with 'enu_to_xyz_local'.

    Parameters
    ----------
    ITRF: Array (n_ant, 3)
        Antenna positions in the ITRF frame in units of metres.
    h0: Array (n_time,)
        The hour angle of the target in decimal degrees.
    dec: float
        The declination of the target in decimal degrees.

    Returns
    -------
    uvw: Array (n_time, n_ant, 3)
        The uvw coordinates of the antennas for a given observer
        location, time and target (ra,dec).
    """

    itrf = jnp.atleast_2d(itrf)
    itrf = itrf - itrf[0,None,:]
    
    h0 = jnp.deg2rad(jnp.atleast_1d(h0))
    dec = jnp.deg2rad(jnp.asarray(dec))
    ones = jnp.ones_like(h0)
    
    R = jnp.array([
        [jnp.sin(h0), jnp.cos(h0), jnp.zeros_like(h0)],
        [-jnp.sin(dec)*jnp.cos(h0), jnp.sin(dec)*jnp.sin(h0), jnp.cos(dec)*ones],
        [jnp.cos(dec)*jnp.cos(h0), -jnp.cos(dec)*jnp.sin(h0), jnp.sin(dec)*ones]
    ])
    
    uvw = jnp.einsum("ijt,aj->tai", R, itrf)

    return uvw

@jit_with_doc
def enu_to_uvw(enu: Array,
    latitude: float,
    longitude: float,
    elevation: float,
    ra: float,
    dec: float,
    times: Array,
) -> Array:
    """
    Convert antenna coordinates in the ENU frame to the UVW coordinates, where
    w points at the phase centre defined by (ra,dec), at specific times for a
    telescope at a specifc latitude and longitude.

    Parameters
    ----------
    enu: ndarray (n_ant, 3)
        The East, North, Up coordindates of each antenna relative to the
        position defined by the latitude and longitude.
    latitude: float
        Latitude of the telescope.
    longitude: float
        Longitude of the telescope.
    ra: float
        Right Ascension of the phase centre.
    dec: float
        Declination of the phase centre.
    times: ndarray (n_time,)
        Times, in seconds, at which to calculate the UVW coordinates.

    Returns
    -------
    uvw: ndarray (n_time, n_ant, 3)
        UVW coordinates, in metres, of the individual antennas at each time.
    """

    enu = jnp.atleast_2d(jnp.asarray(enu))
    latitude = jnp.asarray(latitude).flatten()[0]
    longitude = jnp.asarray(longitude).flatten()[0]
    elevation = jnp.asarray(elevation).flatten()[0]
    ra = jnp.asarray(ra).flatten()[0]
    dec = jnp.asarray(dec).flatten()[0]
    times = jnp.atleast_1d(times)

    # xyz = enu_to_xyz_local(enu, latitude)
    # lh0 = lst_sec2deg(times) + longitude - ra # local hour angle
    # uvw = itrf_to_uvw(xyz, lh0, dec)

    itrf = enu_to_itrf(enu, latitude, longitude, elevation)
    gh0 = lst_sec2deg(times) - ra # Greenwich hour angle
    uvw = itrf_to_uvw(itrf, gh0, dec)

    return uvw


@jit_with_doc
def calculate_fringe_frequency(times: Array, freq: float, rfi_xyz: Array, ants_itrf: Array, ants_u: Array, dec: float) -> Array:
    """Calculate the fringe frequency of an RFI source.

    Parameters
    ----------
    times : Array (n_time,)
        Times are which the RFI and antenna positions are given in seconds.
    freq : float
        Observational frequency in Hz.
    rfi_xyz : Array (n_time, 3)
        Position of the RFI source in the ECI frame in metres.
    ants_itrf : Array (n_ant, 3)
        Antenna positions in the ITRF (ECEF) frame in metres. 
    ants_u : Array (n_time,)
        U component of the antennas in UVW frame in metres.
    dec : float
        Phase centre declination in degrees.

    Returns
    -------
    Array (n_time, n_bl)
        Fringe frequencies on each baseline.
    """

    lam = C / freq
    gsa = lst_sec2deg(times)

    r_ecef = xyz_to_itrf(rfi_xyz, gsa)
    s_ecef = r_ecef - jnp.mean(ants_itrf, axis=0)
    s_hat_ecef = s_ecef / jnp.linalg.norm(s_ecef, axis=-1, keepdims=True)
    s_hat_dot = jnp.gradient(s_hat_ecef, times, axis=0)

    a1, a2 = jnp.triu_indices(len(ants_itrf), 1)
    bl_ecef = ants_itrf[a1] - ants_itrf[a2]
    bl_u = ants_u[:,a1] - ants_u[:,a2]

    fringe_move = jnp.einsum("bi,ti->tb", bl_ecef, s_hat_dot) / lam
    fringe_stat = - bl_u * Omega_e * jnp.cos(jnp.deg2rad(dec)) / lam
    fringe_freq = fringe_move - fringe_stat

    return fringe_freq


@jit_with_doc
def calculate_sat_corr_time(sat_xyz: Array, ants_xyz: Array, orbit_el: float, lat: float, dish_d: float, freqs: Array) -> float:
    """Calculate the expected correlation time for a Gaussian process model of the RFI signal due to a satellite moving through the sidelobes of the primary beam.

    Parameters
    ----------
    rfi_xyz : Array (n_time, 3)
        Position of the RFI source over time in metres
    ants_xyz : Array (n_time, n_ant, 3)
        Positions of the antennas over time in metres.
    orbit_el : float
        Orbit elevation in metres.
    lat : float
        Latitude of the antennas in degrees.
    dish_d : float
        Dish diameter in metres. The primary beam is modelled by an Airy disk based on this.
    freqs : Array (n_freq,)
        Observational frequencies in Hz.

    Returns
    -------
    float
        Expected minimum correlation time in seconds.
    """

    r = sat_xyz - ants_xyz.mean(axis=1)
    R = jnp.linalg.norm(r, axis=-1).min()

    R_sat = earth_radius(lat) + orbit_el
    v_sat = jnp.sqrt(G * M_e / R_sat)

    l = jnp.min(C / freqs) * R / (dish_d * v_sat) / 4

    return l

@jit_with_doc
def angular_separation(
    rfi_xyz: Array, ants_xyz: Array, ra: float, dec: float
) -> Array:
    """
    Calculate the angular separation between the pointing direction of each
    antenna and the satellite source.

    Parameters
    ----------
    rfi_xyz: ndarray (n_src, n_time, 3)
        Position of the RFI sources in ECEF reference frame over time. This
        is the same frame as `ants_xyz`.
    ants_xyz: ndarray (n_time, n_ant, 3)
        Position of the antennas in ECEF reference frame over time.
    ra: float
        Right ascension of the pointing in decimal degrees.
    dec: float
        Declination of the pointing in decimal degrees.

    Returns
    -------
    angles: ndarray (n_src, n_time, n_ant)
        Angular separation (in degrees) between the pointing direction and RFI source for each antenna.
    """
    rfi_xyz = jnp.asarray(rfi_xyz)
    ants_xyz = jnp.asarray(ants_xyz)
    ra = jnp.asarray(ra).flatten()[0]
    dec = jnp.asarray(dec).flatten()[0]
    src_xyz = radec_to_XYZ(ra, dec)
    ant_to_sat_xyz = rfi_xyz[:, :, None, :] - ants_xyz[None, :, :, :]
    costheta = jnp.einsum("i,ljki->ljk", src_xyz, ant_to_sat_xyz) / jnp.linalg.norm(
        ant_to_sat_xyz, axis=-1
    )
    angles = jnp.rad2deg(jnp.arccos(costheta))
    return angles


@jit_with_doc
def Rotx(theta: float) -> Array:
    """
    Define a rotation matrix about the 'x-axis' by an angle theta, in degrees.

    Parameters
    ----------
    theta: float
        Rotation angle in degrees.

    Returns
    -------
    R: ndarray (3, 3)
        Rotation matrix.
    """
    theta = jnp.asarray(theta).flatten()[0]
    c = jnp.cos(jnp.deg2rad(theta))
    s = jnp.sin(jnp.deg2rad(theta))
    Rx = jnp.array([[1, 0, 0], [0, c, -s], [0, s, c]])

    return Rx


@jit_with_doc
def Rotz(theta: float) -> Array:
    """
    Define a rotation matrix about the 'z-axis' by an angle theta, in degrees.

    Parameters
    ----------
    theta: float
        Rotation angle in degrees.

    Returns
    -------
    R: ndarray (3, 3)
        Rotation matrix.
    """
    theta = jnp.asarray(theta).flatten()[0]
    c = jnp.cos(jnp.deg2rad(theta))
    s = jnp.sin(jnp.deg2rad(theta))
    Rz = jnp.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return Rz


@jit_with_doc
def orbit(
    times: Array,
    elevation: float,
    inclination: float,
    lon_asc_node: float,
    periapsis: float,
) -> Array:
    """
    Calculate orbital path of a satellite in perfect circular orbit.

    Parameters
    ----------
    times: ndarray (n_time,)
        Times at which to evaluate the positions.
    elevation: float
        Elevation/Altitude of the orbit in metres.
    inclination: float
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: float
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: float
        Perisapsis of the orbit. This is the angular starting point of the
        orbit at t = 0.

    Returns
    -------
    positions: ndarray (n_time, 3)
         The position vector of the orbiting object at each specified time.
    """
    times = jnp.asarray(times)
    elevation = jnp.asarray(elevation).flatten()[0]
    inclination = jnp.asarray(inclination).flatten()[0]
    lon_asc_node = jnp.asarray(lon_asc_node).flatten()[0]
    periapsis = jnp.asarray(periapsis).flatten()[0]
    R = R_e + elevation
    omega = jnp.sqrt(G * M_e / R**3)
    r = R * jnp.array(
        [jnp.cos(omega * (times)), jnp.sin(omega * (times)), jnp.zeros(len(times))]
    )
    R1 = Rotz(periapsis)
    R2 = Rotx(inclination)
    R3 = Rotz(lon_asc_node)
    rt = (R3 @ R2 @ R1 @ r).T

    return rt


def orbit_vmap(
    times: Array,
    elevation: Array,
    inclination: Array,
    lon_asc_node: Array,
    periapsis: Array,
) -> Array:
    """Calculate orbital path of a satellite in perfect circular orbit.

    Parameters
    ----------
    times: ndarray (n_time,)
        Times at which to evaluate the positions.
    elevation: nd_array (n_orbits,)
        Elevation/Altitude of the orbit in metres.
    inclination: nd_array (n_orbits,)
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: nd_array (n_orbits,)
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: nd_array (n_orbits,)
        Perisapsis of the orbit. This is the angular starting point of the
        orbit at t = 0.

    Returns
    -------
    positions: ndarray (n_orbits, n_time, 3)
         The position vectors of the orbiting objects at each specified time.
    """
    return vmap(orbit, in_axes=(None, 0, 0, 0, 0))(
        times, elevation, inclination, lon_asc_node, periapsis
    )


@jit_with_doc
def orbit_velocity(
    times: Array,
    elevation: float,
    inclination: float,
    lon_asc_node: float,
    periapsis: float,
) -> Array:
    """
    Calculate the velocity of a circular orbit at specific times.

    Parameters
    ----------
    times: ndarray (n_time,)
        Times at which to evaluate the rotation matrices.
    elevation: float
        Elevation/Altitude of the orbit in metres.
    inclination: float
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: float
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: float
        Perisapsis of the orbit. This is the angular starting point of the orbit
        at t = 0.

    Returns
    -------
    velocity: ndarray (n_time, 3)
        The velocity vector at the specified times.
    """
    times = jnp.asarray(times)
    elevation = jnp.asarray(elevation)
    inclination = jnp.asarray(inclination)
    lon_asc_node = jnp.asarray(lon_asc_node)
    periapsis = jnp.asarray(periapsis)
    vel_vmap = vmap(jacrev(orbit, argnums=(0)), in_axes=(0, None, None, None, None))
    velocity = vel_vmap(
        times[:, None], elevation, inclination, lon_asc_node, periapsis
    ).reshape(len(times), 3)

    return velocity


@jit_with_doc
def R_uvw(
    times: Array,
    elevation: float,
    inclination: float,
    lon_asc_node: float,
    periapsis: float,
) -> Array:
    """
    Calculate the rotation matrices at each time step to transform from an Earth
    centric reference frame (ECI) to a satellite centric frame given the
    parameters of its (circular) orbit.

    Parameters
    ----------
    times: ndarray (n_time,)
        Times at which to evaluate the rotation matrices.
    elevation: float
        Elevation/Altitude of the orbit in metres.
    inclination: float
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: float
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: float
        Perisapsis of the orbit. This is the angular starting point of the orbit
        at t = 0.

    Returns
    -------
    R: ndarray (3, 3)
        The rotation matrix to orient to a satellite centric (RIC) frame defined
        by the Radial, In-track, Cross-track components.
    """
    times = jnp.asarray(times)
    elevation = jnp.asarray(elevation)
    inclination = jnp.asarray(inclination)
    lon_asc_node = jnp.asarray(lon_asc_node)
    periapsis = jnp.asarray(periapsis)
    position = orbit(times, elevation, inclination, lon_asc_node, periapsis)
    velocity = orbit_velocity(times, elevation, inclination, lon_asc_node, periapsis)
    vel_hat = velocity / jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    u_hat = position / jnp.linalg.norm(position, axis=-1, keepdims=True)
    w_hat = jnp.cross(u_hat, vel_hat)
    v_hat = jnp.cross(w_hat, u_hat)
    R = jnp.stack([u_hat.T, v_hat.T, w_hat.T])

    return R


@jit_with_doc
def RIC_dev(
    times: Array,
    true_orbit_params: Array,
    estimated_orbit_params: Array,
) -> Array:
    """
    Calculate the Radial (R), In-track (I) and Cross-track (C) deviations
    between two circular orbits given their orbit parameters at many time steps.

    Parameters
    ----------
    times: array (n_time,)
        Times at which to evaluate the RIC deviations.
    true_orb_params: ndarray (4,)
        Orbit parameters (elevation, inclination, lon_asc_node, periapsis) of
        the object of most interest. The more accurate measurement when
        comparing the same object.
    est_orb_params: array (elevation, inclination, lon_asc_node, periapsis)
        Estimated orbit parameters.

    Returns
    -------
    RIC: ndarray (n_time, 3)
        Radial, In-track and Cross-track coordinates, in metres, of the orbiting
        body relative to the orbit defined by 'true_orbit_params'.
    """
    times = jnp.asarray(times)
    true_orbit_params = jnp.asarray(true_orbit_params)
    estimated_orbit_params = jnp.asarray(estimated_orbit_params)
    true_xyz = orbit(times, *true_orbit_params)
    est_xyz = orbit(times, *estimated_orbit_params)
    R = R_uvw(times, *true_orbit_params)
    true_uvw = vmap(lambda x, y: x @ y, in_axes=(-1, 0))(R, true_xyz)
    est_uvw = vmap(lambda x, y: x @ y, in_axes=(-1, 0))(R, est_xyz)

    return est_uvw - true_uvw


@jit_with_doc
def orbit_fisher(
    times: Array, orbit_params: Array, RIC_std: Array
) -> Array:
    """
    Calculate the inverse covariance (Fisher) matrix in orbital elements
    induced by errors in the RIC frame of an orbiting object. This is
    essentially linear uncertainty propagation assuming Gaussian errors.

    Parameters
    ----------
    times: ndarray (n_time,)
        Times at which the RIC covariance/standard deviations is defined.
    orbit_params: ndarray (4,)
        Orbit parameters (elevation, inclination, lon_asc_node, periapsis) for
        the orbit defining the origin of the RIC frame.
    RIC_std: ndarray (3,)
        The standard deviation in the Radial, In-track and Cross-track
        directions.

    Returns
    -------
    fisher: ndarray (4, 4)
        The inverse covariance (Fisher) matrix for the orbit parameters induced
        by the orbit uncertainties in the RIC frame.
    """
    times = jnp.asarray(times)
    orbit_params = jnp.asarray(orbit_params)
    RIC_std = jnp.asarray(RIC_std)
    J = jacrev(RIC_dev, argnums=(2,))(times, orbit_params, orbit_params)[0]
    F = jnp.diag(1.0 / RIC_std**2)

    def propagate(J, F):
        return J.T @ F @ J

    fisher = vmap(propagate, in_axes=(0, None))(J, F).sum(axis=0)

    return fisher
