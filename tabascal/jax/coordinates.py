import jax.numpy as jnp
from jax import jit, vmap, jacrev, config, Array

config.update("jax_enable_x64", True)

# Constants
G = 6.67408e-11  # Gravitational constant
M_e = 5.9722e24  # Mass of the Earth
R_e = 6.371e6  # Average radius of the Earth
T_s = 86164.0905  # Sidereal day in seconds


@jit
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


@jit
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


@jit
def radec_to_altaz(ra: float, dec: float, latitude: float, longitude: float, times: Array) -> Array:
    """
    !! Do not use this function - Needs to be checked !!
    Convert Right ascension and Declination to unit vector in ECI coordinates.

    Parameters
    ----------
    ra : float
        Right-ascension in degrees.
    dec : float
        Declination in degrees.
    latitude: float
        The latitude of the observer in degrees.
    times: ndarray (n_time,)
        The time of each position in seconds.

    Returns
    -------
    altaz: ndarray (n_time, 2)
        The altiude and azimuth of the source at each time.
    """
    ra, dec = jnp.deg2rad(jnp.asarray([ra, dec]))
    lat, lon = jnp.deg2rad(jnp.asarray([latitude, longitude]))
    times = jnp.asarray(times)
    gmst = 2.0 * jnp.pi * times / T_s
    ha = (gmst + lon - ra) % 2 * jnp.pi

    alt = jnp.arcsin(
        jnp.sin(dec) * jnp.sin(lat) + jnp.cos(dec) * jnp.cos(lat) * jnp.cos(ha)
    )

    az = jnp.arctan2(
        -jnp.sin(ha), jnp.cos(ha) * jnp.sin(lat) - jnp.tan(dec) * jnp.cos(lat)
    )

    # az = jnp.arccos(
    #     (jnp.sin(dec) - jnp.sin(alt) * jnp.sin(lat)) / (jnp.cos(alt) * jnp.cos(lat))
    # )
    # az = jnp.where(ha < 0, 2 * jnp.pi - az, az)

    # az = jnp.arcsin(
    #     jnp.sin(dec) * jnp.sin(lat) + jnp.cos(dec) * jnp.cos(lat) * jnp.cos(ha)
    # )
    # az = jnp.where(ha < jnp.pi, 2 * jnp.pi - az, az)

    return jnp.rad2deg(jnp.array([alt, az]).T)


@jit
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
    d_lon = jnp.rad2deg(
        jnp.arcsin(enu[:, 1] / (R_e * jnp.cos(jnp.deg2rad(geo_ref[0]))))
    )
    d_lat = jnp.rad2deg(jnp.arcsin(enu[:, 0] / R_e))
    geo_ants = jnp.array(geo_ref)[None, :] + jnp.array([d_lat, d_lon, enu[:, -1]]).T

    return geo_ants


@jit
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
    # r = dist_centre(lat) + elevation
    r = R_e + elevation
    lat = jnp.deg2rad(lat)
    lon = jnp.deg2rad(lon)
    omega = 2.0 * jnp.pi / T_s

    x = r * jnp.cos(lon + (omega * times)) * jnp.cos(lat)
    y = r * jnp.sin(lon + (omega * times)) * jnp.cos(lat)
    z = r * jnp.sin(lat)

    return jnp.array([x, y, z]).T


@jit
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


@jit
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


@jit
def alt_az_of_source(lst: Array, lat: float, ra: float, dec: float) -> Array:
    """
    https://astronomy.stackexchange.com/questions/14492/need-simple-equation-for-rise-transit-and-set-time
    """

    h0 = ra - jnp.atleast_1d(lst)
    ones = jnp.ones_like(h0)
    lat, h0, dec = jnp.deg2rad(jnp.array([lat*ones, h0, dec*ones]))

    a = jnp.cos(lat)*jnp.sin(dec) - jnp.cos(dec)*jnp.sin(lat)*jnp.cos(h0)
    b = jnp.cos(dec)*jnp.sin(h0)
    c = jnp.sin(dec)*jnp.sin(lat) + jnp.cos(dec)*jnp.cos(lat)*jnp.cos(h0)

    alt = jnp.rad2deg(jnp.arctan2(c, jnp.sqrt( a**2 + b**2 )))
    az = jnp.rad2deg(jnp.arctan2(b, a))

    return jnp.array([alt, az]).T


@jit
def rise_and_set_of_source(lat: float, ra: float, dec: float) -> Array:

    lat, dec = jnp.deg2rad(jnp.array([lat, dec]))
    
    a = jnp.rad2deg(jnp.arccos(-jnp.tan(dec)*jnp.tan(lat)))

    return jnp.array([ra-a, ra+a])


@jit
def lst_deg2sec(lst: Array) -> Array:
    """Convert sidereal time in degrees to seconds.

    Parameters
    ----------
    lst : Array
        Sidereal time in degrees.

    Returns
    -------
    Array
        Sidereal time in seconds.
    """
    
    return lst / 360 * T_s


@jit
def lst_sec2deg(lst: Array) -> Array:

    return lst / T_s * 360


@jit
def gmst_to_lst(gmst: Array, lon: float) -> Array:

    lst = gmst + lst_deg2sec(lon)

    return lst

def time_above_horizon(lat: float, dec: float) -> Array:
    """
    The number of degrees an object is above the horizon in a given day.
    """
    
    lat, dec = jnp.deg2rad(jnp.array([lat, dec]))
    
    if jnp.cos(dec)==0 or jnp.cos(lat)==0:
        return jnp.inf
        
    H = 2*jnp.rad2deg(jnp.arccos( -jnp.tan(lat)*jnp.tan(dec) ))
                   
    return H


@jit
def transit_altitude(lat: float, dec: float) -> float:

    alt = 90 - jnp.abs(dec-lat)

    return alt


@jit
def earth_radius(lat: float) -> float:
    a = 6378137.0 # equitorial radius
    b = 6356752.3 # polar radius
    lat = jnp.deg2rad(lat)
    cos = jnp.cos(lat)
    sin = jnp.sin(lat)
    r = jnp.sqrt(( (a**2*cos)**2 + (b**2*sin)**2 ) / ( (a*cos)**2 + (b*sin)**2 ) )

    return r


@jit
def enu_to_itrf(enu: Array, lat: float, lon: float, el: float) -> Array:
    """
    Calculate ITRF coordinates from ENU coordinates of antennas given the
    latitude and longitude of the antenna array centre.

    Paramters
    ---------
    enu: ndarray (n_ant, 3)
        The East, North, Up coordinates of each antenna.
    lat: float
        The latitude of the observer/telescope.
    lon: float
        The longitude of the observer/telescope.
    el: float
        The elevation of the observer/telescope.

    Returns
    -------
    itrf: jnp.array (n_ant, 3)
        The ITRF coordinates of the antennas.
    """

    enu = jnp.atleast_2d(enu)
    R_e = earth_radius(lat)
    lat, lon = jnp.deg2rad(jnp.array([lat, lon]))
    
    r0 = (R_e+el)*jnp.array([
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


@jit
def itrf_to_uvw(itrf: Array, h0: Array, dec: float) -> Array:
    """
    Calculate uvw coordinates from ITRF/ECEF coordinates,
    source local hour angle and declination.

    Parameters
    ----------
    ITRF: Array (n_ant, 3)
        Antenna positions in the ITRF frame in units of metres.
    h0: float
        The hour angle of the target in decimal degrees.
    dec: float
        The declination of the target in decimal degrees.

    Returns
    -------
    uvw: Array (n_ant, 3)
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

@jit
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

    itrf = enu_to_itrf(enu, latitude, longitude, elevation)
    h0 = lst_sec2deg(times) + longitude - ra
    uvw = itrf_to_uvw(itrf, h0, dec)

    return uvw

@jit
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
        Angular separation between the pointing direction and RFI source for each antenna.
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


@jit
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


@jit
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


@jit
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


@jit
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


@jit
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


@jit
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


@jit
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
