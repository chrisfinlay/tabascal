import jax.numpy as jnp
from jax import jit, vmap, jacrev
from jax.config import config

config.update("jax_enable_x64", True)

# Constants
G = 6.67408e-11  # Gravitational constant
M_e = 5.9722e24  # Mass of the Earth
R_e = 6.371e6  # Average radius of the Earth
T_s = 86164.0905  # Sidereal day in seconds


def radec_to_lmn(
    ra: jnp.ndarray, dec: jnp.ndarray, phase_centre: jnp.ndarray
) -> jnp.ndarray:
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


def radec_to_XYZ(ra: jnp.ndarray, dec: jnp.ndarray) -> jnp.ndarray:
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


def ENU_to_GEO(geo_ref: jnp.ndarray, enu: jnp.ndarray) -> jnp.ndarray:
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


def GEO_to_XYZ(geo: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
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


def GEO_to_XYZ_vmap0(geo: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
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


def GEO_to_XYZ_vmap1(geo: jnp.ndarray, times: jnp.ndarray) -> jnp.ndarray:
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


def ENU_to_ITRF(enu: jnp.ndarray, lat: float, lon: float) -> jnp.ndarray:
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

    Returns
    -------
    itrf: jnp.array (n_ant, 3)
        The ITRF coordinates of the antennas.
    """
    enu = jnp.asarray(enu)
    lat = jnp.asarray(lat)
    lon = jnp.asarray(lon)
    E, L = jnp.deg2rad(jnp.array([lon, lat]))
    sL, cL = jnp.sin(L), jnp.cos(L)
    sE, cE = jnp.sin(E), jnp.cos(E)

    R = jnp.array([[-sL, -cL * sE, cL * cE], [cL, -sL * sE, sL * cE], [0.0, cE, sE]])

    itrf = jnp.dot(R, enu.T).T

    return itrf


def ITRF_to_UVW(
    ITRF: jnp.ndarray, ra: float, dec: float, lon: float, time: float
) -> jnp.ndarray:
    """
    Calculate uvw coordinates from ITRF/ECEF coordinates,
    longitude a Greenwich Mean Sidereal Time.

    Parameters
    ----------
    ITRF: jnp.array (n_ant, 3)
        Antenna positions in the ITRF frame in units of metres.
    ra: float
        The right ascension of the target in decimal degrees.
    dec: float
        The declination of the target in decimal degrees.
    lon: float
        The longitude at the observation location in decimal degrees.
    time: float
        Time of day in seconds past 12am.

    Returns
    -------
    uvw: jnp.array (n_ant, 3)
        The uvw coordinates of the antennas for a given observer
        location, time and target (ra,dec).
    """
    ITRF = jnp.asarray(ITRF)
    ra = jnp.asarray(ra).flatten()[0]
    dec = jnp.asarray(dec).flatten()[0]
    lon = jnp.asarray(lon).flatten()[0]
    time = jnp.asarray(time)
    gmst = 360.0 * (time / T_s)

    H0 = gmst + lon - ra
    d0 = dec

    H0, d0 = jnp.deg2rad(jnp.array([H0, d0]))
    sH0, cH0 = jnp.sin(H0), jnp.cos(H0)
    sd0, cd0 = jnp.sin(d0), jnp.cos(d0)

    R = jnp.array(
        [[sH0, cH0, 0.0], [-sd0 * cH0, sd0 * sH0, cd0], [cd0 * cH0, -cd0 * sH0, sd0]]
    )

    uvw = jnp.dot(R, ITRF.T).T

    return uvw


# @jit
def ENU_to_UVW(
    enu: jnp.ndarray,
    latitude: float,
    longitude: float,
    ra: float,
    dec: float,
    times: jnp.ndarray,
) -> jnp.ndarray:
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
    enu = jnp.asarray(enu)
    latitude = jnp.asarray(latitude).flatten()[0]
    longitude = jnp.asarray(longitude).flatten()[0]
    ra = jnp.asarray(ra).flatten()[0]
    dec = jnp.asarray(dec).flatten()[0]
    times = jnp.asarray(times)
    times = jnp.array([times]) if times.ndim < 1 else times
    itrf = ENU_to_ITRF(enu, latitude, longitude)
    UVW = vmap(ITRF_to_UVW, in_axes=(None, None, None, None, 0))
    uvw = UVW(itrf, ra, dec, longitude, times)

    return uvw


def angular_separation(
    rfi_xyz: jnp.ndarray, ants_xyz: jnp.ndarray, ra: float, dec: float
) -> jnp.ndarray:
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


# @jit
def Rotx(theta: float) -> jnp.ndarray:
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


# @jit
def Rotz(theta: float) -> jnp.ndarray:
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


# @jit
def orbit(
    times: jnp.ndarray,
    elevation: float,
    inclination: float,
    lon_asc_node: float,
    periapsis: float,
) -> jnp.ndarray:
    """
    Calculate orbital path of a satellite in perfect circular orbit.

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
        Perisapsis of the orbit. This is the angular starting point of the
        orbit at t = 0.

    Returns
    -------
    velocity: ndarray (n_time, 3)
         The velocity vector of the orbiting object at each specified time.
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
    times: jnp.ndarray,
    elevation: jnp.ndarray,
    inclination: jnp.ndarray,
    lon_asc_node: jnp.ndarray,
    periapsis: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate orbital path of a satellite in perfect circular orbit.

    Parameters
    ----------
    times: ndarray (n_time,)
        Times at which to evaluate the rotation matrices.
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
    velocity: ndarray (n_orbits, n_time, 3)
         The velocity vector of the orbiting object at each specified time.
    """
    return vmap(orbit, in_axes=(None, 0, 0, 0, 0))(
        times, elevation, inclination, lon_asc_node, periapsis
    )


# @jit
def orbit_velocity(
    times: jnp.ndarray,
    elevation: float,
    inclination: float,
    lon_asc_node: float,
    periapsis: float,
) -> jnp.ndarray:
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


# @jit
def R_uvw(
    times: jnp.ndarray,
    elevation: float,
    inclination: float,
    lon_asc_node: float,
    periapsis: float,
) -> jnp.ndarray:
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


# @jit
def RIC_dev(
    times: jnp.ndarray,
    true_orbit_params: jnp.ndarray,
    estimated_orbit_params: jnp.ndarray,
) -> jnp.ndarray:
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


# @jit
def orbit_fisher(
    times: jnp.ndarray, orbit_params: jnp.ndarray, RIC_std: jnp.ndarray
) -> jnp.ndarray:
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
