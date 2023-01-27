import jax.numpy as jnp
from jax import jit, vmap, jacrev
from jax.config import config
config.update("jax_enable_x64", True)

# Constants
G = 6.67408e-11                 # Gravitational constant
M_e = 5.9722e24                 # Mass of the Earth
R_e = 6.371e6                   # Average radius of the Earth
T_s = 86164.0905                # Sidereal day in seconds

def radec_to_lmn(ra, dec, phase_centre):
    """
    Convert right-ascension and declination positions of a set of sources to
    direction cosines.

    Parameters
    ----------
    ra : array_like (n_src,)
        Right-ascension in degrees.
    dec : array_like (n_src,)
        Declination in degrees.
    phase_centre : array_like (2,)
        The ra and dec coordinates of the phase centre in degrees.

    Returns
    -------
    lmn : array_like (n_src, 3)
        The direction cosines, (l,m,n), coordinates of each source.
    """
    ra, dec = jnp.deg2rad(jnp.array([ra, dec]))
    phase_centre = jnp.deg2rad(phase_centre)

    delta_ra = ra - phase_centre[0]
    dec_0 = phase_centre[1]

    l = jnp.cos(dec)*jnp.sin(delta_ra)
    m = jnp.sin(dec)*jnp.cos(dec_0) - \
        jnp.cos(dec)*jnp.sin(dec_0)*jnp.cos(delta_ra)
    n = jnp.sqrt(1 - l**2 - m**2) - 1

    return jnp.array([l,m,n]).T

def ENU_to_GEO(geo_ref, enu):
    """
    Convert a set of points in ENU co-ordinates to geographic coordinates i.e.
    (latitude, longitude, elevation).

    Parameters:
    ----------
    geo_ref: array_like (3,)
        The latitude, longitude and elevation, (lat,lon,el), of the reference
        position i.e. ENU = (0,0,0).
    enu: array_like (n_ants, 3)
        The ENU coordinates of each antenna. (East, North, Up).

    Returns:
    --------
    geo_ants: array_like (n_ant, 3)
        The geographic coordinates, (lat,lon,el), of each antenna.
    """
    d_lon = jnp.rad2deg(jnp.arcsin(enu[:,1]/(R_e*jnp.cos(jnp.deg2rad(geo_ref[0])))))
    d_lat = jnp.rad2deg(jnp.arcsin(enu[:,0]/R_e))
    geo_ants = jnp.array(geo_ref)[None,:] + \
                jnp.array([d_lat, d_lon, enu[:,-1]]).T

    return geo_ants


def GEO_to_XYZ(geo, t):
    """
    Convert geographic coordinates to an Earth Centred Inertial (ECI)
    coordinate frame. This is different to ECEF as ECI remains fixed with the
    celestial sphere whereas ECEF coordinates rotate w.r.t. the celestial
    sphere. (0,0,0) is the Earth's centre of mass, +z points to the North Pole
    and +x is in the plane of the Equator passing through the Meridian at t = 0
    and +y is also in the plane of the Equator and passes through 90 degrees
    East at t = 0. ECEF and ECI are aligned when t % T_s = 0.

    Parameters:
    ----------
    geo: array_like (n_time, 3)
        The geographic coordinates, (lat,lon,el), at each point in time.
    t: array_like (n_times,)
        The time of each position in seconds.

    Returns:
    --------
    xyz: array_like (n_time, 3)
        The ECI coordinates at each time, (lat,lon,el), of each antenna.
    """
    lat, lon, el = geo.T
    # r = dist_centre(lat) + el
    r = R_e + el
    lat = jnp.deg2rad(lat)
    lon = jnp.deg2rad(lon)
    omega = 2.0*jnp.pi/T_s

    x = r*jnp.cos(lon+(omega*t))*jnp.cos(lat)
    y = r*jnp.sin(lon+(omega*t))*jnp.cos(lat)
    z = r*jnp.sin(lat)

    return jnp.array([x, y, z]).T


def radec_to_XYZ(ra, dec):
    """
    Convert Right ascension and Declination to unit vector in ECI coordinates.

    Parameters:
    ----------
    ra : array_like (n_src,)
        Right-ascension in degrees.
    dec : array_like (n_src,)
        Declination in degrees.

    Returns:
    --------
    xyz: array_like (n_src, 3) or (3,)
        The ECI coordinate unit vector of each source.
    """
    ra, dec = jnp.deg2rad(jnp.array([ra, dec]))
    x = jnp.cos(ra)*jnp.cos(dec)
    y = jnp.sin(ra)*jnp.cos(dec)
    z = jnp.sin(dec)

    return jnp.array([x,y,z]).T

def ENU_to_ITRF(enu, lat, lon):
    """
    Calculate ITRF coordinates from ENU coordinates of antennas given the
    latitude and longitude of the antenna array centre.

    Paramters:
    ----------
    enu: array_like (n_ant, 3)
        The East, North, Up coordinates of each antenna.
    lat: float
        The latitude of the observer/telescope.
    lon: float
        The longitude of the observer/telescope.

    Returns:
    --------
    itrf: jnp.array (n_ant, 3)
        The ITRF coordinates of the antennas.
    """
    E, L = jnp.deg2rad(jnp.array([lon, lat]))
    sL, cL = jnp.sin(L), jnp.cos(L)
    sE, cE = jnp.sin(E), jnp.cos(E)

    R = jnp.array([[-sL, -cL*sE, cL*cE],
                   [cL,  -sL*sE, sL*cE],
                   [0.0,    cE,    sE ]])

    itrf = jnp.dot(R, enu.T).T

    return itrf


def ITRF_to_UVW(ITRF, ra, dec, lon, time):
    """
    Calculate uvw coordinates from ITRF/ECEF coordinates,
    longitude a Greenwich Mean Sidereal Time.

    Parameters:
    -----------
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

    Returns:
    --------
    uvw: jnp.array (n_ant, 3)
        The uvw coordinates of the antennas for a given observer
        location, time and target (ra,dec).
    """
    gmst = 360.0*(time/T_s)

    H0 = gmst + lon - ra
    d0 = dec

    H0, d0 = jnp.deg2rad(jnp.array([H0, d0]))
    sH0, cH0 = jnp.sin(H0), jnp.cos(H0)
    sd0, cd0 = jnp.sin(d0), jnp.cos(d0)

    R = jnp.array([[sH0,      cH0,      0.0],
                  [-sd0*cH0, sd0*sH0,  cd0],
                  [cd0*cH0,  -cd0*sH0, sd0]])

    uvw = jnp.dot(R, ITRF.T).T

    return uvw

@jit
def ENU_to_UVW(enu, latitude, longitude, ra, dec, times):
    """
    Convert antenna coordinates in the ENU frame to the UVW coordinates, where W
    points at the phase centre defined by (ra,dec), at specific times for a
    telescope at a specifc latitude and longitude.

    Parameters:
    -----------
    enu: array_like (n_ant, 3)
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
    times: array_like (n_time,)
        Times, in seconds, at which to calculate the UVW coordinates.

    Return:
    -------
    uvw: array_like (n_time, n_ant, 3)
        UVW coordinates, in metres, of the individual antennas at each time.
    """
    times = jnp.array([times]) if times.ndim<1 else times
    itrf = ENU_to_ITRF(enu, latitude, longitude)
    uvw = vmap(ITRF_to_UVW, in_axes=(None,None,None,None,0))
    uvw = uvw(itrf, ra, dec, longitude, times)

    return uvw

@jit
def Rotx(theta):
    """
    Define a rotation matrix about the 'x-axis' by an angle theta, in degrees.

    Parameters:
    -----------
    theta: float
        Rotation angle in degrees.

    Returns:
    --------
    R: array_like (3, 3)
        Rotation matrix.
    """
    c = jnp.cos(jnp.deg2rad(theta))
    s = jnp.sin(jnp.deg2rad(theta))
    Rx = jnp.array([[1, 0,  0],
                    [0, c, -s],
                    [0, s,  c]])

    return Rx

@jit
def Rotz(theta):
    """
    Define a rotation matrix about the 'z-axis' by an angle theta, in degrees.

    Parameters:
    -----------
    theta: float
        Rotation angle in degrees.

    Returns:
    --------
    R: array_like (3, 3)
        Rotation matrix.
    """
    c = jnp.cos(jnp.deg2rad(theta))
    s = jnp.sin(jnp.deg2rad(theta))
    Rz = jnp.array([[c, -s, 0],
                    [s,  c, 0],
                    [0,  0, 1]])

    return Rz

@jit
def orbit(t, elevation, inclination, lon_asc_node, periapsis):
    """
    Calculate orbital path of a satellite in perfect circular orbit.

    Parameters:
    -----------
    t: array_like (n_time,)
        Times at which to evaluate the rotation matrices.
    el: float
        Elevation/Altitude of the orbit in metres.
    inclination: float
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: float
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: float
        Perisapsis of the orbit. This is the angular starting point of the orbit
        at t = 0.

    Returns:
    --------
    velocity: array_like (n_time, 3)
         The velocity vector of the orbiting object at each specified time.
    """
    R = R_e + elevation
    omega = jnp.sqrt(G*M_e/R**3)
    r = R*jnp.array([jnp.cos(omega*(t)),
                     jnp.sin(omega*(t)),
                     jnp.zeros(len(t))])
    R1 = Rotz(periapsis)
    R2 = Rotx(inclination)
    R3 = Rotz(lon_asc_node)
    rt = (R3@R2@R1@r).T

    return rt

@jit
def orbit_velocity(times, el, inclination, lon_asc_node, periapsis):
    """
    Calculate the velocity of a circular orbit at specific times.

    Parameters:
    -----------
    times: array_like (n_time,)
        Times at which to evaluate the rotation matrices.
    el: float
        Elevation/Altitude of the orbit in metres.
    inclination: float
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: float
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: float
        Perisapsis of the orbit. This is the angular starting point of the orbit
        at t = 0.

    Returns:
    --------
    velocity: array_like (n_time, 3)
        The velocity vector at the specified times.
    """
    velocity = vmap(jacrev(orbit, argnums=(0)), in_axes=(0,None,None,None,None))
    velocity = velocity(times[:,None], el,
                        inclination, lon_asc_node,
                        periapsis).reshape(len(times),3)

    return velocity

@jit
def R_uvw(times, el, inclination, lon_asc_node, periapsis):
    """
    Calculate the rotation matrices at each time step to transform from an Earth
    centric reference frame (ECI) to a satellite centric frame given the
    parameters of its (circular) orbit.

    Parameters:
    -----------
    times: array_like (n_time,)
        Times at which to evaluate the rotation matrices.
    el: float
        Elevation/Altitude of the orbit in metres.
    inclination: float
        Inclination angle of the orbit relative to the equatorial plane.
    lon_asc_node: float
        Longitude of the ascending node of the orbit. This is the longitude of
        when the orbit crosses the equator from the south to the north.
    periapsis: float
        Perisapsis of the orbit. This is the angular starting point of the orbit
        at t = 0.

    Returns:
    --------
    R: array_like (3, 3)
        The rotation matrix to orient to a satellite centric (RIC) frame defined
        by the Radial, In-track, Cross-track components.
    """
    position = orbit(times, el, inclination, lon_asc_node, periapsis)
    velocity = orbit_velocity(times, el, inclination, lon_asc_node, periapsis)
    vel_hat = sat_vel/jnp.linalg.norm(velocity, axis=-1, keepdims=True)
    u_hat = sat_xyz/jnp.linalg.norm(position, axis=-1, keepdims=True)
    w_hat = jnp.cross(u_hat, vel_hat)
    v_hat = jnp.cross(w_hat, u_hat)
    R = jnp.stack([u_hat.T, v_hat.T, w_hat.T])

    return R

@jit
def RIC_dev(times, true_orbit_params, estimated_orbit_params):
    """
    Calculate the Radial (R), In-track (I) and Cross-track (C) deviations
    between two circular orbits given their orbit parameters at many time steps.

    Parameters:
    -----------
    times: array (n_time,)
        Times at which to evaluate the RIC deviations.
    true_orb_params: array_like (4,)
        Orbit parameters (elevation, inclination, lon_asc_node, periapsis) of
        the object of most interest. The more accurate measurement when
        comparing the same object.
    est_orb_params: array (elevation, inclination, lon_asc_node, periapsis)
        Estimated orbit parameters.

    Returns:
    --------
    RIC: array_like (n_time, 3)
        Radial, In-track and Cross-track coordinates, in metres, of the orbiting
        body relative to the orbit defined by 'true_orbit_params'.
    """
    true_xyz = orbit(times, *true_orbit_params)
    est_xyz = orbit(times, *estimated_orbit_params)
    R = R_uvw(times, *true_orbit_params)
    true_uvw = vmap(lambda x, y: x@y, in_axes=(-1,0))(R, true_xyz)
    est_uvw = vmap(lambda x, y: x@y, in_axes=(-1,0))(R, est_xyz)

    return est_uvw - true_uvw

@jit
def orbit_fisher(times, orbit_params, RIC_std):
    """
    Calculate the inverse covariance (Fisher) matrix in orbital elements
    induced by errors in the RIC frame of an orbiting object. This is
    essentially linear uncertainty propagation assuming Gaussian errors.

    Parameters:
    -----------
    times: array_like (n_time,)
        Times at which the RIC covariance/standard deviations is defined.
    orbit_params: array_like (4,)
        Orbit parameters (elevation, inclination, lon_asc_node, periapsis) for
        the orbit defining the origin of the RIC frame.
    RIC_std: array_like (3,)
        The standard deviation in the Radial, In-track and Cross-track
        directions.

    Returns:
    --------
    fisher: array_like (4, 4)
        The inverse covariance (Fisher) matrix for the orbit parameters induced
        by the orbit uncertainties in the RIC frame.
    """
    J = jacrev(RIC_dev, argnums=(2,))(times, orbit_params, orbit_params)[0]
    F = jnp.diag(1./RIC_std**2)
    propagate = lambda J, F: J.T@F@J
    fisher = vmap(propagate, in_axes=(0,None))(J, F).sum(axis=0)

    return fisher
