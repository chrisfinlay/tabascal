import yaml
import re
import os
import shutil

import collections.abc

import dask.array as da
import xarray as xr
import numpy as np
import pandas as pd

from tabascal.dask.observation import Observation
from tabascal.utils.sky import generate_random_sky
from tabascal.utils.plot import plot_uv, plot_src_alt, plot_angular_seps
from tabascal.utils.write import write_ms, mk_obs_name, mk_obs_dir


# Define normalized yaml simulation config
def get_base_sim_config():
    tel_keys = ["name", "latitude", "longitude", "elevation", "dish_d", "enu_path", "itrf_path"]
    norm_tel = {key: None for key in tel_keys}

    obs_keys = ["target_name", "ra", "dec", "start_time", "int_time", "n_time", 
                "start_freq", "chan_width", "n_freq", "SEFD", "auto_corrs", "nor_w", 
                "random_seed"]
    norm_obs = {key: None for key in obs_keys}

    src_rand_dict = {"n_src": 0, "min_I": "3sigma", "max_I": 1.0, "I_pow_law": 1.6, 
                    "si_mean": 0.7, "si_std": 0.2, "n_beam": 5, "max_sep": 200.0, 
                    "random_seed": 123456}
    rand_extras = [{}, {"major_mean": 30.0, "major_std": 5.0, "minor_mean": 30.0, "minor_std": 5.0}, {"size_mean": 30.0, "size_std": 5.0}]
    src_rands = [{"random": {**src_rand_dict, **extras}} for extras in rand_extras]

    src_dicts = [{"path": None, **src_rand} for src_rand in src_rands]

    norm_ast = {key: src_dict  for key, src_dict in zip(["point", "gauss", "exp"], src_dicts)}

    norm_rfi = {
        "satellite": {"sat_ids": None, "tle_path": None, "circ_path": None, "power_scale": 1, "spec_model": None},
        "stationary": {"loc_ids": None, "geo_path": None, "power_scale": 1, "spec_model": None}
        }
    
    norm_gains = {
        "G0_mean": 1.0, "G0_std": 0.0, "Gt_std_amp": 0.0, 
        "Gt_std_phase": 0.0, "random_seed": 999}

    norm_out= {"path": "./", "zarr": True, "ms": True, 
               "prefix": None, "suffix": None, "overwrite": False}

    norm_diag = {"uv_cov": True, "src_alt": True, "rfi_seps": True}

    norm_dask = {"max_chunk_MB": 100.0}

    sim_keys = ["telescope", "observation", "ast_sources", "rfi_sources", 
                "gains", "output", "diagnostics", "dask"]
    sim_dicts = [norm_tel, norm_obs, norm_ast, norm_rfi, 
                 norm_gains, norm_out, norm_diag, norm_dask]

    base_config = {key: value for key, value in zip(sim_keys, sim_dicts)}

    return base_config


def deep_update(d: dict, u: dict) -> dict:
    """Recursively update a dictionary which includes subdictionaries.

    Parameters
    ----------
    d : dict
        Base dictionary to update.
    u : dict
        Update dictionary.

    Returns
    -------
    dict
        Updated dictionary.
    """
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = deep_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d

loader = yaml.SafeLoader
loader.add_implicit_resolver(
    u'tag:yaml.org,2002:float',
    re.compile(u'''^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$''', re.X),
    list(u'-+0123456789.'))

def yaml_load(path):
    config = yaml.load(open(path), Loader=loader)
    return config

def load_sim_config(path):
    obs_spec = yaml_load(path)
    base_config = get_base_sim_config()
    
    return deep_update(base_config, obs_spec)


def load_sky_model(file_path: str, freqs: da.Array, src_type: str) -> tuple:
    """Read in a skymodel file and return parameters to be used in tabascal simulator.
    A skymodel file is a csv file with the following columns: 
    (src_id, I, Q, U, V, ra, dec, spec_idx, ref_freq, major, minor, post_ang)
    Depending on the 'src_type' the last 3 columns may be required. src_id, Q, U, and V
    are not used. 

    Parameters
    ----------
    file_path : str
        The file path to the skymodel file. 
    freqs : da.Array
        Array of observation frequencies used to build a sky model
    src_type : str
        The source type to return. Options are {'point', 'gauss', 'exp'}.

    Returns
    -------
    tuple
        A tuple of dask arrays including at least (I, ra, dec) and some of 
        (major, minor, pos_ang).

    Raises
    ------
    KeyError
        The error is raised when one of the accepted source types is not chosen.
    """

    freqs = da.atleast_1d(freqs)
    
    def calc_S_nu(I: da.Array, f0: da.Array, si: da.Array):
        return I[:,None,None] * da.ones((len(I), 1, len(freqs))) * (freqs[None,None,:] / f0[:,None,None])**si[:,None,None]

    src_list = [da.atleast_1d(x) for x in np.loadtxt(file_path, skiprows=1, delimiter=",").T]

    
    if src_type=="point":
        src_id, I, Q, U, V, si, f0, ra, dec = src_list[:9]
        S = calc_S_nu (I, f0, si)
        return S, ra, dec
    elif src_type=="gauss":
        src_id, I, Q, U, V, si, f0, ra, dec, major, minor, pos_ang = src_list
        S = calc_S_nu (I, f0, si)
        return S, ra, dec, major, minor, pos_ang
    elif src_type=="exp":
        src_id, I, Q, U, V, si, f0, ra, dec, major = src_list[:10]
        S = calc_S_nu (I, f0, si)
        return S, ra, dec, major
    else:
        raise KeyError("'src_type' must be one of {'point', 'gauss', 'exp'}.")
    

def sigma_value(value: str|float|int, obs: Observation):
    try:
        if isinstance(value, str):
            sigma = (np.mean(obs.noise_std)/np.sqrt(obs.n_time*obs.n_bl)).compute()
            n_sig = float(value.replace("sigma", ""))
            value = n_sig * sigma
            return value
        elif isinstance(value, float) or isinstance(value, int):
            return float(value)
    except:
        raise ValueError()
    

def load_obs(obs_spec: dict) -> Observation:

    tel_ = obs_spec["telescope"]
    obs_ = obs_spec["observation"]
    dask_ = obs_spec["dask"]

    def arange(start: float, delta: float, n: int):
        x = da.arange(start, start + n * delta, delta)
        return x

    times = arange(obs_["start_time"], obs_["int_time"], obs_["n_time"])
    freqs = arange(obs_["start_freq"], obs_["chan_width"], obs_["n_freq"])

    obs = Observation(
        latitude=tel_["latitude"],
        longitude=tel_["longitude"],
        elevation=tel_["elevation"],
        ra=obs_["ra"],
        dec=obs_["dec"],
        times=times,
        freqs=freqs,
        SEFD=obs_["SEFD"],
        ENU_path=tel_["enu_path"],
        ITRF_path=tel_["itrf_path"], 
        dish_d=tel_["dish_d"],
        random_seed=obs_["random_seed"],
        auto_corrs=obs_["auto_corrs"],
        no_w=obs_["no_w"],
        n_int_samples=obs_["n_int"],
        tel_name=tel_["name"],
        target_name=obs_["target_name"],
        max_chunk_MB=dask_["max_chunk_MB"],
    ) 

    return obs

def add_astro_sources(obs: Observation, obs_spec: dict) -> None:
    """Add astronomical sources from the simulation config file to the observation object.

    Parameters
    ----------
    obs : Observation
        Observation object instance.
    obs_spec : dict
        Simulation config dictionary.
    """

    methods = {"point": obs.addAstro, "gauss": obs.addAstroGauss, "exp": obs.addAstroExp}
    ast_ = obs_spec["ast_sources"]

    for key in ast_.keys():
        
        path = ast_[key]["path"]
        if path is not None:
            params = load_sky_model(path, obs.freqs, key)
            print()
            print(f"Adding {len(params[0])} {key} sources from {path} ...")
            methods[key](*params)

        if ast_[key]["random"]["n_src"] > 0:
            rand_ = ast_[key]["random"]
            beam_width = np.min([obs.syn_bw, rand_["max_sep"]/rand_["n_beam"]])
            
            I, d_ra, d_dec = generate_random_sky(
                    n_src=rand_["n_src"],
                    min_I=sigma_value(rand_["min_I"], obs),
                    max_I=sigma_value(rand_["max_I"], obs),
                    freqs=obs.freqs,
                    fov=obs.fov,
                    beam_width=beam_width,
                    random_seed=rand_["random_seed"],
                    n_beam=rand_["n_beam"],
                )
            
            if key=="point":
                print()
                print(f"Adding {rand_['n_src']} random {key} sources ...")
                methods[key](I[:,None,:], obs.ra + d_ra, obs.dec + d_dec)
            elif key=="gauss":
                rng = np.random.default_rng(rand_["random_seed"]+1)
                pos_ang = rng.uniform(low=0.0, high=360.0, size=(rand_["n_src"],))
                major_ = np.abs(rng.normal(loc=rand_["major_mean"], scale=rand_["major_std"], size=(rand_["n_src"],)))
                minor_ = np.abs(rng.normal(loc=rand_["minor_mean"], scale=rand_["minor_std"], size=(rand_["n_src"],)))
                major = np.where(major_>minor_, major_, minor_)
                minor = np.where(major_<minor_, major_, minor_)
                print()
                print(f"Adding {rand_['n_src']} random {key} sources ...")
                methods[key](I[:,None,:], major, minor, pos_ang, obs.ra + d_ra, obs.dec + d_dec)
            elif key=="exp":
                rng = np.random.default_rng(rand_["random_seed"]+1)
                shape = np.abs(rng.normal(loc=rand_["size_mean"], scale=rand_["size_std"], size=(rand_["n_src"],)))
                print()
                print(f"Adding {rand_['n_src']} random {key} sources ...")
                methods[key](I[:,None,:], shape, obs.ra + d_ra, obs.dec + d_dec)


def gauss(A: float, mean: float, sigma: float, x: da.Array) -> da.Array:
    return A * da.exp( -0.5 * ((x-mean)/sigma)**2 )


def generate_spectra(spec_df: pd.DataFrame, freqs: da.Array, id_key: str) -> tuple:

    spectra = []
    ids = []
    for i, spec in spec_df.iterrows():
        if spec["sig_type"].lower()=="gauss":
            ids.append(spec[id_key])
            spectra.append(gauss(spec["power"], spec["freq"], spec["band_width"]/2, freqs))
        else:
            print(f"sig_type : {spec['sig_type']} not supported.")

    return np.array(ids), da.atleast_2d(spectra)


def add_satellite_sources(obs: Observation, obs_spec: dict) -> None:

    sat_ = obs_spec["rfi_sources"]["satellite"]
    if sat_["sat_ids"] is not None:
        ids = sat_["sat_ids"]
        path_ids = []
        if sat_["circ_path"] is None and sat_["tle_path"] is None:
            raise ValueError("'circ_path' or 'tle_path' must be populated to include satellites.")
        # Not yet implemented
        # if sat_["tle_path"] is not None:
        #     tles = read_tle(sat_["tle_path"])
        #     tles = tles[tles["sat_id"].isin(ids)]
        #     path_ids.append(tles["sat_id"].values)
        if sat_["circ_path"] is not None:
            oles = pd.read_csv(sat_["circ_path"])
            oles = oles[oles["sat_id"].isin(ids)]
            path_ids.append(oles["sat_id"].values)

        path_ids = np.concatenate(path_ids)

        sat_spec = pd.read_csv(sat_["spec_model"])
        sat_spec = sat_spec[sat_spec["sat_id"].isin(path_ids)]

        if len(sat_spec) > 0:
            ids, spectra = generate_spectra(sat_spec, obs.freqs, "sat_id")
            uids = np.unique(ids)
            for uid in uids:
                Pv = da.sum(spectra[ids==uid], axis=0)[None,None,:] * da.ones((1, 1, obs.n_freq))
                ole = oles[oles["sat_id"]==uid]
                if len(ole)==1:
                    print()
                    print("Adding satellite RFI source ...")
                    obs.addSatelliteRFI(Pv, ole["elevation"].values, ole["inclination"].values, ole["lon_asc_node"].values, ole["periapsis"].values)
                # tle = tles[tles["sat_id"]==uid]
                # elif len(tle)==1:
                #     obs.addSatelliteRFI_TLE(Pv, tle)
                else:
                    print()
                    print(f"sat_id: {uid} multiply-defined.")


def add_stationary_sources(obs: Observation, obs_spec: dict) -> None:

    stat_ = obs_spec["rfi_sources"]["stationary"]
    if stat_["loc_ids"] is not None:
        ids = stat_["loc_ids"]
        path_ids = []
        if stat_["geo_path"] is not None:
            geos = pd.read_csv(stat_["geo_path"])
            geos = geos[geos["loc_id"].isin(ids)]
            path_ids.append(geos["loc_id"].values)

        path_ids = np.concatenate(path_ids)

        stat_spec = pd.read_csv(stat_["spec_model"])
        stat_spec = stat_spec[stat_spec["loc_id"].isin(ids)]

        if len(stat_spec) > 0:
            ids, spectra = generate_spectra(stat_spec, obs.freqs, "loc_id")
            uids = np.unique(ids)
            for uid in uids:
                Pv = da.sum(spectra[ids==uid], axis=0)[None,None,:] * da.ones((1, 1, obs.n_freq))
                geo = geos[geos["loc_id"]==uid]
                if len(geo)==1:
                    print()
                    print("Adding stationary RFI source ...")
                    obs.addStationaryRFI(Pv, geo["latitude"].values, geo["longitude"].values, geo["elevation"].values)
                else:
                    print()
                    print(f"loc_id: {uid} multiply-defined.")


        
                    
def add_gains(obs: Observation, obs_spec: dict) -> None:

    gains_ = obs_spec["gains"]
    gain_offset = gains_["G0_mean"]!=1 or gains_["G0_std"]!=0
    gain_var = gains_["Gt_std_amp"]!=0 or gains_["Gt_std_phase"]!=0
    if gain_offset or gain_var:
        print()
        print("Adding gains ...")
        obs.addGains(
            gains_["G0_mean"], 
            gains_["G0_std"], 
            gains_["Gt_std_amp"], 
            np.deg2rad(gains_["Gt_std_phase"]), 
            gains_["random_seed"]
            )
    else:
        print()
        print("No gains added ...")


def plot_diagnostics(obs: Observation, obs_spec: dict, save_path: str):

    diag_ = obs_spec["diagnostics"]

    if diag_["src_alt"]:
        print()
        print("Plotting source altitude ...")
        plot_src_alt(obs, save_path)
    if diag_["uv_cov"]:
        print()
        print("Plotting uv coverage ...")
        plot_uv(obs, save_path)
    if diag_["rfi_seps"]:
        print()
        print("Plotting RFI angular separations ...")
        plot_angular_seps(obs, save_path)

def save_data(obs: Observation, obs_spec: dict, zarr_path: str, ms_path: str) -> None:

    if obs_spec["output"]["zarr"] or obs_spec["output"]["ms"]:
        print()
        print("Calculating visibilities ...")
        obs.calculate_vis()
        print()
        print(f"Flag Rate      : {100*obs.flags.mean().compute(): .1f} %")

    overwrite = obs_spec["output"]["overwrite"]
    save_path = os.path.split(zarr_path)[0]

    if obs_spec["output"]["zarr"]:
        print()
        print(f"Writing visibilities to zarr ...")
        obs.write_to_zarr(zarr_path, overwrite)
    if obs_spec["output"]["zarr"] and obs_spec["output"]["ms"]:
        xds = xr.open_zarr(zarr_path)
        print()
        print(f"Writing visibilities to MS ...")
        write_ms(xds, ms_path, overwrite)
    elif obs_spec["output"]["ms"]:
        print()
        print(f"Writing visibilities to MS ...")
        write_ms(obs.dataset, ms_path, overwrite)


def save_inputs(obs_spec: dict, save_path: str) -> None:

    for key in ["enu_path", "itrf_path"]:
        path = obs_spec["telescope"][key]
        if path is not None:
            shutil.copy(path, save_path)

    for key in obs_spec["ast_sources"].keys():
        path = obs_spec["ast_sources"][key]["path"]
        if path is not None:
            shutil.copy(path, save_path)

    key = 3*["satellite",] + 2*["stationary",]
    subkey = ["tle_path", "circ_path", "spec_model", "geo_path", "spec_model"]
    for key1, key2 in zip(key, subkey):
        path = obs_spec["rfi_sources"][key1][key2]
        if path is not None:
            shutil.copy(path, save_path)

    with open(os.path.join(save_path, "sim_config.yaml"), "w") as fp:
        yaml.dump(obs_spec, fp)


def run_sim_config(obs_spec: dict=None, path: str=None) -> Observation:

    if path is not None:
        obs_spec = load_sim_config(path)
    elif obs_spec is None:
        print("obs_spec or path must be defined.")
        return None

    obs = load_obs(obs_spec)
    add_astro_sources(obs, obs_spec)
    add_satellite_sources(obs, obs_spec)
    add_stationary_sources(obs, obs_spec)
    add_gains(obs, obs_spec)

    print(obs)

    obs_name = mk_obs_name(obs_spec["output"]["prefix"], obs, obs_spec["output"]["suffix"])
    save_path, zarr_path, ms_path = mk_obs_dir(obs_spec["output"]["path"], obs_name, obs_spec["output"]["overwrite"])

    input_path = os.path.join(save_path, "input_data")

    os.makedirs(input_path, exist_ok=True)
    save_inputs(obs_spec, input_path)

    print()
    print(f"Writing data to : {save_path}")

    plot_diagnostics(obs, obs_spec, save_path)
    save_data(obs, obs_spec, zarr_path, ms_path)

    return obs