import yaml
import re
import os
import sys
import shutil

import collections.abc
from datetime import datetime

import dask.array as da
import xarray as xr
import numpy as np
import pandas as pd

from tabascal.dask.observation import Observation
from tabascal.utils.sky import generate_random_sky
from tabascal.utils.plot import plot_uv, plot_src_alt, plot_angular_seps
from tabascal.utils.write import write_ms, mk_obs_name, mk_obs_dir
from tabascal.jax.coordinates import calculate_fringe_frequency
from tabascal.utils.tle import get_visible_satellite_tles

from daskms import xds_from_ms

from tqdm import tqdm

JD0 = 2459997.079914223 # GMSA = 0 2023-02-21 13:55:04.589 UTC

# Define normalized yaml simulation config
def get_base_sim_config():
    tel_keys = ["name", "latitude", "longitude", "elevation", "dish_d", "enu_path", "itrf_path", "n_ant"]
    norm_tel = {key: None for key in tel_keys}

    obs_keys = ["target_name", "ra", "dec", "start_time", "start_time_jd", "start_time_isot", "int_time", "n_time", 
                "start_freq", "chan_width", "n_freq", "SEFD", "auto_corrs", "no_w", 
                "random_seed"]
    norm_obs = {key: None for key in obs_keys}

    src_rand_dict = {"n_src": 0, "min_I": "3sigma", "max_I": 1.0, "I_pow_law": 1.6, 
                    "si_mean": 0.7, "si_std": 0.2, "n_beam": 5, "max_sep": 50.0, 
                    "random_seed": 123456}
    rand_extras = [{}, {"major_mean": 30.0, "major_std": 5.0, "minor_mean": 30.0, "minor_std": 5.0}, {"size_mean": 30.0, "size_std": 5.0}]
    src_rands = [{"random": {**src_rand_dict, **extras}} for extras in rand_extras]

    src_dicts = [{"path": None, **src_rand} for src_rand in src_rands]

    norm_ast = {key: src_dict for key, src_dict in zip(["point", "gauss", "exp"], src_dicts)}

    norm_ast.update({"pow_spec": {"path": None, "random": {"type": None, "random_seed": 1234}}})

    norm_rfi = {
        "satellite": {"tle_dir": "./tles", "norad_ids": [], "norad_ids_path": None, "sat_names": [], "norad_spec_model": None,
                      "sat_ids": None, "circ_path": None, 
                      "power_scale": 1, "spec_model": None},
        "stationary": {"loc_ids": None, "geo_path": None, "power_scale": 1, "spec_model": None}
        }
    
    norm_gains = {
        "G0_mean": 1.0, "G0_std": 0.0, "Gt_std_amp": 0.0, 
        "Gt_std_phase": 0.0, "random_seed": 999}

    norm_out= {"path": "./", "zarr": True, "ms": True, 
               "prefix": None, "suffix": None, "overwrite": False}

    norm_diag = {"uv_cov": True, "src_alt": True, "rfi_seps": True}

    norm_dask = {"max_chunk_MB": 100.0}

    norm_st = {"username": None, "password": None}

    sim_keys = ["telescope", "observation", "ast_sources", "rfi_sources", 
                "gains", "output", "diagnostics", "dask", "spacetrack"]
    sim_dicts = [norm_tel, norm_obs, norm_ast, norm_rfi, 
                 norm_gains, norm_out, norm_diag, norm_dask, norm_st]

    base_config = {key: value for key, value in zip(sim_keys, sim_dicts)}

    return base_config

def get_base_extract_config():

    extract_config = {
        "data": {
            "sim_dir": None,
        },
        "ideal": {
            "data_col": "AST_DATA",
            "flag": {
                "type": "perfect",
                "thresh": 0,
            },
        },
        "tab": {
            "data_col": "TAB_DATA",
            "flag": {
                "type": "perfect",
                "thresh": 0,
            },
        },
        "flag1": {
            "data_col": "CAL_DATA",
            "flag": {
                "type": "perfect",
                "thresh": 3.0,
            },
        },
        "flag2": {
            "data_col": "CAL_DATA",
            "flag": {
                "type": "aoflagger",
                "sif_path": None,
                "strategies": None,
            },
        },
        "image": {
            "sif_path": None,
            "params": {
                "size": "256 256",
                "scale": "20amin",
                "niter": 100000,
                "mgain": 0.1,
                "auto-threshold": 0.3,
                "auto_mask": 2.0,
                "pol": "xx",
                "weight": "natural",
            },
        },
        "extract": {
            "sigma_cut": 3.0,
            "beam_cut": 1.0,
            "thresh_isl": 1.5,
            "thresh_pix": 1.5,
        },
    }

    return extract_config


def get_base_tab_config():

    tab_config = {
        "data": {
            "sim_dir": None,
            "sampling": 1,
        },
        "plots": {
            "init": True,
            "truth": True,
            "prior": True,
            "prior_samples": 100,
        },
        "inference": {
            "mcmc": False,
            "opt": True,
            "fisher": False
        },
        "opt": {
            "epsilon": 1e-1,
            "max_iter": 100,
            "dual_run": True,
            "guide": "map",
        },
        "fisher": {
            "n_samples": 1,
            "max_cg_iter": 10000,
        },
        "ast":{
            "init": "prior",    # Options are truth, prior, est, truth_mean
            "mean": 0,          # Options include truth, est, 0, truth_mean
            "pow_spec": {
                "P0": 1e3,
                "k0": 1e-3,
                "gamma": 1.0,
            },
        },
        "rfi": {
            "init": "prior",    # Options are truth, prior, est
            "mean": 0,          # Options are truth, est, 0
            "var": None,        # Jy
            "corr_time": 15,    # seconds
        },
        "gains": {
            "amp_mean": "truth",
            "phase_mean": "truth",
            "amp_std": 1.0,      # %
            "phase_std": 1.0,    # degrees
            "corr_time": 180,    # minutes
        },
    }

    return tab_config

def get_base_pow_spec_config():
    
    ps_config = {
        "ideal": {
            "data_col": "AST_DATA",
            "suffix": "",
            "flag": {
                "type": "perfect",
                "thresh": 0.0,
            },
        },
        "tab": {
            "data_col": "TAB_DATA",
            "suffix": "",
            "flag": {
                "type": "perfect",
                "thresh": 0.0,
            },
        },
        "flag1": {
            "data_col": "CAL_DATA",
            "suffix": "perfect",
            "flag": {
                "type": "perfect",
                "thresh": 3.0,
            },
        },
        "flag2": {
            "data_col": "CAL_DATA",
            "suffix": "aoflagger",
            "flag": {
                "type": "aoflagger",
                "sif_path": None,
                "strategies": None,
            },
        },
        "tge": {
            "n_grid": 256,
            "n_bins": 20,
        },
    }

    return ps_config

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


class Tee(object):
    """https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout
    """
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
    def flush(self):
        pass


def load_config(path: str, config_type: str="sim") -> dict:
    """Load a configuration file and populate default parameters where needed.

    Parameters
    ----------
    path : str
        Path to the yaml config file.
    config_type : str, optional
        Type of configuration file, by default "sim". Options are {"sim", "tab", "extract", "pow_spec"}.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    config = yaml_load(path)
    if config_type=="sim":
        base_config = get_base_sim_config()
    elif config_type=="tab":
        base_config = get_base_tab_config()
    elif config_type=="extract":
        base_config = get_base_extract_config()
    elif config_type=="pow_spec":
        base_config = get_base_pow_spec_config()
    else:
        ValueError("A config type must be specified. Options are {sim, tab, extract, pow_spec}.")
    
    return deep_update(base_config, config)


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

    if obs_["start_time"]:
        start_time = obs_["start_time"]
        start_time_jd = JD0 + start_time/(24*3600)
    elif obs_["start_time_jd"]:
        start_time_jd = obs_["start_time_jd"]
    elif obs_["start_time_isot"]:
        from astropy.time import Time
        start_time_jd = Time(obs_["start_time_isot"], format="isot", scale="ut1").jd
    else:
        ValueError("A start time must be given in either the observation: start_time: or observation: start_time_jd:")

    time_range = arange(0, obs_["int_time"], obs_["n_time"])
    times_jd = start_time_jd + time_range/(24*3600)
    freqs = arange(obs_["start_freq"], obs_["chan_width"], obs_["n_freq"])

    obs = Observation(
        latitude=tel_["latitude"],
        longitude=tel_["longitude"],
        elevation=tel_["elevation"],
        ra=obs_["ra"],
        dec=obs_["dec"],
        times_jd=times_jd,
        freqs=freqs,
        chan_width=obs_["chan_width"],
        SEFD=obs_["SEFD"],
        ENU_path=tel_["enu_path"],
        ITRF_path=tel_["itrf_path"], 
        n_ant=tel_["n_ant"],
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
        
        if key=="pow_spec" and ast_["pow_spec"]["random"]["type"] is not None:

            from tge import simulate_sky, Cl, Pk, beam_constants, lm_to_radec

            rand_ = ast_["pow_spec"]["random"]
            const = beam_constants(D=obs.dish_d.compute(), freq=obs.freqs[0].compute(), dBdT=None, f=1)
            beam = lambda x: 1
            non_ps_keys = ["n_side", "fov_f", "type", "random_seed"]
            ps_args = {key: value for key, value in rand_.items() if key not in non_ps_keys}
            if rand_["type"] == "Cl":
                I, lxy = simulate_sky(rand_["n_side"], rand_["fov_f"]*const["thetaFWHM"], Cl=Cl, PS_args=ps_args, beam=beam, seed=rand_["random_seed"])
            elif rand_["type"] == "Pk":
                I, lxy = simulate_sky(N_side=rand_["n_side"], fov=rand_["fov_f"]*const["thetaFWHM"], Pk=Pk, beam=beam, seed=rand_["random_seed"])
            else:
                ValueError("Keyword 'type' in section pow_spec.random must be one of {Cl, Pk}")
            ra, dec = lm_to_radec(lxy, obs.ra, obs.dec).T
            I = const["dBdT"] * da.asarray(I.reshape(-1, 1))
            obs.addAstro(I[:,None,:], ra, dec)
    

        if "n_src" in ast_[key]["random"]:
            if ast_[key]["random"]["n_src"] > 0:
                rand_ = ast_[key]["random"]
                n_beam = rand_["n_beam"]
                max_beam = rand_["max_sep"]/3600/n_beam
                beam_width = np.min([obs.syn_bw, max_beam])
                print()
                print(f"Generating {rand_['n_src']} sources within {obs.fov:.2f} deg FoV ({obs.fov/2:.2f} radius) ...") 
                print(f"Minimum {n_beam*beam_width*3600:.1f} arcsec separation ...")
                
                I, d_ra, d_dec = generate_random_sky(
                        n_src=rand_["n_src"],
                        min_I=sigma_value(rand_["min_I"], obs),
                        max_I=sigma_value(rand_["max_I"], obs),
                        freqs=obs.freqs,
                        fov=obs.fov,
                        beam_width=beam_width,
                        random_seed=rand_["random_seed"],
                        n_beam=n_beam,
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

    # Circular path based Satellites
    if sat_["sat_ids"] is not None:
        if sat_["circ_path"] is not None:
            oles = pd.read_csv(sat_["circ_path"])
            oles = oles[oles["sat_id"].isin(sat_["sat_ids"])]
            path_ids = oles["sat_id"].values
        else:
            raise ValueError("'circ_path' must be populated to include satellites with circular orbits.")

        sat_spec = pd.read_csv(sat_["spec_model"])
        sat_spec = sat_spec[sat_spec["sat_id"].isin(path_ids)]

        if len(sat_spec) > 0:
            spec_ids, spectra = generate_spectra(sat_spec, obs.freqs, "sat_id")
            uids = np.unique(spec_ids)
            for uid in uids:
                Pv = sat_["power_scale"] * da.sum(spectra[spec_ids==uid], axis=0)[None,None,:] * da.ones((1, 1, obs.n_freq))
                ole = oles[oles["sat_id"]==uid]
                if len(ole)==1:
                    print()
                    print("Adding satellite RFI source ...")
                    obs.addSatelliteRFI(Pv, ole["elevation"].values, ole["inclination"].values, ole["lon_asc_node"].values, ole["periapsis"].values)
                else:
                    print()
                    print(f"sat_id: {uid} multiply-defined.")

def add_tle_satellite_sources(obs: Observation, obs_spec: dict) -> None:

    sat_ = obs_spec["rfi_sources"]["satellite"]
    st_ = obs_spec["spacetrack"]

    # TLE path based Satellites
    tle_cond = [sat_["norad_ids_path"], len(sat_["norad_ids"])>0, len(sat_["sat_names"])>0]
    if np.any(tle_cond):
        
        if sat_["norad_ids_path"] is None:
            norad_ids = sat_["norad_ids"]
        else:
            norad_ids = np.concatenate([sat_["norad_ids"], np.loadtxt(sat_["norad_ids_path"], usecols=0)])
        
        from astropy.time import Time
        times_check = Time(np.arange(obs.times_jd[0], obs.times_jd[-1], sat_["vis_step"]/(24*60)), format="jd")

        norad_ids, tles = get_visible_satellite_tles(
            st_["username"], st_["password"], times_check, 
            obs.latitude, obs.longitude, obs.elevation,
            obs.ra, obs.dec, 
            sat_["max_angular_separation"], sat_["min_elevation"], 
            sat_["sat_names"], norad_ids, sat_["tle_dir"]
            )
        
        print(norad_ids)

        sat_spec = pd.read_csv(sat_["norad_spec_model"])
        sat_spec = sat_spec[sat_spec["norad_id"].isin(norad_ids)]

        if len(sat_spec) > 0:
            print()
            print("Adding TLE-based satellite RFI sources ...")
            ids, spectra = generate_spectra(sat_spec, obs.freqs, "norad_id")
            uids = np.unique(ids)
            for uid in tqdm(uids):
                Pv = sat_["power_scale"] * da.sum(spectra[ids==uid], axis=0)[None,None,:] * da.ones((1, 1, obs.n_freq))
                tle = tles[norad_ids==uid]
                if len(tle)==1:
                    obs.addTLESatelliteRFI(Pv, [uid], tle)
                else:
                    print()
                    print(f"norad_id: {uid} multiply-defined.")  


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
                Pv = stat_["power_scale"] * da.sum(spectra[ids==uid], axis=0)[None,None,:] * da.ones((1, 1, obs.n_freq))
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


def plot_diagnostics(obs: Observation, obs_spec: dict, save_path: str) -> None:

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

def print_signal_specs(vis_rfi: da.Array, vis_ast: da.Array, noise_data: da.Array, flags: da.Array) -> None:

    rfi_amp = da.mean(da.abs(vis_rfi)).compute()
    ast_amp = da.mean(da.abs(vis_ast)).compute()
    noise = da.std(noise_data).compute()
    flag_rate = 100 * da.mean(flags).compute()

    print()
    print(f"Mean RFI Amp.  : {rfi_amp:.2f} Jy")
    print(f"Mean AST Amp.  : {ast_amp:.2f} Jy")
    print(f"Vis Noise Amp. : {noise:.2f} Jy")
    print(f"Flag Rate      : {flag_rate:.1f} %")

def write_to_ms(xds: xr.Dataset, ms_path: str, overwrite: bool) -> None:
    start = datetime.now()
    print()
    print(f"Writing visibilities to MS ...")
    write_ms(xds, ms_path, overwrite)
    end = datetime.now()
    print(f"MS Write Time : {end - start}")

def write_to_zarr(obs: Observation, zarr_path: str, overwrite: bool) -> None:
    start = datetime.now()
    print()
    print(f"Writing visibilities to zarr ...")
    obs.write_to_zarr(zarr_path, overwrite)
    end = datetime.now()
    print(f"zarr Write Time : {end - start}")

def save_data(obs: Observation, obs_spec: dict, zarr_path: str, ms_path: str) -> None:

    if obs_spec["output"]["zarr"] or obs_spec["output"]["ms"]:
        print()
        print("Calculating visibilities ...")
        obs.calculate_vis()

    overwrite = obs_spec["output"]["overwrite"]

    if obs_spec["output"]["zarr"] and obs_spec["output"]["ms"]:
        write_to_zarr(obs, zarr_path, overwrite)
        xds = xr.open_zarr(zarr_path)
        write_to_ms(xds, ms_path, overwrite)
        print_signal_specs(xds.vis_rfi.data, xds.vis_ast.data, xds.noise_data.data, xds.flags.data)
    elif obs_spec["output"]["zarr"]:
        write_to_zarr(obs, zarr_path, overwrite)
        xds = xr.open_zarr(zarr_path)
        print_signal_specs(xds.vis_rfi.data, xds.vis_ast.data, xds.noise_data.data, xds.flags.data)
    elif obs_spec["output"]["ms"]:
        write_to_ms(obs.dataset, ms_path, overwrite)
        xds = xds_from_ms(ms_path)
        print_signal_specs(xds.RFI_MODEL_DATA.data, xds.AST_MODEL_DATA.data, xds.NOISE_DATA.data, xds.FLAG.data)
    else:
        ValueError("No output format has been chosen. output: zarr: or output: ms: must be True.")


def save_inputs(obs_spec: dict, save_path: str) -> None:

    for key in ["enu_path", "itrf_path"]:
        path = obs_spec["telescope"][key]
        if path is not None:
            shutil.copy(path, save_path)

    for key in obs_spec["ast_sources"].keys():
        path = obs_spec["ast_sources"][key]["path"]
        if path is not None:
            shutil.copy(path, save_path)

    key = 4*["satellite",] + 2*["stationary",]
    subkey = ["norad_ids_path", "norad_spec_model", "circ_path", "spec_model", "geo_path", "spec_model"]
    for key1, key2 in zip(key, subkey):
        path = obs_spec["rfi_sources"][key1][key2]
        if path is not None:
            shutil.copy(path, save_path)

    with open(os.path.join(save_path, "sim_config.yaml"), "w") as fp:
        yaml.dump(obs_spec, fp)


def print_fringe_freq_sat(obs: Observation):
    fringe_params = [{
        "times": obs.times_fine[::obs.n_int_samples].compute(),
        "freq": obs.freqs.max().compute(),
        "rfi_xyz": da.concatenate(obs.rfi_satellite_xyz, axis=0)[i,::obs.n_int_samples].compute(),
        "ants_itrf": obs.ITRF.compute(),
        "ants_u": obs.ants_uvw[::obs.n_int_samples,:,0].compute(),
        "dec": obs.dec.compute(),
    } for i in range(obs.n_rfi_satellite)]
    fringe_freq = [calculate_fringe_frequency(**f_params) for f_params in fringe_params]
    f_sample = np.pi * np.max(np.abs(fringe_freq)) * np.max(obs.rfi_satellite_A_app) / np.sqrt(6 * obs.noise_std.mean()).compute()
    n_int = np.ceil(obs.int_time.compute() * f_sample)

    print()
    print(f"Maximum Fringe Frequency is : {np.max(np.abs(fringe_freq)):.2f} Hz")
    print(f"Maximum sampling rate is    : {f_sample:.2f} Hz")
    print(f"Recommended n_int is >=     : {n_int:.0f} ({obs.n_int_samples} used)")

def print_fringe_freq_tle_sat(obs: Observation):
    fringe_params = [{
        "times": obs.times_fine[::obs.n_int_samples].compute(),
        "freq": obs.freqs.max().compute(),
        "rfi_xyz": da.concatenate(obs.rfi_tle_satellite_xyz, axis=0)[i,::obs.n_int_samples].compute(),
        "ants_itrf": obs.ITRF.compute(),
        "ants_u": obs.ants_uvw[::obs.n_int_samples,:,0].compute(),
        "dec": obs.dec.compute(),
    } for i in range(obs.n_rfi_tle_satellite)]
    fringe_freq = [calculate_fringe_frequency(**f_params) for f_params in fringe_params]
    f_sample = np.pi * np.max(np.abs(fringe_freq)) * np.max(obs.rfi_tle_satellite_A_app) / np.sqrt(6 * obs.noise_std.mean()).compute()
    n_int = np.ceil(obs.int_time.compute() * f_sample)

    print()
    print(f"Maximum Fringe Frequency is : {np.max(np.abs(fringe_freq)):.2f} Hz")
    print(f"Maximum sampling rate is    : {f_sample:.2f} Hz")
    print(f"Recommended n_int is >=     : {n_int:.0f} ({obs.n_int_samples} used)")

def run_sim_config(obs_spec: dict=None, path: str=None) -> Observation:

    log = open('log_sim.txt', 'w')
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    start = datetime.now()
    print(datetime.now())

    if path is not None:
        obs_spec = load_config(path, config_type="sim")
    elif obs_spec is None:
        print("obs_spec or path must be defined.")
        return None
    
    obs = load_obs(obs_spec)
    add_astro_sources(obs, obs_spec)
    add_satellite_sources(obs, obs_spec)
    add_tle_satellite_sources(obs, obs_spec)
    add_stationary_sources(obs, obs_spec)
    add_gains(obs, obs_spec)

    # Change this for calculating all RFI sources
    if obs.n_rfi_satellite>0:
        print_fringe_freq_sat(obs)
    if obs.n_rfi_tle_satellite>0:
        print_fringe_freq_tle_sat(obs)

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

    end = datetime.now()
    print()
    print(f"Total simulation time : {end - start}")
    print()
    print(datetime.now())

    log.close()
    shutil.copy("log_sim.txt", save_path)
    os.remove("log_sim.txt")
    sys.stdout = backup

    return obs, save_path