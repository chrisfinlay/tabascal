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

from pathlib import Path

pkg_dir = Path(__file__).parent.absolute()

sim_base_config_path = os.path.join(pkg_dir, "../data/sim_config_base.yaml")
tab_base_config_path = os.path.join(pkg_dir, "../data/tab_config_base.yaml")
extract_base_config_path = os.path.join(pkg_dir, "../data/extract_config_base.yaml")
pow_spec_base_config_path = os.path.join(
    pkg_dir, "../data/extract_pow_spec_config_base.yaml"
)

JD0 = 2459997.079914223  # 2023-02-21 13:55:04.589 UTC => GMSA = 0
MJD0 = JD0 - 2400000.5


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
    "tag:yaml.org,2002:float",
    re.compile(
        """^(?:
     [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
    |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
    |\\.[0-9_]+(?:[eE][-+][0-9]+)?
    |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
    |[-+]?\\.(?:inf|Inf|INF)
    |\\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(path):
    config = yaml.load(open(path), Loader=loader)
    return config


class Tee(object):
    """https://stackoverflow.com/questions/17866724/python-logging-print-statements-while-having-them-print-to-stdout"""

    def __init__(self, *files):
        self.files = files

    def write(self, obj):
        for f in self.files:
            f.write(obj)

    def flush(self):
        pass


def load_config(path: str, config_type: str = "sim") -> dict:
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
    if config_type == "sim":
        base_config = yaml_load(sim_base_config_path)
    elif config_type == "tab":
        base_config = yaml_load(tab_base_config_path)
    elif config_type == "extract":
        base_config = yaml_load(extract_base_config_path)
    elif config_type == "pow_spec":
        base_config = yaml_load(pow_spec_base_config_path)
    else:
        ValueError(
            "A config type must be specified. Options are {sim, tab, extract, pow_spec}."
        )

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
        return (
            I[:, None, None]
            * da.ones((len(I), 1, len(freqs)))
            * (freqs[None, None, :] / f0[:, None, None]) ** si[:, None, None]
        )

    src_list = [
        da.atleast_1d(x) for x in np.loadtxt(file_path, skiprows=1, delimiter=",").T
    ]

    if src_type == "point":
        src_id, I, Q, U, V, si, f0, ra, dec = src_list[:9]
        S = calc_S_nu(I, f0, si)
        return S, ra, dec
    elif src_type == "gauss":
        src_id, I, Q, U, V, si, f0, ra, dec, major, minor, pos_ang = src_list
        S = calc_S_nu(I, f0, si)
        return S, ra, dec, major, minor, pos_ang
    elif src_type == "exp":
        src_id, I, Q, U, V, si, f0, ra, dec, major = src_list[:10]
        S = calc_S_nu(I, f0, si)
        return S, ra, dec, major
    else:
        raise KeyError("'src_type' must be one of {'point', 'gauss', 'exp'}.")


def sigma_value(value: str | float | int, obs: Observation):
    try:
        if isinstance(value, str):
            sigma = (np.mean(obs.noise_std) / np.sqrt(obs.n_time * obs.n_bl)).compute()
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

    if obs_["start_time_jd"]:
        from tabascal.jax.coordinates import jd_to_mjd

        start_time_mjd = jd_to_mjd(obs_["start_time_jd"])
    elif obs_["start_time_isot"]:
        from astropy.time import Time
        from tabascal.jax.coordinates import jd_to_mjd

        start_time_mjd = jd_to_mjd(
            Time(obs_["start_time_isot"], format="isot", scale="ut1").jd
        )
    elif obs_["start_time_lha"] is not None:
        gsa = obs_["start_time_lha"] - tel_["longitude"] + obs_["ra"]
        start_time_mjd = MJD0 + (gsa / 360)
    elif obs_["start_time"] is not None:
        start_time = obs_["start_time"]
        start_time_mjd = MJD0 + start_time / (24 * 3600)
    else:
        ValueError(
            "A start time must be given in either the 'start_time', 'start_time_jd', 'start_time_isot' or 'start_time_lha'"
        )

    time_range = arange(0, obs_["int_time"], obs_["n_time"])
    times_mjd = start_time_mjd + time_range / (24 * 3600)
    freqs = arange(obs_["start_freq"], obs_["chan_width"], obs_["n_freq"])

    obs = Observation(
        latitude=tel_["latitude"],
        longitude=tel_["longitude"],
        elevation=tel_["elevation"],
        ra=obs_["ra"],
        dec=obs_["dec"],
        times_mjd=times_mjd,
        freqs=freqs,
        int_time=obs_["int_time"],
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

    print()
    print(f"Theoretical synthesized beam width : {3600*obs.syn_bw:.0f} arcsec")

    return obs


def add_power_spectrum_sources(obs: Observation, ps_rand: dict) -> None:

    from tge import simulate_sky, Cl, Pk, beam_constants, lm_to_radec

    const = beam_constants(
        D=obs.dish_d.compute(), freq=obs.freqs[0].compute(), dBdT=None, f=1
    )
    beam = lambda x: 1
    cl_ps_keys = ["A", "beta"]
    cl_ps_args = {key: value for key, value in ps_rand.items() if key in cl_ps_keys}
    pk_ps_keys = ["Po", "k0", "gamma"]
    pk_ps_args = {key: value for key, value in ps_rand.items() if key in pk_ps_keys}
    non_ps_keys = ["n_side", "fov_f", "type", "random_seed"]
    # ps_args = {key: value for key, value in ps_rand.items() if key not in non_ps_keys}

    if ps_rand["type"] == "Cl":
        I, lxy = simulate_sky(
            N_side=ps_rand["n_side"],
            fov=ps_rand["fov_f"] * const["thetaFWHM"],
            Cl=Cl,
            PS_args=cl_ps_args,
            beam=beam,
            seed=ps_rand["random_seed"],
        )
    elif ps_rand["type"] == "Pk":
        I, lxy = simulate_sky(
            N_side=ps_rand["n_side"],
            fov=ps_rand["fov_f"] * const["thetaFWHM"],
            Pk=Pk,
            PS_args=pk_ps_args,
            beam=beam,
            seed=ps_rand["random_seed"],
        )
    else:
        ValueError("Keyword 'type' in section pow_spec.random must be one of {Cl, Pk}")

    ra, dec = lm_to_radec(lxy, obs.ra, obs.dec).T
    I = const["dBdT"] * da.asarray(I.reshape(-1, 1))
    obs.addAstro(I[:, None, :], ra, dec)


def add_astro_sources(obs: Observation, obs_spec: dict) -> None:
    """Add astronomical sources from the simulation config file to the observation object.

    Parameters
    ----------
    obs : Observation
        Observation object instance.
    obs_spec : dict
        Simulation config dictionary.
    """

    methods = {
        "point": obs.addAstro,
        "gauss": obs.addAstroGauss,
        "exp": obs.addAstroExp,
    }
    ast_ = obs_spec["ast_sources"]

    if ast_["pow_spec"]["random"]["type"]:
        add_power_spectrum_sources(obs, ast_["pow_spec"]["random"])

    for key in ast_.keys():

        path = ast_[key]["path"]
        if path is not None:
            params = load_sky_model(path, obs.freqs, key)
            print()
            print(f"Adding {len(params[0])} {key} sources from {path} ...")
            methods[key](*params)

        if "n_src" in ast_[key]["random"]:
            if ast_[key]["random"]["n_src"] > 0:
                rand_ = ast_[key]["random"]
                n_beam = rand_["n_beam"]
                max_beam = rand_["max_sep"] / 3600 / n_beam
                beam_width = np.min([obs.syn_bw, max_beam])
                fov = np.min([obs.fov, 180.0])
                print()
                print(
                    f"Generating {rand_['n_src']} sources within {fov:.2f} deg FoV ({fov/2:.2f} radius) ..."
                )
                print(f"Minimum {n_beam*beam_width*3600:.1f} arcsec separation ...")

                I, d_ra, d_dec = generate_random_sky(
                    n_src=rand_["n_src"],
                    min_I=sigma_value(rand_["min_I"], obs),
                    max_I=sigma_value(rand_["max_I"], obs),
                    freqs=obs.freqs,
                    fov=fov,
                    beam_width=beam_width,
                    random_seed=rand_["random_seed"],
                    n_beam=n_beam,
                )
                ra = (obs.ra + d_ra) % 360
                dec = obs.dec + d_dec
                dec = np.where(dec > 90, 180 - dec, dec)
                dec = np.where(dec < -90, -180 - dec, dec)

                if key == "point":
                    print()
                    print(f"Adding {rand_['n_src']} random {key} sources ...")
                    methods[key](I[:, None, :], ra, dec)
                elif key == "gauss":
                    rng = np.random.default_rng(rand_["random_seed"] + 1)
                    pos_ang = rng.uniform(low=0.0, high=360.0, size=(rand_["n_src"],))
                    major_ = np.abs(
                        rng.normal(
                            loc=rand_["major_mean"],
                            scale=rand_["major_std"],
                            size=(rand_["n_src"],),
                        )
                    )
                    minor_ = np.abs(
                        rng.normal(
                            loc=rand_["minor_mean"],
                            scale=rand_["minor_std"],
                            size=(rand_["n_src"],),
                        )
                    )
                    major = np.where(major_ > minor_, major_, minor_)
                    minor = np.where(major_ < minor_, major_, minor_)
                    print()
                    print(f"Adding {rand_['n_src']} random {key} sources ...")
                    methods[key](
                        I[:, None, :],
                        major,
                        minor,
                        pos_ang,
                        ra,
                        dec,
                    )
                elif key == "exp":
                    rng = np.random.default_rng(rand_["random_seed"] + 1)
                    shape = np.abs(
                        rng.normal(
                            loc=rand_["size_mean"],
                            scale=rand_["size_std"],
                            size=(rand_["n_src"],),
                        )
                    )
                    print()
                    print(f"Adding {rand_['n_src']} random {key} sources ...")
                    methods[key](I[:, None, :], shape, ra, dec)


def gauss(A: float, mean: float, sigma: float, x: da.Array) -> da.Array:
    return A * da.exp(-0.5 * ((x - mean) / sigma) ** 2)


def generate_spectra(spec_df: pd.DataFrame, freqs: da.Array, id_key: str) -> tuple:

    spectra = []
    ids = []
    for i, spec in spec_df.iterrows():
        if spec["sig_type"].lower() == "gauss":
            ids.append(spec[id_key])
            spectra.append(
                gauss(spec["power"], spec["freq"], spec["band_width"] / 2, freqs)
            )
        else:
            print(f"sig_type : {spec['sig_type']} not supported.")

    return np.array(ids), da.atleast_2d(spectra)


def add_satellite_sources(obs: Observation, obs_spec: dict) -> None:

    sat_ = obs_spec["rfi_sources"]["satellite"]

    # Circular path based Satellites
    if len(sat_["sat_ids"]) > 0:
        if sat_["circ_path"] is not None:
            oles = pd.read_csv(sat_["circ_path"])
            oles = oles[oles["sat_id"].isin(sat_["sat_ids"])]
            path_ids = oles["sat_id"].values
        else:
            raise ValueError(
                "'circ_path' must be populated to include satellites with circular orbits."
            )

        sat_spec = pd.read_csv(sat_["spec_model"])
        sat_spec = sat_spec[sat_spec["sat_id"].isin(path_ids)]

        if len(sat_spec) > 0:
            spec_ids, spectra = generate_spectra(sat_spec, obs.freqs, "sat_id")
            uids = np.unique(spec_ids)
            for uid in uids:
                Pv = (
                    sat_["power_scale"]
                    * da.sum(spectra[spec_ids == uid], axis=0)[None, None, :]
                    * da.ones((1, 1, obs.n_freq))
                )
                ole = oles[oles["sat_id"] == uid]
                if len(ole) == 1:
                    print()
                    print("Adding satellite RFI source ...")
                    obs.addSatelliteRFI(
                        Pv,
                        ole["elevation"].values,
                        ole["inclination"].values,
                        ole["lon_asc_node"].values,
                        ole["periapsis"].values,
                    )
                else:
                    print()
                    print(f"sat_id: {uid} multiply-defined.")
        else:
            print("No 'sat_ids' matching in 'spec_model' file given.")


def add_tle_satellite_sources(
    obs: Observation, obs_spec: dict, spacetrack_path: str
) -> None:

    sat_ = obs_spec["rfi_sources"]["tle_satellite"]

    # TLE path based Satellites
    tle_cond = [
        sat_["norad_ids_path"],
        len(sat_["norad_ids"]) > 0,
        len(sat_["sat_names"]) > 0,
    ]
    if np.any(tle_cond):

        if spacetrack_path:
            st_ = yaml_load(spacetrack_path)
        else:
            raise ValueError(
                "Space-Track login details must be given to simulate TLE based satellites"
            )

        if sat_["norad_ids_path"] is None:
            norad_ids = sat_["norad_ids"]
        else:
            norad_ids = np.concatenate(
                [sat_["norad_ids"], np.loadtxt(sat_["norad_ids_path"], usecols=0)]
            )

        from astropy.time import Time

        jd_step = sat_["vis_step"] / (24 * 60)
        times_check = Time(
            np.arange(obs.times_mjd[0], obs.times_mjd[-1] + jd_step, jd_step),
            format="mjd",
        )

        norad_ids, tles = get_visible_satellite_tles(
            st_["username"],
            st_["password"],
            times_check,
            obs.latitude,
            obs.longitude,
            obs.elevation,
            obs.ra,
            obs.dec,
            sat_["max_ang_sep"],
            sat_["min_alt"],
            sat_["sat_names"],
            norad_ids,
            sat_["tle_dir"],
        )

        print(f"NORAD IDs includeed : {norad_ids}")

        sat_spec = pd.read_csv(sat_["norad_spec_model"])
        sat_spec = sat_spec[sat_spec["norad_id"].isin(norad_ids)]

        print(f"Spectral models for {len(sat_spec)} TLE satellites found.")

        if len(sat_spec) > 0:
            print()
            print("Adding TLE-based satellite RFI sources ...")
            ids, spectra = generate_spectra(sat_spec, obs.freqs, "norad_id")
            uids = np.unique(ids)
            for uid in tqdm(uids[: sat_["max_n_sat"]]):
                Pv = (
                    sat_["power_scale"]
                    * da.sum(spectra[ids == uid], axis=0)[None, None, :]
                    * da.ones((1, 1, obs.n_freq))
                )
                tle = tles[norad_ids == uid]
                if len(tle) == 1:
                    obs.addTLESatelliteRFI(Pv, [uid], tle)
                else:
                    print()
                    print(f"norad_id: {uid} multiply-defined.")
        else:
            print("No NORAD IDs matching in 'norad_spec_model' file given.")


def add_stationary_sources(obs: Observation, obs_spec: dict) -> None:

    stat_ = obs_spec["rfi_sources"]["stationary"]
    if len(stat_["loc_ids"]) > 0:
        ids = stat_["loc_ids"]
        path_ids = []
        if stat_["geo_path"] is not None:
            geos = pd.read_csv(stat_["geo_path"])
            geos = geos[geos["loc_id"].isin(ids)]
            path_ids.append(geos["loc_id"].values)
        else:
            raise ValueError(
                "'geo_path' must be populated to include stationary RFI sources."
            )

        if len(path_ids) == 0:
            print("No location IDs matching in 'geo_path' file given.")
        else:
            path_ids = np.concatenate(path_ids)

            stat_spec = pd.read_csv(stat_["spec_model"])
            stat_spec = stat_spec[stat_spec["loc_id"].isin(ids)]

            if len(stat_spec) > 0:
                ids, spectra = generate_spectra(stat_spec, obs.freqs, "loc_id")
                uids = np.unique(ids)
                for uid in uids:
                    Pv = (
                        stat_["power_scale"]
                        * da.sum(spectra[ids == uid], axis=0)[None, None, :]
                        * da.ones((1, 1, obs.n_freq))
                    )
                    geo = geos[geos["loc_id"] == uid]
                    if len(geo) == 1:
                        print()
                        print("Adding stationary RFI source ...")
                        obs.addStationaryRFI(
                            Pv,
                            geo["latitude"].values,
                            geo["longitude"].values,
                            geo["elevation"].values,
                        )
                    else:
                        print()
                        print(f"loc_id: {uid} multiply-defined.")
            else:
                print("No locations IDs matching in 'spec_model' file given.")


def add_gains(obs: Observation, obs_spec: dict) -> None:

    gains_ = obs_spec["gains"]
    gain_offset = gains_["G0_mean"] != 1 or gains_["G0_std"] != 0
    gain_var = gains_["Gt_std_amp"] != 0 or gains_["Gt_std_phase"] != 0
    if gain_offset or gain_var:
        print()
        print("Adding gains ...")
        obs.addGains(
            gains_["G0_mean"],
            gains_["G0_std"],
            gains_["Gt_std_amp"],
            gains_["Gt_std_phase"],
            gains_["Gt_corr_amp"],
            gains_["Gt_corr_phase"],
            gains_["random_seed"],
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


def print_signal_specs(
    vis_rfi: da.Array, vis_ast: da.Array, noise_data: da.Array, flags: da.Array
) -> None:

    rfi_amp = da.mean(da.abs(vis_rfi)).compute()
    ast_amp = da.mean(da.abs(vis_ast)).compute()
    noise = da.std(noise_data.real).compute()
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
        obs.calculate_vis(flags=obs_spec["output"]["flag_data"])

    overwrite = obs_spec["output"]["overwrite"]

    if obs_spec["output"]["zarr"] and obs_spec["output"]["ms"]:
        write_to_zarr(obs, zarr_path, overwrite)
        xds = xr.open_zarr(zarr_path)
        write_to_ms(xds, ms_path, overwrite)
        print_signal_specs(
            xds.vis_rfi.data, xds.vis_ast.data, xds.noise_data.data, xds.flags.data
        )
    elif obs_spec["output"]["zarr"]:
        write_to_zarr(obs, zarr_path, overwrite)
        xds = xr.open_zarr(zarr_path)
        print_signal_specs(
            xds.vis_rfi.data, xds.vis_ast.data, xds.noise_data.data, xds.flags.data
        )
    elif obs_spec["output"]["ms"]:
        write_to_ms(obs.dataset, ms_path, overwrite)
        xds = xds_from_ms(ms_path)
        print_signal_specs(
            xds.RFI_MODEL_DATA.data,
            xds.AST_MODEL_DATA.data,
            xds.NOISE_DATA.data,
            xds.FLAG.data,
        )
    else:
        ValueError(
            "No output format has been chosen. output: zarr: or output: ms: must be True."
        )


def save_inputs(obs: Observation, obs_spec: dict, save_path: str) -> None:

    for key in ["enu_path", "itrf_path"]:
        path = obs_spec["telescope"][key]
        if path is not None:
            shutil.copy(path, save_path)

    for key in obs_spec["ast_sources"].keys():
        path = obs_spec["ast_sources"][key]["path"]
        if path is not None:
            shutil.copy(path, save_path)

    key = (
        2
        * [
            "tle_satellite",
        ]
        + 2
        * [
            "satellite",
        ]
        + 2
        * [
            "stationary",
        ]
    )
    subkey = [
        "norad_ids_path",
        "norad_spec_model",
        "circ_path",
        "spec_model",
        "geo_path",
        "spec_model",
    ]
    for key1, key2 in zip(key, subkey):
        path = obs_spec["rfi_sources"][key1][key2]
        if path is not None:
            shutil.copy(path, save_path)

    np.savetxt(os.path.join(save_path, "norad_ids.yaml"), obs.norad_ids, fmt="%i")

    with open(os.path.join(save_path, "sim_config.yaml"), "w") as fp:
        yaml.dump(obs_spec, fp)


def print_fringe_freq_sat(obs: Observation):
    if len(obs.times_fine) == obs.n_int_samples:
        n_int = int(obs.n_int_samples) - 1
    elif obs.times[-1] - obs.times[0] > 120:
        n_int = int(60 / obs.int_time) * int(obs.n_int_samples)
    else:
        n_int = obs.n_int_samples
    fringe_params = [
        {
            "times": obs.times_fine[::n_int].compute(),
            "freq": obs.freqs.max().compute(),
            "rfi_xyz": da.concatenate(obs.rfi_satellite_xyz, axis=0)[
                i, ::n_int
            ].compute(),
            "ants_itrf": obs.ITRF.compute(),
            "ants_u": obs.ants_uvw[::n_int, :, 0].compute(),
            "dec": obs.dec.compute(),
        }
        for i in range(obs.n_rfi_satellite)
    ]
    fringe_freq = [calculate_fringe_frequency(**f_params) for f_params in fringe_params]
    f_sample = (
        np.pi
        * np.max(np.abs(fringe_freq))
        * np.max(obs.rfi_satellite_A_app)
        / np.sqrt(6 * obs.noise_std.mean()).compute()
    )
    n_int = int(np.ceil(obs.int_time * f_sample))

    print()
    print(f"Maximum Fringe Frequency is : {np.max(np.abs(fringe_freq)):.2f} Hz")
    print(f"Maximum sampling rate is    : {f_sample:.2f} Hz")
    print(f"Recommended n_int is >=     : {n_int:.0f} ({obs.n_int_samples} used)")


def print_fringe_freq_tle_sat(obs: Observation):
    if len(obs.times_fine) == obs.n_int_samples:
        n_int = obs.n_int_samples - 1
    elif obs.times[-1] - obs.times[0] > 120:
        n_int = int(60 / obs.int_time) * obs.n_int_samples
    else:
        n_int = obs.n_int_samples
    fringe_params = [
        {
            "times_mjd": obs.times_mjd_fine[::n_int].compute(),
            "freq": obs.freqs.max().compute(),
            "rfi_xyz": da.concatenate(obs.rfi_tle_satellite_xyz, axis=0)[
                i, ::n_int
            ].compute(),
            "ants_itrf": obs.ITRF.compute(),
            "ants_u": obs.ants_uvw[::n_int, :, 0].compute(),
            "dec": obs.dec.compute(),
        }
        for i in range(obs.n_rfi_tle_satellite)
    ]
    fringe_freq = [calculate_fringe_frequency(**f_params) for f_params in fringe_params]
    f_sample = (
        np.pi
        * np.max(np.abs(fringe_freq))
        * np.max(obs.rfi_tle_satellite_A_app)
        / np.sqrt(6 * obs.noise_std.mean()).compute()
    )
    n_int = int(np.ceil(obs.int_time * f_sample))

    print()
    print(f"Maximum Fringe Frequency is : {np.max(np.abs(fringe_freq)):.2f} Hz")
    print(
        f"Maximum RFI Amplitude       : {np.max(obs.rfi_tle_satellite_A_app):.5f} sqrt(Jy)"
    )
    print(f"Maximum sampling rate is    : {f_sample:.2f} Hz")
    print(f"Recommended n_int is >=     : {n_int:.0f} ({obs.n_int_samples} used)")


def run_sim_config(
    obs_spec: dict = None, config_path: str = None, spacetrack_path: str = None
) -> Observation:

    from tabascal.utils.tle import id_generator

    log_path = f"log_sim_{id_generator()}.txt"
    log = open(log_path, "w")
    backup = sys.stdout
    sys.stdout = Tee(sys.stdout, log)

    start = datetime.now()
    print(datetime.now())

    if config_path is not None:
        obs_spec = load_config(config_path, config_type="sim")
    elif obs_spec is None:
        print("obs_spec or path must be defined.")
        return None

    obs = load_obs(obs_spec)
    add_astro_sources(obs, obs_spec)
    add_satellite_sources(obs, obs_spec)
    if obs_spec["rfi_sources"]["tle_satellite"]["max_n_sat"] != 0:
        add_tle_satellite_sources(obs, obs_spec, spacetrack_path)
    add_stationary_sources(obs, obs_spec)
    add_gains(obs, obs_spec)

    # Change this for calculating all RFI sources
    if obs.n_rfi_satellite > 0:
        print_fringe_freq_sat(obs)
    if obs.n_rfi_tle_satellite > 0:
        print_fringe_freq_tle_sat(obs)

    print(obs)

    obs_name = mk_obs_name(
        obs_spec["output"]["prefix"], obs, obs_spec["output"]["suffix"]
    )
    save_path, zarr_path, ms_path = mk_obs_dir(
        obs_spec["output"]["path"], obs_name, obs_spec["output"]["overwrite"]
    )

    input_path = os.path.join(save_path, "input_data")

    os.makedirs(input_path, exist_ok=True)
    save_inputs(obs, obs_spec, input_path)

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
    shutil.copy(log_path, save_path)
    os.remove(log_path)
    sys.stdout = backup

    return obs, save_path
