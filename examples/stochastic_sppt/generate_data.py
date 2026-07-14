"""Generate identical-twin truth datasets (Mode A recovery / Mode B model-error)."""
import argparse
import pickle

import jax
import jax.numpy as jnp

from gfs_dynamical_core.jax.states import SpectralState
from gfs_dynamical_core.jax.transforms import spectral_to_grid

from . import sppt
from .config import ExperimentConfig, ModelConfig
from .diagnostics import extract_fields
from .held_suarez import hs_tendencies
from .model import build_model, rest_state, spectral_truncate, step

_FIELDS = ("u850", "v850", "t500", "vort500", "ps")


def _deterministic_advance(bundle, spec, n_steps):
    def body(s, _):
        grid, _g = spectral_to_grid(s, bundle.trans_config)
        return step(bundle, s, hs_tendencies(grid, bundle)), None
    out, _ = jax.lax.scan(body, spec, None, length=int(n_steps))
    return out


def nature_run(bundle, key, n_days, save_stride_days):
    steps_per_day = int(round(86400.0 / bundle.dt))
    stride = int(round(save_stride_days * steps_per_day))
    spec = rest_state(bundle, key, t0=280.0)
    spec = _deterministic_advance(bundle, spec, int(n_days * steps_per_day))
    return spec, stride, steps_per_day


def _sppt_advance(bundle, spec, params, key, n_steps):
    mu = sppt.vertical_taper(bundle)
    r = sppt.init_pattern(params, key, bundle.trans_config, bundle.dt)
    keys = jax.random.split(key, int(n_steps))

    def body(carry, k):
        s, rl = carry
        grid, _ = spectral_to_grid(s, bundle.trans_config)
        phys = sppt.apply_sppt(hs_tendencies(grid, bundle),
                               sppt.pattern_to_grid(rl, bundle.trans_config), mu, params)
        s = step(bundle, s, phys)
        rl = sppt.pattern_step(rl, k, params, bundle.trans_config, bundle.dt)
        return (s, rl), None

    (out, _), _ = jax.lax.scan(body, (spec, r), keys)
    return out


def _lead_steps(exp, bundle):
    spd = int(round(86400.0 / bundle.dt))
    return tuple(int(d) * spd for d in exp.lead_days)


def _sample_case_ics(bundle, key, exp):
    """Spin up then sample n_cases states separated by case_stride_days."""
    spec, _stride, spd = nature_run(bundle, key, exp.spinup_days, exp.case_stride_days)
    ics = [spec]
    for _ in range(exp.n_cases - 1):
        spec = _deterministic_advance(bundle, spec, int(exp.case_stride_days * spd))
        ics.append(spec)
    return ics


def mode_b_dataset(exp: ExperimentConfig):
    hi = build_model(ModelConfig(resolution=exp.truth_resolution))
    lo = build_model(ModelConfig(resolution=exp.forecast_resolution))
    key = jax.random.PRNGKey(exp.seed)
    lead_steps_hi = _lead_steps(exp, hi)

    case_ics_hi = _sample_case_ics(hi, key, exp)
    ic_specs, truth = [], {k: [] for k in _FIELDS}
    for ic_hi in case_ics_hi:
        ic_specs.append(spectral_truncate(ic_hi, hi, lo))
        prev, spec = 0, ic_hi
        per_lead = {k: [] for k in _FIELDS}
        for ls in lead_steps_hi:
            spec = _deterministic_advance(hi, spec, ls - prev)
            f = extract_fields(spectral_truncate(spec, hi, lo), lo)
            for k in _FIELDS:
                per_lead[k].append(f[k])
            prev = ls
        for k in _FIELDS:
            truth[k].append(jnp.stack(per_lead[k]))
    return _pack(ic_specs, truth, exp, lo)


def mode_a_dataset(exp: ExperimentConfig, true_params):
    bundle = build_model(ModelConfig(resolution=exp.forecast_resolution))
    key = jax.random.PRNGKey(exp.seed)
    lead_steps = _lead_steps(exp, bundle)

    case_ics = _sample_case_ics(bundle, key, exp)
    ic_specs, truth = [], {k: [] for k in _FIELDS}
    for i, ic in enumerate(case_ics):
        ic_specs.append(ic)
        mkey = jax.random.fold_in(key, 1000 + i)   # one truth draw from the true model
        prev, spec = 0, ic
        per_lead = {k: [] for k in _FIELDS}
        for ls in lead_steps:
            spec = _sppt_advance(bundle, spec, true_params,
                                 jax.random.fold_in(mkey, prev), ls - prev)
            f = extract_fields(spec, bundle)
            for k in _FIELDS:
                per_lead[k].append(f[k])
            prev = ls
        for k in _FIELDS:
            truth[k].append(jnp.stack(per_lead[k]))
    return _pack(ic_specs, truth, exp, bundle)


def _stack_specs(specs):
    return SpectralState(
        vorticity=jnp.stack([s.vorticity for s in specs]),
        divergence=jnp.stack([s.divergence for s in specs]),
        temperature=jnp.stack([s.temperature for s in specs]),
        log_surface_pressure=jnp.stack([s.log_surface_pressure for s in specs]),
        tracers=jnp.stack([s.tracers for s in specs]),
    )


def _pack(ic_specs, truth, exp, bundle):
    return {
        "ic_specs": _stack_specs(ic_specs),
        "truth": {k: jnp.stack(v) for k, v in truth.items()},   # (n_case,n_lead,lat,lon)
        "lead_steps": _lead_steps(exp, bundle),
        "forecast_resolution": exp.forecast_resolution,
        "n_lev": bundle.n_lev,
    }


def save_dataset(path, data):
    with open(path, "wb") as f:
        pickle.dump(jax.device_get(data), f)


def load_dataset(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["A", "B"], required=True)
    ap.add_argument("--forecast", default="T42")
    ap.add_argument("--truth", default="T127")
    ap.add_argument("--n-cases", type=int, default=16)
    ap.add_argument("--members", type=int, default=8)
    ap.add_argument("--spinup-days", type=float, default=200.0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    exp = ExperimentConfig(mode=a.mode, forecast_resolution=a.forecast,
                           truth_resolution=(a.truth if a.mode == "B" else a.forecast),
                           n_members=a.members, n_cases=a.n_cases, spinup_days=a.spinup_days)
    if a.mode == "A":
        data = mode_a_dataset(exp, sppt.default_params())
    else:
        data = mode_b_dataset(exp)
    save_dataset(a.out, data)
    print(f"wrote {a.out}: {exp.n_cases} cases, leads {exp.lead_days} d")


if __name__ == "__main__":
    main()
