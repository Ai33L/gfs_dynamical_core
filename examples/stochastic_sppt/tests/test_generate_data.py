import os
import jax.numpy as jnp
from examples.stochastic_sppt.config import ExperimentConfig
from examples.stochastic_sppt import generate_data


def _tiny(mode):
    return ExperimentConfig(
        mode=mode, forecast_resolution="T21",
        truth_resolution=("T42" if mode == "B" else "T21"),
        n_members=3, lead_days=(1,), n_cases=2, spinup_days=1.0,
        case_stride_days=1.0, seed=0,
    )


def test_mode_a_dataset_shapes(tmp_path):
    from examples.stochastic_sppt import sppt
    true_p = sppt.default_params()
    data = generate_data.mode_a_dataset(_tiny("A"), true_p)
    assert data["truth"]["t500"].shape[0] == 2       # n_cases
    assert data["truth"]["t500"].shape[1] == 1       # n_leads
    p = tmp_path / "ds.npz"
    generate_data.save_dataset(str(p), data)
    back = generate_data.load_dataset(str(p))
    assert set(back["truth"]) == {"u850", "v850", "t500", "vort500", "ps"}


def test_mode_b_truncates_truth_to_forecast_grid():
    data = generate_data.mode_b_dataset(_tiny("B"))
    # forecast grid is T21 -> n_lat=32, n_lon=63
    assert data["truth"]["ps"].shape[-2:] == (32, 63)
    assert data["ic_specs"].temperature.shape[-2:] == (32, 63)
