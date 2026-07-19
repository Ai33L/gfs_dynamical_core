from examples.stochastic_sppt.config import ModelConfig, ExperimentConfig, TrainConfig, RESOLUTIONS


def test_resolutions_present():
    assert RESOLUTIONS["T42"] == (64, 42, 1200.0)
    assert set(RESOLUTIONS) == {"T21", "T42", "T85", "T127"}


def test_model_config_derived():
    mc = ModelConfig(resolution="T42")
    assert mc.L == 64 and mc.ntrunc == 42 and mc.dt == 1200.0 and mc.n_lev == 20


def test_experiment_and_train_defaults():
    ec = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21")
    assert ec.n_members >= 2 and ec.lead_days[-1] == 10
    tc = TrainConfig()
    assert 0.0 < tc.alpha <= 1.0


def test_experiment_config_has_n_draws_default_one():
    ec = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21")
    assert ec.n_draws == 1
    ec2 = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_draws=32)
    assert ec2.n_draws == 32
