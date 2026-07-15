import jax
import jax.numpy as jnp
from examples.stochastic_sppt.config import ExperimentConfig, ModelConfig, TrainConfig
from examples.stochastic_sppt import generate_data, sppt, train


def test_mode_a_training_moves_params_toward_truth():
    exp = ExperimentConfig(mode="A", forecast_resolution="T21", truth_resolution="T21",
                           n_members=4, lead_days=(1,), n_cases=4, spinup_days=1.0,
                           case_stride_days=1.0, seed=0)
    true_p = sppt.SPPTParams(log_sigma=jnp.log(0.6), log_tau=jnp.log(6 * 3600.0),
                             log_len=jnp.log(500e3))
    data = generate_data.mode_a_dataset(exp, true_p)

    tc = TrainConfig(alpha=0.95, lr=1e-1, n_opt_steps=15, batch_cases=4, log_every=5, seed=0)
    # start with a deliberately wrong sigma
    init = sppt.SPPTParams(log_sigma=jnp.log(0.2), log_tau=jnp.log(6 * 3600.0),
                           log_len=jnp.log(500e3))
    params, history = train.train(data, tc, init_params=init)
    # loss decreased and sigma moved toward the truth (0.6)
    assert history[-1] < history[0]
    assert float(jnp.exp(params.log_sigma)) > float(jnp.exp(init.log_sigma))
