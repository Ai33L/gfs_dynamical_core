"""Almost-fair CRPS (Lang et al. 2024) and ensemble-calibration diagnostics."""
import jax.numpy as jnp


def afcrps(ensemble, truth, alpha=0.95):
    """Almost-fair CRPS (Lang et al. 2024, eq. 3/4). alpha=1 -> fair CRPS.
    ensemble: (M, ...), truth: (...). Returns the mean score over trailing dims."""
    M = ensemble.shape[0]
    eps = (1.0 - alpha) / M
    e1 = jnp.mean(jnp.abs(ensemble - truth[None]), axis=0)
    diff = jnp.abs(ensemble[:, None] - ensemble[None, :])   # (M, M, ...)
    e2 = jnp.sum(diff, axis=(0, 1)) / (2.0 * M * (M - 1))
    return jnp.mean(e1 - (1.0 - eps) * e2)


def fair_crps(ensemble, truth):
    return afcrps(ensemble, truth, alpha=1.0)


def spread(ensemble):
    """RMS ensemble standard deviation (finite-M unbiased, ddof=1)."""
    var = jnp.var(ensemble, axis=0, ddof=1)
    return jnp.sqrt(jnp.mean(var))


def rmse_of_mean(ensemble, truth):
    err = jnp.mean(ensemble, axis=0) - truth
    return jnp.sqrt(jnp.mean(err ** 2))


def spread_error_ratio(ensemble, truth):
    """Reliable ensemble => ratio ~ 1 (finite-M correction on the error side)."""
    M = ensemble.shape[0]
    return spread(ensemble) / (jnp.sqrt(M) * rmse_of_mean(ensemble, truth))


def rank_histogram(ensemble, truth):
    """Counts of the truth's rank among the M members (length M+1)."""
    M = ensemble.shape[0]
    rank = jnp.sum(ensemble < truth[None], axis=0).ravel()   # 0..M
    return jnp.bincount(rank, length=M + 1)
