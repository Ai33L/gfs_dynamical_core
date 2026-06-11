import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from flax import struct
from s2fft.precompute_transforms import spherical as _pre_spherical
from s2fft.precompute_transforms.construct import (
    spin_spherical_kernel_jax as _build_spin_kernel,
)

from .states import GridGradients, GridState, SpectralState, SpectralTendencies

# ---------------------------------------------------------------------------
# Precompute-kernel transform wrappers.
#
# s2fft's on-the-fly Wigner recursions dominate the dycore cost (~7 ms per
# transform at L=64 on CPU). The precompute kernels are only ~4 MB each at
# this resolution and reduce a transform to a dense contraction (~0.6 ms,
# ~12x faster), matching the recursive results to ~1e-12. Kernels are cached
# per (L, sampling, spin, direction); under jit they become baked-in
# constants.
# ---------------------------------------------------------------------------

_KERNEL_CACHE: dict = {}


def _kernel(L: int, sampling: str, spin: int, forward: bool):
    key = (L, sampling, spin, forward)
    if key not in _KERNEL_CACHE:
        # Kernel construction is not jit-traceable (data-dependent nonzero):
        # it must happen eagerly. prebuild_kernels() does this; the lazy path
        # here covers direct (non-jit) use in tests and scripts.
        _KERNEL_CACHE[key] = _build_spin_kernel(
            L, spin=spin, sampling=sampling, forward=forward
        )
    return _KERNEL_CACHE[key]


def prebuild_kernels(L: int, sampling: str = "gl"):
    """Eagerly build every kernel the dycore uses, so that jit-compiled
    code only ever reads the cache. Must be called before tracing."""
    for spin, forward in [(0, True), (0, False), (1, True), (1, False), (-1, True)]:
        _kernel(L, sampling, spin, forward)


def s2_forward(f, L: int, sampling: str, spin: int = 0):
    """Forward spherical-harmonic transform via precompute kernel."""
    return _pre_spherical.forward_transform_jax(
        f, _kernel(L, sampling, spin, True), L, sampling, False, spin, None
    )


def s2_inverse(flm, L: int, sampling: str, spin: int = 0):
    """Inverse spherical-harmonic transform via precompute kernel."""
    return _pre_spherical.inverse_transform_jax(
        flm, _kernel(L, sampling, spin, False), L, sampling, False, spin, None
    )


def get_grid_dimensions(L: int, sampling: str) -> tuple[int, int]:
    """Returns (n_lat, n_lon) for a given L and sampling."""
    if sampling == "gl":
        return L, 2 * L - 1
    elif sampling == "mw":
        return L, 2 * L - 1
    elif sampling == "dh":
        return 2 * L, 2 * L
    return L, 2 * L


@struct.dataclass
class TransformConfig:
    """Configuration for spectral transforms.

    Grid dimensions are derived directly from L and sampling:
      - GL/MW:  n_lat = L,   n_lon = 2*L - 1
      - DH:     n_lat = 2*L, n_lon = 2*L

    No external n_lat/n_lon needed; all grid-space arrays inside the
    dynamics pipeline use the native s2fft sizes.

    The ``ntrunc`` parameter controls the *physical* triangular truncation
    used for de-aliasing.  It defaults to ``int(n_lon / 3 - 2)`` which
    matches the Fortran GFS convention and satisfies the standard 2/3
    de-aliasing rule for quadratic nonlinearities.  Spectral arrays still
    have shape ``(L, 2*L-1)`` but coefficients with ``l > ntrunc`` or
    ``|m| > ntrunc`` are zeroed after every forward transform via
    :func:`enforce_triangular_truncation`.
    """

    L: int = struct.field(pytree_node=False)  # Bandlimit for s2fft transforms
    sampling: str = struct.field(
        pytree_node=False, default="gl"
    )  # Gauss-Legendre for GFS
    ntrunc: int = struct.field(
        pytree_node=False, default=None
    )  # Physical truncation (de-aliased)
    radius: float = 6371000.0

    @property
    def truncation(self):
        """Physical triangular truncation wavenumber (de-aliased).

        Defaults to ``int(n_lon / 3 - 2)`` (the Fortran GFS convention)
        when *ntrunc* is not explicitly set.
        """
        if self.ntrunc is not None:
            return self.ntrunc
        # Fortran GFS convention: ntrunc = int(nlons / 3 - 2)
        return int(self.n_lon / 3 - 2)

    @property
    def n_lat(self):
        return get_grid_dimensions(self.L, self.sampling)[0]

    @property
    def n_lon(self):
        return get_grid_dimensions(self.L, self.sampling)[1]


def get_gaussian_latitudes(L: int) -> jnp.ndarray:
    """Compute Gaussian quadrature latitudes (in radians) for GL sampling.

    s2fft GL sampling uses colatitudes theta from ``np.polynomial.legendre.leggauss``.
    We convert to geographic latitude phi = pi/2 - theta, so that phi ranges from
    approximately +pi/2 (north pole) to -pi/2 (south pole).

    Returns:
        jnp.ndarray: Latitudes in radians, shape ``(L,)``, north-to-south
            (matching s2fft's GL row order which goes theta = small -> large,
            i.e. north -> south).
    """
    # s2fft thetas for GL: flip(arccos(leggauss(L)[0]))  -> theta ascending (N->S)
    cos_theta, _ = np.polynomial.legendre.leggauss(L)
    thetas = np.flip(np.arccos(cos_theta))  # colatitude, ascending
    latitudes = np.pi / 2.0 - thetas  # geographic latitude, N->S descending
    return jnp.array(latitudes)


def spectral_to_grid(
    spec_state: SpectralState, config: TransformConfig
) -> tuple[GridState, GridGradients]:
    """
    Transforms spectral state to grid space, including gradients.

    All output arrays use native s2fft grid dimensions (L, 2*L-1) for GL sampling.
    No longitude resampling is performed.
    """
    L = config.L
    sampling = config.sampling
    radius = config.radius

    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))
    # Avoid division by zero at l=0
    inv_l_factor = jnp.where(l_arr > 0, 1.0 / l_factor, 0.0)

    def transform_level(vort, div, temp, tracers):
        # 1. Scalar transforms
        grid_t = s2_inverse(temp, L, sampling)
        grid_vort = s2_inverse(vort, L, sampling)
        grid_div = s2_inverse(div, L, sampling)

        # 2. Vector transforms (vort, div -> u, v)
        # F1_lm = (D_lm + i zeta_lm) * radius / sqrt(l(l+1))
        F1_lm = inv_l_factor[:, None] * (div + 1j * vort) * radius
        f_spin1 = s2_inverse(F1_lm, L, sampling, spin=1)
        # u = Imag(f_spin1), v = -Real(f_spin1)
        grid_u = f_spin1.imag
        grid_v = -f_spin1.real

        # 3. Tracers (scalar inverse transforms)
        grid_tracers = jax.vmap(
            lambda flm: s2_inverse(flm, L, sampling)
        )(tracers)

        # 4. Temperature Gradients
        F1_t_lm = -l_factor[:, None] * temp
        f_t_spin1 = s2_inverse(F1_t_lm, L, sampling, spin=1)
        grad_t_x = f_t_spin1.imag / radius
        grad_t_y = -f_t_spin1.real / radius

        # 5. Tracer Gradients (same spin-1 approach as temperature)
        # tracers shape here: (n_tracers, L, 2*L-1)
        def _tracer_grad(tracer_flm):
            F1_lm_ = -l_factor[:, None] * tracer_flm
            f_spin1_ = s2_inverse(F1_lm_, L, sampling, spin=1)
            return f_spin1_.imag / radius, -f_spin1_.real / radius

        grad_tracers_x, grad_tracers_y = jax.vmap(_tracer_grad)(tracers)
        # grad_tracers_x/y shape: (n_tracers, n_lat, n_lon)

        return (
            grid_u,
            grid_v,
            grid_t,
            grid_vort,
            grid_div,
            grid_tracers,
            grad_t_x,
            grad_t_y,
            grad_tracers_x,
            grad_tracers_y,
        )

    (
        grid_u,
        grid_v,
        grid_t,
        grid_vort,
        grid_div,
        grid_tracers,
        grad_t_x,
        grad_t_y,
        grad_tracers_x,
        grad_tracers_y,
    ) = jax.vmap(transform_level)(
        spec_state.vorticity,
        spec_state.divergence,
        spec_state.temperature,
        spec_state.tracers.transpose(1, 0, 2, 3),
    )
    # After vmap:
    #   grad_tracers_x / grad_tracers_y shape: (levels, n_tracers, n_lat, n_lon)
    # Reorder to (n_tracers, levels, n_lat, n_lon) to match tracers convention
    grad_tracers_x = grad_tracers_x.transpose(1, 0, 2, 3)
    grad_tracers_y = grad_tracers_y.transpose(1, 0, 2, 3)

    lnps = s2_inverse(spec_state.log_surface_pressure, L, sampling)

    # Log surface pressure gradients
    F1_lnps_lm = -l_factor[:, None] * spec_state.log_surface_pressure
    f_lnps_spin1 = s2_inverse(F1_lnps_lm, L, sampling, spin=1)
    grad_lnps_x = f_lnps_spin1.imag / radius
    grad_lnps_y = -f_lnps_spin1.real / radius

    grid_state = GridState(
        u=grid_u,
        v=grid_v,
        temperature=grid_t,
        vorticity=grid_vort,
        divergence=grid_div,
        log_surface_pressure=lnps,
        tracers=grid_tracers.transpose(1, 0, 2, 3),
    )

    grid_grads = GridGradients(
        d_log_ps_d_lambda=grad_lnps_x,
        d_log_ps_d_phi=grad_lnps_y,
        d_t_d_lambda=grad_t_x,
        d_t_d_phi=grad_t_y,
        d_tracers_d_lambda=grad_tracers_x,
        d_tracers_d_phi=grad_tracers_y,
    )

    return grid_state, grid_grads


def enforce_triangular_truncation(flm, L, T):
    """
    Zero out spherical harmonics where l > T or |m| > T, and enforce the
    reality condition f_{l,-m} = (-1)^m conj(f_{l,+m}) (with Im f_{l,0} = 0).

    The reality enforcement is essential for long-term stability: s2fft
    stores the full (l, +-m) rectangle and treats fields as general
    complex functions, so roundoff (mainly in the dual-spin +- forward
    combinations) seeds a tiny violation of the reality condition. That
    violation is exactly an imaginary-valued grid field, which evolves as
    an undamped tangent-linear perturbation of the flow — it e-folds with
    the jet's fastest instability (~5.5 h for the JW06 jet at T40) until
    it couples back into the real part through nonlinear terms and blows
    up the model around day 7-8. SHTNS/Fortran enforces the symmetry by
    construction (it only stores m >= 0); this makes the JAX state do the
    same. All real-coefficient linear ops (RK updates, implicit solve,
    diffusion) preserve the symmetry exactly, so applying it at every
    forward-transform output keeps the state symmetric for all time.

    flm shape: (..., L, 2L-1)
    """
    l_arr = jnp.arange(L)
    m_arr = jnp.arange(-L + 1, L)
    l_grid, m_grid = jnp.meshgrid(l_arr, m_arr, indexing="ij")
    mask = (l_grid <= T) & (jnp.abs(m_grid) <= T) & (jnp.abs(m_grid) <= l_grid)

    # Expand mask for prepended dimensions
    for _ in range(flm.ndim - 2):
        mask = jnp.expand_dims(mask, axis=0)

    flm = jnp.where(mask, flm, 0.0)

    # Reality condition: rebuild negative-m modes from positive-m modes
    # (matching Fortran/SHTNS, which only stores m >= 0), zero Im at m=0.
    m_vals = jnp.arange(-L + 1, L)  # m along the last axis
    sign = jnp.where(m_vals % 2 == 0, 1.0, -1.0)  # (-1)^|m| = (-1)^m
    mirrored = sign * jnp.conj(flm[..., ::-1])  # (-1)^m conj(f_{l,-m->+m})
    flm = jnp.where(m_vals < 0, mirrored, flm)
    flm = jnp.where(m_vals == 0, flm.real.astype(flm.dtype), flm)
    return flm


def grid_to_spectral(grid_state: GridState, config: TransformConfig) -> SpectralState:
    """
    Transforms grid state to spectral state.

    Input arrays are expected in native s2fft grid dimensions (L, 2*L-1) for GL.
    No longitude resampling is performed.
    """
    L = config.L
    T = config.truncation
    sampling = config.sampling
    radius = config.radius

    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))

    def transform_level(u, v, temp, tracers):
        # 1. Scalar transforms
        flm_temp = s2_forward(temp, L, sampling)

        # 2. Vector transforms (u, v -> vort, div)
        # Use BOTH spin+1 and spin-1 forward transforms to correctly
        # decompose divergence and vorticity spectral coefficients.
        #
        # spin+1 forward of (-v + i*u) gives F1_lm  where l_factor*F1/R  = D + i*zeta
        # spin-1 forward of ( v + i*u) gives Fm1_lm where l_factor*Fm1/R = D - i*zeta
        #
        # For m=0, D_lm and zeta_lm are real, so a single transform and
        # .real/.imag would suffice. But for m>=1 they are complex:
        #   D_lm = a+bi, zeta_lm = c+di
        #   D + i*zeta = (a-d) + i*(b+c)  -> .real/.imag loses information
        #
        # With both transforms we can solve exactly:
        #   D_lm    = (result_p + result_m) / 2
        #   zeta_lm = (result_p - result_m) / (2i)
        f_plus = -v + 1j * u  # spin +1 input
        f_minus = v + 1j * u  # spin -1 input

        F1_lm = s2_forward(f_plus, L, sampling, spin=1)
        Fm1_lm = s2_forward(f_minus, L, sampling, spin=-1)

        result_p = l_factor[:, None] * F1_lm / radius  # D + i*zeta
        result_m = l_factor[:, None] * Fm1_lm / radius  # D - i*zeta

        flm_div = (result_p + result_m) / 2
        flm_vort = (result_p - result_m) / (2j)

        # 3. Tracers
        flm_tracers = jax.vmap(lambda f: s2_forward(f, L, sampling))(
            tracers
        )

        return flm_vort, flm_div, flm_temp, flm_tracers

    flm_vort, flm_div, flm_temp, flm_tracers = jax.vmap(transform_level)(
        grid_state.u,
        grid_state.v,
        grid_state.temperature,
        grid_state.tracers.transpose(1, 0, 2, 3),
    )

    flm_lnps = s2_forward(grid_state.log_surface_pressure, L, sampling)

    # Enforce exact triangular truncation limit
    flm_vort = enforce_triangular_truncation(flm_vort, L, T)
    flm_div = enforce_triangular_truncation(flm_div, L, T)
    flm_temp = enforce_triangular_truncation(flm_temp, L, T)
    flm_lnps = enforce_triangular_truncation(flm_lnps, L, T)
    flm_tracers = enforce_triangular_truncation(flm_tracers, L, T)

    spec_state = SpectralState(
        vorticity=flm_vort,
        divergence=flm_div,
        temperature=flm_temp,
        log_surface_pressure=flm_lnps,
        tracers=flm_tracers.transpose(1, 0, 2, 3),
    )

    return spec_state


def grid_to_spectral_tendencies(
    grid_tends, config: TransformConfig
) -> SpectralTendencies:
    """
    Transforms grid-space tendencies back to spectral space.

    Input arrays are expected in native s2fft grid dimensions (L, 2*L-1) for GL.
    No longitude resampling is performed.
    """
    L = config.L
    T = config.truncation
    sampling = config.sampling
    radius = config.radius

    l_arr = jnp.arange(L)
    l_factor = jnp.sqrt(l_arr * (l_arr + 1))

    # temp, lnps, tracers are scalars
    flm_temp = jax.vmap(lambda f: s2_forward(f, L, sampling))(
        grid_tends.temp_tend
    )

    flm_lnps = s2_forward(grid_tends.log_ps_tend, L, sampling)

    def forward_tracers(tracers):
        return jax.vmap(lambda f: s2_forward(f, L, sampling))(tracers)

    flm_tracers = jax.vmap(forward_tracers)(grid_tends.tracer_tends)

    def transform_vector_tendencies(u_flux, v_flux):
        # Use BOTH spin+1 and spin-1 forward transforms (same fix as
        # grid_to_spectral) to correctly decompose the divergence and
        # curl of the flux vector for all m values.
        f_plus = -v_flux + 1j * u_flux  # spin +1 input
        f_minus = v_flux + 1j * u_flux  # spin -1 input

        F1_lm = s2_forward(f_plus, L, sampling, spin=1)
        Fm1_lm = s2_forward(f_minus, L, sampling, spin=-1)

        result_p = l_factor[:, None] * F1_lm / radius  # div(flux) + i*curl(flux)
        result_m = l_factor[:, None] * Fm1_lm / radius  # div(flux) - i*curl(flux)

        div_of_flux = (result_p + result_m) / 2
        curl_of_flux = (result_p - result_m) / (2j)

        # The momentum equation tendencies are:
        #   d(vorticity)/dt  = -div(flux)
        #   d(divergence)/dt =  curl(flux)
        #
        # This matches the Fortran convention where getvrtdivspec computes
        # (vort_of_flux, div_of_flux) and then:
        #   ddivspecdt  =  vort_of_flux   (= curl of flux)
        #   dvrtspecdt  = -div_of_flux    (sign-flipped divergence of flux)
        d_vort = -div_of_flux
        d_div = curl_of_flux
        return d_vort, d_div

    flm_vort, flm_div = jax.vmap(transform_vector_tendencies)(
        grid_tends.u_flux, grid_tends.v_flux
    )

    # Add KE Laplacian to divergence tendency
    def add_ke_laplacian(ke, div_tend):
        ke_lm = s2_forward(ke, L, sampling)
        laplacian_ke = -(l_arr * (l_arr + 1))[:, None] * ke_lm / (radius**2)
        return div_tend - laplacian_ke

    flm_div = jax.vmap(add_ke_laplacian)(grid_tends.kinetic_energy, flm_div)

    # Enforce exact triangular truncation limit
    flm_vort = enforce_triangular_truncation(flm_vort, L, T)
    flm_div = enforce_triangular_truncation(flm_div, L, T)
    flm_temp = enforce_triangular_truncation(flm_temp, L, T)
    flm_lnps = enforce_triangular_truncation(flm_lnps, L, T)
    flm_tracers = enforce_triangular_truncation(flm_tracers, L, T)

    return SpectralTendencies(
        d_vorticity_d_t=flm_vort,
        d_divergence_d_t=flm_div,
        d_temperature_d_t=flm_temp,
        d_log_surface_pressure_d_t=flm_lnps,
        d_tracers_d_t=flm_tracers,
    )
