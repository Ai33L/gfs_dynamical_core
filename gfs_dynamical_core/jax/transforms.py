import jax
import jax.numpy as jnp
import numpy as np
import s2fft
from flax import struct

from .states import GridGradients, GridState, SpectralState, SpectralTendencies


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
        grid_t = s2fft.inverse_jax(temp, L, sampling=sampling)
        grid_vort = s2fft.inverse_jax(vort, L, sampling=sampling)
        grid_div = s2fft.inverse_jax(div, L, sampling=sampling)

        # 2. Vector transforms (vort, div -> u, v)
        # F1_lm = (D_lm + i zeta_lm) * radius / sqrt(l(l+1))
        F1_lm = inv_l_factor[:, None] * (div + 1j * vort) * radius
        f_spin1 = s2fft.inverse_jax(F1_lm, L, spin=1, sampling=sampling)
        # u = Imag(f_spin1), v = -Real(f_spin1)
        grid_u = f_spin1.imag
        grid_v = -f_spin1.real

        # 3. Tracers (scalar inverse transforms)
        grid_tracers = jax.vmap(
            lambda flm: s2fft.inverse_jax(flm, L, sampling=sampling)
        )(tracers)

        # 4. Temperature Gradients
        F1_t_lm = -l_factor[:, None] * temp
        f_t_spin1 = s2fft.inverse_jax(F1_t_lm, L, spin=1, sampling=sampling)
        grad_t_x = f_t_spin1.imag / radius
        grad_t_y = -f_t_spin1.real / radius

        # 5. Tracer Gradients (same spin-1 approach as temperature)
        # tracers shape here: (n_tracers, L, 2*L-1)
        def _tracer_grad(tracer_flm):
            F1_lm_ = -l_factor[:, None] * tracer_flm
            f_spin1_ = s2fft.inverse_jax(F1_lm_, L, spin=1, sampling=sampling)
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

    lnps = s2fft.inverse_jax(spec_state.log_surface_pressure, L, sampling=sampling)

    # Log surface pressure gradients
    F1_lnps_lm = -l_factor[:, None] * spec_state.log_surface_pressure
    f_lnps_spin1 = s2fft.inverse_jax(F1_lnps_lm, L, spin=1, sampling=sampling)
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
    Zero out spherical harmonics where l > T or |m| > T.
    flm shape: (..., L, 2L-1)
    """
    l_arr = jnp.arange(L)
    m_arr = jnp.arange(-L + 1, L)
    l_grid, m_grid = jnp.meshgrid(l_arr, m_arr, indexing="ij")
    mask = (l_grid <= T) & (jnp.abs(m_grid) <= T)

    # Expand mask for prepended dimensions
    for _ in range(flm.ndim - 2):
        mask = jnp.expand_dims(mask, axis=0)

    return jnp.where(mask, flm, 0.0)


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
        flm_temp = s2fft.forward_jax(temp, L, sampling=sampling)

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

        F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=sampling)
        Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)

        result_p = l_factor[:, None] * F1_lm / radius  # D + i*zeta
        result_m = l_factor[:, None] * Fm1_lm / radius  # D - i*zeta

        flm_div = (result_p + result_m) / 2
        flm_vort = (result_p - result_m) / (2j)

        # 3. Tracers
        flm_tracers = jax.vmap(lambda f: s2fft.forward_jax(f, L, sampling=sampling))(
            tracers
        )

        return flm_vort, flm_div, flm_temp, flm_tracers

    flm_vort, flm_div, flm_temp, flm_tracers = jax.vmap(transform_level)(
        grid_state.u,
        grid_state.v,
        grid_state.temperature,
        grid_state.tracers.transpose(1, 0, 2, 3),
    )

    flm_lnps = s2fft.forward_jax(grid_state.log_surface_pressure, L, sampling=sampling)

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
    flm_temp = jax.vmap(lambda f: s2fft.forward_jax(f, L, sampling=sampling))(
        grid_tends.temp_tend
    )

    flm_lnps = s2fft.forward_jax(grid_tends.log_ps_tend, L, sampling=sampling)

    def forward_tracers(tracers):
        return jax.vmap(lambda f: s2fft.forward_jax(f, L, sampling=sampling))(tracers)

    flm_tracers = jax.vmap(forward_tracers)(grid_tends.tracer_tends)

    def transform_vector_tendencies(u_flux, v_flux):
        # Use BOTH spin+1 and spin-1 forward transforms (same fix as
        # grid_to_spectral) to correctly decompose the divergence and
        # curl of the flux vector for all m values.
        f_plus = -v_flux + 1j * u_flux  # spin +1 input
        f_minus = v_flux + 1j * u_flux  # spin -1 input

        F1_lm = s2fft.forward_jax(f_plus, L, spin=1, sampling=sampling)
        Fm1_lm = s2fft.forward_jax(f_minus, L, spin=-1, sampling=sampling)

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
        ke_lm = s2fft.forward_jax(ke, L, sampling=sampling)
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
