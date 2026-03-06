import jax
import jax.numpy as jnp
from flax import struct
from .states import SpectralState, SpectralTendencies
from .dynamics import get_spectral_tendencies, DynamicsConfig
from .transforms import TransformConfig

@struct.dataclass
class StepperConfig:
    """Configuration for the IMEX Runge-Kutta time stepper."""
    # Explicit RK constants
    a21: float = 1.0
    a31: float = 0.25
    a32: float = 0.25
    b1: float = 1.0 / 6.0
    b2: float = 1.0 / 6.0
    b3: float = 2.0 / 3.0
    
    # Implicit RK constants
    aa21: float = 0.635
    aa22: float = 0.365
    aa31: float = 0.3175
    aa32: float = 0.0
    aa33: float = 0.1825
    bb1: float = 0.35
    bb2: float = 0.0
    bb3: float = 0.3
    bb4: float = 0.35
    
    explicit: bool = struct.field(pytree_node=False, default=False)
    
    # Linear diffusion
    disspec: jnp.ndarray = None # (L, 2L-1)
    diff_prof: jnp.ndarray = None # (n_lev,)
    dmp_prof: jnp.ndarray = None # (n_lev,)
    
    # Semi-implicit matrices
    amhyb: jnp.ndarray = None # (n_lev, n_lev)
    bmhyb: jnp.ndarray = None # (n_lev, n_lev)
    tor_hyb: jnp.ndarray = None # (n_lev,)
    svhyb: jnp.ndarray = None # (n_lev,)
    d_hyb_m: jnp.ndarray = None # (3, L, n_lev, n_lev)
    
    dt: float = 1200.0

def advance(
    state: SpectralState, 
    phis_grads: tuple[jnp.ndarray, jnp.ndarray],
    dyn_config: DynamicsConfig,
    trans_config: TransformConfig,
    stepper_config: StepperConfig,
    latitudes: jnp.ndarray
) -> SpectralState:
    """
    Advances the spectral state by one timestep using 3-stage IMEX RK scheme.
    """
    dt = stepper_config.dt
    L = trans_config.L
    
    l_arr = jnp.arange(L)
    # laplacian operator in spectral space is -l(l+1) / R^2, matching Fortran's lap(n)
    # But wait, Fortran's lap is computed via SHTNS, which is -l(l+1). Fortran code does `lap(n)`
    # actually Fortran lap is indeed -l(l+1). The / R^2 is applied explicitly or included?
    # Fortran: `ddivdtlin_orig(n,:) = -lap(n) * (matmul(amhyb, virtempspec) + tor_hyb * lnpsspec)`
    # Let's define lap = -l(l+1).
    lap = -l_arr * (l_arr + 1.0)
    
    def compute_linear_tendencies(div, temp, lnps):
        # Temp linear tendency: dtvdtlin = -matmul(bmhyb, div)
        # div is (n_lev, L, 2L-1), bmhyb is (n_lev, n_lev). We matmul over levs.
        dtvdtlin = -jnp.einsum('ij,j...->i...', stepper_config.bmhyb, div)
        
        # lnps linear tendency: dlnpsdtlin = -sum(svhyb * div)
        dlnpsdtlin = -jnp.einsum('i,i...->...', stepper_config.svhyb, div)
        
        # div linear tendency: ddivdtlin = -lap * (matmul(amhyb, temp) + tor_hyb * lnps)
        temp_term = jnp.einsum('ij,j...->i...', stepper_config.amhyb, temp)
        lnps_term = stepper_config.tor_hyb[:, None, None] * lnps[None, :, :]
        ddivdtlin = -lap[None, :, None] * (temp_term + lnps_term)
        
        return ddivdtlin, dtvdtlin, dlnpsdtlin
    
    def solve_implicit(div_expl, temp_expl, lnps_expl, coeff, stage_idx):
        # rhs = div_expl - coeff * dt * lap * (matmul(amhyb, temp_expl) + tor_hyb * lnps_expl)
        temp_term = jnp.einsum('ij,j...->i...', stepper_config.amhyb, temp_expl)
        lnps_term = stepper_config.tor_hyb[:, None, None] * lnps_expl[None, :, :]
        rhs = div_expl - coeff * dt * lap[None, :, None] * (temp_term + lnps_term)
        
        # div_new = matmul(d_hyb_m, rhs)
        # d_hyb_m has shape (3, L, n_lev, n_lev) -> indexed by stage_idx, l.
        # rhs is (n_lev, L, 2L-1).
        # We need to apply d_hyb_m[stage_idx, l] to rhs[:, l, m].
        # Using einsum: 'lij,jlm->ilm'
        d_mat = stepper_config.d_hyb_m[stage_idx] # (L, n_lev, n_lev)
        div_new = jnp.einsum('lij,jlm->ilm', d_mat, rhs)
        
        # back substitute
        temp_new = temp_expl - coeff * dt * jnp.einsum('ij,j...->i...', stepper_config.bmhyb, div_new)
        lnps_new = lnps_expl - coeff * dt * jnp.einsum('i,i...->...', stepper_config.svhyb, div_new)
        
        return div_new, temp_new, lnps_new

    # --- Stage 1 ---
    tends_orig = get_spectral_tendencies(state, phis_grads, dyn_config, trans_config, latitudes)
    
    jax.debug.print("Max Temp Tend: {}", jnp.max(jnp.abs(tends_orig.d_temperature_d_t)))
    jax.debug.print("Max Div Tend: {}", jnp.max(jnp.abs(tends_orig.d_divergence_d_t)))
    jax.debug.print("Max LnPs Tend: {}", jnp.max(jnp.abs(tends_orig.d_log_surface_pressure_d_t)))
    
    vort1 = state.vorticity + stepper_config.a21 * dt * tends_orig.d_vorticity_d_t
    tracers1 = state.tracers + stepper_config.a21 * dt * tends_orig.d_tracers_d_t
    
    if stepper_config.explicit:
        div1 = state.divergence + stepper_config.a21 * dt * tends_orig.d_divergence_d_t
        temp1 = state.temperature + stepper_config.a21 * dt * tends_orig.d_temperature_d_t
        lnps1 = state.log_surface_pressure + stepper_config.a21 * dt * tends_orig.d_log_surface_pressure_d_t
        
        ddivdtlin_orig = jnp.zeros_like(tends_orig.d_divergence_d_t)
        dtvdtlin_orig = jnp.zeros_like(tends_orig.d_temperature_d_t)
        dlnpsdtlin_orig = jnp.zeros_like(tends_orig.d_log_surface_pressure_d_t)
    else:
        ddivdtlin_orig, dtvdtlin_orig, dlnpsdtlin_orig = compute_linear_tendencies(
            state.divergence, state.temperature, state.log_surface_pressure
        )
        ddivspecdt_orig_nl = tends_orig.d_divergence_d_t - ddivdtlin_orig
        dtvspecdt_orig_nl = tends_orig.d_temperature_d_t - dtvdtlin_orig
        dlnpsspecdt_orig_nl = tends_orig.d_log_surface_pressure_d_t - dlnpsdtlin_orig
        
        div_expl = state.divergence + dt * (stepper_config.a21 * ddivspecdt_orig_nl + stepper_config.aa21 * ddivdtlin_orig)
        temp_expl = state.temperature + dt * (stepper_config.a21 * dtvspecdt_orig_nl + stepper_config.aa21 * dtvdtlin_orig)
        lnps_expl = state.log_surface_pressure + dt * (stepper_config.a21 * dlnpsspecdt_orig_nl + stepper_config.aa21 * dlnpsdtlin_orig)
        
        div1, temp1, lnps1 = solve_implicit(div_expl, temp_expl, lnps_expl, stepper_config.aa22, 0)
        
        # Override tendencies with nonlinear part for subsequent stages
        tends_orig = SpectralTendencies(
            d_vorticity_d_t=tends_orig.d_vorticity_d_t,
            d_divergence_d_t=ddivspecdt_orig_nl,
            d_temperature_d_t=dtvspecdt_orig_nl,
            d_log_surface_pressure_d_t=dlnpsspecdt_orig_nl,
            d_tracers_d_t=tends_orig.d_tracers_d_t
        )

    state1 = SpectralState(vorticity=vort1, divergence=div1, temperature=temp1, log_surface_pressure=lnps1, tracers=tracers1)
    
    # --- Stage 2 ---
    tends1 = get_spectral_tendencies(state1, phis_grads, dyn_config, trans_config, latitudes)
    
    vort2 = state.vorticity + dt * (stepper_config.a31 * tends_orig.d_vorticity_d_t + stepper_config.a32 * tends1.d_vorticity_d_t)
    tracers2 = state.tracers + dt * (stepper_config.a31 * tends_orig.d_tracers_d_t + stepper_config.a32 * tends1.d_tracers_d_t)
    
    if stepper_config.explicit:
        div2 = state.divergence + dt * (stepper_config.a31 * tends_orig.d_divergence_d_t + stepper_config.a32 * tends1.d_divergence_d_t)
        temp2 = state.temperature + dt * (stepper_config.a31 * tends_orig.d_temperature_d_t + stepper_config.a32 * tends1.d_temperature_d_t)
        lnps2 = state.log_surface_pressure + dt * (stepper_config.a31 * tends_orig.d_log_surface_pressure_d_t + stepper_config.a32 * tends1.d_log_surface_pressure_d_t)
        
        ddivdtlin1 = jnp.zeros_like(tends1.d_divergence_d_t)
        dtvdtlin1 = jnp.zeros_like(tends1.d_temperature_d_t)
        dlnpsdtlin1 = jnp.zeros_like(tends1.d_log_surface_pressure_d_t)
    else:
        ddivdtlin1, dtvdtlin1, dlnpsdtlin1 = compute_linear_tendencies(div1, temp1, lnps1)
        ddivspecdt1_nl = tends1.d_divergence_d_t - ddivdtlin1
        dtvspecdt1_nl = tends1.d_temperature_d_t - dtvdtlin1
        dlnpsspecdt1_nl = tends1.d_log_surface_pressure_d_t - dlnpsdtlin1
        
        div_expl = state.divergence + dt * (stepper_config.a31 * tends_orig.d_divergence_d_t + stepper_config.aa31 * ddivdtlin_orig + 
                                            stepper_config.a32 * ddivspecdt1_nl + stepper_config.aa32 * ddivdtlin1)
        temp_expl = state.temperature + dt * (stepper_config.a31 * tends_orig.d_temperature_d_t + stepper_config.aa31 * dtvdtlin_orig + 
                                              stepper_config.a32 * dtvspecdt1_nl + stepper_config.aa32 * dtvdtlin1)
        lnps_expl = state.log_surface_pressure + dt * (stepper_config.a31 * tends_orig.d_log_surface_pressure_d_t + stepper_config.aa31 * dlnpsdtlin_orig + 
                                                       stepper_config.a32 * dlnpsspecdt1_nl + stepper_config.aa32 * dlnpsdtlin1)
        
        div2, temp2, lnps2 = solve_implicit(div_expl, temp_expl, lnps_expl, stepper_config.aa33, 1)
        
        tends1 = SpectralTendencies(
            d_vorticity_d_t=tends1.d_vorticity_d_t,
            d_divergence_d_t=ddivspecdt1_nl,
            d_temperature_d_t=dtvspecdt1_nl,
            d_log_surface_pressure_d_t=dlnpsspecdt1_nl,
            d_tracers_d_t=tends1.d_tracers_d_t
        )

    state2 = SpectralState(vorticity=vort2, divergence=div2, temperature=temp2, log_surface_pressure=lnps2, tracers=tracers2)
    
    # --- Stage 3 ---
    tends2 = get_spectral_tendencies(state2, phis_grads, dyn_config, trans_config, latitudes)
    
    vort3 = state.vorticity + dt * (stepper_config.b1 * tends_orig.d_vorticity_d_t + stepper_config.b2 * tends1.d_vorticity_d_t + stepper_config.b3 * tends2.d_vorticity_d_t)
    tracers3 = state.tracers + dt * (stepper_config.b1 * tends_orig.d_tracers_d_t + stepper_config.b2 * tends1.d_tracers_d_t + stepper_config.b3 * tends2.d_tracers_d_t)
    
    if stepper_config.explicit:
        div3 = state.divergence + dt * (stepper_config.b1 * tends_orig.d_divergence_d_t + stepper_config.b2 * tends1.d_divergence_d_t + stepper_config.b3 * tends2.d_divergence_d_t)
        temp3 = state.temperature + dt * (stepper_config.b1 * tends_orig.d_temperature_d_t + stepper_config.b2 * tends1.d_temperature_d_t + stepper_config.b3 * tends2.d_temperature_d_t)
        lnps3 = state.log_surface_pressure + dt * (stepper_config.b1 * tends_orig.d_log_surface_pressure_d_t + stepper_config.b2 * tends1.d_log_surface_pressure_d_t + stepper_config.b3 * tends2.d_log_surface_pressure_d_t)
    else:
        ddivdtlin2, dtvdtlin2, dlnpsdtlin2 = compute_linear_tendencies(div2, temp2, lnps2)
        ddivspecdt2_nl = tends2.d_divergence_d_t - ddivdtlin2
        dtvspecdt2_nl = tends2.d_temperature_d_t - dtvdtlin2
        dlnpsspecdt2_nl = tends2.d_log_surface_pressure_d_t - dlnpsdtlin2
        
        div_expl = state.divergence + dt * (stepper_config.b1 * tends_orig.d_divergence_d_t + stepper_config.bb1 * ddivdtlin_orig + 
                                            stepper_config.b2 * tends1.d_divergence_d_t + stepper_config.bb2 * ddivdtlin1 +
                                            stepper_config.b3 * ddivspecdt2_nl + stepper_config.bb3 * ddivdtlin2)
        temp_expl = state.temperature + dt * (stepper_config.b1 * tends_orig.d_temperature_d_t + stepper_config.bb1 * dtvdtlin_orig + 
                                              stepper_config.b2 * tends1.d_temperature_d_t + stepper_config.bb2 * dtvdtlin1 +
                                              stepper_config.b3 * dtvspecdt2_nl + stepper_config.bb3 * dtvdtlin2)
        lnps_expl = state.log_surface_pressure + dt * (stepper_config.b1 * tends_orig.d_log_surface_pressure_d_t + stepper_config.bb1 * dlnpsdtlin_orig + 
                                                       stepper_config.b2 * tends1.d_log_surface_pressure_d_t + stepper_config.bb2 * dlnpsdtlin1 +
                                                       stepper_config.b3 * dlnpsspecdt2_nl + stepper_config.bb3 * dlnpsdtlin2)
        
        if jnp.abs(stepper_config.bb4) > 1e-5:
            div3, temp3, lnps3 = solve_implicit(div_expl, temp_expl, lnps_expl, stepper_config.bb4, 2)
        else:
            div3, temp3, lnps3 = div_expl, temp_expl, lnps_expl
            
    # Forward implicit linear damping/diffusion
    if stepper_config.disspec is not None and stepper_config.diff_prof is not None:
        disspec = stepper_config.disspec[None, :, :]
        diff_prof = stepper_config.diff_prof[:, None, None]
        dmp_prof = stepper_config.dmp_prof[:, None, None] if stepper_config.dmp_prof is not None else jnp.zeros_like(diff_prof)
        
        # denom = 1.0 - (disspec * diff_prof - dmp_prof) * dt
        denom_vort_div = 1.0 - (disspec * diff_prof - dmp_prof) * dt
        denom_temp_tracers = 1.0 - (disspec * diff_prof) * dt
        
        vort3 = vort3 / denom_vort_div
        div3 = div3 / denom_vort_div
        temp3 = temp3 / denom_temp_tracers
        tracers3 = tracers3 / denom_temp_tracers[None, :, :, :]
        
    return SpectralState(vorticity=vort3, divergence=div3, temperature=temp3, log_surface_pressure=lnps3, tracers=tracers3)
