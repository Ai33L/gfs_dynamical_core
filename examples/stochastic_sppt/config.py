"""Configuration dataclasses for the trainable-SPPT pipeline."""
from dataclasses import dataclass, field

# (L, ntrunc, dt_seconds). L = s2fft bandlimit (= n_lat, n_lon = 2L-1);
# ntrunc = physical triangular truncation; dt chosen for CFL at each resolution.
RESOLUTIONS: dict[str, tuple[int, int, float]] = {
    "T21": (32, 21, 1800.0),
    "T42": (64, 42, 1200.0),
    "T85": (128, 85, 600.0),
    "T127": (192, 127, 450.0),
}


@dataclass(frozen=True)
class ModelConfig:
    resolution: str = "T42"
    n_lev: int = 20

    @property
    def L(self) -> int:
        return RESOLUTIONS[self.resolution][0]

    @property
    def ntrunc(self) -> int:
        return RESOLUTIONS[self.resolution][1]

    @property
    def dt(self) -> float:
        return RESOLUTIONS[self.resolution][2]


@dataclass(frozen=True)
class ExperimentConfig:
    mode: str                       # "A" (parameter recovery) or "B" (model-error)
    forecast_resolution: str        # e.g. "T42"
    truth_resolution: str           # e.g. "T127" (Mode B) or == forecast (Mode A)
    n_members: int = 8
    lead_days: tuple = (3, 5, 7, 10)
    n_cases: int = 16
    spinup_days: float = 200.0
    case_stride_days: float = 5.0
    ic_perturb_amp: float = 0.0     # 0 => SPPT-only spread
    seed: int = 0


@dataclass(frozen=True)
class TrainConfig:
    alpha: float = 0.95
    lr: float = 5e-2
    n_opt_steps: int = 300
    batch_cases: int = 4
    log_every: int = 10
    checkpoint_dir: str = "sppt_checkpoints"
    seed: int = 0
