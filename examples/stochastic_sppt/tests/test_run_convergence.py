import argparse
import pickle

import pytest

from examples.stochastic_sppt.figures import run_convergence


def test_check_sig_guard(tmp_path):
    p = tmp_path / "art.pkl"
    # matching signature -> no raise
    run_convergence._check_sig("dataset", {"draws": 2}, {"draws": 2}, p)
    # missing signature (artifact built before the guard) -> warn + reuse, no raise
    run_convergence._check_sig("dataset", None, {"draws": 2}, p)
    # mismatched signature -> refuse, naming the difference
    with pytest.raises(RuntimeError, match="different config"):
        run_convergence._check_sig("dataset", {"draws": 2}, {"draws": 3}, p)


def test_ensure_dataset_rejects_stale_config(tmp_path, monkeypatch):
    # A cached dataset built with draws=2, but the caller now asks for draws=3:
    # the guard must refuse BEFORE any (expensive) rebuild rather than silently
    # training on the wrong dataset.
    ds = tmp_path / "ds.pkl"
    monkeypatch.setattr(run_convergence, "DATASET", ds)
    with open(ds, "wb") as f:
        pickle.dump({"truth": {}, "_sig": {"cases": 2, "draws": 2,
                     "lead_days": 1, "spinup_days": 1.0}}, f)
    a = argparse.Namespace(cases=2, draws=3, members=2, lead_days=1, spinup_days=1.0)
    with pytest.raises(RuntimeError, match="different config"):
        run_convergence.ensure_dataset(a)


def test_convergence_smoke_and_resume(tmp_path, monkeypatch):
    # Point artifacts at a temp dir so the test is hermetic.
    monkeypatch.setattr(run_convergence, "ART", tmp_path)
    monkeypatch.setattr(run_convergence, "DATASET", tmp_path / "ds.pkl")
    monkeypatch.setattr(run_convergence, "CKPT", tmp_path / "ck.pkl")

    argv = ["--cases", "2", "--draws", "2", "--members", "2", "--batch", "2",
            "--target-steps", "1", "--lead-days", "1", "--spinup-days", "1",
            "--budget-seconds", "600"]
    run_convergence.main(argv)
    assert (tmp_path / "ck.pkl").exists()
    assert (tmp_path / "ds.pkl").exists()

    # Re-invoking with a higher target resumes from the checkpoint (step advances).
    import pickle
    with open(tmp_path / "ck.pkl", "rb") as f:
        step_after_first = pickle.load(f)["step"]
    assert step_after_first == 1
    run_convergence.main(argv[:-2] + ["--target-steps", "2", "--budget-seconds", "600"])
    with open(tmp_path / "ck.pkl", "rb") as f:
        assert pickle.load(f)["step"] == 2
