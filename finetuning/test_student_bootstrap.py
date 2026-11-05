"""Tests for finetuning/student_bootstrap.py on small synthetic arrays (no data files)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import student_bootstrap as SB  # noqa: E402


def test_metrics_single_select():
    # three classes as one-hot rows; 3 of 4 right
    y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
    p = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0], [1, 0, 0]], dtype=bool)
    m = SB.metrics(y, p, single=True)
    assert m["exact"] == 0.75
    # F1: class0 2*2/(4+1)=0.8, class1 1.0, class2 0.0 -> mean 0.6
    assert abs(m["macro_f1"] - 0.6) < 1e-12
    assert "micro_f1" not in m


def test_metrics_multi_select_and_weights():
    y = np.array([[1, 1, 0], [0, 1, 0]], dtype=bool)
    p = np.array([[1, 0, 0], [0, 1, 1]], dtype=bool)
    m = SB.metrics(y, p, single=False)
    assert m["exact"] == 0.0
    # tp=2, fp=1, fn=1 -> micro 4/6
    assert abs(m["micro_f1"] - 4 / 6) < 1e-12
    # a weight of 2 on the second row equals duplicating it
    w = SB.metrics(y, p, single=False, w=np.array([1.0, 2.0]))
    d = SB.metrics(np.vstack([y, y[1:]]), np.vstack([p, p[1:]]), single=False)
    assert all(abs(w[k] - d[k]) < 1e-12 for k in w)


def test_binomial():
    assert SB.binom_two_sided(0, 0) == 1.0
    assert abs(SB.binom_two_sided(1, 1) - 1.0) < 1e-12
    assert SB.binom_two_sided(0, 10) < 0.002


def test_student_key():
    assert SB.student("P000_P2_01") == "P000"
