from __future__ import annotations

from pathlib import Path
import sys
import site
import re


def find_site_packages() -> Path:
    candidates: list[Path] = []

    try:
        candidates.extend(Path(p) for p in site.getsitepackages())
    except Exception:
        pass

    try:
        user_site = site.getusersitepackages()
        if user_site:
            candidates.append(Path(user_site))
    except Exception:
        pass

    for p in sys.path:
        if "site-packages" in p:
            candidates.append(Path(p))

    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for sp in unique_candidates:
        if (sp / "pySOT").exists():
            return sp

    raise FileNotFoundError("Nepodarilo sa nájsť site-packages s balíkom pySOT.")


def backup_file(path: Path) -> None:
    backup = path.with_suffix(path.suffix + ".bak")
    if not backup.exists():
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")


def replace_exact(text: str, old: str, new: str, label: str) -> tuple[str, bool]:
    if old in text:
        return text.replace(old, new), True
    if new in text:
        return text, False
    raise ValueError(f"Nepodarilo sa nájsť očakávaný blok pre patch: {label}")


def ensure_contains(text: str, needle: str, label: str) -> None:
    if needle not in text:
        raise ValueError(f"V súbore po patchi chýba očakávaný obsah: {label}")


def patch_ei_merit(site_packages: Path) -> str:
    path = site_packages / "pySOT" / "auxiliary_problems" / "ei_merit.py"
    backup_file(path)
    text = path.read_text(encoding="utf-8")

    old = """import numpy as np
import scipy.spatial as scpspatial
from scipy.stats import norm


def ei_merit(X, surrogate, fX, XX=None, dtol=0):
    \"\"\"Compute the expected improvement merit function.

    :param X: Points where to compute EI, of size n x dim
    :type X: numpy.array
    :param surrogate: Surrogate model object, must implement predict_std
    :type surrogate: object
    :param fX: Values at previously evaluated points, of size m x 1
    :type fX: numpy.array
    :param XX: Previously evaluated points, of size m x 1
    :type XX: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float

    :return: Evaluate the expected improvement for points X
    :rtype: numpy.array of length X.shape[0]
    \"\"\"
    mu, sig = surrogate.predict(X), surrogate.predict_std(X)
    gamma = (np.min(fX) - mu) / sig
    beta = gamma * norm.cdf(gamma) + norm.pdf(gamma)
    ei = sig * beta

    if dtol > 0:
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(dists, axis=1, keepdims=True)
        ei[dmerit < dtol] = 0.0
    return ei
"""
    new = """import numpy as np
import scipy.spatial as scpspatial
from scipy.stats import norm


def ei_merit(X, surrogate, fX, XX=None, dtol=0):
    \"\"\"Compute the expected improvement merit function.

    :param X: Points where to compute EI, of size n x dim
    :type X: numpy.array
    :param surrogate: Surrogate model object, must implement predict_std
    :type surrogate: object
    :param fX: Values at previously evaluated points, of size m x 1
    :type fX: numpy.array
    :param XX: Previously evaluated points, of size m x 1
    :type XX: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float

    :return: Evaluate the expected improvement for points X
    :rtype: numpy.array of length X.shape[0]
    \"\"\"
    X = np.asarray(X)
    fX = np.asarray(fX).reshape(-1)

    mu = np.asarray(surrogate.predict(X)).reshape(-1)
    sig = np.asarray(surrogate.predict_std(X)).reshape(-1)

    sig = np.maximum(sig, 1e-12)

    gamma = (np.min(fX) - mu) / sig
    beta = gamma * norm.cdf(gamma) + norm.pdf(gamma)
    ei = sig * beta
    ei = np.asarray(ei).reshape(-1)

    if dtol > 0 and XX is not None:
        XX = np.asarray(XX)
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(dists, axis=1)
        ei[dmerit < dtol] = 0.0
    return ei
"""
    text, _ = replace_exact(text, old, new, "ei_merit full block")
    ensure_contains(text, "fX = np.asarray(fX).reshape(-1)", "ei_merit flatten fX")
    ensure_contains(text, "dmerit = np.amin(dists, axis=1)", "ei_merit dmerit 1D")
    path.write_text(text, encoding="utf-8")
    return str(path)


def patch_lcb_merit(site_packages: Path) -> str:
    path = site_packages / "pySOT" / "auxiliary_problems" / "lcb_merit.py"
    backup_file(path)
    text = path.read_text(encoding="utf-8")

    old = """import numpy as np
import scipy.spatial as scpspatial


def lcb_merit(X, surrogate, fX, XX=None, dtol=0.0, kappa=2.0):
    \"\"\"Compute the lcb merit function.

    :param X: Points where to compute LCB, of size n x dim
    :type X: numpy.array
    :param surrogate: Surrogate model object, must implement predict_std
    :type surrogate: object
    :param fX: Values at previously evaluated points, of size m x 1
    :type fX: numpy.array
    :param XX: Previously evaluated points, of size m x 1
    :type XX: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float
    :param kappa: Constant in front of standard deviation
        Default: 2.0
    :type kappa: float

    :return: Evaluate the lower confidence bound for points X
    :rtype: numpy.array of length X.shape[0]
    \"\"\"
    mu, sig = surrogate.predict(X), surrogate.predict_std(X)
    lcb = mu - kappa * sig

    if dtol > 0:
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(dists, axis=1, keepdims=True)
        lcb[dmerit < dtol] = np.inf
    return lcb
"""
    new = """import numpy as np
import scipy.spatial as scpspatial


def lcb_merit(X, surrogate, fX, XX=None, dtol=0.0, kappa=2.0):
    \"\"\"Compute the lcb merit function.

    :param X: Points where to compute LCB, of size n x dim
    :type X: numpy.array
    :param surrogate: Surrogate model object, must implement predict_std
    :type surrogate: object
    :param fX: Values at previously evaluated points, of size m x 1
    :type fX: numpy.array
    :param XX: Previously evaluated points, of size m x 1
    :type XX: numpy.array
    :param dtol: Minimum distance between evaluated and pending points
    :type dtol: float
    :param kappa: Constant in front of standard deviation
        Default: 2.0
    :type kappa: float

    :return: Evaluate the lower confidence bound for points X
    :rtype: numpy.array of length X.shape[0]
    \"\"\"
    X = np.asarray(X)

    mu = np.asarray(surrogate.predict(X)).reshape(-1)
    sig = np.asarray(surrogate.predict_std(X)).reshape(-1)

    lcb = mu - kappa * sig
    lcb = np.asarray(lcb).reshape(-1)

    if dtol > 0 and XX is not None:
        XX = np.asarray(XX)
        dists = scpspatial.distance.cdist(X, XX)
        dmerit = np.amin(dists, axis=1)
        lcb[dmerit < dtol] = np.inf
    return lcb
"""
    text, _ = replace_exact(text, old, new, "lcb_merit full block")
    ensure_contains(text, "mu = np.asarray(surrogate.predict(X)).reshape(-1)", "lcb_merit flatten mu")
    ensure_contains(text, "dmerit = np.amin(dists, axis=1)", "lcb_merit dmerit 1D")
    path.write_text(text, encoding="utf-8")
    return str(path)


def patch_candidate_srbf(site_packages: Path) -> str:
    path = site_packages / "pySOT" / "auxiliary_problems" / "candidate_srbf.py"
    backup_file(path)
    text = path.read_text(encoding="utf-8")

    text = text.replace(
        "dmerit = np.amin(dists, axis=1, keepdims=True)",
        "dmerit = np.amin(dists, axis=1)",
    )

    if "merit = np.asarray(merit).reshape(-1)" not in text:
        text = text.replace(
            "merit = w * fvals + (1.0 - w) * (1.0 - unit_rescale(np.copy(dmerit)))",
            "merit = w * fvals + (1.0 - w) * (1.0 - unit_rescale(np.copy(dmerit)))\n"
            "    merit = np.asarray(merit).reshape(-1)\n"
            "    dmerit = np.asarray(dmerit).reshape(-1)",
        )

    ensure_contains(text, "merit = np.asarray(merit).reshape(-1)", "candidate_srbf flatten merit")
    ensure_contains(text, "dmerit = np.asarray(dmerit).reshape(-1)", "candidate_srbf flatten dmerit")
    path.write_text(text, encoding="utf-8")
    return str(path)


def patch_utils(site_packages: Path) -> str:
    path = site_packages / "pySOT" / "utils.py"
    backup_file(path)
    text = path.read_text(encoding="utf-8")

    text = text.replace("dtype=np.int)", "dtype=int)")
    text = text.replace("dtype=np.int ", "dtype=int ")
    text = text.replace("dtype=np.int,", "dtype=int,")

    ensure_contains(text, "if not (len(np.intersect1d(contvar, intvar)) == 0):", "utils intersect1d intact")
    if "dtype=np.int" in text:
        raise ValueError("V utils.py zostal neopravený výskyt dtype=np.int")

    path.write_text(text, encoding="utf-8")
    return str(path)


def patch_sop_strategy(site_packages: Path) -> str:
    path = site_packages / "pySOT" / "strategy" / "sop_strategy.py"
    backup_file(path)
    text = path.read_text(encoding="utf-8")

    text = text.replace("dtype=np.int)", "dtype=int)")
    text = text.replace("dtype=np.int ", "dtype=int ")
    text = text.replace("dtype=np.int,", "dtype=int,")

    if "dtype=np.int" in text:
        raise ValueError("V sop_strategy.py zostal neopravený výskyt dtype=np.int")

    path.write_text(text, encoding="utf-8")
    return str(path)


def main() -> None:
    site_packages = find_site_packages()
    patched: list[str] = []

    patched.append(patch_ei_merit(site_packages))
    patched.append(patch_lcb_merit(site_packages))
    patched.append(patch_candidate_srbf(site_packages))
    patched.append(patch_utils(site_packages))
    patched.append(patch_sop_strategy(site_packages))

    print("pySOT patches applied successfully:")
    for p in patched:
        print(f" - {p}")


if __name__ == "__main__":
    main()