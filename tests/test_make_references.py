"""Tests for scripts/make_references.py (numbered citations and the IEEE-style reference list)."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def M():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("make_references", ROOT / "scripts" / "make_references.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BIB = r"""
Some Markdown text.
```bibtex
@article{hanley1982,       % status: used (AUC standard error)
  author  = {Hanley, J. A. and McNeil, B. J.},
  title   = {The meaning and use of the area under a receiver operating characteristic ({ROC}) curve},
  journal = {Radiology}, volume = {143}, number = {1}, pages = {29--36}, year = {1982}
}
@book{hastie2009, % status: used
  author = {Hastie, T. and Tibshirani, R. and Friedman, J.},
  title = {The Elements of Statistical Learning}, edition = {2nd}, publisher = {Springer}, year = {2009}}
@article{sklearn, % status: used
  author = {Pedregosa, F. and Varoquaux, G. and others}, title = {Scikit-learn: Machine Learning in {P}ython},
  journal = {Journal of Machine Learning Research}, volume = {12}, pages = {2825--2830}, year = {2011}}
@misc{jax, % status: used
  author = {Bradbury, J. and Frostig, R.}, title = {{JAX}: Composable Transformations of {P}ython+{N}um{P}y Programs},
  howpublished = {\url{https://github.com/jax-ml/jax}}, year = {2018}}
@article{seaborn, % status: candidate
  author = {Waskom, M. L.}, title = {seaborn}, journal = {JOSS}, year = {2021}}
```
"""


@pytest.fixture(scope="module")
def REG(M):
    return {e["key"]: e for e in M.parse_bibtex(BIB)}


def test_entries_in_ieee_style(M, REG):
    assert REG["hanley1982"]["status"] == "used" and REG["seaborn"]["status"] == "candidate"
    assert M.format_entry(REG["hanley1982"]) == (
        "J. A. Hanley and B. J. McNeil, “The Meaning and Use of the Area Under a Receiver Operating "
        "Characteristic (ROC) Curve,” *Radiology*, vol. 143, no. 1, pp. 29–36, 1982.")
    assert M.format_entry(REG["hastie2009"]) == (
        "T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning*, 2nd ed. Springer, 2009.")
    assert M.format_entry(REG["sklearn"]).startswith(
        "F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” *J. Mach. Learn. Res.*, vol. 12,")
    assert M.format_entry(REG["jax"]).endswith(
        "2018. [Online]. Available: [https://github.com/jax-ml/jax](https://github.com/jax-ml/jax)")


def test_citations_are_numbered_by_first_appearance(M, REG):
    core = ("Scores [@hanley1982]. Trees [@sklearn; @hastie2009]. Again [@hanley1982].\n"
            "<!-- [@seaborn] in a comment is ignored --> `[@jax]` in code is ignored.")
    ext = "Leakage [@hastie2009, Sec. 7.10.2]; software [@jax; @sklearn; @hanley1982]; unknown [@nobody]."
    (c, e), order, warnings = M.resolve([core, ext], REG)
    assert order == ["hanley1982", "sklearn", "hastie2009", "jax"]
    assert c.startswith("Scores&nbsp;[1]. Trees&nbsp;[2, 3]. Again&nbsp;[1].")
    assert "<!-- [@seaborn] in a comment is ignored -->" in c and "`[@jax]`" in c
    assert "Leakage&nbsp;[3, Sec. 7.10.2]" in e and "software&nbsp;[1–2, 4]" not in e
    assert "software&nbsp;[1, 2, 4]" in e and "unknown&nbsp;[?]" in e      # runs of 3+ only become ranges
    assert warnings == ["citation key @nobody is not in registry/references.md"]
    assert M._compress([5, 3, 4, 9]) == "3–5, 9"


def test_reference_list_part(M, REG):
    md = M.render_list(["hanley1982", "sklearn"], REG)
    assert md.startswith("# References\n\n<!-- small -->\n[1] J. A. Hanley and B. J. McNeil,")
    body = md.split("<!-- small -->\n")[1].split("\n<!-- /small -->")[0]
    assert body.split("\n")[1].startswith("[2] F. Pedregosa et al.") and body.split("\n")[0].endswith("\\")
