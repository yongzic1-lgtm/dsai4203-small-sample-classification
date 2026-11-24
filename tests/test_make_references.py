"""Tests for scripts/make_references.py (reference list of the report)."""
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
@misc{jax, % status: used
  author = {Bradbury, J. and Frostig, R. and Hawkins, P. and Johnson, M. J. and Leary, C. and Maclaurin, D. and Necula, G.},
  title = {{JAX}: Composable Transformations of {P}ython+{N}um{P}y Programs},
  howpublished = {\url{https://github.com/jax-ml/jax}}, year = {2018}}
@article{kohavi1995, % status: candidate (cited by E1 only)
  author = {Kohavi, R.}, title = {A Study of Cross-Validation}, journal = {Proc. IJCAI}, year = {1995}}
@article{seaborn, % status: candidate
  author = {Waskom, M. L.}, title = {seaborn}, journal = {JOSS}, year = {2021}}
```
"""


def test_entries_are_complete_and_in_one_style(M):
    E = {e["key"]: e for e in M.parse_bibtex(BIB)}
    assert E["hanley1982"]["status"] == "used" and E["kohavi1995"]["status"] == "candidate"
    assert M.format_entry(E["hanley1982"]) == (
        "Hanley, J. A., & McNeil, B. J. (1982). The Meaning and Use of the Area Under a Receiver Operating "
        "Characteristic (ROC) Curve. *Radiology*, *143*(1), 29–36.")
    assert M.format_entry(E["hastie2009"]) == (
        "Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Springer.")
    jax = M.format_entry(E["jax"])                       # seven authors: six, then et al.; URL as a link
    assert jax.startswith("Bradbury, J., Frostig, R., Hawkins, P., Johnson, M. J., Leary, C., Maclaurin, D., et al. (2018).")
    assert jax.endswith("[https://github.com/jax-ml/jax](https://github.com/jax-ml/jax)")


def test_selection_follows_status_and_citations(M):
    entries = M.parse_bibtex(BIB)
    cited = M.citations("As shown by Hanley and McNeil (1982) and (Kohavi, 1995; Hastie et al., 2009, Sec. 7). "
                        "<!-- (Waskom, 2021) in a comment does not count --> seed 2026, Run 20261022-063908.")
    assert cited == {("hanley", "1982"), ("kohavi", "1995"), ("hastie", "2009")}
    chosen, warnings = M.select(entries, cited | {("nobody", "1999")})
    assert [e["key"] for e in chosen] == ["jax", "hanley1982", "hastie2009", "kohavi1995"]   # sorted by surname
    assert any("nobody 1999" in w for w in warnings)                  # a citation without an entry
    assert any(w.startswith("jax is listed") for w in warnings)        # a used entry that no part cites
    assert not any("seaborn" in w for w in warnings)                   # uncited candidates are simply left out


def test_rendered_list_is_one_hard_broken_block_in_small_type(M):
    entries = [e for e in M.parse_bibtex(BIB) if e["status"] == "used"]
    md = M.render(sorted(entries, key=M.sort_key))
    assert "# References" in md and md.count("<!-- small -->") == 1 and md.count("<!-- /small -->") == 1
    body = md.split("<!-- small -->\n")[1].split("\n<!-- /small -->")[0]
    lines = body.split("\n")
    assert len(lines) == 3 and all(l.endswith("\\") for l in lines[:-1]) and not lines[-1].endswith("\\")
