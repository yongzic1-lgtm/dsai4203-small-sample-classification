"""scripts/package_deliverables.py: e-mail redaction of project.yaml in the team zip.

The addresses are built at run time (AT = "@") so that this file itself holds none and the
packager's own e-mail check stays clean.
"""

from __future__ import annotations

import pytest

from ssc import paths

AT = "@"


@pytest.fixture
def P(monkeypatch):
    monkeypatch.syspath_prepend(str(paths.ROOT / "scripts"))
    import package_deliverables
    return package_deliverables


def test_redact_emails_replaces_every_address(P):
    text = f"accounts:\n  pro1: first.last{AT}gmail.com   # default\n  pro2: x_y+2{AT}mail.uni.edu.hk\n"
    out, n = P.redact_emails(text)
    assert n == 2
    assert AT not in out
    assert out.count("<removed>") == 2
    assert "# default" in out


def test_redact_emails_leaves_other_text_alone(P):
    text = f"pip install pkg{AT}1.0 and R{AT}home are not addresses; neither is a{AT}b"
    out, n = P.redact_emails(text)
    assert (out, n) == (text, 0)


def test_emails_in_skips_ai_attribution_test_domains_and_binary(P, tmp_path):
    md = tmp_path / "note.md"
    md.write_text(f"Co-Authored-By: Claude <noreply{AT}anthropic.com>\ntest: a{AT}example.com\n"
                  f"contact: someone{AT}uni.edu.hk\n", encoding="utf-8")
    assert P.emails_in(md) == [f"someone{AT}uni.edu.hk"]
    png = tmp_path / "fig.png"
    png.write_bytes(f"PNG someone{AT}uni.edu.hk".encode())
    assert P.emails_in(png) == []
