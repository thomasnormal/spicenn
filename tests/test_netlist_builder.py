from pathlib import Path

from spice.netlist_builder import Netlist, param_ref, pwl, v


def test_netlist_builder_renders_common_spice_elements(tmp_path: Path):
    deck = Netlist("smoke deck")
    deck.param("CSTATE", 1e-12)
    deck.vsource("in", "in", "0", pwl([(0.0, 0.0), (1e-9, 1.0)]))
    deck.resistor("leak", "in", "0", "1G")
    deck.capacitor("state", "state", "0", param_ref("CSTATE"), ic=0)
    deck.bsource("copy", "state", "0", "V", v("in"))
    deck.options("method=gear", "maxord=2")
    deck.control("tran 1p 1n uic", "wrdata out.dat V(state)")
    deck.end()

    text = deck.render()
    assert text.startswith("* smoke deck\n")
    assert ".param CSTATE=1e-12" in text
    assert "Vin in 0 PWL(0 0 1e-09 1)" in text
    assert "Rleak in 0 1G" in text
    assert "Cstate state 0 {CSTATE} IC=0" in text
    assert "Bcopy state 0 V = V(in)" in text
    assert text.endswith(".end\n")

    out = tmp_path / "deck.cir"
    deck.write(out)
    assert out.read_text() == text


def test_bsource_rejects_unknown_kind():
    deck = Netlist()
    try:
        deck.bsource("bad", "x", "0", "P", "1")
    except ValueError as exc:
        assert "must be I or V" in str(exc)
    else:
        raise AssertionError("expected ValueError")
