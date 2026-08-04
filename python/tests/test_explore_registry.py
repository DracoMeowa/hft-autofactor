"""Explore-lane registry tests: metadata completeness, naming, spec loading."""
from __future__ import annotations

import textwrap
from pathlib import Path

import polars as pl
import pytest

from hft_autofactor.explore.registry import (
    Prototype,
    PrototypeError,
    PrototypeRegistry,
    default_registry,
    explore_prototype,
    load_prototype_spec,
)


def _compute(part: pl.DataFrame) -> pl.Series:
    return part["mid_px"]


def _kwargs(**overrides):
    base = dict(
        name="test_proto",
        mechanism="an economic mechanism",
        info_set="mid_px",
        inspiration="a prior paper",
        compute=_compute,
    )
    base.update(overrides)
    return base


# --------------------------------------------------------------------- #
# metadata completeness gate                                            #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("missing", ["name", "mechanism", "info_set", "inspiration"])
def test_registration_refuses_missing_metadata(missing):
    kwargs = _kwargs(**{missing: None})
    with pytest.raises(PrototypeError, match=missing):
        explore_prototype(**kwargs)


@pytest.mark.parametrize("blank", ["", "   ", "\n"])
def test_registration_refuses_blank_metadata(blank):
    with pytest.raises(PrototypeError, match="mechanism"):
        explore_prototype(**_kwargs(mechanism=blank))


def test_registration_refuses_non_callable_compute():
    with pytest.raises(PrototypeError, match="compute"):
        explore_prototype(**_kwargs(compute="pl.col('mid_px')"))


def test_registration_error_lists_every_missing_field():
    with pytest.raises(PrototypeError) as excinfo:
        explore_prototype(
            name="", mechanism=None, info_set="", inspiration=None, compute=None
        )
    msg = str(excinfo.value)
    for field in ("name", "mechanism", "info_set", "inspiration", "compute"):
        assert field in msg


# --------------------------------------------------------------------- #
# name rules                                                            #
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "bad_name", ["OIR", "9lives", "with-dash", "has space", "_lead"]
)
def test_name_must_be_lower_snake_case(bad_name):
    with pytest.raises(PrototypeError, match="invalid"):
        explore_prototype(**_kwargs(name=bad_name))


@pytest.mark.parametrize(
    "reserved",
    ["oir", "mid_px", "ts_ms", "channel", "fwd_mid_ret_15s", "rv_60s"],
)
def test_name_must_not_shadow_panel_columns(reserved):
    with pytest.raises(PrototypeError, match="reserved"):
        explore_prototype(**_kwargs(name=reserved))


# --------------------------------------------------------------------- #
# registry behavior                                                     #
# --------------------------------------------------------------------- #
def test_registry_register_get_contains_iter():
    reg = PrototypeRegistry()
    proto = explore_prototype(**_kwargs())
    reg.register(proto)
    assert "test_proto" in reg
    assert reg.get("test_proto") is proto
    assert reg.names() == ["test_proto"]
    assert list(reg) == [proto]
    assert len(reg) == 1


def test_registry_refuses_duplicates_unless_overwrite():
    reg = PrototypeRegistry([explore_prototype(**_kwargs())])
    with pytest.raises(PrototypeError, match="already registered"):
        reg.register(explore_prototype(**_kwargs(mechanism="other")))
    reg.register(explore_prototype(**_kwargs(mechanism="other")), overwrite=True)
    assert reg.get("test_proto").mechanism == "other"


def test_registry_get_unknown_raises_with_available_names():
    reg = PrototypeRegistry([explore_prototype(**_kwargs())])
    with pytest.raises(PrototypeError, match="test_proto"):
        reg.get("nope")


def test_registry_register_rejects_non_prototype():
    reg = PrototypeRegistry()
    with pytest.raises(PrototypeError, match="expected a Prototype"):
        reg.register({"name": "x"})  # type: ignore[arg-type]


# --------------------------------------------------------------------- #
# built-in seed registry                                                #
# --------------------------------------------------------------------- #
def test_default_registry_is_metadata_complete():
    reg = default_registry()
    assert len(reg) >= 3
    for proto in reg:
        assert proto.mechanism.strip()
        assert proto.info_set.strip()
        assert proto.inspiration.strip()
        assert callable(proto.compute)
        assert proto.source == "builtin"


def test_default_prototypes_pass_causality_on_synthetic_panel():
    from hft_autofactor.explore.causality import panel_prefix_check

    n = 40
    rows = []
    for date in ("20250602", "20250603"):
        for inst in ("A", "B"):
            for i in range(n):
                rows.append(
                    {
                        "date": date,
                        "exchange": "sse",
                        "instrument": inst,
                        "ts_ms": 34_200_000 + i * 3000,
                        "snap_seq": i,
                        "flags": 0,
                        "mid_px": 4.0 + 0.001 * i,
                        "last_px": 4.0,
                        "bid1_px": 3.999,
                        "ask1_px": 4.001,
                        "bid1_qty": 100,
                        "ask1_qty": 90,
                        "depth_bid5": 1000 + i,
                        "depth_ask5": 900,
                        "quoted_spread_ticks": 2.0,
                    }
                )
    panel = pl.DataFrame(rows)
    for proto in default_registry():
        report = panel_prefix_check(panel, proto, k=4)
        assert report.passed, f"{proto.name}: {report.diffs}"


# --------------------------------------------------------------------- #
# spec-file loading                                                     #
# --------------------------------------------------------------------- #
_SPEC_TEMPLATE = textwrap.dedent(
    """
    import polars as pl
    from hft_autofactor.explore.registry import explore_prototype

    PROTOTYPE = explore_prototype(
        name="spec_proto",
        mechanism="spec mechanism",
        info_set="mid_px",
        inspiration="spec inspiration",
        compute=lambda part: part["mid_px"].diff(1),
    )
    """
)


def test_load_prototype_spec_object_form(tmp_path):
    spec = tmp_path / "spec_proto.py"
    spec.write_text(_SPEC_TEMPLATE, encoding="utf-8")
    proto = load_prototype_spec(spec)
    assert isinstance(proto, Prototype)
    assert proto.name == "spec_proto"
    assert proto.source == str(spec)


def test_load_prototype_spec_dict_form(tmp_path):
    spec = tmp_path / "dict_proto.py"
    spec.write_text(
        textwrap.dedent(
            """
            PROTOTYPE = dict(
                name="dict_proto",
                mechanism="m",
                info_set="mid_px",
                inspiration="i",
                compute=lambda part: part["mid_px"],
            )
            """
        ),
        encoding="utf-8",
    )
    proto = load_prototype_spec(spec)
    assert proto.name == "dict_proto"


@pytest.mark.parametrize(
    "body, match",
    [
        ("X = 1\n", "must define PROTOTYPE"),
        (
            "PROTOTYPE = dict(name='p', mechanism='m', info_set='i', "
            "inspiration='', compute=lambda part: part['mid_px'])\n",
            "inspiration",
        ),
        ("raise RuntimeError('boom')\n", "failed to import"),
    ],
)
def test_load_prototype_spec_refusal_modes(tmp_path, body, match):
    spec = tmp_path / "bad.py"
    spec.write_text(body, encoding="utf-8")
    with pytest.raises(PrototypeError, match=match):
        load_prototype_spec(spec)


def test_load_prototype_spec_missing_file(tmp_path):
    with pytest.raises(PrototypeError, match="not found"):
        load_prototype_spec(tmp_path / "absent.py")
