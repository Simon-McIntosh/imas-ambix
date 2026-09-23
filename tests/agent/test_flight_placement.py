"""The project flight layer declares clive worker placement.

The project flight layer merges over the host layer and is the only place the
placement of a clive worker is declared: it names the scheduler the placement
runs under and the two scheduler queries dispatch uses to ask about the job.
Dispatch resolves the job from the project's own published reservation record,
so the declaration carries a query with a ``{job}`` substitution point and no
literal identifier -- a literal would go stale on a resubmit or a cancel and
re-hold, silently, with the launch landing back on the login node while every
record still claimed it was placed.

These checks hold the declaration's shape.

They do not verify that a worker is in fact placed, and they cannot: that is
decided by the published reservation and by dispatch, not by this file. What
they catch is the declaration losing the property that makes placement
resolvable -- the scheduler, the queries, or the absence of a baked-in job
identifier.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml

FLIGHT_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "state"
    / "imas-ambix"
    / "flight.yaml"
)

# A run of five or more digits not adjacent to further digits is the shape a
# scheduler job identifier takes, wherever it appears. The declaration must
# never carry one; it must carry a query with a literal "{job}" for dispatch to
# substitute the resolved job into.
#
# The run is matched inside a string as well as across the whole of one, so an
# identifier embedded in a query argument ("--jobid=1276262") is caught and not
# only a bare scalar. It is also why an unquoted YAML integer must be checked
# separately: safe_load hands an unquoted 1276262 back as an int, so a check
# that only looks at str values passes the very thing this test forbids.
_JOB_IDENTIFIER_RUN = re.compile(r"(?<![0-9])[0-9]{5,}(?![0-9])")
# For a numeric scalar the whole decimal form must be the run: a job identifier
# is never negative and never carries a sign, so "-1276262" is not one.
_JOB_IDENTIFIER_INTEGER = re.compile(r"[0-9]{5,}")


def _is_job_identifier(value: Any) -> bool:
    # bool first: a boolean is an int in Python, and True must never read as an
    # equivalent of 1.
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return _JOB_IDENTIFIER_INTEGER.fullmatch(str(value)) is not None
    # A float reaches here for the same reason an int does: a scheduler
    # identifier written without a decimal point is not the only spelling
    # yaml.safe_load can produce, and "1276262.0" is the same job as
    # "1276262". A non-integral float is not an identifier at all.
    if isinstance(value, float) and value.is_integer():
        return _JOB_IDENTIFIER_INTEGER.fullmatch(str(int(value))) is not None
    if isinstance(value, str):
        return _JOB_IDENTIFIER_RUN.search(value) is not None
    return False


def _load_flight_config() -> dict[str, Any]:
    return yaml.safe_load(FLIGHT_CONFIG.read_text(encoding="utf-8"))


def _iter_scalars(node: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Yield every scalar in the document with its location.

    Keys are walked alongside values: a job identifier written as a mapping key
    is as absent from the resolver as one written as a value, and the
    declaration is checked at the document level rather than at one
    conveniently-dotted level.
    """
    if isinstance(node, dict):
        for key, value in node.items():
            yield from _iter_scalars(key, f"{path}/<key>")
            yield from _iter_scalars(value, f"{path}/{key}")
    elif isinstance(node, list):
        for index, item in enumerate(node):
            yield from _iter_scalars(item, f"{path}[{index}]")
    else:
        yield path or "<document>", node


def test_flight_config_is_present_and_parses() -> None:
    assert FLIGHT_CONFIG.is_file(), f"project flight layer not found at {FLIGHT_CONFIG}"
    assert isinstance(_load_flight_config(), dict)


def test_clive_placement_declares_the_scheduler() -> None:
    config = _load_flight_config()
    placement = config["backends"]["clive"]["placement"]
    assert placement["scheduler"] == "srun"


def test_placement_queries_carry_the_job_substitution_point() -> None:
    placement = _load_flight_config()["backends"]["clive"]["placement"]
    # The argv is asserted whole, not just for the presence of "{job}". The
    # output field is what a consumer reads the answer out of: the state query
    # must ask for the state and the reason query for the reason, and a change
    # to either field silently changes what is read. Swapping the state query's
    # %T for %j, %t or %R, or the reason query's %r for %N, leaves the job
    # substitution point intact while making a liveness reading that consumes
    # these answers read a running job as dead. Any change here is therefore a
    # deliberate change to what the fleet observes, and must be made as one.
    expected = {
        "state_query": ["squeue", "-h", "-j", "{job}", "-o", "%T"],
        "reason_query": ["squeue", "-h", "-j", "{job}", "-o", "%r"],
    }
    for name, argv in expected.items():
        assert placement[name] == argv, (
            f"{name} must be exactly {argv!r}: the output field is what reckon's "
            f"liveness and kill-class reading consumes, so a different field "
            f"changes what the fleet observes rather than merely how it is "
            f"spelled; got {placement[name]!r}"
        )


def test_no_job_identifier_is_written_into_the_declaration() -> None:
    offenders = [
        f"{path} = {value!r}"
        for path, value in _iter_scalars(_load_flight_config())
        if _is_job_identifier(value)
    ]
    assert not offenders, (
        "a scheduler job identifier must never be written into the project flight "
        "layer -- dispatch resolves the job from the published reservation, so a "
        "literal goes stale on a resubmit or a cancel and re-hold; found: "
        + "; ".join(offenders)
    )
