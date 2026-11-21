"""Small, reusable assertions/helpers for V.92 DCE qualification."""

from __future__ import annotations

from v8 import V8_PROTOCOL_LAPM_V42


def lapm_detect_required(protocols: int, *, force_detect: bool = False) -> bool:
    """Return whether ODP/ADP detection is needed after V.8 negotiation.

    A common V.8 LAPM protocol selection means both sides start LAPM directly.
    ``force_detect`` exists only for controlled comparison captures.
    """
    if isinstance(protocols, bool) or not isinstance(protocols, int):
        raise TypeError("V.8 protocol mask must be an integer")
    if type(force_detect) is not bool:
        raise TypeError("force_detect must be bool")
    return force_detect or not bool(protocols & V8_PROTOCOL_LAPM_V42)


def lapm_payload_drained(link) -> bool:
    """Whether application data, rather than control acknowledgements, drained."""
    if link is None:
        return False
    stats = getattr(link, "stats", None)
    if not isinstance(stats, dict):
        return False
    return bool(link.pending() == 0 and not link._unacked
                and stats.get("wire_out") == stats.get("wire_acked"))
