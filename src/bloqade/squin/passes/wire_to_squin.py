"""Convert wire methods and their callees back to squin."""

from dataclasses import dataclass

from kirin import ir

from bloqade.squin.rewrite.wire.from_wire import FromWire


@dataclass
class WireToSquin:
    """Convert a wire squin method and every wire callee back to squin kernels."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return a squin kernel that runs the same program as `method`."""
        rewrite = FromWire()
        return rewrite.convert(method)
