"""Convert wire methods and their callees back to squin."""

from dataclasses import dataclass

from kirin import ir
from kirin.rewrite import Walk

from bloqade.squin.rewrite.wire.driver import reachable
from bloqade.squin.rewrite.wire.from_wire import FromWire
from bloqade.squin.rewrite.wire.normalize import BoundListLengths
from bloqade.squin.analysis.validation.wire import WireValidation


@dataclass
class WireToSquin:
    """Convert a wire squin method and every wire callee back to squin kernels."""

    def emit(self, method: ir.Method) -> ir.Method:
        """Return a squin kernel that runs the same program as `method`."""
        for callee in reachable(method):
            WireValidation().check(callee)
        rewrite = FromWire()
        result = rewrite.convert(method)
        for copy in rewrite.kernels.values():
            Walk(BoundListLengths()).rewrite(copy.code)
            copy.verify()
            copy.verify_type()
        return result
