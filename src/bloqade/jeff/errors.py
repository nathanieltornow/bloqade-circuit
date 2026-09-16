"""This module holds the errors that the jeff package raises."""

from kirin import ir
from kirin.lowering import BuildError
from kirin.ir.exception import ValidationErrorGroup


class JeffImportError(BuildError):
    """Signal that the loader cannot import a file or module as a jeff program."""


class Refusal(ValidationErrorGroup, Exception):
    """An error group that a conversion raises when it cannot express a program."""

    def __init__(self, message: str, errors: list[ir.ValidationError]) -> None:
        """Group `errors` under one `message`."""
        super().__init__(message, errors)
        self.message = message

    def __str__(self) -> str:
        """Return the message followed by one indented line for each error."""
        return "\n".join([self.message, *(f"  {error}" for error in self.errors)])


class JeffToSquinError(ir.ValidationError):
    """An error that marks a jeff statement or block that squin cannot express."""
