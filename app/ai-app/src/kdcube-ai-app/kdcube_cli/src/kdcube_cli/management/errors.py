from __future__ import annotations


class ManagementCliError(RuntimeError):
    """A fixed, user-safe KDCube management failure."""

    def __init__(self, code: str, message: str, *, exit_code: int = 2) -> None:
        super().__init__(message)
        self.code = str(code or "management_error")
        self.message = str(message or "KDCube could not complete the request.")
        self.exit_code = int(exit_code)

    def __str__(self) -> str:
        return self.message


__all__ = ["ManagementCliError"]
