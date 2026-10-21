"""Small shared validation helpers; independent of Torch and Hugging Face."""


def format_errors(errors: dict[str, str], heading: str) -> str:
    return heading + ":\n" + "\n".join(f"  - {name}: {message}" for name, message in errors.items())


class ValidationError(ValueError):
    heading = "Invalid values"

    def __init__(self, errors: dict[str, str]):
        self.errors = errors
        super().__init__(format_errors(errors, self.heading))


def validate_rules(rules: dict[str, tuple[bool, str]], *, error_type=ValidationError) -> None:
    """Collect independent failures. Validate prerequisites before dependent rules."""
    errors = {name: message for name, (valid, message) in rules.items() if not valid}
    if errors:
        raise error_type(errors)
