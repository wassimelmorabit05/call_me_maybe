import json

from .models import FunctionCallResult, FunctionDefinition


def parse_arguments(arguments: str) -> dict[str, object]:
    """Parse generated JSON arguments."""

    data = json.loads(arguments)

    if not isinstance(data, dict):
        raise ValueError(
            "Generated arguments must be a JSON object."
        )

    return data


def validate_parameter_names(
    parameters: dict[str, object],
    function: FunctionDefinition,
) -> None:
    """Validate generated parameter names."""

    expected_names = set(function.parameters.keys())
    actual_names = set(parameters.keys())

    if actual_names != expected_names:
        raise ValueError(
            "Generated parameter names do not match "
            "the function definition."
        )


def validate_parameter_types(
    parameters: dict[str, object],
    function: FunctionDefinition,
) -> None:
    """Validate generated parameter types."""

    for name, definition in function.parameters.items():
        value = parameters[name]

        if definition.type == "string":
            if not isinstance(value, str):
                raise ValueError(
                    f"Parameter '{name}' must be a string."
                )

        elif definition.type == "number":
            if isinstance(value, bool) or not isinstance(
                value,
                (int, float),
            ):
                raise ValueError(
                    f"Parameter '{name}' must be a number."
                )

        elif definition.type == "integer":
            if isinstance(value, bool) or not isinstance(
                value,
                int,
            ):
                raise ValueError(
                    f"Parameter '{name}' must be an integer."
                )

        else:
            raise ValueError(
                f"Unsupported parameter type: {definition.type}"
            )


def validate_arguments(
    arguments: str,
    function: FunctionDefinition,
) -> dict[str, int | float | str]:
    """Parse and validate generated function arguments."""

    parameters = parse_arguments(arguments)

    validate_parameter_names(
        parameters,
        function,
    )

    validate_parameter_types(
        parameters,
        function,
    )

    validated: dict[str, int | float | str] = {}

    for name, definition in function.parameters.items():
        value = parameters[name]

        if definition.type == "number":
            if isinstance(value, (int, float)) and not isinstance(
                value,
                bool,
            ):
                validated[name] = float(value)

        elif definition.type == "integer":
            if isinstance(value, int) and not isinstance(
                value,
                bool,
            ):
                validated[name] = value

        elif definition.type == "string":
            if isinstance(value, str):
                validated[name] = value

    return validated


def build_function_call_result(
    user_prompt: str,
    function: FunctionDefinition,
    arguments: str,
) -> FunctionCallResult:
    """Build a validated function call result."""

    parameters = validate_arguments(
        arguments,
        function,
    )

    return FunctionCallResult(
        prompt=user_prompt,
        name=function.name,
        parameters=parameters,
    )
