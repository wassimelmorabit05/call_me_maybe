from .models import FunctionDefinition


FEW_SHOT = (
    "User request: What is the sum of 6 and 7?\n\n"
    "Function: fn_add\n"
    "Description: Add two numbers.\n"
    "Parameters:\n"
    "- a: number\n"
    "- b: number\n\n"
    'JSON arguments: {"a":6,"b":7}\n\n'
    "User request: Replace the word 'red' with 'blue' in 'a red car'\n\n"
    "Function: fn_replace\n"
    "Description: Replace text using a regular expression.\n"
    "Parameters:\n"
    "- text: string\n"
    "- pattern: string\n"
    "- replacement: string\n\n"
    'JSON arguments: {"text":"a red car","pattern":"red",'
    '"replacement":"blue"}\n\n'
    'User request: Replace all numbers in "Room 12 costs 30 dollars" '
    "with X\n\n"
    "Function: fn_replace\n"
    "Description: Replace text using a regular expression.\n"
    "Parameters:\n"
    "- text: string\n"
    "- pattern: string\n"
    "- replacement: string\n\n"
    'JSON arguments: {"text":"Room 12 costs 30 dollars",'
    '"pattern":"\\\\d+","replacement":"X"}\n\n'
    "User request: Replace all spaces in 'a b c' with underscores\n\n"
    "Function: fn_replace\n"
    "Description: Replace text using a regular expression.\n"
    "Parameters:\n"
    "- text: string\n"
    "- pattern: string\n"
    "- replacement: string\n\n"
    'JSON arguments: {"text":"a b c","pattern":"\\\\s",'
    '"replacement":"_"}\n\n'
)


def build_arguments_prompt(
    user_prompt: str,
    function: FunctionDefinition,
) -> str:
    """Build a prompt for function argument generation."""

    parameters_text = ""

    for name, definition in function.parameters.items():
        parameters_text += f"- {name}: {definition.type}\n"

    return (
        "Extract the arguments of the function call as one JSON object.\n"
        "Copy text from the request exactly. Do not execute the function.\n"
        "Regex uses Python syntax: no slashes, no flags. "
        "A literal word is its own regex.\n\n"
        f"{FEW_SHOT}"
        f"User request: {user_prompt}\n\n"
        f"Function: {function.name}\n"
        f"Description: {function.description}\n"
        "Parameters:\n"
        f"{parameters_text}\n"
        "JSON arguments: "
    )


def build_function_selection_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition],
) -> str:
    """Build a prompt for function selection."""

    functions_text = ""

    for function in functions:
        functions_text += (
            f"Name: {function.name}\n"
            f"Description: {function.description}\n\n"
        )

    return (
        "Choose the most appropriate function for the user request.\n"
        "Respond with only the function name.\n\n"
        "Available functions:\n"
        f"{functions_text}"
        f"User request: {user_prompt}\n"
        "Selected function: "
    )
