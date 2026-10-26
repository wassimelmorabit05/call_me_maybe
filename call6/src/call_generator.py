"""Let the LLM choose a function and generate its arguments."""

import json
from typing import Any

from src.decoding import generate_value
from src.models import FunctionCallResult, Parameter, Prompt

BLUE = "\033[34m"
GREEN = "\033[32m"
RESET = "\033[0m"


def build_prompt(prompt: str, functions: list, no_match: str) -> str:
    """Build Qwen's chat format with reasoning disabled."""
    definitions = []
    for function in functions:
        arguments = []
        for name, schema in function.parameters.items():
            detail = name + ":" + schema.type
            if schema.enum is not None:
                detail += "=" + json.dumps(schema.enum, ensure_ascii=False)
            if schema.description:
                detail += " " + schema.description
            arguments.append(detail)
        definitions.append(function.name + "(" + ", ".join(arguments)
                           + "): " + function.description)
    instructions = (
        "Return a JSON function call with name and parameters. "
        "Extract arguments, do not compute the result. Copy strings exactly. "
        f"If no function fits, choose {no_match}.\n"
        + "\n".join(definitions)
    )
    return (
        "<|im_start|>system\n" + instructions + "<|im_end|>\n"
        "<|im_start|>user\n" + prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def generate_function_call(prompt: Prompt, functions: list,
                           model: Any, tokens: dict,
                           visualize: bool = False) -> FunctionCallResult:
    """Return one call with every argument checked against its schema."""
    by_name = {function.name: function for function in functions}
    no_match = "no_matching_function"
    while no_match in by_name:
        no_match += "_"
    context = build_prompt(prompt.prompt, functions, no_match) + '{"name": '
    name_schema = Parameter(type="string", enum=[*by_name, no_match])
    if visualize:
        print(f"  {BLUE}Generating function name...{RESET}", flush=True)
    name = generate_value(context, name_schema, ', "parameters": {',
                          model, tokens)
    if name == no_match:
        raise ValueError("The model found no matching function")
    if visualize:
        print(f"    {GREEN}-> {name}{RESET}", flush=True)
    function = by_name[name]
    context = build_prompt(prompt.prompt, [function], no_match)
    context += '{"name": ' + json.dumps(name) + ', "parameters": {'
    parameters: dict = {}
    items = list(function.parameters.items())
    for index, (key, schema) in enumerate(items):
        context += json.dumps(key, ensure_ascii=False) + ": "
        boundary = "}}"
        if index + 1 < len(items):
            boundary = ", " + json.dumps(items[index + 1][0]) + ": "
        value = generate_value(context, schema, boundary, model, tokens)
        if visualize:
            print(f"  {BLUE}{key}:{RESET} {GREEN}{value!r}{RESET}", flush=True)
        parameters[key] = value
        context += json.dumps(value, ensure_ascii=False, allow_nan=False)
        context += ", " if index + 1 < len(items) else ""
    return FunctionCallResult(prompt=prompt.prompt, name=name,
                              parameters=parameters)
