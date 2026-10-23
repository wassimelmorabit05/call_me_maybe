import re
import numpy as np
from typing import Any
from .schema import FunctionCallResult
from .loader import Prompt, FunctionDefinition
from .c_decoding import get_valid_next_tokens, mask_logits


def build_prompt(
    user_prompt: str,
    functions: list[FunctionDefinition]
) -> str:
    """
    Build the input prompt for the LLM.

    Include the available functions, their descriptions and parameters,
    followed by the user's request and a response header.

    Return:
        The complete prompt as a string.
    """
    result = "You are a function calling assistant.\n"
    result += "Available functions:\n"

    for func in functions:
        name = func.name
        desc = func.description
        params = []
        for k, v in func.parameters.items():
            params.append(f"{k} ({v.type})")
            result += f"{name} - {desc} - Parameters: {', '.join(params)}\n"

    result += f"\nUser request: {user_prompt}\n"
    result += '\nResponse:\n'

    return result


def is_valid_prompt(prompt: str) -> bool:
    """Check if a prompt is valid and meaningful.

    Args:
        prompt: User input string to validate.

    Returns:
        True if prompt appears valid, False otherwise.
    """
    if len(prompt.strip()) < 3:
        return False

    words = prompt.strip().split()
    real_words = [
        w for w in words if w.isalpha() and len(w) > 12 and not w.isdigit()]
    if len(real_words) > 0:
        return False
    return True


def get_numeric_and_stop_token_ids(
    token_to_id: dict[str, int]
) -> dict[str, set[int]]:
    """
    Identify vocabulary tokens allowed during numeric generation.

    Collect token IDs used to stop generation and token IDs containing
    only digits, decimal points, or minus signs, with at most one
    decimal point and one minus sign per token.

    Return:
        A dictionary containing the stop-token IDs and numeric-token IDs.
    """
    stop_tokens = set()
    stop = [",", "}", "Ġ", " ", " ,", "\n"]
    stop_tokens = {token_to_id[c] for c in stop if c in token_to_id}
    numbers_tokens = set()
    for k, v in token_to_id.items():
        if (all(c in "0123456789.-" for c in k)
                and k.count('.') <= 1 and k.count('-') <= 1):
            numbers_tokens.add(v)
    result = {
        "stop_tokens": stop_tokens,
        "all_tokens": numbers_tokens
    }
    return result


def generate_number_value(
    input_ids: list[int],
    model: Any,
    token_to_id: dict[str, int],
    id_to_token: dict[int, str]
) -> tuple[float, list[int]]:
    """
    Generate a numeric parameter value using constrained decoding.

    Repeatedly predict the next token and restrict selection to numeric
    tokens and stopping tokens. Accumulate numeric tokens until a
    stopping token is selected, then convert the result to a float.

    Return:
        A tuple containing the generated float and its token IDs.
        Return 0.0 if the generated text is empty or cannot be
        converted to a float.
    """
    all_valid = get_numeric_and_stop_token_ids(token_to_id)
    curr_number = ""
    gen_ids: list[int] = []
    while True:
        full_ids = input_ids + gen_ids
        logits = model.get_logits_from_input_ids(full_ids)
        masked = mask_logits(
            logits, all_valid["all_tokens"] | all_valid["stop_tokens"])
        max_token_id = int(np.argmax(masked))
        max_token = id_to_token[max_token_id]
        if max_token_id in all_valid["stop_tokens"]:
            break
        gen_ids.append(max_token_id)
        curr_number += max_token
    curr_number = curr_number.strip()
    try:
        val = float(curr_number) if curr_number else 0.0
    except ValueError:
        val = 0.0

    return val, gen_ids


def generate_string_value(
    input_ids: list[int],
    model: Any,
    id_to_token: dict[int, str]
) -> tuple[str, list[int]]:
    """
    Generate a string parameter value using the LLM.

    Repeatedly select the highest-scoring token and append it to the
    generated string until a token containing a double quote is found.
    Replace tokenizer space markers and escaped apostrophes in the result.

    Return:
        A tuple containing the generated string and its token IDs,
        excluding the token that stopped generation.
    """
    generated_token_ids: list[int] = []
    result: str = ""
    while True:
        full_ids = input_ids + generated_token_ids
        logits = model.get_logits_from_input_ids(full_ids)
        max_token_id: int = int(np.argmax(logits))
        if '"' in id_to_token[int(max_token_id)]:
            break
        generated_token_ids.append(max_token_id)
        result += id_to_token[max_token_id]

    return result.replace("Ġ", " ").replace("\\'", "'"), generated_token_ids


def generate_function_call(
    prompt: Prompt,
    functions: list[FunctionDefinition],
    model: Any,
    token_to_id: dict[str, int],
    id_to_token: dict[int, str],
    visualize: bool = False
) -> FunctionCallResult:
    """
    Generate a function call from a user prompt using the LLM.

    Build and tokenize the input prompt, then use constrained decoding
    to select a function name from the available functions.

    Generate the selected function's parameter values according to
    their types and construct a validated function-call result.

    Return:
        A FunctionCallResult object containing the original prompt,
        selected function name, and generated parameters.
        Return an invalid result if no function name is matched.
    """
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    reset = "\033[0m"

    if not is_valid_prompt(prompt.prompt):
        return FunctionCallResult(
            prompt=prompt.prompt, name="Invalid!",
            parameters={"error": "prompt not recognized"})

    prompt_text = build_prompt(prompt.prompt, functions)
    input_ids = model.encode(prompt_text)[0].tolist()

    # Model's Response Prefix
    input_ids.append(token_to_id['{'])
    input_ids.append(token_to_id['"'])
    input_ids.append(token_to_id['name'])
    input_ids.append(token_to_id['":'])
    input_ids.append(token_to_id['Ġ'])
    input_ids.append(token_to_id['"'])

    # Constrained Decoding:
    current_function_prefix = ""
    gen_ids: list[int] = []
    function_names = [fn.name for fn in functions]

    if visualize:
        print(f"      {yellow}Generating function name ...{reset}")

    for _ in range(30):
        full_input_ids = input_ids + gen_ids
        logits = model.get_logits_from_input_ids(full_input_ids)
        logits_numpy = np.array(logits)
        valid_token_ids = get_valid_next_tokens(
            current_function_prefix, function_names, token_to_id
        )
        masked_logits = mask_logits(logits_numpy, valid_token_ids)
        next_token_id = int(np.argmax(masked_logits))
        next_token = id_to_token[next_token_id]
        gen_ids.append(next_token_id)

        current_function_prefix += next_token
        if current_function_prefix in function_names:
            if visualize:
                print(
                    f'        - {green}Found:{reset} '
                    f'"{current_function_prefix}"'
                    )
                print(
                    "\n      "
                    f"{yellow}Generating function's parameters ...{reset}"
                    )

            selected_function = next(
                fn for fn in functions if fn.name == current_function_prefix)
            input_ids.extend(gen_ids)
            input_ids.append(token_to_id['"'])
            input_ids.append(token_to_id[','])
            input_ids.append(token_to_id['Ġ'])
            input_ids.append(token_to_id['"'])
            input_ids.append(token_to_id['parameters'])
            input_ids.append(token_to_id['":'])
            input_ids.append(token_to_id['Ġ'])
            input_ids.append(token_to_id['{'])
            params = list(selected_function.parameters.items())
            parameters: dict[str, float | str | int] = {}
            for i, (name, value) in enumerate(params):
                if visualize:
                    print(
                        "        "
                        f"{blue}Generating parameter {i + 1} ...{reset}"
                        )
                input_ids.append(token_to_id['"'])
                for chr_id in model.encode(name)[0].tolist():
                    input_ids.append(chr_id)
                input_ids.append(token_to_id['"'])
                input_ids.append(token_to_id[':'])
                input_ids.append(token_to_id['Ġ'])
                if value.type == "number" or value.type == "integer":
                    has_negative = bool(re.search(r"-\d+", prompt.prompt))
                    if has_negative:
                        extracted_numbers = re.findall(
                            r"-?\d+(?:\.\d+)?", prompt.prompt)
                        if value.type == "number":
                            parameters[name] = float(extracted_numbers[i])
                        else:
                            parameters[name] = int(extracted_numbers[i])
                    else:
                        val, ids = generate_number_value(
                            input_ids, model, token_to_id, id_to_token)
                        if value.type == "number":
                            parameters[name] = val
                        else:
                            parameters[name] = int(val)
                        input_ids += ids

                    if visualize:
                        print(
                            f"          - {green}Found:{reset} {name}: "
                            f"{parameters[name]}"
                            )

                    if i < len(params) - 1:
                        input_ids.append(token_to_id[','])
                        input_ids.append(token_to_id['Ġ'])
                    else:
                        input_ids.append(token_to_id['}'])
                        input_ids.append(token_to_id['}'])
                elif value.type == "string":
                    input_ids.append(token_to_id['"'])
                    (str_val, ids) = generate_string_value(
                        input_ids, model, id_to_token)
                    parameters[name] = str_val
                    input_ids += ids
                    if visualize:
                        print(
                            "          "
                            f"- {green}Found:{reset} {name}: {str_val}"
                            )

            return FunctionCallResult(
                prompt=prompt.prompt,
                name=current_function_prefix,
                parameters=parameters
            )

    return FunctionCallResult(
        prompt=prompt.prompt,
        name="invalid",
        parameters={"error": "no matching function found"}
    )
