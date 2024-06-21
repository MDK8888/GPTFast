def add_default_parameter(func_str, func_name, default_parameter_name, parameter_type, default_value, has_terminating_comma: bool = False):
    lines = func_str.split('\n')
    signature_end = None
    num_spaces = -1

    # Find the line containing the function name
    for i, line in enumerate(lines):
        if f"def {func_name}" in line:
            start_index = i
            break

    # Proceed with the rest of the function string after finding the function name
    for i in range(start_index, len(lines)):
        num_spaces = max(num_spaces, len(lines[i]) - len(lines[i].lstrip()))
        if ')' in lines[i]:
            signature_end = i
            break

    if signature_end is not None and num_spaces != -1:
        new_param_str = f"{default_parameter_name}: {parameter_type} = {default_value}"
        indent = ' ' * num_spaces
        new_signature = f"{indent}{new_param_str}"
        lines.insert(signature_end, new_signature)

        # Append comma to the line preceding the signature end
        if not has_terminating_comma:
            if signature_end > 0:
                lines[signature_end - 1] += ','

    return '\n'.join(lines[start_index:])