from .shift_left import shift_left

def add_input_pos_to_func_str(func_str:str, submodule_name:str, new_arg_string:str):
    lines = func_str.split('\n')
    found_submodule_forward = False
    submodule_indent = -1

    # Find the forward pass of the specified submodule
    for i, line in enumerate(lines):
        if found_submodule_forward:
            submodule_indent = max(submodule_indent, len(line) - len(line.lstrip()))
            if ')' in line:
                break
        elif submodule_name in line and (submodule_name + '(' in line or submodule_name + '.forward(' in line):
            found_submodule_forward = True

    if not found_submodule_forward:
        raise ValueError("Submodule forward pass not found.")

    # Add the new argument to the forward pass
    for i in range(i, len(lines)):
        if lines[i].strip().startswith(')'):
            lines.insert(i, f"{' ' * submodule_indent}{new_arg_string},")
            break

    res_str = '\n'.join(lines)

    return shift_left(res_str)