def shift_right(func_str:str, spaces:int=4) -> str:
    # Split the func string into lines
    lines = func_str.split('\n')

    # Shift each line to the right by the specified number of spaces
    for i in range(len(lines)):
        lines[i] = ' ' * spaces + lines[i]

    # Join the lines back into a string
    result_str = '\n'.join(lines)

    return result_str