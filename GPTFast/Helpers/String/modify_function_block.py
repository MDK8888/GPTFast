from .shift_left import shift_left

def modify_function_block(method_str:str, line_to_search:str, num_lines_to_replace:int, replace_list:list[str]) -> str:
    # Split the source code of the function
    source_code_split = method_str.split('\n')

    # Find the index of the line to search for
    line_index = None
    for index, line in enumerate(source_code_split):
        if line_to_search in line:
            line_index = index
            break

    if line_index is not None:
        # Count the number of spaces that the line is indented over
        indent = len(source_code_split[line_index]) - len(source_code_split[line_index].lstrip())

        # Adjust each element in replace_list by the calculated indent
        for i in range(len(replace_list)):
            replace_list[i] = ' ' * indent + replace_list[i]

        # Replace lines from line_index to line_index + num_lines_to_replace with replace_list
        source_code_split = source_code_split[:line_index] + replace_list + source_code_split[line_index + num_lines_to_replace:]

    # Join the modified source code and return
    modified_code = '\n'.join(source_code_split)
    return shift_left(modified_code)