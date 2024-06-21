from .shift_left import shift_left

def modify_if_block(method_str:str, condition:str, new_condition_strings_list: list[str]):

    # Find the index of the if condition
    if_condition_index = method_str.find(f"{condition}:")

    if if_condition_index != -1:
        # Extract the line containing the if condition
        lines = method_str.split('\n')
        for line_num, line in enumerate(lines):
            if condition in line:
                if_condition_line = line
                left_index = line_num
                break

        # Find the indentation level of the if condition
        if_condition_indentation = len(if_condition_line) - len(if_condition_line.lstrip())

        # Find the index where the indentation level decreases
        for right_index, line in enumerate(lines[left_index+1:], start=left_index+1):
            if line.strip() and len(line) - len(line.lstrip()) <= if_condition_indentation:
                break

        # Calculate the indentation on lines[right_index]
        if_condition_indentation = len(lines[right_index]) - len(lines[right_index].lstrip())

        # Shift new_condition_strings_list to the right by if_condition_indentation + 4
        total_spaces = (if_condition_indentation + 4) * " "
        for i in range(len(new_condition_strings_list)):
            new_condition_strings_list[i] = total_spaces + new_condition_strings_list[i]

        # Concatenate the modified block
        lines = lines[:left_index + 1] + new_condition_strings_list + lines[right_index:]
        method_str = "\n".join(lines)

    method_str = shift_left(method_str)
    return method_str