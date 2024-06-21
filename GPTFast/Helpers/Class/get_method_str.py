import inspect

def get_method_str(object_instance:object, method_name:str) -> str:
    if not hasattr(object_instance, method_name):
        return f"Method '{method_name}' not found in the object."

    # Get the original method
    original_method = getattr(object_instance, method_name)

    # Convert the method to a string
    method_str = inspect.getsource(original_method)

    return method_str