import types

def add_str_as_func(obj:object, method_name:str, func_str:str, imports:list[str] = []):
    import_str = "\n".join(imports)

    # Combine imports and func string
    complete_func_str = import_str + "\n" + func_str

    func_code = compile(complete_func_str, "<string>", "exec")
    namespace = {}
    exec(func_code, namespace)

    # Extract the func from the namespace
    my_func = namespace[method_name]

    # Attach the func to the object
    setattr(obj, method_name, types.MethodType(my_func, obj))

    return obj