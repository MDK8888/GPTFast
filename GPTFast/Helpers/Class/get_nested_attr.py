def get_nested_attr(obj, attr_path:str):
    attrs = attr_path.split('.')
    for attr in attrs:
        if hasattr(obj, attr):
            obj = getattr(obj, attr)
        else:
            return None
    return obj