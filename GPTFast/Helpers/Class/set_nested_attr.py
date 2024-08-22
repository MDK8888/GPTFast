def set_nested_attr(obj, attr_path, value):
    attrs = attr_path.split('.')
    for attr in attrs[:-1]:
        obj = getattr(obj, attr)
    setattr(obj, attrs[-1], value)