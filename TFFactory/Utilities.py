# These just don't belong elsewhere

def findAndApply(obj, found, apply):
    if found(obj):
        return apply(obj)
    if isinstance(obj, list):
        for i, o in enumerate(obj):
            obj[i] = findAndApply(o, found, apply)
    elif isinstance(obj, dict):
        for k, v in obj.items():
            obj[k] = findAndApply(v, found, apply)
    return obj