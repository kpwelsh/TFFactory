from TFFactory.Pointer import Pointer

def __findPointers(obj, pointers):
    if isinstance(obj, list):
        for i, o in enumerate(obj):
            next = {}
            pointer = __findPointers(o, next)
            if pointer is not None:
                pointers[i] = pointer
            elif len(next.keys()) > 0:
                pointers[i] = next
    elif isinstance(obj, dict):
        if obj.get('_type') == 'pointer':
            value = obj.get('value')
            return Pointer(value)
        else:
            for k, v in obj.items():
                next = {}
                pointer = __findPointers(v, next)
                if pointer is not None:
                    pointers[k] = pointer
                elif len(next.keys()) > 0:
                    pointers[k] = next
    return None

def mergeObj(o1, o2):
    if o2 is None:
        return o1
    if o1 is None:
        return o2
    if isinstance(o2, dict):
        for k, v in o2.items():
            o1[k] = mergeObj(o1[k], v)
        return o1
    elif isinstance(o2, list):
        for i, v in enumerate(o2):
            o1[i] = mergeObj(o1[i], v)
        return o1
    else:
        return o2
    return None


if __name__ == '__main__':
    o1 = {
        'dict' : {
            'key1' : 'value',
            'key2' : 'more value',
            'list' : [
                {
                    'value' : 'inAList',
                    '_type' : 'pointer'
                },
                {
                    'value' : 'inAList',
                    '_type' : 'pointer'
                }
            ]
        },
        'key' : 'value',
        'heres one' : {
                    'value' : 'inAList',
                    '_type' : 'pointer'
                }
    }

    print(o1)
    pointers = {}
    __findPointers(o1, pointers)
    print(pointers)
    print(mergeObj(o1, pointers))