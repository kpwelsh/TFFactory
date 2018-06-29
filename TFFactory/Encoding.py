import json

TENSORFLOW_MAP = {
    tensorflow.float16 : 'tensorflow.float16',
    tensorflow.float32 : 'tensorflow.float32',
    tensorflow.float64 : 'tensorflow.float64',
    tensorflow.int8 : 'tensorflow.int8',
    tensorflow.int16 : 'tensorflow.int16',
    tensorflow.int32 : 'tensorflow.int32',
    tensorflow.int64 : 'tensorflow.int64'
}

class Encoder(json.JSONEncoder):
    def EncodeGraph(self, graph):
        d = {}
        for n in graph:
            d.update({
                n.ID : {
                    'type' : n.Type,
                    'inputs' : n.Inputs
                }
            })
        return self.encode(d)

    def default(self, obj):
        if isinstance(obj, Pointer):
            return {
                'value' : obj.ID,
                '_type' : 'pointer'
            }
        elif isinstance(obj, Hashable) and obj in TENSORFLOW_MAP:
            return {
                'value' : SERIALIZE_MAP[obj],
                '_type' : 'tensorflow'
            }

        return json.JSONEncoder.default(self, obj)

DESERIALIZE_MAP = {
    'tensorflow.float16': tensorflow.float16,
    'tensorflow.float32': tensorflow.float32,
    'tensorflow.float64': tensorflow.float64,
    'tensorflow.int8': tensorflow.int8,
    'tensorflow.int16': tensorflow.int16,
    'tensorflow.int32': tensorflow.int32,
    'tensorflow.int64': tensorflow.int64 
}

class Decoder(json.JSONDecoder):
    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
        return

    def object_hook(self, obj):
        if '_type' in obj:
            t = obj['_type']
            v = obj['value']
            if t == 'tensorflow' :
                return DESERIALIZE_MAP[v]
            if t == 'pointer':
                return Pointer(str(v))
        return obj

if __name__ == '__main__':
    obj = {
        'dict' : {
            'string' : 'value',
            'int' : 2
        },
        'string' : 'value',
        'int' : 1,
        'array' : [JSONNode(1), JSONNode(2)],
        'obj' : JSONNode(3)
    }
    print(json.dumps(obj, cls = Encoder, indent = 2))