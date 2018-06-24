import tensorflow
import sys
import json
from collections import defaultdict, Hashable
from types import ModuleType
from TFFactory.SupportedFunctions import *

ID_COUNTER = defaultdict(int)
CURRENT_GRAPH = {}

PYTHON_FUNCTIONS = {
    'fileSource' : lambda FilePath, NRows : readFile(FilePath, ',', NRows),
    'parser' : splitFile
}

MOCKED_FUNCTIONS = [ 
    'tensorflow.placeholder',
    'tensorflow.Variable',
    'tensorflow.abs',
    'tensorflow.add',
    'tensorflow.subtract',
    'tensorflow.multiply',
    'tensorflow.divide',
    'tensorflow.log',
    'tensorflow.tanh',
    'tensorflow.tensordot',
    'tensorflow.reduce_sum',
    'tensorflow.shape',
    'tensorflow.transpose',
    'tensorflow.expand_dims',
    'tensorflow.nn.relu',
    'tensorflow.nn.l2_loss',
    'tensorflow.nn.softmax',
    'tensorflow.layers.conv2d',
    'tensorflow.layers.batch_normalization',
    'tensorflow.layers.dense',
    'tensorflow.distributions.Dirichlet',
]

SERIALIZE_MAP = {
    tensorflow.float16 : 'tensorflow.float16',
    tensorflow.float32 : 'tensorflow.float32',
    tensorflow.float64 : 'tensorflow.float64',
    tensorflow.int8 : 'tensorflow.int8',
    tensorflow.int16 : 'tensorflow.int16',
    tensorflow.int32 : 'tensorflow.int32',
    tensorflow.int64 : 'tensorflow.int64'
}

DESERIALIZE_MAP = {
    'tensorflow.float16': tensorflow.float16,
    'tensorflow.float32': tensorflow.float32,
    'tensorflow.float64': tensorflow.float64,
    'tensorflow.int8': tensorflow.int8,
    'tensorflow.int16': tensorflow.int16,
    'tensorflow.int32': tensorflow.int32,
    'tensorflow.int64': tensorflow.int64 
}


def Serialize(value):
    type = 'unknown'
    if isinstance(value, JSONNode):
        v = value.ID
        type = 'ref'
    elif isinstance(value, Hashable):
        v = SERIALIZE_MAP.get(value, value)
    else:
        v = value
    
    d = {
        'value' : v,
        'type' : type
    }
    
    return d

def Deserialize(value):
    """
        Given the value object with format :
        value = {
            'value' : <>,
            'type' : <>
        }

        returns (the deserialized value, whether or not it is a node reference)
    """
    v = value['value']
    if value['type'] == 'ref':
        return (v, True)
    if isinstance(v, Hashable):
        v = DESERIALIZE_MAP.get(v, v)

    return (v, False)

def NewGraph():
    global ID_COUNTER, CURRENT_GRAPH
    ID_COUNTER = defaultdict(int)
    CURRENT_GRAPH = {}
    return

def MockFunction(funcName):
    def MockedFunction(*args, **kwargs):
        global ID_COUNTER, CURRENT_GRAPH
        name = kwargs.get('name','Variable')
        count = ID_COUNTER[name]
        ID_COUNTER[name] += 1
        if count > 0:
            name = '{}_{}'.format(name, count)
        n = JSONNode(name, funcName, args, kwargs)
        CURRENT_GRAPH.update(n.asDict())
        return n
    return MockedFunction

this = sys.modules[__name__]
for f in MOCKED_FUNCTIONS:
    attrs = f.split('.')
    curObj = this
    for v in attrs[1:-1]:
        if not hasattr(curObj, v):
            setattr(curObj, v, ModuleType(v))
        curObj = getattr(curObj, v)
    setattr(curObj, attrs[-1], MockFunction(f))


class JSONNode:
    def __init__(self, id, type, args, kwargs):
        self.ID = str(id)
        self.type = str(type)
        self.Inputs = {
            'args' : [],
            'kwargs' : {}
        }
        for name, value in kwargs.items():
            self.Inputs['kwargs'][name] = Serialize(value)
        self.Inputs['args'] = []
        for value in args:
            self.Inputs['args'].append(Serialize(value))
        return
    
    def asDict(self):
        d = {
            self.ID : {
                'type' : self.type,
                'inputs' : self.Inputs
            }
        }
        return d
    
    def __neg__(self):
        return multiply(-1, self)
    def __pos__(self):
        return abs(self)

    def __add__(self, other):
        return add(self, other)
    def __radd__(self, other):
        return add(other, self)
    __iadd__ = __add__

    def __sub__(self, other):
        return subtract(self, other)
    def __rsub__(self, other):
        return subtract(other, self)
    __isub__ = __sub__

    def __mul__(self, other):
        return multiply(self, other)
    def __rmul__(self, other):
        return multiply(other, self)
    __imul__ = __mul__

    def __truediv__(self, other):
        return divide(self, other)
    def __rtruediv__(self, other):
        return divide(other, self)
    __itruediv__ = __truediv__

    def __str__(self):
        return json.dumps(self.asDict())

    def __eq__(self, other):
        return other.ID == self.ID

    def __hash__(self):
        return hash(self.ID)