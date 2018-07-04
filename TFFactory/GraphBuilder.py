import tensorflow
import sys
import json
from collections import defaultdict, Hashable
from types import ModuleType
import TFFactory.SupportedFunctions as functions
from TFFactory.Pointer import Pointer
from .Utilities import findAndApply

ID_COUNTER = defaultdict(int)
CURRENT_GRAPH = {}


def NewGraph():
    global ID_COUNTER, CURRENT_GRAPH
    ID_COUNTER = defaultdict(int)
    CURRENT_GRAPH = {}
    return


def _assignFunctions(this, functions, type):
    for f in functions:
        attrs = f.split('.')
        curObj = this
        for v in attrs[1:-1]:
            if not hasattr(curObj, v):
                setattr(curObj, v, ModuleType(v))
            curObj = getattr(curObj, v)
        setattr(curObj, attrs[-1], _mockFunction(f, type))


def _mockFunction(funcName, type):
    def MockedFunction(*args, **kwargs):
        global ID_COUNTER, CURRENT_GRAPH
        name = kwargs.get('name', 'unnamed')
        shape = kwargs.pop('_shape', None)
        count = ID_COUNTER[name]
        ID_COUNTER[name] += 1
        if count > 0:
            name = '{}:{}'.format(name, count)
        n = JSONNode(name, funcName, list(args), kwargs, shape, type)
        CURRENT_GRAPH.update({n.ID: n})
        return n
    return MockedFunction


this = sys.modules[__name__]
PYTHON_FUNCTIONS = [
    'SupportedFunctions.fileSource',
    'SupportedFunctions.parser',
    'SupportedFunctions.testAdd'
]
_assignFunctions(this, PYTHON_FUNCTIONS, 'pythonNode')

COMPOSITE_FUNCTIONS = [
    'SupportedFunctions.AdamOptimizer',
    'SupportedFunctions.MomentumOptimizer',
    'SupportedFunctions.GradientDescentOptimizer',
    'SupportedFunctions.SampleDirichlet',
    'SupportedFunctions.GetItem'
]
_assignFunctions(this, COMPOSITE_FUNCTIONS, 'tensorflowNode')

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
    'tensorflow.square',
    'tensorflow.sqrt',
    'tensorflow.reduce_sum',
    'tensorflow.reduce_mean',
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
    'tensorflow.summary.tensor_summary',
    'tensorflow.summary.scalar',
    'tensorflow.summary.histogram',
    'tensorflow.summary.image',
    'tensorflow.summary.text',
    'tensorflow.summary.audio',
    'tensorflow.summary.merge',
    'tensorflow.summary.merge_all'
]
_assignFunctions(this, MOCKED_FUNCTIONS, 'tensorflowNode')


class Encoder(json.JSONEncoder):
    SERIALIZE_MAP = {
        tensorflow.float16: 'tensorflow.float16',
        tensorflow.float32: 'tensorflow.float32',
        tensorflow.float64: 'tensorflow.float64',
        tensorflow.int8: 'tensorflow.int8',
        tensorflow.int16: 'tensorflow.int16',
        tensorflow.int32: 'tensorflow.int32',
        tensorflow.int64: 'tensorflow.int64'
    }

    def default(self, obj):
        if isinstance(obj, JSONNode):
            return {
                '_type': obj.Type,
                'funcName': obj.FuncName,
                'inputs': obj.Inputs
            }
        elif isinstance(obj, Pointer):
            return {
                'value': obj.Ref,
                '_type': 'pointer'
            }
        elif isinstance(obj, slice):
            return {
                'value': [obj.start, obj.stop, obj.step],
                '_type': 'slice'
            }
        elif isinstance(obj, Hashable) and obj in Encoder.SERIALIZE_MAP:
            return {
                'value': Encoder.SERIALIZE_MAP[obj],
                '_type': 'tensorflow'
            }

        return json.JSONEncoder.default(self, obj)


class Decoder(json.JSONDecoder):
    DESERIALIZE_MAP = {
        'tensorflow.float16': tensorflow.float16,
        'tensorflow.float32': tensorflow.float32,
        'tensorflow.float64': tensorflow.float64,
        'tensorflow.int8': tensorflow.int8,
        'tensorflow.int16': tensorflow.int16,
        'tensorflow.int32': tensorflow.int32,
        'tensorflow.int64': tensorflow.int64
    }

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(
            self, object_hook=self.object_hook, *args, **kwargs)
        return

    def object_hook(self, obj):
        if '_type' in obj:
            t = obj['_type']
            if t == 'tensorflow':
                return Decoder.DESERIALIZE_MAP[obj['value']]
            elif t == 'pointer':
                return Pointer(obj['value'])
            elif t == 'slice':
                return slice(*obj['value'])
        return obj


class JSONNode:
    def __init__(self, id, funcName, args, kwargs, shape, type):
        self.ID = str(id)
        self.FuncName = funcName
        self.Type = str(type)
        self.Inputs = {
            'args': args,
            'kwargs': kwargs,
            '_shape': shape
        }
        self.Inputs = findAndApply(
            self.Inputs, self._shouldBePointer, self._replaceWithPointer)

        return

    @staticmethod
    def _shouldBePointer(obj):
        return isinstance(obj, JSONNode)

    @staticmethod
    def _replaceWithPointer(obj):
        return Pointer(obj.ID)

    def __getitem__(self, key):
        return GetItem(self, key)

    def __neg__(self):
        return multiply(-1.0, self)

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

    def __eq__(self, other):
        return other.ID == self.ID

    def __hash__(self):
        return hash(self.ID)
