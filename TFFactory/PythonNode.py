from .Node import Node
from .Utilities import findAndApply
from functools import partial
from .GraphBuilder import Encoder, Decoder
from .Pointer import Pointer

class PythonNode(Node):
    Encoder = Encoder()
    Decoder = Decoder()
    def __init__(self, id, evalFunc, tensor = None,
                    args = None, kwargs = None, pointerMap = None, needToFeed = None):
        
        self.EvalFunc = evalFunc
        self.Tensor = tensor
        self.Args = args if args is not None else []
        self.Kwargs = kwargs if kwargs is not None else {}
        self.AllArgs = {
            'args' : self.Args,
            'kwargs' : self.Kwargs
        }
        self.NeedToFeed = needToFeed if needToFeed is not None else {}
        self.PointerMap = pointerMap if pointerMap is not None else {}
        return super().__init__(id)

    def _eval(self, session = None, feed_dict = {}, newContext = True):
        pointers = PythonNode.Decoder.decode(PythonNode.Encoder.encode(self.PointerMap))
        print(pointers)

        pointers = findAndApply(pointers, Pointer.IsInstance, 
                                partial(self._feed,
                                        nodeMap = self.NeedToFeed,
                                        session = session, 
                                        feed_dict = feed_dict,
                                        newContext = newContext))
        self._mergeObj(self.AllArgs, pointers)
        print(self.AllArgs)
        val = self.EvalFunc(*self.AllArgs['args'], **self.AllArgs['kwargs'])
        return val
        
    @staticmethod
    def _feed(nodeMap, session, feed_dict, newContext, pointer):
        return nodeMap[pointer.Ref].Eval(session, feed_dict, newContext)

    @staticmethod
    def _mergeObj(o1, o2):
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