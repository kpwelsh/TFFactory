import tensorflow
import numpy as np
import json
import TFFactory.GraphBuilder as GraphBuilder
import TFFactory.SupportedFunctions as SupportedFunctions
from .TFFactoryException import TFFactoryException
from .Pointer import Pointer
from .PythonNode import PythonNode
from .TFNode import TFNode
from .Utilities import findAndApply
from functools import partial

def CreateTFGraph(graph):
    nodes = {}
    for key, _ in graph.items():
        __buildBranch(graph, key, nodes)

    return nodes

def __buildBranch(graph, key, allNodes):
    """
        @graph -- The full json graph for traversal/reference
        @key -- the key to access the desired eval end point in @graph
        @allNodes -- dict to store references to all nodes in scope. This function will add nodes here.
        @needtofeed -- the keys of the nodes that need to be supplied to the feed_dict for evaluation
            @allNodes[key] will hold a reference to the node.
    """
    if key in allNodes: 
        return allNodes[key]

    _type = graph[key].get('_type')

    node = None
    if _type == 'pythonNode':
        node = __buildPythonNode(key, graph, allNodes)
    elif _type == 'tensorflowNode':
        node = __buildTFNode(key, graph, allNodes)
        
    if node is None:
        raise AssertionError('Unsupported node type: {}'.format(_type))
    allNodes[key] = node
    return node

def __buildPythonNode(key, graph, allNodes):
    graphNode = graph[key]
    funcName = graphNode['funcName']
    inputs = graphNode['inputs']
    func = __resolveRef(funcName)
    needToFeed = {}

    allArgs = {
        'args' : inputs.get('args', []),
        'kwargs' : inputs.get('kwargs', {})
    }
    dependencies = set()
    allArgs = findAndApply(allArgs, Pointer.IsInstance, 
                            partial(__markPythonDependency, 
                                dependencies = dependencies,
                                grpah = graph,
                                allNodes = allNodes))
    
    args = allArgs['args']
    kwargs = allArgs['kwargs']
    for d in dependencies:
        needToFeed[d.ID] = d

    placeholder = None
    if '_shape' in inputs:
        placeholder = tensorflow.placeholder(tensorflow.float32, 
                                            shape = inputs['_shape'], 
                                            name = 'Placeholder_{}'.format(key))
    
    node = PythonNode(key,
                tensor = placeholder,
                evalFunc = func,
                args = args,
                kwargs = kwargs,
                needToFeed = needToFeed)

    allNodes[key] = node
    return node

def __markPythonDependency(pointer, dependencies, graph, allNodes):
    node = __buildBranch(graph, pointer.Ref, allNodes)
    dependencies.add(node)
    return pointer

def __buildTFNode(key, graph, allNodes):
    graphNode = graph[key]
    funcName = graphNode['funcName']
    inputs = graphNode['inputs']
    tfOp = __resolveRef(funcName)
    needToFeed = {}

    allArgs = {
        'args' : inputs.get('args', []),
        'kwargs' : inputs.get('kwargs', {})
    }
    dependencies = set()
    allArgs = findAndApply(allArgs, Pointer.IsInstance, 
                            partial(__markTFDependency, 
                                dependencies = dependencies,
                                graph = graph,
                                allNodes = allNodes))

    args = allArgs['args']
    kwargs = allArgs['kwargs']
    for d in dependencies:
        if isinstance(d, PythonNode):
            # Add any python node dependencies directly to the list.
            needToFeed[d.ID] = d
        elif isinstance(d, TFNode):
            # For all of the TF node dependencies
            # Add all of their dependencies to the list. 
            needToFeed.update(**d.NeedToFeed)
    node = TFNode(key, tfOp(*args, **kwargs), needToFeed = needToFeed)
    if tfOp == tensorflow.placeholder:
        # It can't eval itself without being fed. So hack that in, I guess?
        # Will cause an infinite loop if things break. Wont if they dont!
        node.NeedToFeed.update({key : node}) 
    return node

def __markTFDependency(pointer, dependencies, graph, allNodes):
    node = __buildBranch(graph, pointer.Ref, allNodes)
    if node not in dependencies:
        dependencies.add(node)
    if isinstance(node, TFNode) or isinstance(node, PythonNode):
        return node.Tensor
    return None

def __resolveRef(ref):
    if 'tensorflow' not in ref and 'SupportedFunctions' not in ref:
        return None
    try:
        obj = eval(ref)
        if not callable(obj):
            obj = None
    except:
        obj = None

    return obj

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

    
