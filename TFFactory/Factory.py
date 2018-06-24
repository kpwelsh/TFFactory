import tensorflow
import numpy as np
import json
from .TFFactoryException import TFFactoryException
from .DataLoader import DataLoader
import TFFactory.GraphBuilder as GB


class Node:
    EvalContext = None
    def __init__(self, id, backingVariable = None, evalFunc = None, positionalArgs = [], dictParams = {}, needtoFeed = {}):
        if backingVariable is None and evalFunc is None:
            raise AssertionError('Node cannot be created without a backing function or backing variable')
        assert id is not None, 'Nodes need a unique ID'

        self.BackVariable = backingVariable
        self.EvalFunc = evalFunc
        self.PositionalArgs = positionalArgs
        self.DictParams = dictParams
        self.NeedtoFeed = needtoFeed
        self.LastContext = {}
        self.JSONRep = {}
        self.ID = id
        return

    def eval(self, session = None, feed_dict = {}, newContext = True):
        if self.ID in feed_dict:
            return feed_dict[self.ID]
        elif self.ID in self.NeedtoFeed:
            raise TFFactoryException('Node {} was not fed during execution.'.format(self.ID))

        if newContext:
            Node.EvalContext = np.random.random()
        if Node.EvalContext not in self.LastContext:
            self.LastContext = {}
        if Node.EvalContext is not None:
            if Node.EvalContext in self.LastContext:
                return self.LastContext[Node.EvalContext]
            
        val = None
        if self.EvalFunc is not None:
            args = []
            for a in self.PositionalArgs:
                if isinstance(a, Node):
                    args.append(a.eval(session = session, feed_dict = feed_dict, newContext = False))
                else:
                    args.append(a)
            _feed_dict = {}
            for name, node in self.NeedtoFeed.items():
                _feed_dict[name] = node.eval(session = session, feed_dict = feed_dict, newContext = False)
            val = self.EvalFunc(*args, **self.DictParams, **_feed_dict)
        elif self.BackVariable is not None:
            if len(self.NeedtoFeed) == 0:
                val = self.BackVariable.eval(session = session)
            else:
                _feed_dict = {}
                # Swap out placeholders with node eval function results.
                for key, node in self.NeedtoFeed.items():
                    _feed_dict[node.BackVariable] = node.eval(session = session, feed_dict = feed_dict, newContext = False) 
                val = self.BackVariable.eval(feed_dict = _feed_dict, session = session)
        if Node.EvalContext is not None:
            self.LastContext[Node.EvalContext] = val
        if val is None: raise AssertionError('Good luck with this one.')
        return val

    def __str__(self):
        return json.dumps(self.JSONRep)


def CreateTFGraph(graph):
    nodes = {}

    for key in graph:
        __buildBranch(graph, key, nodes)

    return nodes

def __buildBranch(graph, key, allNodes, needtofeed = None):
    """
        @graph -- The full json graph for traversal/reference
        @key -- the key to access the desired eval end point in @graph
        @allNodes -- dict to store references to all nodes in scope. This function will add nodes here.
        @needtofeed -- the keys of the nodes that need to be supplied to the feed_dict for evaluation
            @allNodes[key] will hold a reference to the node.
    """
    if needtofeed is None:
        needtofeed = {}
    if key in allNodes: # Just be done, but update the feeding dependencies for the caller.
        if allNodes[key].EvalFunc is not None: # If it a non-TF op, it has a non-TF feed dict, and needs to be fed. Also it needs to have a backing variable.
            needtofeed.update({key : allNodes[key]})
        else: # If it is a TF op, then compile its child dependencies.
            needtofeed.update(**allNodes[key].NeedtoFeed)
        return

    key = str(key)
    graphNode = graph[key] # Stop it with the data type key errors...
    type = graphNode['type']
    inputs = graphNode['inputs']
    node = None

    tfOp = __resolveTFRef(type)
    # Build a function reference to eval when it comes time.
    if type in GB.PYTHON_FUNCTIONS:
        func = GB.PYTHON_FUNCTIONS[type]
        dictParams = {}
        args = []
        childNeeds = {}
        # Parse the provided kwargs
        for name, value in inputs.get('kwargs', {}).items():
            v, isRef = GB.Deserialize(value)
            if isRef:
                __buildBranch(graph, v, allNodes)
                childNeeds[name] = allNodes[v]
            else:
                dictParams[name] = v
        # Parse the provided args
        for p in inputs.get('args', []):
            v, isRef = GB.Deserialize(p)
            if isRef:
                __buildBranch(graph, v, allNodes)
                childNeeds[name] = allNodes[v]
                args.append(allNodes[v])
            else:
                args.append(p['value'])

        placeholder = None
        if 'Shape' in inputs:
            placeholder = tensorflow.placeholder(tensorflow.float32, 
                                                shape = inputs['Shape']['value'], 
                                                name = 'Placeholder_{}'.format(key))
        # Here the childNeeds is a dict of [param name] : node.
        # This is because we don't actually call it recursively here, and still need to 
        # actually call the function. 
        node = Node(key,
                    placeholder,
                    evalFunc = func,
                    positionalParams = args,
                    dictParams = dictParams,
                    needtoFeed = childNeeds)
        if placeholder is not None:
            needtofeed[key] = node
        allNodes[key] = node
    # Apply TF functions to get a reference to a tensor
    elif tfOp is not None:
        params = {}
        args = []
        childNeeds = {}
        # Parse kwargs
        for name, p in inputs.get('kwargs', {}).items():
            v, isRef = GB.Deserialize(p)
            if isRef:
                __buildBranch(graph, v, allNodes, childNeeds)
                params[name] = allNodes[v].BackVariable
            else:
                params[name] = v
        # Parse args
        for p in inputs.get('args', []):
            v, isRef = GB.Deserialize(p)
            if isRef:
                __buildBranch(graph, v, allNodes, childNeeds)
                args.append(allNodes[v].BackVariable)
            else:
                args.append(v)
        # We don't need the name of a parameter here, because we have already called the function
        # If there is something we need to feed, it has a tensor attached, and we will feed that in 
        # at runtime. We just need to know which nodes need evaluation/replacing.
        node = Node(key, tfOp(*args, **params), needtoFeed = childNeeds)
        needtofeed.update(**childNeeds)
        if tfOp == tensorflow.placeholder:
            needtofeed.update({key : node})
            # It can't eval itself without being fed. So hack that in, I guess?
            # Will cause an infinite loop if things break. Wont if they dont!
            node.NeedtoFeed.update({key : node}) 

    if node is None:
        raise AssertionError('Unsupported node type: {}'.format(type))
    node.JSONRep = {key : graphNode}
    allNodes[key] = node
    return

def __resolveTFRef(ref):
    if 'tensorflow' not in ref:
        return None
    try:
        obj = eval(ref)
        if not callable(obj):
            obj = None
    except:
        obj = None

    return obj

