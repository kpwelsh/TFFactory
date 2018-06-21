import tensorflow
import numpy as np
import json
from TFFactoryException import TFFactoryException
from DataLoader import DataLoader

class JSONNode:
    def __init__(self, id, type, params):
        self.ID = str(id)
        self.type = str(type)
        self.Inputs = {}
        for name, value in params.items():
            if isinstance(value, JSONNode):
                self.Inputs[name] = {'value' : value.ID, 'type' : 'ref'}
            else:
                self.Inputs[name] = {'value' : json.loads(json.dumps(value)), 'type' : 'dynamic'}
        return
    
    def asDict(self):
        d = {
            self.ID : {
                'type' : self.type,
                'inputs' : self.Inputs
            }
        }
        return d
    
    def __str__(self):
        return json.dumps(self.asDict())

    def __eq__(self, other):
        return other.ID == self.ID

    def __hash__(self):
        return hash(self.ID)
    
class Node:
    EvalContext = None
    def __init__(self, id, backingVariable = None, evalFunc = None, dictParams = {}, needtoFeed = {}):
        if backingVariable is None and evalFunc is None:
            raise AssertionError('Node cannot be created without a backing function or backing variable')
        assert id is not None, 'Nodes need a unique ID'

        self.BackVariable = backingVariable
        self.EvalFunc = evalFunc
        self.DictParams = dictParams
        self.NeedtoFeed = needtoFeed
        self.LastContext = {}
        self.JSONRep = {}
        self.ID = id
        return

    def eval(self, session = None, feed_dict = {}, newContext = True):
        if self.ID in feed_dict:
            return feed_dict[self.ID]
        if newContext:
            Node.EvalContext = np.random.random()
        if Node.EvalContext not in self.LastContext:
            self.LastContext = {}
        if Node.EvalContext is not None:
            if Node.EvalContext in self.LastContext:
                return self.LastContext[Node.EvalContext]
            
        val = None
        if self.EvalFunc is not None:
            _feed_dict = {}
            for name, node in self.NeedtoFeed.items():
                _feed_dict[name] = node.eval(session = session, feed_dict = feed_dict, newContext = False)
            val = self.EvalFunc(**self.DictParams, **_feed_dict)
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

class TFFactory:
    def __init__(self):
        self.DataLoader = DataLoader()
        self.FunctionMap = self.functionMap()
        return

    def CreateTFGraph(self, graph):
        nodes = {}

        for key in graph:
            self.buildBranch(graph, key, nodes)

        return nodes

    def buildBranch(self, graph, key, allNodes, needtofeed = None):
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

        tfOp = self.resolveTFRef(type)
        # Build a function reference to eval when it comes time.
        if type in self.FunctionMap['ops']:
            func = self.FunctionMap['ops'][type]
            dictParams = {}
            childNeeds = {}
            for name, obj in inputs.items():
                if obj['type'] == 'ref':
                    self.buildBranch(graph, obj['value'], allNodes)
                    childNeeds[name] = allNodes[obj['value']]
                else:
                    dictParams[name] = obj['value']
            placeholder = None
            if 'Shape' in inputs:
                placeholder = tensorflow.placeholder(tensorflow.float32, shape = inputs['Shape']['value'], name = 'Placeholder_{}'.format(key))
            node = Node(key,
                        placeholder,
                        evalFunc = func,
                        dictParams = dictParams,
                        needtoFeed = childNeeds)
            if placeholder is not None:
                needtofeed[key] = node
            allNodes[key] = node
        # Apply TF functions to get a reference to a tensor
        elif tfOp is not None:
            params = {}
            childNeeds = {}  
            for name, p in inputs.items():
                if p['type'] == 'ref':
                    nodeId = p['value']
                    self.buildBranch(graph, nodeId, allNodes, childNeeds)
                    params[name] = allNodes[nodeId].BackVariable
                else:
                    params[name] = p['value']
            print(params)
            node = Node(key, tfOp(**params), needtoFeed = childNeeds)
            needtofeed = {**needtofeed, **childNeeds}

        if node is None:
            raise AssertionError('Unsupported node type: {}'.format(type))
        node.JSONRep = graphNode
        allNodes[key] = node
        return

    def readFile(self, fp, delim, nRows):
        return self.DataLoader.sampleFile(fp, nRows, delim, caching = True)

    def splitFile(self, Source, SegmentDelimeter, DataDelimeter, SegmentIndex, Shape):
        data = []
        for row in Source:
            data.append(list(map(float, row.split(SegmentDelimeter)[SegmentIndex].split(DataDelimeter))))
        return np.array(data).reshape((*Shape))

    def resolveTFRef(self, ref):
        if 'tensorflow' not in ref:
            return None
        try:
            obj = eval(ref)
            if not callable(obj):
                obj = None
        except:
            obj = None

        return obj

    def placeHolder(self):
        raise TFFactoryException('Place holder variable was not fed during execution.')

    def functionMap(self):
        map = {}
        ops = {}

        ops['fileSource'] = lambda FilePath, NRows : self.readFile(FilePath, ',', NRows)
        ops['parser'] = self.splitFile
        ops['placeHolder'] = self.placeHolder

        map['ops'] = ops
        return map
