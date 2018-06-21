import tensorflow as tf
import numpy as np
from DataLoader import DataLoader

class Node:
    EvalContext = None
    def __init__(self, backingVariable = None, evalFunc = None, positionalParams = (), dictParams = {}, needtoFeed = {}):
        if backingVariable is None and evalFunc is None:
            raise AssertionError('Node cannot be created without a backing function or backing variable')
        self.BackVariable = backingVariable
        self.EvalFunc = evalFunc
        self.PositionalParams = positionalParams
        self.DictParams = dictParams
        self.NeedtoFeed = needtoFeed
        self.LastContext = {}
        return

    def eval(self, session = None):
        if Node.EvalContext not in self.LastContext:
            self.LastContext = {}
        if Node.EvalContext is not None:
            if Node.EvalContext in self.LastContext:
                return self.LastContext[Node.EvalContext]
            
        val = None
        if self.EvalFunc is not None:
            feed_dict = {}
            for name, node in self.NeedtoFeed.items():
                feed_dict[name] = node.eval(session = session)
            val = self.EvalFunc(*self.PositionalParams, **self.DictParams, **feed_dict)
        elif self.BackVariable is not None:
            if len(self.NeedtoFeed) == 0:
                val = self.BackVariable.eval(session = session)
            else:
                feed_dict = {}
                # Swap out placeholders with node eval function results.
                for key, node in self.NeedtoFeed.items():
                    feed_dict[node.BackVariable] = node.eval(session = session) 
                val = self.BackVariable.eval(feed_dict = feed_dict, session = session)
        if Node.EvalContext is not None:
            self.LastContext[Node.EvalContext] = val
        if val is None: raise AssertionError('Good luck with this one.')
        return val

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
                needtofeed.update({allNodes[key].BackVariable : allNodes[key]})
            else: # If it is a TF op, then compile its child dependencies.
                needtofeed.update(**allNodes[key].NeedtoFeed)
            return

        graphNode = graph[str(key)] # Stop it with the data type key errors...
        type = graphNode['type']
        inputs = graphNode['inputs']
        node = None
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
                placeholder = tf.placeholder(tf.float32, shape = inputs['Shape']['value'], name = 'Placeholder_{}'.format(key))
            node = Node(placeholder,
                        evalFunc = func,
                        dictParams = dictParams,
                        needtoFeed = childNeeds)
            if placeholder is not None:
                needtofeed[key] = node
            allNodes[key] = node
        # Apply TF functions to get a reference to a tensor
        elif type in self.FunctionMap['tfops']:
            params = {}
            childNeeds = {}  
            for name, p in inputs.items():
                if p['type'] == 'ref':
                    nodeId = p['value']
                    self.buildBranch(graph, nodeId, allNodes, childNeeds)
                    params[name] = allNodes[nodeId].BackVariable
            node = Node(self.applyTFOp(type, params), needtoFeed = childNeeds)
            needtofeed = {**needtofeed, **childNeeds}
        elif type == 'variable':
            node = Node(tf.Variable(inputs['Init']['value']))

        if node is None:
            raise AssertionError('Unsupported node type: {}'.format(type))
        allNodes[key] = node
        return

    def readFile(self, fp, delim, nRows):
        return self.DataLoader.sampleFile(fp, nRows, delim, caching = True)

    def splitFile(self, Source, SegmentDelimeter, DataDelimeter, SegmentIndex, Shape):
        data = []
        for row in Source:
            data.append(list(map(float, row.split(SegmentDelimeter)[SegmentIndex].split(DataDelimeter))))
        return np.array(data).reshape((*Shape))

    def applyTFOp(self, tfop, params):
        if tfop not in self.FunctionMap['tfops']:
            raise NotImplementedError('Unsupported TF operation: {0}'.format(tfop))

        return self.FunctionMap['tfops'][tfop](**params)


    def functionMap(self):
        map = {}
        tfops = {}
        ops = {}

        tfops['multiply'] = lambda A, B: A * B
        tfops['add'] = lambda A, B: A + B
        tfops['subtract'] = lambda A, B: A - B

        ops['fileSource'] = lambda FilePath, NRows : self.readFile(FilePath, ',', NRows)
        ops['parser'] = self.splitFile

        map['tfops'] = tfops
        map['ops'] = ops
        return map
