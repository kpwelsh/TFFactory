from TFFactory import TFFactory, Node, JSONNode
import tensorflow

ID_COUNTER = 0


def OverridingFunction(funcName):
    func_to_abstract = eval(funcName)
    assert callable(func_to_abstract), \
        'You need to pass a callable object in here'
    def abstractor(func):
        def abstracted(graph, *args, **kwargs):
            assert len(args) == 0, 'Don\'t use positional arguments'
            global ID_COUNTER
            ID_COUNTER += 1
            n = JSONNode(ID_COUNTER, funcName, kwargs)
            graph.update(n.asDict())
            return n
        abstracted.__doc__ = func_to_abstract.__doc__
        return abstracted
    return abstractor

@OverridingFunction('tensorflow.Variable')
def Variable(graph, inital_value):
    pass

@OverridingFunction('tensorflow.multiply')
def Multiply(graph, x, y):
    pass


if __name__ == '__main__':
    graph = {}
    n = Variable(graph, initial_value = [1,2,3])
    b = Variable(graph, initial_value = [4,5,6])
    n = Multiply(graph, x = n, y = b)

    factory = TFFactory()
    print(str(n))
    print(str(graph))
    compiledGraph = factory.CreateTFGraph(graph)

    print(str(compiledGraph))
    
    with tensorflow.Session() as sess:
        tensorflow.global_variables_initializer().run(session = sess)
        for k,v in compiledGraph.items():
            print(v.eval())