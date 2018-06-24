import TFFactory.GraphBuilder as tff
import TFFactory.Factory as factory
import tensorflow as tf
import json

if __name__ == '__main__':
    placeHolder = tff.placeholder(tf.int32, shape = [3], name = 'input')
    n = tff.Variable([-1, -2, -3], name = 'n1')
    b = tff.Variable(initial_value = [4, 5, 6], name = 'b')
    n = n + b + placeHolder

    graph = json.dumps(tff.CURRENT_GRAPH)
    print(graph)
    graph = json.loads(graph)
    compiledGraph = factory.CreateTFGraph(graph)

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session = sess)
        for k,v in compiledGraph.items():
            print('{} = {}'.format(k, v.eval(feed_dict = {'input' : [1,2,3]})))


