import TFFactory.GraphBuilder as tff
import TFFactory.Factory as factory
import tensorflow as tf
import json

if __name__ == '__main__':

    pythonNode = tff.testAdd(1, -100, _shape=[1])

    placeHolder = tff.placeholder(tf.float32, shape=[3], name='input')
    n = tff.Variable([-1, -2, -3], name='n1', dtype=tf.float32)
    sliced = tff.SampleDirichlet([0.2, 0.8], [1, 5], name='dirichlet')[0, :]
    b = tff.Variable(initial_value=[4, 5, 6], name='b', dtype=tf.float32)
    n = n + b + placeHolder + pythonNode
    n = tff.AdamOptimizer(n, 1.0)

    graph = json.dumps(tff.CURRENT_GRAPH, cls=tff.Encoder, indent=2)
    print(graph)
    graph = json.loads(graph, cls=tff.Decoder)
    print('Compiling graph!')
    compiledGraph = factory.CreateTFGraph(graph)

    with tf.Session() as sess:
        tf.global_variables_initializer().run(session=sess)
        for k, v in compiledGraph.items():
            print('{} = {}'.format(
                k, v.Eval(session=sess, feed_dict={'input': [1, 2, 3]})))
