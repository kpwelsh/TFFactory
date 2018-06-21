from TFFactory import TFFactory, Node
import tensorflow as tf


factory = TFFactory()
print('Funciton Map: ')
print(factory.FunctionMap)
print('------------------')
'''
'3': {\
    'type': 'add',\
    'inputs': {\
        'A': {\
            'value': '1',\
            'type': 'ref'\
        },\
        'B': {\
            'value' :'2',\
            'type': 'ref'\
        }\
        }\
},\
'1': { \
    'type': 'variable',\
    'inputs': {\
        'Init': {\
            'value': [2,3],\
            'type': 'array'\
        }\
        }\
},\
'2': {\
    'type': 'variable',\
    'inputs': {\
        'Init': {\
            'value': [1,2],\
            'type': 'array'\
        }\
        }\
},\
'4': {\
    'type': 'fileSource',\
    'inputs': {\
        'FilePath': {\
            'value': 'Server\\file.txt',\
            'type': 'string'\
        },\
        'NRows': {\
            'value' :1,\
            'type': 'int'\
        }
        }\
},'''
testJSON = \
{ \
    '4': {\
        'type': 'fileSource',\
        'inputs': {\
            'FilePath': {\
                'value': 'Server\\file.txt',\
                'type': 'string'\
            },\
            'NRows': {\
                'value' :2,\
                'type': 'int'\
            }
            }\
    },
    '5': {\
        'type': 'parser',\
        'inputs': {\
            'Source': {\
                'value': '4',\
                'type': 'ref'\
            },\
            'SegmentDelimeter': {\
                'value' :';',\
                'type': 'string'\
            },\
            'DataDelimeter': {\
                'value' :',',\
                'type': 'string'\
            },\
            'SegmentIndex': {\
                'value' : 0,\
                'type': 'int'\
            },\
            'Shape': {\
                'value' : [2,5],\
                'type': 'array'\
            }
         }\
    },\
    '6': {\
        'type': 'parser',\
        'inputs': {\
            'Source': {\
                'value': '4',\
                'type': 'ref'\
            },\
            'SegmentDelimeter': {\
                'value' :';',\
                'type': 'string'\
            },\
            'DataDelimeter': {\
                'value' :',',\
                'type': 'string'\
            },\
            'SegmentIndex': {\
                'value' : 1,\
                'type': 'int'\
            },\
            'Shape': {\
                'value' : [2,5],\
                'type': 'array'\
            }
         }\
    },\
    '7': {\
        'type': 'add',\
        'inputs': {\
            'A': {\
                'value': '6',\
                'type': 'ref'\
            },\
            'B': {\
                'value' :'5',\
                'type': 'ref'\
            }\
         }\
    }\
}
print('Testing graph: ')
print(testJSON)
for k in testJSON:
    print('Key: {}'.format(k))
print('Node results: ')
res = factory.CreateTFGraph(testJSON)
print(res)
print()

sess = tf.Session()

tf.global_variables_initializer().run(session = sess)
Node.EvalContext = 1
for n in res:
    print('Evaluating {}:'.format(n)) 
    print(str(res[n].eval(session = sess)))

print('Session 2:')
for n in res:
    print('Evaluating {}:'.format(n))
    print(str(res[n].eval(session = sess)))

sess.close()
