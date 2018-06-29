from .Node import Node
from .TFFactoryException import TFFactoryException

class TFNode(Node):
    def __init__(self, id, tensor, needToFeed = None):
        
        self.Tensor = tensor
        self.NeedToFeed = needToFeed if needToFeed is not None else {}
        return super().__init__(id)

    def _eval(self, session = None, feed_dict = {}, newContext = True):
        if self.ID in self.NeedToFeed:
            raise TFFactoryException('Node {} was not fed during evaluation.'.format(self.ID))

        if len(self.NeedToFeed) == 0:
            val = session.run(self.Tensor)
        else:
            tf_feed_dict = {}
            # Swap out placeholders with node eval function results.
            for key, node in self.NeedToFeed.items():
                tf_feed_dict[node.Tensor] = node.Eval(session, feed_dict, False)
            val = session.run(self.Tensor, feed_dict = tf_feed_dict)
        return val