import numpy as np

class Node:
    EvalContext = None
    def __init__(self, id):
        self.ID = id
        self.LastContext = {}
        return

    def Eval(self, session = None, feed_dict = {}, newContext = True):
        if self.ID in feed_dict:
            return feed_dict[self.ID]

        if newContext:
            Node.EvalContext = np.random.random()
        if Node.EvalContext not in self.LastContext:
            self.LastContext = {}
        if Node.EvalContext is not None:
            if Node.EvalContext in self.LastContext:
                return self.LastContext[Node.EvalContext]
        
        val = self._eval(session = session, feed_dict = feed_dict, newContext = newContext)
        if Node.EvalContext is not None:
            self.LastContext[Node.EvalContext] = val
        return val

    def _eval(self, session = None, feed_dict = {}, newContext = True):
        raise NotImplementedError

    def __hash__(self):
        return hash(self.ID)
    def __eq__(self, other):
        return other.ID == self.ID