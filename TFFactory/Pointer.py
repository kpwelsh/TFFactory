class Pointer():
    def __init__(self, ref):
        self.Ref = ref
    
    @classmethod
    def IsInstance(cls, obj):
        return isinstance(obj, cls)