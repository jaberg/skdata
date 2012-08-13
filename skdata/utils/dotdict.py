class dotdict(dict):
    def __getattr__(self, attr):
        if attr in self:
            return self.get(attr, None)
        else:
            raise KeyError
    __setattr__= dict.__setitem__
    __delattr__= dict.__delitem__
