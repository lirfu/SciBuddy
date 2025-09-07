

class AttrDict(dict):
    """
        Transformes dictionary items to attributes.
    """
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


class AppendArray:
    """
        Array used for non-intrusive array concatenation (allocate once).
    """
    def __init__(self, array):
        """
            Use given array as memory allocation.
        """
        self.array = array
        self.idx = 0

    def __getitem__(self, idx):
        return self.array[idx]

    def __setitem__(self, idx, v):
        self.array[idx] = v

    def __lshift__(self, a):
        self[self.idx:self.idx+len(a), :a.shape[1]] = a
        self.idx += len(a)

    def __len__(self):
        return len(self.array)

    def append_info(self, a):
        print( f'{self.array.shape}({self.idx}) <- {a.shape}' )