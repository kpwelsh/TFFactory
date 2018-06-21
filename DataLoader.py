import numpy as np

class DataLoader(object):
    def __init__(self):
        self.FileCache = {}
        return

    def sampleFile(self, filePath, nRows, delim = ',', caching = True):
        data = []
        if caching:
            if filePath in self.FileCache:
                for i in np.random.randint(0, len(self.FileCache[filePath]), nRows):
                    data.append(self.FileCache[filePath][i])
                return data
            else:
                self.FileCache[filePath] = []
        with open(filePath, 'r') as fin:
            i = 0
            for line in fin:
                row = line.strip('\n')
                if caching:
                    self.FileCache[filePath].append(row)
                if i < nRows:
                    data.append(row)
                else:
                    r = np.random.randint(0, i+1)
                    if r < nRows:
                        data[r] = row
                i+=1
        self.FileCache[filePath] = data
        return data


