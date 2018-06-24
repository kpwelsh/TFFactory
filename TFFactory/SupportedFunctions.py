from .DataLoader import DataLoader
DL = DataLoader()

def readFile(fp, delim, nRows):
    return DL.sampleFile(fp, nRows, delim, caching = True)

def splitFile(Source, SegmentDelimeter, DataDelimeter, SegmentIndex, Shape):
    data = []
    for row in Source:
        data.append(list(map(float, row.split(SegmentDelimeter)[SegmentIndex].split(DataDelimeter))))
    return np.array(data).reshape((*Shape))
