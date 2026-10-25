"""Conservative matrix retrieval with unchanged exact scalar boundary checks."""
import numpy as np


class QueryMatrix:
    def __init__(self, values):
        self.values = list(values)
        self.matrix = np.stack([value[-1] for value in self.values])
        self.norms = np.linalg.norm(self.matrix.astype('float64'),axis=1)
        self.matrix.flags.writeable = False
        self.norms.flags.writeable = False

    def matches(self, query, threshold):
        dimension=len(query)
        unit=np.finfo(np.float32).eps/2
        gamma=dimension*unit/(1-dimension*unit)
        errors=(4*gamma+16*np.finfo(np.float64).eps)*self.norms*float(np.linalg.norm(query.astype('float64')))
        errors+=dimension*np.finfo(np.float32).tiny
        upper=self.matrix@query+errors
        for index in np.flatnonzero(~np.isfinite(upper) | (upper>=threshold)):
            value=self.values[index]
            if float(np.dot(query,value[-1]))>=threshold:
                yield value
