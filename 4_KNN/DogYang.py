import math
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy import stats


X1 = np.ones((250,3))
X2 = np.ones((500,3))
print(X1.shape)
print(X2.shape)
dist =  distance.cdist(X1,X2, 'euclidean')
np.save('dist.npy',dist)
print(dist.shape)