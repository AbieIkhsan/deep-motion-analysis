import numpy as np

X = np.load('HiddenActivations.npz')['Orig']

# 1to1 array for counting
countNumZeros = np.zeros(len(X))
countPercent = np.zeros(len(X))

#Go through only inner matrix
for i in range(len(X)):
    # Go through from 1 to 17924 checking if any arr got a number
    countNumZeros[i] = not X[i].any()
    countPercent[i] = np.non_zero(X[i])


countNumZeros = countNumZeros.astype(int)

assert (np.sum(countNumZeros)==0)

print 'Percent not zero' + str(np.sum(countPercent).astype(float)/len(X))