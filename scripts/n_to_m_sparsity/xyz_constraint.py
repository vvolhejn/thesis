import numpy as np
import itertools

attempt = [[1,1,0],[1,0,1],[0,1,1]]

for i1 in range(3):
    for i2 in range(i1, 3):
        for i3 in range(i2, 3):
            cur = np.array([attempt[i] for i in [i1,i2,i3]]).T
            cur.sort(axis=1)
            score = cur[:,1:].sum()
            print([i1,i2,i3], score)

