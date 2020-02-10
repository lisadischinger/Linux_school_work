import numpy as np

a = np.array([[1], [2], [3]])
a_3d = np.atleast_3d(a)
a_db_stack = np.append(a_3d, np.atleast_3d(a), axis=2)

dud = 45
