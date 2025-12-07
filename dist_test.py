import numpy as np

arr = np.array([[1, 2], [4, 5], [0.3, 0.4], [7, 10]])

for element in arr:
    norm = np.linalg.norm(arr - element, axis=1)
    print(norm[norm != 0])

# x = np.array([[1, 2], [5, 8]])
# y = np.array([1, 3])
# print(x-y)
# print(np.linalg.norm(x-y, axis=1))