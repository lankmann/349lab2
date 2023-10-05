import numpy as np

def NaiveMxMultiplication(a, b):
  num_rows_a = len(a)
  num_cols_a = len(a[0])
  num_rows_b = len(b)
  num_cols_b = len(b[0])

  if num_cols_a != num_rows_b:
    return

  result = np.empty((num_rows_a, num_cols_b))

  for i in range(num_rows_a):
    for j in range(num_cols_b):
      sum = 0
      for k in range(num_cols_a):
        sum += a[i, k] * b[k, j]
      result[i, j] = sum
  return result

a = np.array([[3, 4, -2, 0, 6, -1, 4, 4],
              [2, 2, 0, 3, -3, -2, 6, -2],
              [1, 3, 8, 3, 9, 1, -2, -5],
              [4, -2, 5, -1, 0, 0, -6, 2],
              [3, 4, -1, -3, 3, 0, 8, 1],
              [5, 5, 3, -2, -1, -3, 7, 1],
              [1, -2, -3, 7, 1, 2, 0, 0],
              [0, 0, -1, 0, 2, -2, 0, -3]])
b = np.array([[6, 3, 4, 2, 2, 3, 2, 7],
              [0, 3, 2, -2, -2, 3, 3, 1],
              [0, 3, 1, 0, 0, 4, 5, -2],
              [-2, -2, 0, 5, 0, 7, -6, 0],
              [2, -2, -1, 7, 0, 1, -9, 0],
              [5, -2, -2, 2, -2, -1, 2, 3],
              [0, -2, -4, 1, 4, 0, -2, 1],
              [-1, 0, 0, 6, -5, 2, 0, -2]])

print(NaiveMxMultiplication(a, b))
print(np.matmul(a, b))
