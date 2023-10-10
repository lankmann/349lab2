import numpy as np
from math import ceil

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

def MxMultiply(a, b):
  n = len(a)
  if n == 1:
    return a[0][0] * b[0][0]
  
  half = ceil(n / 2)

  x = a[:half, :half]
  y = a[:half, half:]
  z = a[half:, :half]
  w = a[half:, half:]
  p = b[:half, :half]
  q = b[:half, half:]
  r = b[half:, :half]
  s = b[half:, half:]

  c1 = MatrixSum(MxMultiply(x, p), MxMultiply(y, r))
  c2 = MatrixSum(MxMultiply(x, q), MxMultiply(y, s))
  c3 = MatrixSum(MxMultiply(z, p), MxMultiply(w, r))
  c4 = MatrixSum(MxMultiply(z, q), MxMultiply(w, s))

  result = np.empty((n, n))
  result[:half, :half] = c1
  result[:half, half:] = c2
  result[half:, :half] = c3
  result[half:, half:] = c4

  return result

def MatrixSum(a, b):
    n = len(a)
    result = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = a[i, j] + b[i, j]
    return result

def MatrixDiff(a, b):
    n = len(a)
    result = np.empty((n, n))
    for i in range(n):
        for j in range(n):
            result[i, j] = a[i][j] - b[i][j]
    return result

def Strassen(a, b):
    n = len(a)
    if n == 1:
        return np.array([[a[0, 0] * b[0][0]]])
    
    half = ceil(n / 2)
    x = a[:half, :half]
    y = a[:half, half:]
    z = a[half:, :half]
    w = a[half:, half:]
    p = b[:half, :half]
    q = b[:half, half:]
    r = b[half:, :half]
    s = b[half:, half:]
    p1 = Strassen(x, MatrixDiff(q, s))
    p2 = Strassen(MatrixSum(x, y), s)
    p3 = Strassen(MatrixSum(z, w), p)
    p4 = Strassen(w, MatrixDiff(r, p))
    p5 = Strassen(MatrixSum(x, w), MatrixSum(p, s))
    p6 = Strassen(MatrixDiff(y, w), MatrixSum(r, s))
    p7 = Strassen(MatrixDiff(x, z), MatrixSum(p, q))
    c11 = MatrixSum(MatrixDiff(MatrixSum(p5, p4), p2), p6)
    c12 = MatrixSum(p1, p2)
    c21 = MatrixSum(p3, p4)
    c22 = MatrixDiff(MatrixSum(p1, p5), MatrixSum(p3, p7))

    result = np.empty((n, n))
    result[:half, :half] = c11
    result[:half, half:] = c12
    result[half:, :half] = c21
    result[half:, half:] = c22
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

# print(NaiveMxMultiplication(a, b))
print(MxMultiply(a, b))
print(np.matmul(a, b))
print(Strassen(a, b))
