import unittest
import numpy as np
from matrix import LinearMxMultiply, DncMxMultiply, StrassenMxMultiply
from matplotlib import pyplot as plt

class MaxTest(unittest.TestCase):
  def setUp(self):
    self.test_values = []
    for i in range(32):
      rows = np.random.randint(1, 64)
      cols_a = np.random.randint(1, 64)
      cols_b = np.random.randint(1, 64)
      a = np.random.randint(-9, 9, (rows, cols_a))
      b = np.random.randint(-9, 9, (cols_a, cols_b))
      self.test_values.append((a, b))

  def test_linear_matrix_multiplication(self):
    self.matrix_multiplication(LinearMxMultiply)

  def test_DNC_matrix_multiplication(self):
    self.matrix_multiplication(DncMxMultiply)

  def test_strassen_matrix_multiplication(self):
    self.matrix_multiplication(StrassenMxMultiply)

  def matrix_multiplication(self, f):
    for a, b in self.test_values:
      expected = np.matmul(a, b)
      actual = f(a, b)
      self.assertSequenceEqual(np.shape(expected), np.shape(actual))
      for i in range(len(actual)):
        self.assertListEqual(expected[i].tolist(), actual[i].tolist())
      
    

if __name__ == '__main__':
  unittest.main()