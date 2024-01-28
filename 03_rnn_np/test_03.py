import unittest
import numpy as np
import math
import sys
import os
import main
from scipy import special as sp
import scipy

class TestOps(unittest.TestCase):

    def test_softmax(self):
        x:np.ndarray=np.random.uniform(-1.00, 1.00, size=(10,5))
        model=main.RNN()
        accuracy=10
        self.assertTrue(np.array_equal(
            np.round(sp.softmax(x), decimals=accuracy),
            np.round(model.softmax(x), decimals=accuracy)
            )
        )

if __name__ == '__main__':
    unittest.main()
