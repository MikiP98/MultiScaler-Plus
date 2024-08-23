import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src/')))

import main
import unittest


class TestDummy(unittest.TestCase):
    def test_dummy(self):
        self.assertTrue(True)
