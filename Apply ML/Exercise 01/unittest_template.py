# -*- coding: utf-8 -*-
"""

Small example for testing in Python using unittest

run this script, or call:
>>python test_add_fun_unittest.py

https://docs.python.org/3/library/unittest.html

See also YouTube video by Corey Schafer
https://www.youtube.com/watch?v=6tNS--WetLI


This script will test a simple function that adds two numbers:

def adding_function(a, b):
    # Add function
    
    # catch some errors if no numbers are handed over
    if (type(a) is not float) and (type(a) is not int):
        raise TypeError('input a is not of type int or float')
        
    if (type(b) is not float) and (type(b) is not int):
        raise TypeError('input b is not of type int or float')
        
    return a+b


(C) Merten Stender, TU Berlin

"""
import unittest
from add_fun import adding_function as add


# create some test cases by creating a test class
class TestAddingFunction(unittest.TestCase):
    # Requires the syntax test_ to recognize test
    
    # overview on test cases:
    # https://docs.python.org/3/library/unittest.html#test-cases
    
    def test_floats(self) -> None:
        # test the addition of two floats
        self.assertAlmostEqual(add(1.5, 1.5), 3)  # avoid rounding errors
        
    def test_ints(self) -> None:
        # test the addition of two ints
        self.assertEqual(add(1, 2), 3)
        
    def test_negatives(self) -> None:
        # test adding a negative and a positive number
        self.assertEqual(add(-1, 1), 0)
        
    def test_types(self) -> None:
        # test whether an exception is raised for non-numeric inputs
        # using the context manager
        with self.assertRaises(TypeError):
            add(5, 'five')
        


if __name__ == "__main__":
    # use the main to run this script directly from your editor
    unittest.main()