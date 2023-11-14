# -*- coding: utf-8 -*-
"""
Small illustrative example for Test-Driven Development in Python

(C) Merten Stender, TU Berlin
merten.stender@tu-berlin.de

"""

def adding_function(a, b):
    """Add function"""
    
    # catch some errors if no numbers are handed over
    if (type(a) is not float) and (type(a) is not int):
        raise ValueError('input a is not of type int or float')
        
    if (type(b) is not float) and (type(b) is not int):
        raise ValueError('input b is not of type int or float')
        
    return a+b