#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:52:08 2020

@author: Vincent
"""
 

def RunFunction(Fonction, args, return_dict):
    """Run a function in order to performe multi processing
    """
    #print(*args)
    out = Fonction(*args)
    #verboseprint(out)
    return_dict['output'] = out
    return 


