# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     : Zijiang Yang                                   
# @file       : YamlRules.py
# @Time       : 2021/12/9 10:30

import yaml


# define custom string join func
def string_join(loader, node):
    seq = loader.construct_sequence(node)
    return ''.join([str(i) for i in seq])

