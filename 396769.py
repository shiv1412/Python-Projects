# -*- coding: utf-8 -*-
"""
Created on Sat May 23 20:57:41 2020

@author: sharm
"""

def func(str):
    if str[0] == "?" :
        if str[1] != "?" :
            if int(str[1])> 3 :
                str = "1" + str[1:]
            else:
                str = "2" + str[1:]
        else:
            str = "2" + str[1:]
    if str[1] == "?" :
        if(str[0] == "2"):
            str = str[:1] + "3" + str[2:]
        else:
            str = str[:1] + "9" + str[2:]
        if str[3] == "?" :
            str = str[:3] + "5" + str[4:]
        if s[4] == "?" :
            str = str[:4] + "9"
        return str
    print(str)
            