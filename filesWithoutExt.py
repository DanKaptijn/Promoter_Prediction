"""
author: Dan Kaptijn
date: 11/03/2020
PyVersion: 3.7.3

aim: make a script that will use regex to get a list of file names without their extensions (e.g. a text file without .txt)
"""

import os
import re

Path = "../Atom Code/"
for file in os.listdir(Path):
    result = re.match(".+?\.", file)
    if result:
        print(result.group(0))
