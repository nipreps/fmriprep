#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-11-19 16:44:27
"""A simple script to print current mriqc version"""
from __future__ import print_function, absolute_import
import os
import sys

def main():
    """Import the file directly"""
    from info import __version__
    print(__version__)

if __name__ == '__main__':
    sys.path = [os.path.join(os.getcwd(), 'fmriprep')]
    main()
