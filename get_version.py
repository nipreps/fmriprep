#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
# @Author: oesteban
# @Date:   2017-06-13 09:42:38
import sys
import os.path as op


def main():
    sys.path.insert(0, op.abspath('.'))
    from fmriprep.__about__ import __version__
    print(__version__)


if __name__ == '__main__':
    main()
