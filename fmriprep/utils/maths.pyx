#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Cython math extension

"""

from __future__ import division
import numpy as np
cimport numpy as np
cimport cython

from libc.math cimport fabs


cdef double c_bspl(double x) nogil:
    cdef:
        double x_t = fabs(x)

    if x_t >= 2.0:
        return(0.0)
    if x_t <= 1.0:
        return(2.0 / 3.0 - x_t**2 + 0.5 * x_t**3)
    elif x_t <= 2.0:
        return((2 - x_t)**3 / 6.0)


# PURE PYTHON IMPLEMENTATION:
# def bspl(x):
#     """Univariate cubic bspline implementation"""
#     if x >= 2.0:
#         return 0.0
#     if x <= 1.0:
#         return 2.0 / 3.0 - x**2 + 0.5 * x**3
#     elif x <= 2.0:
#         return (2 - x)**3 / 6.0

def bspl(double x):
  """
  Evaluate the univariate cubic bspline at x
  """
  return(c_bspl(x))
