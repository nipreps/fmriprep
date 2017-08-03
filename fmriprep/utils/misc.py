#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
from __future__ import print_function, division, absolute_import, unicode_literals


def genfname(in_file, suffix=None, path=None, ext=None):
    from os import getcwd
    import os.path as op

    fname, fext = op.splitext(op.basename(in_file))
    if fext == '.gz':
        fname, fext2 = op.splitext(fname)
        fext = fext2 + fext

    if path is None:
        path = getcwd()

    if ext is None:
        ext = fext

    if ext != '' and not ext.startswith('.'):
        ext = '.' + ext

    if suffix is None:
        suffix = 'mod'

    return op.join(path, '{}_{}{}'.format(fname, suffix, ext))


def _first(inlist):
    if not isinstance(inlist, (list, tuple)):
        inlist = [inlist]

    return sorted(inlist)[0]


def fix_multi_T1w_source_name(in_files):
    import os
    # in case there are multiple T1s we make up a generic source name
    if isinstance(in_files, list):
        subject_label = in_files[0].split(os.sep)[-1].split("_")[0].split("-")[-1]
        base, _ = os.path.split(in_files[0])
        return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)
    else:
        return in_files


def add_suffix(in_files, suffix):
    import os.path as op
    from niworkflows.nipype.utils.filemanip import fname_presuffix, filename_to_list
    return op.basename(fname_presuffix(filename_to_list(in_files)[0],
                                       suffix=suffix))


if __name__ == '__main__':
    pass
