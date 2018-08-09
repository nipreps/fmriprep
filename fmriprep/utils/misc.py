#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Miscellaneous utilities
"""


def fix_multi_T1w_source_name(in_files):
    """
    Make up a generic source name when there are multiple T1s

    >>> fix_multi_T1w_source_name([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'])
    '/path/to/sub-045_T1w.nii.gz'

    """
    import os
    from nipype.utils.filemanip import filename_to_list
    base, in_file = os.path.split(filename_to_list(in_files)[0])
    subject_label = in_file.split("_", 1)[0].split("-")[1]
    return os.path.join(base, "sub-%s_T1w.nii.gz" % subject_label)


def add_suffix(in_files, suffix):
    """
    Wrap nipype's fname_presuffix to conveniently just add a prefix

    >>> add_suffix([
    ...     '/path/to/sub-045_ses-test_T1w.nii.gz',
    ...     '/path/to/sub-045_ses-retest_T1w.nii.gz'], '_test')
    'sub-045_ses-test_T1w_test.nii.gz'

    """
    import os.path as op
    from nipype.utils.filemanip import fname_presuffix, filename_to_list
    return op.basename(fname_presuffix(filename_to_list(in_files)[0],
                                       suffix=suffix))


if __name__ == '__main__':
    pass
