# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Multi-echo EPI utilities."""


def combine_meepi_source(in_files):
    """
    Create a new source name when optimally
    combining multiple multi-echo EPIs

    >>> combine_meepi_source([
    ...     'sub-01_run-01_echo-1_bold.nii.gz',
    ...     'sub-01_run-01_echo-2_bold.nii.gz',
    ...     'sub-01_run-01_echo-3_bold.nii.gz',])
    'sub-01_run-01_bold.nii.gz'

    """
    import os
    from nipype.utils.filemanip import filename_to_list
    base, in_file = os.path.split(filename_to_list(in_files)[0])
    entities = [ent for ent in in_file.split('_') if not ent.startswith('echo-')]
    basename = '_'.join(entities)
    return os.path.join(base, basename)
