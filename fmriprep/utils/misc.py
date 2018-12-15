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


def meepi_optimal_comb_source_name(in_files):
    """
    Create a new source name when optimally
    combining multiple multi-echo EPIs

    >>> meepi_optimal_comb_source_name([
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


def read_crashfile(path):
    if path.endswith('.pklz'):
        return _read_pkl(path)
    elif path.endswith('.txt'):
        return _read_txt(path)


def _read_pkl(path):
    from nipype.utils.filemanip import loadcrash
    crash_data = loadcrash(path)
    data = {'file': path,
            'traceback': ''.join(crash_data['traceback'])}
    if 'node' in crash_data:
        data['node'] = crash_data['node']
        if data['node'].base_dir:
            data['node_dir'] = data['node'].output_dir()
        else:
            data['node_dir'] = "Node crashed before execution"
        data['inputs'] = sorted(data['node'].inputs.trait_get().items())
    return data


def _read_txt(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()
    data = {'file': path}
    traceback_start = 0
    if lines[0].startswith('Node'):
        data['node'] = lines[0].split(': ', 1)[1].strip()
        data['node_dir'] = lines[1].split(': ', 1)[1].strip()
        inputs = []
        cur_key = ''
        cur_val = ''
        for i, line in enumerate(lines[5:]):
            if line[0].isspace():
                cur_val += line
                continue

            if cur_val:
                inputs.append((cur_key, cur_val.strip()))

            if line.startswith("Traceback ("):
                traceback_start = i + 5
                break

            cur_key, cur_val = tuple(line.split(' = ', 1))

        data['inputs'] = sorted(inputs)
    else:
        data['node_dir'] = "Node crashed before execution"
    data['traceback'] = ''.join(lines[traceback_start:]).strip()
    return data


if __name__ == '__main__':
    pass
