# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
A tool to generate a tasks_list.sh file for running fmriprep
on subjects downloaded with datalad with sample_openfmri.py


"""

import os
import glob

CMDLINE = """\
{fmriprep_cmd} {bids_dir}/{dataset_dir} {output_dir}/{dataset_dir} participant \
-w {dataset_dir}/work --participant_label {participant_label} \
"""


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(
        description='OpenfMRI participants sampler, for FMRIPREP\'s testing purposes',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument('openfmri_dir', action='store',
                        help='the root folder of a the openfmri dataset')

    parser.add_argument('output_dir', action='store',
                        help='the directory where outputs should be stored')

    parser.add_argument('sample_file', action='store',
                        help='a YAML file containing the subsample schedule')

    # optional arguments
    parser.add_argument('--anat-only', action='store_true', default=False,
                        help='run only anatomical workflow')
    parser.add_argument('--nthreads', action='store', type=int,
                        help='number of total threads')
    parser.add_argument('--omp_nthreads', action='store', type=int,
                        help='number of threads for OMP-based interfaces')
    parser.add_argument('--mem-gb', action='store', type=int,
                        help='available memory in GB')
    parser.add_argument('--tasks-list-file', default='tasks_list.sh',
                        action='store', help='write output file')
    parser.add_argument('-t', '--tasks-filter', action='store', nargs='*',
                        help='run only specific tasks')
    parser.add_argument('--cmd-call', action='store', help='command to be run')
    return parser


def main():
    """Entry point"""
    import yaml
    opts = get_parser().parse_args()

    with open(opts.sample_file) as sfh:
        sampledict = yaml.load(sfh)

    cmdline = CMDLINE
    if opts.anat_only:
        cmdline += ' --anat-only'

    if opts.nthreads:
        cmdline += '--nthreads %d' % opts.nthreads

    if opts.omp_nthreads:
        cmdline += '--omp-nthreads %d' % opts.omp_nthreads

    if opts.mem_gb:
        cmdline += '--mem_mb %d' % (opts.mem_gb * 1000)

    if opts.tasks_filter:
        cmdline += '-t %s' % ' '.join(opts.tasks_filter)

    fmriprep_cmd = 'fmriprep'
    if opts.cmd_call is None:
        singularity_dir = os.getenv('SINGULARITY_BIN')
        singularity_img = sorted(
            glob.glob(os.path.join(singularity_dir, 'poldracklab_fmriprep_*')))
        if singularity_img:
            fmriprep_cmd = 'singularity run %s' % singularity_img[-1]

    task_cmds = []

    # Try to make this Python 2 compatible
    try:
        os.makedirs(opts.output_dir)
    except OSError as e:
        if e.errno != os.errno.EEXIST:
            raise

    for dset, sublist in sampledict.items():
        for sub in sublist:
            cmd = cmdline.format(
                fmriprep_cmd=fmriprep_cmd,
                bids_dir=opts.openfmri_dir,
                dataset_dir=dset,
                output_dir=opts.output_dir,
                participant_label=sub,
            )
            task_cmds.append(cmd)

    with open(opts.tasks_list_file, 'w') as tlfile:
        tlfile.write('\n'.join(task_cmds))


if __name__ == '__main__':
    main()
