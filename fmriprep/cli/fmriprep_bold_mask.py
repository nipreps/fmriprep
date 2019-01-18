# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""Run the BOLD reference+mask workflow"""
import os
from nipype.utils.filemanip import hash_infile


def get_parser():
    """Build parser object"""
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter

    parser = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)

    parser.add_argument('input_file', action='store', help='the input file')
    return parser


def main():
    """Entry point"""
    from fmriprep.workflows.bold.util import init_bold_reference_wf
    opts = get_parser().parse_args()

    wf = init_bold_reference_wf(1, name=hash_infile(opts.input_file), gen_report=True)
    wf.inputs.inputnode.bold_file = opts.input_file
    wf.base_dir = os.getcwd()
    wf.run()


if __name__ == '__main__':
    raise RuntimeError("fmriprep/cli/run.py should not be run directly;\n"
                       "Please `pip install` fmriprep and use the `fmriprep` command")
