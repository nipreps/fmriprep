#!/usr/bin/env python
"""
The fMRIPrep on Docker wrapper


This is a lightweight Python wrapper to run fMRIPrep.
Docker must be installed and running. This can be checked
running ::

  docker info

Please report any feedback to our GitHub repository
(https://github.com/poldracklab/fmriprep) and do not
forget to credit all the authors of software that fMRIPrep
uses (https://fmriprep.readthedocs.io/en/latest/citing.html).
"""
import sys
import os
import re
import subprocess
from warnings import warn

__version__ = '99.99.99'
__packagename__ = 'fmriprep-docker'
__author__ = 'The CRN developers'
__copyright__ = 'Copyright 2018, Center for Reproducible Neuroscience, Stanford University'
__credits__ = ['Craig Moodie', 'Ross Blair', 'Oscar Esteban', 'Chris Gorgolewski',
               'Shoshana Berleant', 'Christopher J. Markiewicz', 'Russell A. Poldrack']
__license__ = '3-clause BSD'
__maintainer__ = 'Christopher J. Markiewicz'
__email__ = 'crn.poldracklab@gmail.com'
__url__ = 'https://github.com/poldracklab/fmriprep'
__bugreports__ = 'https://github.com/poldracklab/fmriprep/issues'

__description__ = """\
fMRIprep is a functional magnetic resonance image pre-processing pipeline \
that is designed to provide an easily accessible, state-of-the-art interface \
that is robust to differences in scan acquisition protocols and that requires \
minimal user input, while providing easily interpretable and comprehensive \
error and output reporting."""
__longdesc__ = """\
This package is a basic wrapper for fMRIprep that generates the appropriate
Docker commands, providing an intuitive interface to running the fMRIprep
workflow in a Docker environment."""

DOWNLOAD_URL = (
    'https://pypi.python.org/packages/source/{name[0]}/{name}/{name}-{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))

CLASSIFIERS = [
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Science/Research',
    'License :: OSI Approved :: BSD License',
    'Programming Language :: Python :: 2.7',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
]


MISSING = """
Image '{}' is missing
Would you like to download? [Y/n] """
PKG_PATH = '/usr/local/miniconda/lib/python3.7/site-packages'

# Monkey-patch Py2 subprocess
if not hasattr(subprocess, 'DEVNULL'):
    subprocess.DEVNULL = -3

if not hasattr(subprocess, 'run'):
    # Reimplement minimal functionality for usage in this file
    def _run(args, stdout=None, stderr=None):
        from collections import namedtuple
        result = namedtuple('CompletedProcess', 'stdout stderr returncode')

        devnull = None
        if subprocess.DEVNULL in (stdout, stderr):
            devnull = open(os.devnull, 'r+')
            if stdout == subprocess.DEVNULL:
                stdout = devnull
            if stderr == subprocess.DEVNULL:
                stderr = devnull

        proc = subprocess.Popen(args, stdout=stdout, stderr=stderr)
        stdout, stderr = proc.communicate()
        res = result(stdout, stderr, proc.returncode)

        if devnull is not None:
            devnull.close()

        return res
    subprocess.run = _run


# De-fang Python 2's input - we don't eval user input
try:
    input = raw_input
except NameError:
    pass


def check_docker():
    """Verify that docker is installed and the user has permission to
    run docker images.

    Returns
    -------
    -1  Docker can't be found
     0  Docker found, but user can't connect to daemon
     1  Test run OK
     """
    try:
        ret = subprocess.run(['docker', 'version'], stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
    except OSError as e:
        from errno import ENOENT
        if e.errno == ENOENT:
            return -1
        raise e
    if ret.stderr.startswith(b"Cannot connect to the Docker daemon."):
        return 0
    return 1


def check_image(image):
    """Check whether image is present on local system"""
    ret = subprocess.run(['docker', 'images', '-q', image],
                         stdout=subprocess.PIPE)
    return bool(ret.stdout)


def check_memory(image):
    """Check total memory from within a docker container"""
    ret = subprocess.run(['docker', 'run', '--rm', '--entrypoint=free',
                          image, '-m'],
                         stdout=subprocess.PIPE)
    if ret.returncode:
        return -1

    mem = [line.decode().split()[1]
           for line in ret.stdout.splitlines()
           if line.startswith(b'Mem:')][0]
    return int(mem)


def merge_help(wrapper_help, target_help):
    # Matches all flags with up to one nested square bracket
    opt_re = re.compile(r'(\[--?[\w-]+(?:[^\[\]]+(?:\[[^\[\]]+\])?)?\])')
    # Matches flag name only
    flag_re = re.compile(r'\[--?([\w-]+)[ \]]')

    # Normalize to Unix-style line breaks
    w_help = wrapper_help.rstrip().replace('\r', '')
    t_help = target_help.rstrip().replace('\r', '')

    w_usage, w_details = w_help.split('\n\n', 1)
    w_groups = w_details.split('\n\n')
    t_usage, t_details = t_help.split('\n\n', 1)
    t_groups = t_details.split('\n\n')

    w_posargs = w_usage.split('\n')[-1].lstrip()
    t_posargs = t_usage.split('\n')[-1].lstrip()

    w_options = opt_re.findall(w_usage)
    w_flags = sum(map(flag_re.findall, w_options), [])
    t_options = opt_re.findall(t_usage)
    t_flags = sum(map(flag_re.findall, t_options), [])

    # The following code makes this assumption
    assert w_flags[:2] == ['h', 'version']
    assert w_posargs.replace(']', '').replace('[', '') == t_posargs

    # Make sure we're not clobbering options we don't mean to
    overlap = set(w_flags).intersection(t_flags)
    expected_overlap = set(['h', 'version', 'w', 'template-resampling-grid',
                            'output-grid-reference', 'fs-license-file', 'use-plugin'])
    assert overlap == expected_overlap, "Clobbering options: {}".format(
        ', '.join(overlap - expected_overlap))

    sections = []

    # Construct usage
    start = w_usage[:w_usage.index(' [')]
    indent = ' ' * len(start)
    new_options = sum((
        w_options[:2],
        [opt for opt, flag in zip(t_options, t_flags) if flag not in overlap],
        w_options[2:]
    ), [])
    opt_line_length = 79 - len(start)
    length = 0
    opt_lines = [start]
    for opt in new_options:
        opt = ' ' + opt
        olen = len(opt)
        if length + olen <= opt_line_length:
            opt_lines[-1] += opt
            length += olen
        else:
            opt_lines.append(indent + opt)
            length = olen
    opt_lines.append(indent + ' ' + t_posargs)
    sections.append('\n'.join(opt_lines))

    # Use target description and positional args
    sections.extend(t_groups[:2])

    for line in t_groups[2].split('\n')[1:]:
        content = line.lstrip().split(',', 1)[0]
        if content[1:] not in overlap:
            w_groups[2] += '\n' + line

    sections.append(w_groups[2])

    # All remaining sections, show target then wrapper (skipping duplicates)
    sections.extend(t_groups[3:] + w_groups[6:])
    return '\n\n'.join(sections)


def get_parser():
    """Defines the command line interface of the wrapper"""
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        add_help=False)

    # Standard FMRIPREP arguments
    parser.add_argument('bids_dir', nargs='?', type=os.path.abspath,
                        default='')
    parser.add_argument('output_dir', nargs='?', type=os.path.abspath,
                        default='')
    parser.add_argument('analysis_level', nargs='?', choices=['participant'],
                        default='participant')

    parser.add_argument('-h', '--help', action='store_true',
                        help="show this help message and exit")
    parser.add_argument('--version', action='store_true',
                        help="show program's version number and exit")

    # Allow alternative images (semi-developer)
    parser.add_argument('-i', '--image', metavar='IMG', type=str,
                        default='poldracklab/fmriprep:{}'.format(__version__),
                        help='image name')

    # Options for mapping files and directories into container
    # Update `expected_overlap` variable in merge_help() when adding to this
    g_wrap = parser.add_argument_group(
        'Wrapper options',
        'Standard options that require mapping files into the container')
    g_wrap.add_argument('-w', '--work-dir', action='store', type=os.path.abspath,
                        help='path where intermediate results should be stored')
    g_wrap.add_argument(
        '--output-grid-reference', required=False, action='store', type=os.path.abspath,
        help='Deprecated after FMRIPREP 1.0.8. Please use --template-resampling-grid instead.')
    g_wrap.add_argument(
        '--template-resampling-grid', required=False, action='store', type=str,
        help='Keyword ("native", "1mm", or "2mm") or path to an existing file. '
             'Allows to define a reference grid for the resampling of BOLD images in template '
             'space. Keyword "native" will use the original BOLD grid as reference. '
             'Keywords "1mm" and "2mm" will use the corresponding isotropic template '
             'resolutions. If a path is given, the grid of that image will be used. '
             'It determines the field of view and resolution of the output images, '
             'but is not used in normalization.')
    g_wrap.add_argument(
        '--fs-license-file', metavar='PATH', type=os.path.abspath,
        default=os.getenv('FS_LICENSE', None),
        help='Path to FreeSurfer license key file. Get it (for free) by registering'
             ' at https://surfer.nmr.mgh.harvard.edu/registration.html')
    g_wrap.add_argument(
        '--use-plugin', metavar='PATH', action='store', default=None,
        type=os.path.abspath, help='nipype plugin configuration file')

    # Developer patch/shell options
    g_dev = parser.add_argument_group(
        'Developer options',
        'Tools for testing and debugging FMRIPREP')
    g_dev.add_argument('-f', '--patch-fmriprep', metavar='PATH',
                       type=os.path.abspath,
                       help='working fmriprep repository')
    g_dev.add_argument('-n', '--patch-niworkflows', metavar='PATH',
                       type=os.path.abspath,
                       help='working niworkflows repository')
    g_dev.add_argument('-p', '--patch-nipype', metavar='PATH',
                       type=os.path.abspath,
                       help='working nipype repository')
    g_dev.add_argument('--shell', action='store_true',
                       help='open shell in image instead of running FMRIPREP')
    g_dev.add_argument('--config', metavar='PATH', action='store',
                       type=os.path.abspath, help='Use custom nipype.cfg file')
    g_dev.add_argument('-e', '--env', action='append', nargs=2, metavar=('ENV_VAR', 'value'),
                       help='Set custom environment variable within container')
    g_dev.add_argument('-u', '--user', action='store',
                       help='Run container as a given user/uid')

    return parser


def main():
    """Entry point"""

    parser = get_parser()
    # Capture additional arguments to pass inside container
    opts, unknown_args = parser.parse_known_args()

    # Set help if no directories set
    if (opts.bids_dir, opts.output_dir, opts.version) == ('', '', False):
        opts.help = True

    # Stop if no docker / docker fails to run
    check = check_docker()
    if check < 1:
        if opts.version:
            print('fmriprep wrapper {!s}'.format(__version__))
        if opts.help:
            parser.print_help()
        if check == -1:
            print("fmriprep: Could not find docker command... Is it installed?")
        else:
            print("fmriprep: Make sure you have permission to run 'docker'")
        return 1

    # For --help or --version, ask before downloading an image
    if not check_image(opts.image):
        resp = 'Y'
        if opts.version:
            print('fmriprep wrapper {!s}'.format(__version__))
        if opts.help:
            parser.print_help()
        if opts.version or opts.help:
            try:
                resp = input(MISSING.format(opts.image))
            except KeyboardInterrupt:
                print()
                return 1
        if resp not in ('y', 'Y', ''):
            return 0
        print('Downloading. This may take a while...')

    # Warn on low memory allocation
    mem_total = check_memory(opts.image)
    if mem_total == -1:
        print('Could not detect memory capacity of Docker container.\n'
              'Do you have permission to run docker?')
        return 1
    if not (opts.help or opts.version or '--reports-only' in unknown_args) and mem_total < 8000:
        print('Warning: <8GB of RAM is available within your Docker '
              'environment.\nSome parts of fMRIPrep may fail to complete.')
        if '--mem_mb' not in unknown_args:
            resp = 'N'
            try:
                resp = input('Continue anyway? [y/N]')
            except KeyboardInterrupt:
                print()
                return 1
            if resp not in ('y', 'Y', ''):
                return 0

    ret = subprocess.run(['docker', 'version', '--format', "{{.Server.Version}}"],
                         stdout=subprocess.PIPE)
    docker_version = ret.stdout.decode('ascii').strip()

    command = ['docker', 'run', '--rm', '-it', '-e',
               'DOCKER_VERSION_8395080871=%s' % docker_version]

    # Patch working repositories into installed package directories
    for pkg in ('fmriprep', 'niworkflows', 'nipype'):
        repo_path = getattr(opts, 'patch_' + pkg)
        if repo_path is not None:
            command.extend(['-v',
                            '{}:{}/{}:ro'.format(repo_path, PKG_PATH, pkg)])

    if opts.env:
        for envvar in opts.env:
            command.extend(['-e', '%s=%s' % tuple(envvar)])

    if opts.user:
        command.extend(['-u', opts.user])

    if opts.fs_license_file:
        command.extend([
            '-v', '{}:/opt/freesurfer/license.txt:ro'.format(
                opts.fs_license_file)])

    main_args = []
    if opts.bids_dir:
        command.extend(['-v', ':'.join((opts.bids_dir, '/data', 'ro'))])
        main_args.append('/data')
    if opts.output_dir:
        command.extend(['-v', ':'.join((opts.output_dir, '/out'))])
        main_args.append('/out')
    main_args.append(opts.analysis_level)

    if opts.work_dir:
        command.extend(['-v', ':'.join((opts.work_dir, '/scratch'))])
        unknown_args.extend(['-w', '/scratch'])

    if opts.config:
        command.extend(['-v', ':'.join((opts.config,
                                        '/root/.nipype/nipype.cfg', 'ro'))])

    if opts.use_plugin:
        command.extend(['-v', ':'.join((opts.use_plugin, '/tmp/plugin.yml',
                                        'ro'))])
        unknown_args.extend(['--use-plugin', '/tmp/plugin.yml'])

    template_target = opts.template_resampling_grid or opts.output_grid_reference
    if template_target is not None:
        if opts.output_grid_reference is not None:
            warn('Option --output-grid-reference is deprecated, please use '
                 '--template-resampling-grid', DeprecationWarning)
        if template_target not in ['native', '2mm' '1mm']:
            target = '/imports/' + os.path.basename(template_target)
            command.extend(['-v', ':'.join((os.path.abspath(
                template_target), target, 'ro'))])
        unknown_args.extend(['--template-resampling-grid', template_target])

    if opts.shell:
        command.append('--entrypoint=bash')

    command.append(opts.image)

    # Override help and version to describe underlying program
    # Respects '-i' flag, so will retrieve information from any image
    if opts.help:
        command.append('-h')
        targethelp = subprocess.check_output(command).decode()
        print(merge_help(parser.format_help(), targethelp))
        return 0
    elif opts.version:
        # Get version to be run and exit
        command.append('--version')
        ret = subprocess.run(command)
        return ret.returncode

    if not opts.shell:
        command.extend(main_args)
        command.extend(unknown_args)

    print("RUNNING: " + ' '.join(command))
    ret = subprocess.run(command)
    if ret.returncode:
        print("fMRIPrep: Please report errors to {}".format(__bugreports__))
    return ret.returncode


if __name__ == '__main__':
    sys.exit(main())
