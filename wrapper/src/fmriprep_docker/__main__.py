#!/usr/bin/env python
"""
The *fMRIPrep* on Docker wrapper

This is a lightweight Python wrapper to run *fMRIPrep*.
Docker must be installed and running. This can be checked
running ::

  docker info

Please acknowledge this work using the citation boilerplate that *fMRIPrep* includes
in the visual report generated for every subject processed.
For a more detailed description of the citation boilerplate and its relevance,
please check out the
`NiPreps documentation <https://www.nipreps.org/intro/transparency/#citation-boilerplates>`__.
Please report any feedback to our `GitHub repository <https://github.com/nipreps/fmriprep>`__.
"""
import os
import re
import subprocess
import sys

try:
    from ._version import __version__
except ImportError:
    __version__ = '0+unknown'

__copyright__ = 'Copyright 2023, Center for Reproducible Neuroscience, Stanford University'
__credits__ = [
    'Craig Moodie',
    'Ross Blair',
    'Oscar Esteban',
    'Chris Gorgolewski',
    'Shoshana Berleant',
    'Christopher J. Markiewicz',
    'Russell A. Poldrack',
]
__bugreports__ = 'https://github.com/nipreps/fmriprep/issues'


MISSING = """
Image '{}' is missing
Would you like to download? [Y/n] """
PKG_PATH = '/opt/conda/lib/python3.9/site-packages'
TF_TEMPLATES = (
    'MNI152Lin',
    'MNI152NLin2009cAsym',
    'MNI152NLin6Asym',
    'MNI152NLin6Sym',
    'MNIInfant',
    'MNIPediatricAsym',
    'NKI',
    'OASIS30ANTs',
    'PNC',
    'UNCInfant',
    'fsLR',
    'fsaverage',
    'fsaverage5',
    'fsaverage6',
)
NONSTANDARD_REFERENCES = ('anat', 'T1w', 'run', 'func', 'sbref', 'fsnative')

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
        ret = subprocess.run(['docker', 'version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    ret = subprocess.run(['docker', 'images', '-q', image], stdout=subprocess.PIPE)
    return bool(ret.stdout)


def check_memory(image):
    """Check total memory from within a docker container"""
    ret = subprocess.run(
        ['docker', 'run', '--rm', '--entrypoint=free', image, '-m'], stdout=subprocess.PIPE
    )
    if ret.returncode:
        return -1

    mem = [
        line.decode().split()[1] for line in ret.stdout.splitlines() if line.startswith(b'Mem:')
    ][0]
    return int(mem)


def merge_help(wrapper_help, target_help):
    def _get_posargs(usage):
        """
        Extract positional arguments from usage string.

        This function can be used by both native fmriprep (`fmriprep -h`)
        and the docker wrapper (`fmriprep-docker -h`).
        """
        posargs = []
        for targ in usage.split('\n')[-3:]:
            line = targ.lstrip()
            if line.startswith('usage'):
                continue
            if line[0].isalnum() or line[0] == "{":
                posargs.append(line)
            elif line[0] == '[' and (line[1].isalnum() or line[1] == "{"):
                posargs.append(line)
        return " ".join(posargs)

    # Matches all flags with up to two nested square brackets
    # I'm sorry.
    opt_re = re.compile(r'(\[--?[\w-]+(?:[^\[\]]+(?:\[(?:[^\[\]]+(?:\[[^\[\]]+\])?)+\])?)?\])')
    # Matches flag name only
    flag_re = re.compile(r'\[--?([\w-]+)[ \]]')

    # Normalize to Unix-style line breaks
    w_help = wrapper_help.rstrip().replace('\r', '')
    t_help = target_help.rstrip().replace('\r', '')

    w_usage, w_details = w_help.split('\n\n', 1)
    w_groups = w_details.split('\n\n')
    t_usage, t_details = t_help.split('\n\n', 1)
    t_groups = t_details.split('\n\n')

    w_posargs = _get_posargs(w_usage)
    t_posargs = _get_posargs(t_usage)

    w_options = opt_re.findall(w_usage)
    w_flags = sum(map(flag_re.findall, w_options), [])
    t_options = opt_re.findall(t_usage)
    t_flags = sum(map(flag_re.findall, t_options), [])

    # The following code makes this assumption
    assert w_flags[:2] == ['h', 'version']
    assert w_posargs.replace(']', '').replace('[', '') == t_posargs

    # Make sure we're not clobbering options we don't mean to
    overlap = set(w_flags).intersection(t_flags)
    expected_overlap = {
        'anat-derivatives',
        'bids-database-dir',
        'bids-filter-file',
        'fs-license-file',
        'fs-subjects-dir',
        'output-spaces',
        'config-file',
        'h',
        'use-plugin',
        'version',
        'w',
    }

    assert overlap == expected_overlap, "Clobbering options: {}".format(
        ', '.join(overlap - expected_overlap)
    )

    sections = []

    # Construct usage
    start = w_usage[: w_usage.index(' [')]
    indent = ' ' * len(start)
    new_options = sum(
        (
            w_options[:2],
            [opt for opt, flag in zip(t_options, t_flags) if flag not in overlap],
            w_options[2:],
        ),
        [],
    )
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


def is_in_directory(filepath, directory):
    return os.path.realpath(filepath).startswith(os.path.realpath(directory) + os.sep)


def get_parser():
    """Defines the command line interface of the wrapper"""
    import argparse
    from functools import partial

    class ToDict(argparse.Action):
        def __call__(self, parser, namespace, values, option_string=None):
            d = {}
            for kv in values:
                k, v = kv.split("=")
                d[k] = os.path.abspath(v)
            setattr(namespace, self.dest, d)

    def _is_file(path, parser):
        """Ensure a given path exists and it is a file."""
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            raise parser.error("Path should point to a file (or symlink of file): <%s>." % path)
        return path

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False
    )

    IsFile = partial(_is_file, parser=parser)

    # Standard FMRIPREP arguments
    parser.add_argument('bids_dir', nargs='?', type=os.path.abspath, default='')
    parser.add_argument('output_dir', nargs='?', type=os.path.abspath, default='')
    parser.add_argument(
        'analysis_level', nargs='?', choices=['participant'], default='participant'
    )

    parser.add_argument(
        '-h', '--help', action='store_true', help="show this help message and exit"
    )
    parser.add_argument(
        '--version', action='store_true', help="show program's version number and exit"
    )

    # Allow alternative images (semi-developer)
    parser.add_argument(
        '-i',
        '--image',
        metavar='IMG',
        type=str,
        default='nipreps/fmriprep:{}'.format(__version__),
        help='image name',
    )

    # Options for mapping files and directories into container
    # Update `expected_overlap` variable in merge_help() when adding to this
    g_wrap = parser.add_argument_group(
        'Wrapper options',
        'Standard options that require mapping files into the container; see fmriprep '
        'usage for complete descriptions',
    )
    g_wrap.add_argument(
        '-w',
        '--work-dir',
        action='store',
        type=os.path.abspath,
    )
    g_wrap.add_argument(
        '--output-spaces',
        nargs="*",
    )

    g_wrap.add_argument(
        '--fs-license-file',
        metavar='PATH',
        type=IsFile,
        default=os.getenv('FS_LICENSE', None),
    )
    g_wrap.add_argument(
        '--fs-subjects-dir',
        metavar='PATH',
        type=os.path.abspath,
    )
    g_wrap.add_argument(
        '--config-file',
        metavar='PATH',
        type=os.path.abspath,
    )
    g_wrap.add_argument(
        '--anat-derivatives',
        metavar='PATH',
        type=os.path.abspath,
    )
    g_wrap.add_argument(
        '--use-plugin',
        metavar='PATH',
        action='store',
        default=None,
        type=os.path.abspath,
    )
    g_wrap.add_argument(
        '--bids-database-dir',
        metavar='PATH',
        type=os.path.abspath,
    )
    g_wrap.add_argument(
        '--bids-filter-file',
        metavar='PATH',
        type=os.path.abspath,
    )

    # Developer patch/shell options
    g_dev = parser.add_argument_group(
        'Developer options', 'Tools for testing and debugging FMRIPREP'
    )
    g_dev.add_argument(
        '--patch',
        nargs="+",
        metavar="PACKAGE=PATH",
        action=ToDict,
        help='Sequence of PACKAGE=PATH specifications to patch a Python package into the '
        'container Python environment.',
    )
    g_dev.add_argument(
        '--shell', action='store_true', help='Open shell in image instead of running FMRIPREP'
    )
    g_dev.add_argument(
        '--config',
        metavar='PATH',
        action='store',
        type=os.path.abspath,
        help='Use custom nipype.cfg file',
    )
    g_dev.add_argument(
        '-e',
        '--env',
        action='append',
        nargs=2,
        metavar=('ENV_VAR', 'value'),
        help='Set custom environment variables within container',
    )
    g_dev.add_argument(
        '-u',
        '--user',
        action='store',
        help='Run container as a given user/uid. Additionally, group/gid can be'
        'assigned, (i.e., --user <UID>:<GID>)',
    )
    g_dev.add_argument(
        '--network',
        action='store',
        help='Run container with a different network driver '
        '("none" to simulate no internet connection)',
    )
    g_dev.add_argument('--no-tty', action='store_true', help='Run docker without TTY flag -it')

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
        print(
            'Could not detect memory capacity of Docker container.\n'
            'Do you have permission to run docker?'
        )
        return 1
    if not (opts.help or opts.version or '--reports-only' in unknown_args) and mem_total < 8000:
        print(
            'Warning: <8GB of RAM is available within your Docker '
            'environment.\nSome parts of fMRIPrep may fail to complete.'
        )
        if '--mem_mb' not in unknown_args:
            resp = 'N'
            try:
                resp = input('Continue anyway? [y/N]')
            except KeyboardInterrupt:
                print()
                return 1
            if resp not in ('y', 'Y', ''):
                return 0

    ret = subprocess.run(
        ['docker', 'version', '--format', "{{.Server.Version}}"], stdout=subprocess.PIPE
    )
    docker_version = ret.stdout.decode('ascii').strip()

    command = ['docker', 'run', '--rm', '-e', 'DOCKER_VERSION_8395080871=%s' % docker_version]

    if not opts.no_tty:
        if opts.help:
            # TTY can corrupt stdout
            command.append('-i')
        else:
            command.append('-it')

    # Patch working repositories into installed package directories
    if opts.patch:
        for pkg, repo_path in opts.patch.items():
            command.extend(['-v', '{}:{}/{}:ro'.format(repo_path, PKG_PATH, pkg)])

    if opts.env:
        for envvar in opts.env:
            command.extend(['-e', '%s=%s' % tuple(envvar)])

    if opts.user:
        command.extend(['-u', opts.user])

    if opts.fs_license_file:
        command.extend(['-v', '{}:/opt/freesurfer/license.txt:ro'.format(opts.fs_license_file)])

    main_args = []
    if opts.bids_dir:
        command.extend(['-v', ':'.join((opts.bids_dir, '/data', 'ro'))])
        main_args.append('/data')
    if opts.output_dir:
        if not os.path.exists(opts.output_dir):
            # create it before docker does
            os.makedirs(opts.output_dir)
        command.extend(['-v', ':'.join((opts.output_dir, '/out'))])
        main_args.append('/out')
    main_args.append(opts.analysis_level)

    if opts.fs_subjects_dir:
        command.extend(['-v', '{}:/opt/subjects'.format(opts.fs_subjects_dir)])
        unknown_args.extend(['--fs-subjects-dir', '/opt/subjects'])

    if opts.config_file:
        command.extend(['-v', '{}:/tmp/config.toml'.format(opts.config_file)])
        unknown_args.extend(['--config-file', '/tmp/config.toml'])

    if opts.anat_derivatives:
        command.extend(['-v', '{}:/opt/smriprep/subjects'.format(opts.anat_derivatives)])
        unknown_args.extend(['--anat-derivatives', '/opt/smriprep/subjects'])

    if opts.work_dir:
        command.extend(['-v', ':'.join((opts.work_dir, '/scratch'))])
        unknown_args.extend(['-w', '/scratch'])

    # Check that work_dir is not a child of bids_dir
    if opts.work_dir and opts.bids_dir:
        if is_in_directory(opts.work_dir, opts.bids_dir):
            print(
                'The selected working directory is a subdirectory of the input BIDS folder. '
                'Please modify the output path.'
            )
            return 1

    if opts.config:
        command.extend(['-v', ':'.join((opts.config, '/home/fmriprep/.nipype/nipype.cfg', 'ro'))])

    if opts.use_plugin:
        command.extend(['-v', ':'.join((opts.use_plugin, '/tmp/plugin.yml', 'ro'))])
        unknown_args.extend(['--use-plugin', '/tmp/plugin.yml'])

    if opts.bids_database_dir:
        command.extend(['-v', ':'.join((opts.bids_database_dir, '/tmp/bids_db'))])
        unknown_args.extend(['--bids-database-dir', '/tmp/bids_db'])

    if opts.bids_filter_file:
        command.extend(['-v', ':'.join((opts.bids_filter_file, '/tmp/bids_filter.json'))])
        unknown_args.extend(['--bids-filter-file', '/tmp/bids_filter.json'])

    if opts.output_spaces:
        spaces = []
        for space in opts.output_spaces:
            if space.split(':')[0] not in (TF_TEMPLATES + NONSTANDARD_REFERENCES):
                tpl = os.path.basename(space)
                if not tpl.startswith('tpl-'):
                    raise RuntimeError("Custom template %s requires a `tpl-` prefix" % tpl)
                target = '/home/fmriprep/.cache/templateflow/' + tpl
                command.extend(['-v', ':'.join((os.path.abspath(space), target, 'ro'))])
                spaces.append(tpl[4:])
            else:
                spaces.append(space)
        unknown_args.extend(['--output-spaces'] + spaces)

    if opts.shell:
        command.append('--entrypoint=bash')

    if opts.network:
        command.append('--network=' + opts.network)

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
    if '__main__.py' in sys.argv[0]:
        from . import __name__ as module

        sys.argv[0] = '%s -m %s' % (sys.executable, module)
    sys.exit(main())
