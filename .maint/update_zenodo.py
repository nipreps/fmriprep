#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fuzzywuzzy",
#     "python-levenshtein",
# ]
# ///
"""Update and sort the creators list of the zenodo record."""

import json
import sys
from pathlib import Path

from fuzzywuzzy import fuzz, process

# These ORCIDs should go last
CREATORS_LAST = ['Poldrack, Russell A.', 'Gorgolewski, Krzysztof J.']
CONTRIBUTORS_LAST = ['Ghosh, Satrajit S.']


def sort_contributors(entries, git_lines, exclude=None, last=None):
    """Return a list of author dictionaries, ordered by contribution."""
    last = last or []
    sorted_authors = sorted(entries, key=lambda i: i['name'])

    first_last = [' '.join(val['name'].split(',')[::-1]).strip() for val in sorted_authors]
    first_last_excl = [' '.join(val['name'].split(',')[::-1]).strip() for val in exclude or []]

    unmatched = []
    author_matches = []
    position = 1
    for ele in git_lines:
        matches = process.extract(ele, first_last, scorer=fuzz.token_sort_ratio, limit=2)
        # matches is a list [('First match', % Match), ('Second match', % Match)]
        if matches[0][1] > 80:
            val = sorted_authors[first_last.index(matches[0][0])]
        else:
            # skip unmatched names
            if ele not in first_last_excl:
                unmatched.append(ele)
            continue

        if val not in author_matches:
            val['position'] = position
            author_matches.append(val)
            position += 1

    names = {' '.join(val['name'].split(',')[::-1]).strip() for val in author_matches}
    for missing_name in first_last:
        if missing_name not in names:
            missing = sorted_authors[first_last.index(missing_name)]
            missing['position'] = position
            author_matches.append(missing)
            position += 1

    all_names = [val['name'] for val in author_matches]
    for last_author in last:
        author_matches[all_names.index(last_author)]['position'] = position
        position += 1

    author_matches = sorted(author_matches, key=lambda k: k['position'])

    return author_matches, unmatched


def get_git_lines(fname='line-contributors.txt'):
    """Run git-line-summary."""
    import shutil
    import subprocess as sp

    contrib_file = Path(fname)

    lines = []
    if contrib_file.exists():
        print('WARNING: Reusing existing line-contributors.txt file.', file=sys.stderr)
        lines = contrib_file.read_text().splitlines()

    cmd = [shutil.which('git-line-summary')]
    if cmd == [None]:
        cmd = [shutil.which('git-summary'), '--line']
    if not lines and cmd[0]:
        print(f'Running {" ".join(cmd)!r} on repo')
        lines = sp.check_output(cmd).decode().splitlines()
        lines = [line for line in lines if 'Not Committed Yet' not in line]
        contrib_file.write_text('\n'.join(lines))

    if not lines:
        raise RuntimeError(
            """\
Could not find line-contributors from git repository.{}""".format(
                """ \
git-(line-)summary not found, please install git-extras. """
                * (cmd[0] is None)
            )
        )
    return [' '.join(line.strip().split()[1:-1]) for line in lines if '%' in line]


def loads_table_from_markdown(s):
    """Read the first table from a Markdown text."""
    table = []
    header = None
    for line in s.splitlines():
        if line.startswith('|'):
            if not header:
                # read header and strip bold
                header = [item.strip('* ') for item in line.split('|')[1:-1]]
            else:
                values = [item.strip() for item in line.split('|')[1:-1]]
                if any(any(c != '-' for c in item) for item in values):
                    table.append(dict(zip(header, values, strict=False)))
        elif header:
            # we have already seen a table, we're past the end of that table
            break
    return table


def loads_contributors(s):
    """Reformat contributors read from the Markdown table."""
    return [
        {
            'affiliation': contributor['Affiliation'],
            'name': '{}, {}'.format(contributor['Lastname'], contributor['Name']),
            'orcid': contributor['ORCID'],
        }
        for contributor in loads_table_from_markdown(s)
    ]


if __name__ == '__main__':
    data = get_git_lines()

    zenodo_file = Path('.zenodo.json')
    zenodo = json.loads(zenodo_file.read_text())

    creators = json.loads(Path('.maint/developers.json').read_text())
    zen_creators, miss_creators = sort_contributors(
        creators,
        data,
        exclude=json.loads(Path('.maint/former.json').read_text()),
        last=CREATORS_LAST,
    )
    contributors = loads_contributors(Path('.maint/CONTRIBUTORS.md').read_text())
    zen_contributors, miss_contributors = sort_contributors(
        contributors,
        data,
        exclude=json.loads(Path('.maint/former.json').read_text()),
        last=CONTRIBUTORS_LAST,
    )
    zenodo['creators'] = zen_creators
    zenodo['contributors'] = zen_contributors

    print(
        'Some people made commits, but are missing in .maint/ files: {}.'.format(
            ', '.join(set(miss_creators).intersection(miss_contributors))
        ),
        file=sys.stderr,
    )

    # Remove position
    for creator in zenodo['creators']:
        del creator['position']
        if isinstance(creator['affiliation'], list):
            creator['affiliation'] = creator['affiliation'][0]

    for creator in zenodo['contributors']:
        creator['type'] = 'Researcher'
        del creator['position']
        if isinstance(creator['affiliation'], list):
            creator['affiliation'] = creator['affiliation'][0]

    zenodo_file.write_text(f'{json.dumps(zenodo, indent=2, ensure_ascii=False)}\n')
