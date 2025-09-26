#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
#     "fuzzywuzzy",
#     "python-levenshtein",
# ]
# ///
"""Update and sort the creators list of the zenodo record."""

import json
import sys
from pathlib import Path

import click
from fuzzywuzzy import fuzz, process

CREATORS_LAST = ['Poldrack, Russell A.', 'Gorgolewski, Krzysztof J.']
CONTRIBUTORS_LAST = ['Ghosh, Satrajit S.']


def read_md_table(md_text):
    """
    Extract the first table found in a markdown document as a Python dict.

    Examples
    --------
    >>> read_md_table('''
    ... # Some text
    ...
    ... More text
    ...
    ... | **Header1** | **Header2** |
    ... | --- | --- |
    ... | val1 | val2 |
    ... |  | val4 |
    ...
    ... | **Header3** | **Header4** |
    ... | --- | --- |
    ... | val1 | val2 |
    ... |  | val4 |
    ... ''')
    [{'header1': 'val1', 'header2': 'val2'}, {'header2': 'val4'}]

    """
    prev = None
    keys = None
    retval = []
    for line in md_text.splitlines():
        if line.strip().startswith('| --- |'):
            keys = (k.replace('*', '').strip() for k in prev.split('|'))
            keys = [k.lower() for k in keys if k]
            continue
        elif not keys:
            prev = line
            continue

        if not line or not line.strip().startswith('|'):
            break

        values = [v.strip() or None for v in line.split('|')][1:-1]
        retval.append({k: v for k, v in zip(keys, values, strict=False) if v})

    return retval


def sort_contributors(entries, git_lines, exclude=None):
    """Return a list of author dictionaries, ordered by contribution."""
    sorted_authors = sorted(entries)

    # Match on First Last
    first_last = [' '.join(name.split(',')[::-1]).strip() for name in sorted_authors]
    first_last_excl = {' '.join(name.split(',')[::-1]).strip() for name in exclude or []}

    indices = []
    unmatched = set()
    for committer in git_lines:
        matches = process.extract(committer, first_last, scorer=fuzz.token_sort_ratio, limit=2)
        if matches[0][1] > 80:
            indices.append(first_last.index(matches[0][0]))
        elif committer not in first_last_excl:
            unmatched.add(committer)

    # Return Last, First
    matches = dict.fromkeys([sorted_authors[i] for i in indices])
    # Add any remaining authors not matched in git_lines
    matches.update(dict.fromkeys(sorted_authors))

    return matches, unmatched


def get_git_lines(fname='line-contributors.txt'):
    """Run git-line-summary."""
    import shutil
    import subprocess as sp

    contrib_file = Path(fname)

    lines = []
    if contrib_file.exists():
        print('WARNING: Reusing existing line-contributors.txt file.', file=sys.stderr)
        lines = contrib_file.read_text().splitlines()

    git_line_summary_path = shutil.which('git-line-summary')
    if not git_line_summary_path:
        git_line_summary_path = 'git summary --dedup-by-email'.split(' ')
    else:
        git_line_summary_path = [git_line_summary_path]

    if not lines and git_line_summary_path:
        print('Running git-line-summary on repo')
        lines = sp.check_output(git_line_summary_path).decode().splitlines()
        lines = [line for line in lines if 'Not Committed Yet' not in line]
        contrib_file.write_text('\n'.join(lines))

    if not lines:
        _msg = ': git-line-summary not found, please install git-extras ' * (
            git_line_summary_path is None
        )
        raise RuntimeError(f'Could not find line-contributors from git repository{_msg}.')
    return [' '.join(line.strip().split()[1:-1]) for line in lines if '%' in line]


def _namelast(inlist):
    retval = []
    for i in inlist:
        i['name'] = (f'{i.pop("lastname", "")}, {i.pop("name", "")}').strip()
        if not i['name']:
            i['name'] = i.get('handle', '<Unknown Name>')
        retval.append(i)
    return retval


def load(path):
    return {
        entry['name']: dict(sorted(entry.items()))
        for entry in _namelast(read_md_table(Path(path).read_text()))
    }


@click.group()
def cli():
    """Generate authorship boilerplates."""
    pass


@cli.command()
@click.option('-z', '--zenodo-file', type=click.Path(exists=True), default='.zenodo.json')
@click.option('-m', '--maintainers', type=click.Path(exists=True), default='.maint/MAINTAINERS.md')
@click.option(
    '-c', '--contributors', type=click.Path(exists=True), default='.maint/CONTRIBUTORS.md'
)
@click.option('--pi', type=click.Path(exists=True), default='.maint/PIs.md')
@click.option('-f', '--former-file', type=click.Path(exists=True), default='.maint/FORMER.md')
def zenodo(
    zenodo_file,
    maintainers,
    contributors,
    pi,
    former_file,
):
    """Generate a new Zenodo payload file."""
    zenodo = json.loads(Path(zenodo_file).read_text())

    maint = load(maintainers)
    contrib = load(contributors)
    pis = load(pi)
    former = load(former_file)

    total_order, misses = sort_contributors(
        maint.keys() | contrib.keys() | pis.keys(),
        get_git_lines(),
        exclude=former,
    )

    # Sort
    creator_names = maint.keys() - set(CREATORS_LAST)
    creator_names = [name for name in total_order if name in creator_names] + CREATORS_LAST

    skip = set(creator_names) | set(CONTRIBUTORS_LAST)
    contrib_names = [name for name in total_order if name not in skip] + CONTRIBUTORS_LAST

    entries = contrib | maint | pis

    zenodo['creators'] = [entries[name] for name in creator_names]
    zenodo['contributors'] = [entries[name] for name in contrib_names]

    if misses:
        print(
            f'Some people made commits, but are missing in .maint/ files: {", ".join(misses)}',
            file=sys.stderr,
        )

    # Remove position
    for creator in zenodo['creators']:
        creator.pop('position', None)
        creator.pop('handle', None)
        if 'affiliation' not in creator:
            creator['affiliation'] = 'Unknown affiliation'
        elif isinstance(creator['affiliation'], list):
            creator['affiliation'] = creator['affiliation'][0]

    for creator in zenodo['contributors']:
        creator.pop('handle', None)
        creator['type'] = 'Researcher'
        creator.pop('position', None)

        if 'affiliation' not in creator:
            creator['affiliation'] = 'Unknown affiliation'
        elif isinstance(creator['affiliation'], list):
            creator['affiliation'] = creator['affiliation'][0]

    Path(zenodo_file).write_text(f'{json.dumps(zenodo, indent=2, ensure_ascii=False)}\n')


@cli.command()
@click.option('-m', '--maintainers', type=click.Path(exists=True), default='.maint/MAINTAINERS.md')
@click.option(
    '-c', '--contributors', type=click.Path(exists=True), default='.maint/CONTRIBUTORS.md'
)
@click.option('--pi', type=click.Path(exists=True), default='.maint/PIs.md')
@click.option('-f', '--former-file', type=click.Path(exists=True), default='.maint/FORMER.md')
def publication(
    maintainers,
    contributors,
    pi,
    former_file,
):
    """Generate the list of authors and affiliations for papers."""
    maint = load(maintainers)
    contrib = load(contributors)
    former = load(former_file)

    hits, misses = sort_contributors(
        maint.keys() | contrib.keys(),
        get_git_lines(),
        exclude=former,
    )

    pis = load(pi)
    entries = contrib | maint

    authors = [entries[name] for name in hits.keys() if name not in pis]
    authors += pis.values()

    def _aslist(value):
        if isinstance(value, (list, tuple)):
            return value
        return [value]

    # Remove position
    affiliations = []
    for item in authors:
        item.pop('position', None)
        for a in _aslist(item.get('affiliation', 'Unaffiliated')):
            if a not in affiliations:
                affiliations.append(a)

    aff_indexes = [
        ', '.join(
            [
                '%d' % (affiliations.index(a) + 1)
                for a in _aslist(author.get('affiliation', 'Unaffiliated'))
            ]
        )
        for author in authors
    ]

    if misses:
        print(
            f'Some people made commits, but are missing in .maint/ files: {", ".join(misses)}',
            file=sys.stderr,
        )

    print(f'Authors ({len(authors)}):')
    print(
        '; '.join(
            f'{i["name"]} \\ :sup:`{idx}`\\ ' for i, idx in zip(authors, aff_indexes, strict=False)
        )
        + '.'
    )

    print('\n\nAffiliations:')
    print('\n'.join(f'{i + 1: >2}. {a}' for i, a in enumerate(affiliations)))


if __name__ == '__main__':
    """ Install entry-point """
    cli()
