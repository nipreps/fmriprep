#!/usr/bin/env python3
"""Update and sort the creators list of the zenodo record."""

import json
import sys
from pathlib import Path

import click
from fuzzywuzzy import fuzz, process


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


def sort_contributors(entries, git_lines, exclude=None, last=None):
    """Return a list of author dictionaries, ordered by contribution."""
    last = last or []
    sorted_authors = sorted(entries, key=lambda i: i['name'])

    first_last = [' '.join(val['name'].split(',')[::-1]).strip() for val in sorted_authors]
    first_last_excl = [' '.join(val['name'].split(',')[::-1]).strip() for val in exclude or []]

    unmatched = []
    author_matches = []
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
            author_matches.append(val)

    names = {' '.join(val['name'].split(',')[::-1]).strip() for val in author_matches}
    for missing_name in first_last:
        if missing_name not in names:
            missing = sorted_authors[first_last.index(missing_name)]
            author_matches.append(missing)

    position_matches = []
    for i, item in enumerate(author_matches):
        pos = item.pop('position', None)
        if pos is not None:
            position_matches.append((i, int(pos)))

    for i, pos in position_matches:
        if pos < 0:
            pos += len(author_matches) + 1
        author_matches.insert(pos, author_matches.pop(i))

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

    git_line_summary_path = shutil.which('git-line-summary')
    if not lines and git_line_summary_path:
        print('Running git-line-summary on repo')
        lines = sp.check_output([git_line_summary_path]).decode().splitlines()
        lines = [line for line in lines if 'Not Committed Yet' not in line]
        contrib_file.write_text('\n'.join(lines))

    if not lines:
        raise RuntimeError(
            """\
Could not find line-contributors from git repository.{}""".format(
                """ \
git-line-summary not found, please install git-extras. """
                * (git_line_summary_path is None)
            )
        )
    return [' '.join(line.strip().split()[1:-1]) for line in lines if '%' in line]


def _namelast(inlist):
    retval = []
    for i in inlist:
        i['name'] = (f"{i.pop('name', '')} {i.pop('lastname', '')}").strip()
        retval.append(i)
    return retval


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
    data = get_git_lines()

    zenodo = json.loads(Path(zenodo_file).read_text())

    former = _namelast(read_md_table(Path(former_file).read_text()))
    zen_creators, miss_creators = sort_contributors(
        _namelast(read_md_table(Path(maintainers).read_text())),
        data,
        exclude=former,
    )

    zen_contributors, miss_contributors = sort_contributors(
        _namelast(read_md_table(Path(contributors).read_text())), data, exclude=former
    )

    zen_pi = _namelast(
        sorted(
            read_md_table(Path(pi).read_text()),
            key=lambda v: (int(v.get('position', -1)), v.get('lastname')),
        )
    )

    zenodo['creators'] = zen_creators
    zenodo['contributors'] = zen_contributors + zen_pi

    misses = set(miss_creators).intersection(miss_contributors)
    if misses:
        print(
            f"Some people made commits, but are missing in .maint/ files: {', '.join(misses)}",
            file=sys.stderr,
        )

    # Remove position
    for creator in zenodo['creators']:
        creator.pop('position', None)
        creator.pop('handle', None)
        if isinstance(creator['affiliation'], list):
            creator['affiliation'] = creator['affiliation'][0]

    for creator in zenodo['contributors']:
        creator.pop('handle', None)
        creator['type'] = 'Researcher'
        creator.pop('position', None)

        if isinstance(creator['affiliation'], list):
            creator['affiliation'] = creator['affiliation'][0]

    Path(zenodo_file).write_text(f'{json.dumps(zenodo, indent=2)}\n')


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
    members = _namelast(read_md_table(Path(maintainers).read_text())) + _namelast(
        read_md_table(Path(contributors).read_text())
    )

    hits, misses = sort_contributors(
        members,
        get_git_lines(),
        exclude=_namelast(read_md_table(Path(former_file).read_text())),
    )

    pi_hits = _namelast(
        sorted(
            read_md_table(Path(pi).read_text()),
            key=lambda v: (int(v.get('position', -1)), v.get('lastname')),
        )
    )

    pi_names = [pi['name'] for pi in pi_hits]
    hits = [hit for hit in hits if hit['name'] not in pi_names] + pi_hits

    def _aslist(value):
        if isinstance(value, list | tuple):
            return value
        return [value]

    # Remove position
    affiliations = []
    for item in hits:
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
        for author in hits
    ]

    if misses:
        print(
            f"Some people made commits, but are missing in .maint/ files: {', '.join(misses)}",
            file=sys.stderr,
        )

    print('Authors (%d):' % len(hits))
    print(
        '{}.'.format(
            '; '.join(
                [
                    rf'{i["name"]} \ :sup:`{idx}`\ '
                    for i, idx in zip(hits, aff_indexes, strict=False)
                ]
            )
        )
    )

    print(
        '\n\nAffiliations:\n{}'.format(
            '\n'.join([f'{i + 1: >2}. {a}' for i, a in enumerate(affiliations)])
        )
    )


if __name__ == '__main__':
    """ Install entry-point """
    cli()
