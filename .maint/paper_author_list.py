#!/usr/bin/env python3
"""Generate an author list for a new paper or abstract."""

import json
import sys
from pathlib import Path

from update_zenodo import get_git_lines, sort_contributors

# These authors should go last
AUTHORS_LAST = ['Gorgolewski, Krzysztof J.', 'Poldrack, Russell A.', 'Esteban, Oscar']


def _aslist(inlist):
    if not isinstance(inlist, list):
        return [inlist]
    return inlist


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
            'affiliation': contributor['Affiliation'] if 'Affiliation' in contributor else None,
            'name': f'{contributor["Lastname"]}, {contributor["Name"]}',
            'orcid': contributor['ORCID'] if 'ORCID' in contributor else None,
        }
        for contributor in loads_table_from_markdown(s)
    ]


if __name__ == '__main__':
    devs = json.loads(Path('.maint/developers.json').read_text())
    contribs = loads_contributors(Path('.maint/CONTRIBUTORS.md').read_text())

    author_matches, unmatched = sort_contributors(
        devs + contribs,
        get_git_lines(),
        exclude=loads_contributors(Path('.maint/FORMER.md').read_text()),
        last=AUTHORS_LAST,
    )
    # Remove position
    affiliations = []
    for item in author_matches:
        del item['position']
        for a in _aslist(item.get('affiliation', 'Unaffiliated')):
            if a not in affiliations:
                affiliations.append(a)

    aff_indexes = [
        ', '.join(
            [
                str(affiliations.index(a) + 1)
                for a in _aslist(author.get('affiliation', 'Unaffiliated'))
            ]
        )
        for author in author_matches
    ]

    print(
        f'Some people made commits, but are missing in .maint/ files: {", ".join(unmatched)}.',
        file=sys.stderr,
    )

    print(f'Authors ({len(author_matches)}):')
    authors = '; '.join(
        f'{i["name"]} \\ :sup:`{idx}`\\ ' for i, idx in zip(hits, aff_indexes, strict=False)
    )
    print(f'{authors}.')

    lines = '\n'.join(f'{i + 1: >2}. {a}' for i, a in enumerate(affiliations))
    print(f'\n\nAffiliations:\n{lines}')
