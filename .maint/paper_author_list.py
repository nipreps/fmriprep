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


if __name__ == '__main__':
    devs = json.loads(Path('.maint/developers.json').read_text())
    contribs = json.loads(Path('.maint/contributors.json').read_text())

    author_matches, unmatched = sort_contributors(
        devs + contribs,
        get_git_lines(),
        exclude=json.loads(Path('.maint/former.json').read_text()),
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
