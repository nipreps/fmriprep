#!/usr/bin/env python3
"""Update and sort the creators list of the zenodo record."""
import sys
import shutil
from pathlib import Path
import json
from fuzzywuzzy import fuzz, process
import subprocess as sp

# These ORCIDs should go last
CREATORS_LAST = ['Poldrack, Russell A.', 'Gorgolewski, Krzysztof J.']
# for entries not found in line-contributions
MISSING_ENTRIES = [
]

if __name__ == '__main__':
    contrib_file = Path('line-contributors.txt')
    lines = []
    if contrib_file.exists():
        print('WARNING: Reusing existing line-contributors.txt file.', file=sys.stderr)
        lines = contrib_file.read_text().splitlines()

    git_line_summary_path = shutil.which('git-line-summary')
    if not lines and git_line_summary_path:
        print("Running git-line-summary on repo")
        lines = sp.check_output([git_line_summary_path]).decode().splitlines()
        contrib_file.write_text('\n'.join(lines))

    if not lines:
        raise RuntimeError("""\
Could not find line-contributors from git repository.%s""" % """ \
git-line-summary not found, please install git-extras. """ * (git_line_summary_path is None))

    data = [' '.join(line.strip().split()[1:-1]) for line in lines if '%' in line]

    # load zenodo from master
    zenodo_file = Path('.zenodo.json')
    zenodo = json.loads(zenodo_file.read_text())
    zen_names = [' '.join(val['name'].split(',')[::-1]).strip()
                 for val in zenodo['creators']]
    total_names = len(zen_names) + len(MISSING_ENTRIES)

    author_matches = []
    position = 1
    for ele in data:
        matches = process.extract(ele, zen_names, scorer=fuzz.token_sort_ratio,
                                  limit=2)
        # matches is a list [('First match', % Match), ('Second match', % Match)]
        if matches[0][1] > 80:
            val = zenodo['creators'][zen_names.index(matches[0][0])]
        else:
            # skip unmatched names
            if ele != "Not Committed Yet":
                print("Author missing in .zenodo.json file:", ele)
            continue

        if val not in author_matches:
            val['position'] = position
            author_matches.append(val)
            position += 1

    names = {' '.join(val['name'].split(',')[::-1]).strip() for val in author_matches}
    for missing_name in zen_names:
        if missing_name not in names:
            missing = zenodo['creators'][zen_names.index(missing_name)]
            missing['position'] = position
            author_matches.append(missing)
            position += 1

    all_names = [val['name'] for val in author_matches]
    for last_author in CREATORS_LAST:
        author_matches[all_names.index(last_author)]['position'] = position
        position += 1

    zenodo['creators'] = sorted(author_matches, key=lambda k: k['position'])
    # Remove position
    for creator in zenodo['creators']:
        del creator['position']

    zenodo_file.write_text('%s\n' % json.dumps(zenodo, indent=2, sort_keys=True))
