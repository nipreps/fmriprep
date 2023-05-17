#!/bin/bash
#
# Collects the pull-requests since the latest release and
# aranges them in the CHANGES.rst.txt file.
#
# This is a script to be run before releasing a new version.
#
# Usage /bin/bash update_changes.sh 1.0.1
#

# Setting      # $ help set
set -u         # Treat unset variables as an error when substituting.
set -x         # Print command traces before executing command.

# Check whether the Upcoming release header is present
head -1 CHANGES.rst | grep -q Upcoming
UPCOMING=$?
if [[ "$UPCOMING" == "0" ]]; then
    head -n3  CHANGES.rst >> newchanges
fi

# Elaborate today's release header
HEADER="$1 ($(date '+%B %d, %Y'))"
echo $HEADER >> newchanges
echo $( printf "%${#HEADER}s" | tr " " "=" ) >> newchanges
echo "" >> newchanges

# Search for PRs since previous release
MERGE_COMMITS=$( git log --grep="Merge pull request\|(#.*)$" `git describe --tags --abbrev=0`..HEAD --pretty='format:%h' )
for COMMIT in ${MERGE_COMMITS//\n}; do
    SUB=$( git log -n 1 --pretty="format:%s" $COMMIT )
    if ( echo $SUB | grep "^Merge pull request" ); then
        # Merge commit
        PR=$( echo $SUB | sed -e "s/Merge pull request \#\([0-9]*\).*/\1/" )
        TITLE=$( git log -n 1 --pretty="format:%b" $COMMIT )
    else
        # Squashed merge
        PR=$( echo $SUB | sed -e "s/.*(\#\([0-9]*\))$/\1/" )
        TITLE=$( echo $SUB | sed -e "s/\(.*\) (\#[0-9]*)$/\1/" )
    fi
    echo "  * $TITLE (#$PR)" >> newchanges
done
echo >> newchanges
echo >> newchanges

# Add back the Upcoming header if it was present
if [[ "$UPCOMING" == "0" ]]; then
    tail -n+4 CHANGES.rst >> newchanges
else
    cat CHANGES.rst >> newchanges
fi

# Replace old CHANGES.rst with new file
mv newchanges CHANGES.rst

