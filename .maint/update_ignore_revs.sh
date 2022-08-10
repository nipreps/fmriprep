#!/bin/bash

# Tag each rev with a simple log entry
for SHA in $( grep -v "^#" .git-blame-ignore-revs ); do
    git log --pretty=format:"# %ad - %ae - %s%n$SHA%n" -n 1 --date short $SHA
done > git-blame-ignore-revs

# Two-step to avoid the original getting truncated before it's read
mv git-blame-ignore-revs .git-blame-ignore-revs
