#!/bin/sh
DS_FULLPATH=$1
NUMTERMS=$2

function help() {
  >&2 echo "Usage: $0 <data source file path> <# terms to generate>"
  >&2 echo "*look under processed directory for json file"
  exit 1
}

if [ -z $DS_FULLPATH ] || \
   [ -z $NUMTERMS ]
then
  >&2 echo "missing arguments"
  help
fi

python3<<!

import json
import random

if __name__ == '__main__':
    corpus = set([])
    metadata = []
    numterms = $NUMTERMS
    with open('$DS_FULLPATH', 'r') as dsfh:
        metadata = json.load(dsfh)
    for data in metadata[0]:
        corpus.update(data['tokens'])
    corpusl = list(corpus)
    maxidx = corpusl.index(corpusl[-1])
    randidxs = random.sample(range(0, maxidx+1), numterms)
    randquery = ' '.join([ corpusl[idx] for idx in randidxs ])
    print(rf'\"{randquery}\"')
!
