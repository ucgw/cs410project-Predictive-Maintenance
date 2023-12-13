#!/bin/sh
DS_FULLPATH=$1

function help() {
  >&2 echo "Usage: $0 <data source file path>"
  >&2 echo "*look under processed directory for json file"
  exit 1
}

if [ -z $DS_FULLPATH ]
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
    with open('$DS_FULLPATH', 'r') as dsfh:
        metadata = json.load(dsfh)
        for idx, data in enumerate(metadata):
            for recdata in metadata[idx]:
                corpus.update(recdata['tokens'])
    corpusl = list(corpus)
    for term in corpusl:
        print(term)
!
