#!/usr/bin/env python

import sys
from sasmodels.compare import parse_opts, show_docs, explore, compare

def main(*argv):
    # type: (*str) -> None
    """
    Main program.
    """
    opts = parse_opts(argv)
    if opts is not None:
        if opts['html']:
            show_docs(opts)
        elif opts['explore']:
            explore(opts)
        else:
            compare(opts)

if __name__ == "__main__":
    main(*sys.argv[1:])
