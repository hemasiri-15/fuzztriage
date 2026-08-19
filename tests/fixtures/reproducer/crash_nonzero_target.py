#!/usr/bin/env python3
"""Fixture: exits non-zero with a plain (non-ASan) error message."""
import sys

def main():
    sys.stderr.write("fatal: could not parse input (not a valid image)\n")
    sys.exit(1)

if __name__ == "__main__":
    main()
