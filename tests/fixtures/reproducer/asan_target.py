#!/usr/bin/env python3
"""Fixture: emits an ASan-style error banner. Fictional address/line, for
detection testing only -- not a real sanitizer report from a real crash."""
import sys

def main():
    sys.stderr.write(
        "==999==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xdeadbeef\n"
        "WRITE of size 4 at 0xdeadbeef thread T0\n"
        "    #0 0x1 in fixture_fn fixture.c:1:1\n"
    )
    sys.exit(1)

if __name__ == "__main__":
    main()
