#!/usr/bin/env python3
"""
Fixture: a richer target for Phase 12 end-to-end pipeline tests.

Unlike the other fixture targets (which behave identically regardless
of input), this one's behavior genuinely depends on the input file's
first byte -- mimicking how a real fuzzed target's crash behavior
depends on the mutated input content. This lets a single pipeline run
exercise multiple distinct evidence combinations from real subprocess
execution, not hardcoded pipeline test data.

    first byte 'A' -> heap-buffer-overflow, WRITE, function decode_a
    first byte 'B' -> heap-buffer-overflow, WRITE, function decode_a
                      (SAME evidence as 'A' -- these should dedup together)
    first byte 'C' -> use-after-free, READ, function free_c
    anything else  -> exits cleanly, no crash
"""
import sys


def main():
    input_path = sys.argv[-1]
    with open(input_path, "rb") as f:
        data = f.read()

    first = data[:1]

    if first in (b"A", b"B"):
        sys.stderr.write(
            "==1001==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xaaaa1111\n"
            "WRITE of size 4 at 0xaaaa1111 thread T0\n"
            "    #0 0x1111aaaa in decode_a campaign.c:100:5\n"
            "    #1 0x2222bbbb in main campaign.c:10:2\n"
        )
        sys.exit(1)

    if first == b"C":
        sys.stderr.write(
            "==1002==ERROR: AddressSanitizer: heap-use-after-free on address 0xbbbb2222\n"
            "READ of size 8 at 0xbbbb2222 thread T0\n"
            "    #0 0x3333cccc in free_c campaign.c:200:9\n"
            "    #1 0x4444dddd in main campaign.c:10:2\n"
        )
        sys.exit(1)

    print("decoded OK")
    sys.exit(0)


if __name__ == "__main__":
    main()
