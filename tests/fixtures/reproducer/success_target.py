#!/usr/bin/env python3
"""Fixture: always succeeds. Mimics a target that decodes valid input cleanly."""
import sys

def main():
    input_path = sys.argv[-1]
    with open(input_path, "rb") as f:
        data = f.read()
    print(f"decoded {len(data)} bytes OK")
    sys.exit(0)

if __name__ == "__main__":
    main()
