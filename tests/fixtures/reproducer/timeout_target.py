#!/usr/bin/env python3
"""Fixture: hangs. Mimics an AFL++ hang artifact."""
import time

def main():
    time.sleep(30)

if __name__ == "__main__":
    main()
