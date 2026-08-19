#!/usr/bin/env python3
"""Fixture: deliberately kills itself with SIGSEGV to test signal capture."""
import os
import signal

def main():
    os.kill(os.getpid(), signal.SIGSEGV)

if __name__ == "__main__":
    main()
