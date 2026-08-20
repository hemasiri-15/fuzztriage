TEST FIXTURE EXECUTABLES — Phase 6 (reproducer.py) only.

These are small, deterministic Python scripts standing in for a real
fuzzing target's CLI behavior, so reproducer.py's tests don't depend
on the DGX or on any real compiled target being present. They are
never presented as vulnerabilities discovered by AFL++ — none of them
came from a real fuzzing campaign.
