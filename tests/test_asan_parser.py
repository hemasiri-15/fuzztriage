"""
Unit tests for app.services.asan_parser, using the fixture ASan
reports at tests/fixtures/asan/.

These are illustrative fixture reports (fictional addresses/paths),
not output from a real crash — the real DGX campaign has 0 crashes
so far.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.asan_parser import parse_asan_report  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "asan"


def _read(name: str) -> str:
    return (FIXTURES / name).read_text()


def test_heap_buffer_overflow_error_class():
    report = parse_asan_report(_read("heap_buffer_overflow.txt"))
    assert report.is_asan is True
    assert report.error_class == "heap-buffer-overflow"
    assert report.memory_region == "HEAP"


def test_heap_buffer_overflow_access_info():
    report = parse_asan_report(_read("heap_buffer_overflow.txt"))
    assert report.access_type == "WRITE"
    assert report.access_size == 4


def test_heap_buffer_overflow_faulting_function_and_location():
    report = parse_asan_report(_read("heap_buffer_overflow.txt"))
    assert report.faulting_function == "decode_mcu_block"
    assert report.source_file == "/home/user/libjpeg-turbo/src/jdhuff.c"
    assert report.source_line == 341


def test_heap_buffer_overflow_stack_trace_frames():
    report = parse_asan_report(_read("heap_buffer_overflow.txt"))
    assert len(report.stack_trace) >= 3
    assert report.stack_trace[0].function == "decode_mcu_block"
    assert report.stack_trace[1].function == "decode_image"
    assert report.stack_trace[2].function == "jpeg_read_header"


def test_heap_buffer_overflow_allocation_stack_captured():
    report = parse_asan_report(_read("heap_buffer_overflow.txt"))
    assert len(report.allocation_stack) >= 1
    assert any(f.function == "alloc_small" for f in report.allocation_stack)


def test_use_after_free_error_class():
    report = parse_asan_report(_read("use_after_free.txt"))
    assert report.is_asan is True
    assert report.error_class == "heap-use-after-free"
    assert report.memory_region == "HEAP"


def test_use_after_free_access_info():
    report = parse_asan_report(_read("use_after_free.txt"))
    assert report.access_type == "READ"
    assert report.access_size == 8


def test_use_after_free_faulting_function():
    report = parse_asan_report(_read("use_after_free.txt"))
    assert report.faulting_function == "jpeg_free_large"
    assert report.source_line == 1103


def test_use_after_free_deallocation_stack_captured():
    report = parse_asan_report(_read("use_after_free.txt"))
    assert len(report.deallocation_stack) >= 1
    assert any(f.function == "free_pool" for f in report.deallocation_stack)


def test_use_after_free_allocation_stack_also_captured():
    # This fixture has a "previously allocated by" section after the
    # "freed by" section — both must still be captured correctly.
    report = parse_asan_report(_read("use_after_free.txt"))
    assert len(report.allocation_stack) >= 1
    assert any(f.function == "alloc_small" for f in report.allocation_stack)


def test_non_asan_plain_crash_returns_is_asan_false():
    report = parse_asan_report(_read("not_asan_plain_crash.txt"))
    assert report.is_asan is False
    assert report.error_class is None
    assert report.faulting_function is None
    assert report.stack_trace == []


def test_empty_text_returns_is_asan_false_not_exception():
    report = parse_asan_report("")
    assert report.is_asan is False


def test_never_fabricates_missing_fields():
    # A minimal, deliberately sparse ASan-looking report with no stack
    # frames at all — every downstream field must stay None, not be
    # guessed from partial information.
    text = "==999==ERROR: AddressSanitizer: double-free on address 0xdeadbeef\n"
    report = parse_asan_report(text)
    assert report.is_asan is True
    assert report.error_class == "double-free"
    assert report.memory_region == "HEAP"
    assert report.faulting_function is None
    assert report.source_file is None
    assert report.stack_trace == []


def test_unknown_error_class_still_captured_not_dropped():
    # Simulates a hypothetical future ASan error class not in our
    # known-set mapping — we should still record the string, just
    # without a memory_region guess.
    text = "==1==ERROR: AddressSanitizer: some-future-error-class on address 0x1\n"
    report = parse_asan_report(text)
    assert report.is_asan is True
    assert report.error_class == "some-future-error-class"
    assert report.memory_region is None
