"""
Phase 8 tests — app.services.stack_normalizer.

Uses real Phase 5 parsing (existing fixtures under tests/fixtures/asan/)
for the integration-level tests, and small inline synthetic frame
lists (dicts, matching CrashFeatures.raw_stack_trace's exact shape)
for the specifically engineered edge cases. Nothing here is presented
as a real crash — same fixture-only discipline as every prior phase.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.asan_parser import parse_asan_report                     # noqa: E402
from app.services.feature_extractor import extract_features                 # noqa: E402
from app.services.stack_normalizer import (                                  # noqa: E402
    normalize_stack,
    normalize_crash_features_stack,
    STACK_SIGNATURE_VERSION,
)

ASAN_FIXTURES = Path(__file__).parent / "fixtures" / "asan"


def _heap_overflow_frames():
    report = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    return report.stack_trace


# ---------------------------------------------------------------------------
# TEST 1 — basic normalization
# ---------------------------------------------------------------------------

def test_basic_normalization_produces_expected_canonical_form():
    result = normalize_stack(_heap_overflow_frames())
    assert result.frame_count >= 3
    assert result.stack_signature is not None
    assert len(result.stack_signature) == 64  # sha256 hex digest length
    assert "decode_mcu_block" in result.normalized_stack
    assert "src/jdhuff.c:341" in result.normalized_stack  # last-2-segments path form


def test_signature_version_present():
    result = normalize_stack(_heap_overflow_frames())
    assert result.stack_signature_version == STACK_SIGNATURE_VERSION == "1.0"


# ---------------------------------------------------------------------------
# TEST 2 — same semantic stack, different memory addresses
# ---------------------------------------------------------------------------

def test_different_addresses_same_signature():
    # Two raw ASan reports, identical semantic stack, deliberately
    # different fictional addresses throughout.
    text_a = (
        "==111==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xaaaa1111\n"
        "WRITE of size 4 at 0xaaaa1111 thread T0\n"
        "    #0 0x1111aaaa in decode_mcu_block src/jdhuff.c:341:12\n"
        "    #1 0x2222bbbb in decode_image src/jdmaster.c:78:5\n"
    )
    text_b = (
        "==222==ERROR: AddressSanitizer: heap-buffer-overflow on address 0xffff9999\n"
        "WRITE of size 4 at 0xffff9999 thread T0\n"
        "    #0 0x9999ffff in decode_mcu_block src/jdhuff.c:341:12\n"
        "    #1 0x8888eeee in decode_image src/jdmaster.c:78:5\n"
    )
    result_a = normalize_stack(parse_asan_report(text_a).stack_trace)
    result_b = normalize_stack(parse_asan_report(text_b).stack_trace)

    assert result_a.normalized_stack == result_b.normalized_stack
    assert result_a.stack_signature == result_b.stack_signature


# ---------------------------------------------------------------------------
# TEST 3 — same semantic stack, different frame numbers
# ---------------------------------------------------------------------------

def test_different_frame_numbers_same_signature():
    frames_a = [
        {"index": 0, "function": "decode_mcu_block", "source_file": "src/jdhuff.c", "source_line": 341},
        {"index": 1, "function": "decode_image", "source_file": "src/jdmaster.c", "source_line": 78},
    ]
    frames_b = [
        {"index": 5, "function": "decode_mcu_block", "source_file": "src/jdhuff.c", "source_line": 341},
        {"index": 6, "function": "decode_image", "source_file": "src/jdmaster.c", "source_line": 78},
    ]
    assert normalize_stack(frames_a).stack_signature == normalize_stack(frames_b).stack_signature


# ---------------------------------------------------------------------------
# TEST 4 — whitespace differences
# ---------------------------------------------------------------------------

def test_whitespace_differences_same_signature():
    frames_a = [{"index": 0, "function": "decode_mcu_block", "source_file": "jdhuff.c", "source_line": 341}]
    frames_b = [{"index": 0, "function": "  decode_mcu_block  ", "source_file": " jdhuff.c ", "source_line": 341}]
    # Note: function/source_file here simulate a hypothetical
    # whitespace-dirty upstream value; normalize_stack does not
    # re-parse ASan text (that's Phase 5's job), it works on whatever
    # field values it receives, so this test also exercises basic
    # value equivalence when Phase 5 itself would strip such
    # whitespace during its own parsing.
    result_a = normalize_stack(frames_a)
    result_b = normalize_stack([
        {"index": 0, "function": "decode_mcu_block", "source_file": "jdhuff.c", "source_line": 341}
    ])
    assert result_a.stack_signature == result_b.stack_signature


# ---------------------------------------------------------------------------
# TEST 5 — different function -> different signature
# ---------------------------------------------------------------------------

def test_different_function_different_signature():
    frames_a = [{"index": 0, "function": "decode_mcu_block", "source_file": "jdhuff.c", "source_line": 341}]
    frames_b = [{"index": 0, "function": "decode_huffman_block", "source_file": "jdhuff.c", "source_line": 341}]
    assert normalize_stack(frames_a).stack_signature != normalize_stack(frames_b).stack_signature


# ---------------------------------------------------------------------------
# TEST 6 — different source file -> different signature
# ---------------------------------------------------------------------------

def test_different_source_file_different_signature():
    frames_a = [{"index": 0, "function": "parse_marker", "source_file": "jdmarker.c", "source_line": 889}]
    frames_b = [{"index": 0, "function": "parse_marker", "source_file": "jdhuff.c", "source_line": 889}]
    assert normalize_stack(frames_a).stack_signature != normalize_stack(frames_b).stack_signature


def test_same_filename_different_directory_does_not_collide():
    # Path normalization strategy check: two files both named utils.c
    # in different subdirectories must NOT collide (pure-basename
    # would have collided here; last-2-segments does not).
    frames_a = [{"index": 0, "function": "helper", "source_file": "src/decode/utils.c", "source_line": 10}]
    frames_b = [{"index": 0, "function": "helper", "source_file": "src/encode/utils.c", "source_line": 10}]
    assert normalize_stack(frames_a).stack_signature != normalize_stack(frames_b).stack_signature


# ---------------------------------------------------------------------------
# TEST 7 — different source line
# ---------------------------------------------------------------------------

def test_different_source_line_different_signature():
    """
    Documented choice: source_line IS part of the signature. A stack
    ending at the same function but a different line can represent a
    different code path or a different bug -- Phase 8 favors precision
    over aggressive deduplication, per the project's explicit
    "high precision + explainability" principle. Phase 9 (not Phase 8)
    is where evidence-based merging decisions belong.
    """
    frames_a = [{"index": 0, "function": "decode_mcu_block", "source_file": "jdhuff.c", "source_line": 341}]
    frames_b = [{"index": 0, "function": "decode_mcu_block", "source_file": "jdhuff.c", "source_line": 342}]
    assert normalize_stack(frames_a).stack_signature != normalize_stack(frames_b).stack_signature


# ---------------------------------------------------------------------------
# TEST 8 — missing source information does not crash
# ---------------------------------------------------------------------------

def test_missing_source_info_does_not_crash():
    frames = [{"index": 0, "function": "decode_mcu_block", "source_file": None, "source_line": None}]
    result = normalize_stack(frames)
    assert result.stack_signature is not None
    assert result.frames[0].source_file is None
    assert result.frames[0].source_line is None
    assert "<unknown>" in result.normalized_stack


def test_missing_function_does_not_crash():
    frames = [{"index": 0, "function": None, "source_file": "jdhuff.c", "source_line": 341}]
    result = normalize_stack(frames)
    assert result.frames[0].function is None
    assert "<unknown>" in result.normalized_stack


# ---------------------------------------------------------------------------
# TEST 9 — recursive/repeated frames preserved
# ---------------------------------------------------------------------------

def test_recursive_frames_are_preserved_not_collapsed():
    frames = [
        {"index": 0, "function": "foo", "source_file": "rec.c", "source_line": 5},
        {"index": 1, "function": "foo", "source_file": "rec.c", "source_line": 5},
        {"index": 2, "function": "foo", "source_file": "rec.c", "source_line": 5},
        {"index": 3, "function": "bar", "source_file": "rec.c", "source_line": 1},
    ]
    result = normalize_stack(frames)
    assert result.frame_count == 4
    assert len(result.normalized_stack.splitlines()) == 4
    lines = result.normalized_stack.splitlines()
    assert lines[0] == lines[1] == lines[2]
    assert lines[3] != lines[0]


def test_frame_order_is_preserved_not_sorted():
    frames = [
        {"index": 0, "function": "zzz_last_alphabetically", "source_file": "a.c", "source_line": 1},
        {"index": 1, "function": "aaa_first_alphabetically", "source_file": "b.c", "source_line": 2},
    ]
    result = normalize_stack(frames)
    lines = result.normalized_stack.splitlines()
    assert "zzz_last_alphabetically" in lines[0]
    assert "aaa_first_alphabetically" in lines[1]


# ---------------------------------------------------------------------------
# TEST 10 — empty stack
# ---------------------------------------------------------------------------

def test_empty_stack_is_explicit_not_a_fake_signature():
    result = normalize_stack([])
    assert result.stack_signature is None
    assert result.normalized_stack == ""
    assert result.frame_count == 0


def test_none_stack_is_also_explicit_empty_state():
    result = normalize_stack(None)
    assert result.stack_signature is None
    assert result.frame_count == 0


# ---------------------------------------------------------------------------
# TEST 11 — path normalization
# ---------------------------------------------------------------------------

def test_path_normalization_local_vs_dgx_absolute_paths_match():
    frames_local = [{
        "index": 0, "function": "decode_mcu_block",
        "source_file": "/home/user/libjpeg-turbo/src/jdhuff.c", "source_line": 341,
    }]
    frames_dgx = [{
        "index": 0, "function": "decode_mcu_block",
        "source_file": "/dgxa_home/se24ucse043/project/src/jdhuff.c", "source_line": 341,
    }]
    result_local = normalize_stack(frames_local)
    result_dgx = normalize_stack(frames_dgx)
    assert result_local.stack_signature == result_dgx.stack_signature
    assert result_local.frames[0].source_file == "src/jdhuff.c"
    assert result_dgx.frames[0].source_file == "src/jdhuff.c"


def test_path_normalization_single_segment_path():
    frames = [{"index": 0, "function": "f", "source_file": "jdhuff.c", "source_line": 1}]
    result = normalize_stack(frames)
    assert result.frames[0].source_file == "jdhuff.c"


# ---------------------------------------------------------------------------
# TEST 12 — determinism
# ---------------------------------------------------------------------------

def test_determinism_repeated_calls_identical_result():
    frames = _heap_overflow_frames()
    first = normalize_stack(frames)
    second = normalize_stack(frames)
    assert first.normalized_stack == second.normalized_stack
    assert first.stack_signature == second.stack_signature


def test_determinism_across_independently_parsed_reports():
    text = (ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text()
    sig_1 = normalize_stack(parse_asan_report(text).stack_trace).stack_signature
    sig_2 = normalize_stack(parse_asan_report(text).stack_trace).stack_signature
    assert sig_1 == sig_2


# ---------------------------------------------------------------------------
# Malformed-frame robustness (from the follow-up requirements doc)
# ---------------------------------------------------------------------------

def test_malformed_frame_dict_missing_keys_entirely():
    result = normalize_stack([{"index": 0}])  # no function/source_file/source_line keys at all
    assert result.frames[0].function is None
    assert result.stack_signature is not None


def test_truncated_frame_only_index():
    result = normalize_stack([{"index": 0, "function": None, "source_file": None, "source_line": None}])
    assert result.normalized_stack == "<unknown>!<unknown>|<unknown>:<unknown>"


def test_double_question_mark_frame_preserved_verbatim():
    """
    ASan's own "symbol unknown" marker ("??") is preserved as-is — it
    is real information ASan reported, distinct from Phase 5 simply
    not having captured a function at all (which stays None ->
    "<unknown>"). The two must never be conflated.
    """
    result = normalize_stack([{"index": 0, "function": "??", "source_file": None, "source_line": None}])
    assert result.frames[0].function == "??"
    assert "??" in result.normalized_stack
    assert result.normalized_stack != "<unknown>!<unknown>|<unknown>:<unknown>"


def test_missing_function_specifically():
    result = normalize_stack([{"index": 0, "function": None, "source_file": "a.c", "source_line": 5}])
    assert result.frames[0].function is None


def test_missing_source_file_specifically():
    result = normalize_stack([{"index": 0, "function": "f", "source_file": None, "source_line": 5}])
    assert result.frames[0].source_file is None


def test_missing_source_line_specifically():
    result = normalize_stack([{"index": 0, "function": "f", "source_file": "a.c", "source_line": None}])
    assert result.frames[0].source_line is None


def test_missing_all_debug_information():
    result = normalize_stack([{"index": 0, "function": None, "source_file": None, "source_line": None}])
    assert result.stack_signature is not None  # still produces a deterministic signature, not a crash


def test_mixed_valid_and_invalid_frames():
    frames = [
        {"index": 0, "function": "decode_mcu_block", "source_file": "jdhuff.c", "source_line": 341},
        {"index": 1, "function": None, "source_file": None, "source_line": None},
        {"index": 2, "function": "main", "source_file": "djpeg.c", "source_line": 44},
    ]
    result = normalize_stack(frames)
    assert result.frame_count == 3
    lines = result.normalized_stack.splitlines()
    assert "decode_mcu_block" in lines[0]
    assert lines[1] == "<unknown>!<unknown>|<unknown>:<unknown>"
    assert "main" in lines[2]


def test_shared_library_module_frame():
    """
    Forward-compatible module support: if a frame-like input already
    carries a "module" field (e.g. a future richer Phase 5
    representation, or a caller-constructed dict), Phase 8 preserves
    it. Phase 5's current StackFrame does not capture module identity
    at all, so this test exercises the dict-input path directly rather
    than going through parse_asan_report().
    """
    frames = [{
        "index": 3, "function": "__libc_start_main", "source_file": None, "source_line": None,
        "module": "libc.so.6",
    }]
    result = normalize_stack(frames)
    assert result.frames[0].module == "libc.so.6"
    assert result.normalized_stack.startswith("libc.so.6!__libc_start_main")


def test_namespaced_cpp_function_name_not_mangled_or_stripped():
    frames = [{"index": 0, "function": "myproject::decoder::Frame::decode", "source_file": "decoder.cpp", "source_line": 88}]
    result = normalize_stack(frames)
    assert result.frames[0].function == "myproject::decoder::Frame::decode"
    assert "myproject::decoder::Frame::decode" in result.normalized_stack


# ---------------------------------------------------------------------------
# Integration with Phase 7 (CrashFeatures)
# ---------------------------------------------------------------------------

def test_normalize_crash_features_stack_integration():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    features = extract_features(asan=asan)
    result = normalize_crash_features_stack(features)

    assert result.stack_signature is not None
    assert result.frame_count == features.stack_depth
    # Raw evidence on CrashFeatures must remain completely untouched.
    assert features.raw_stack_trace[0]["function"] == "decode_mcu_block"


def test_normalize_does_not_mutate_crash_features_raw_stack_trace():
    asan = parse_asan_report((ASAN_FIXTURES / "heap_buffer_overflow.txt").read_text())
    features = extract_features(asan=asan)
    original = [dict(f) for f in features.raw_stack_trace]

    normalize_crash_features_stack(features)

    assert features.raw_stack_trace == original


def test_signature_does_not_depend_on_index_field_value():
    frames_a = [{"index": 0, "function": "f", "source_file": "a.c", "source_line": 1}]
    frames_b = [{"index": 99, "function": "f", "source_file": "a.c", "source_line": 1}]
    assert normalize_stack(frames_a).stack_signature == normalize_stack(frames_b).stack_signature
    assert "0" not in normalize_stack(frames_a).normalized_stack.split("!")[0]  # index not in serialized text
