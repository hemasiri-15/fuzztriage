"""
Unit tests for app.services.afl_parser, using the fixture fuzzer_stats
file at tests/fixtures/afl-output/default/fuzzer_stats.

These fixture values are fictional test data, not the real DGX
campaign numbers.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.afl_parser import parse_fuzzer_stats, parse_fuzzer_stats_text  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures"
FUZZER_STATS_PATH = FIXTURES / "afl-output" / "default" / "fuzzer_stats"


def test_parses_real_fixture_file():
    stats = parse_fuzzer_stats(FUZZER_STATS_PATH)
    assert not stats.is_empty
    assert stats.execs_done == 1050000
    assert stats.corpus_count == 1432
    assert stats.saved_crashes == 0
    assert stats.saved_hangs == 2
    assert stats.edges_found == 280
    assert stats.total_edges == 7377


def test_percentage_fields_are_coerced_to_float():
    stats = parse_fuzzer_stats(FUZZER_STATS_PATH)
    assert stats.stability == 100.00
    assert stats.bitmap_cvg == 4.62


def test_float_exec_per_sec():
    stats = parse_fuzzer_stats(FUZZER_STATS_PATH)
    assert stats.execs_per_sec == 165.10


def test_raw_dict_contains_every_field():
    stats = parse_fuzzer_stats(FUZZER_STATS_PATH)
    assert "afl_banner" in stats.raw
    assert stats.raw["afl_version"] == "++5.03a"
    # command_line isn't promoted to a typed attribute, but must still
    # be preserved in raw — nothing should be silently dropped.
    assert "command_line" in stats.raw


def test_missing_file_returns_empty_stats_not_an_exception():
    stats = parse_fuzzer_stats(FIXTURES / "afl-output" / "does-not-exist" / "fuzzer_stats")
    assert stats.is_empty
    assert stats.execs_done is None
    assert stats.saved_crashes is None


def test_tolerates_missing_fields():
    text = "execs_done       : 500\ncorpus_count     : 10\n"
    stats = parse_fuzzer_stats_text(text)
    assert stats.execs_done == 500
    assert stats.corpus_count == 10
    assert stats.saved_crashes is None  # not present, must stay None, not 0


def test_tolerates_unknown_extra_fields_without_crashing():
    text = "execs_done : 5\nsome_future_afl6_field : whatever\n"
    stats = parse_fuzzer_stats_text(text)
    assert stats.execs_done == 5
    assert stats.raw["some_future_afl6_field"] == "whatever"


def test_tolerates_blank_lines_and_whitespace():
    text = "\n\n   execs_done   :   42   \n\n"
    stats = parse_fuzzer_stats_text(text)
    assert stats.execs_done == 42


def test_tolerates_unparsable_numeric_value_gracefully():
    text = "execs_done : not-a-number\n"
    stats = parse_fuzzer_stats_text(text)
    assert stats.execs_done is None
    assert stats.raw["execs_done"] == "not-a-number"


def test_empty_text_returns_empty_stats():
    stats = parse_fuzzer_stats_text("")
    assert stats.is_empty
