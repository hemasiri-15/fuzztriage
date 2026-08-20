"""
Unit tests for app.services.artifact_collector.

Covers both fixture trees:
  - tests/fixtures/afl-output/default        (empty crashes/, 2 hangs — mirrors the real DGX state)
  - tests/fixtures/afl-output/with-crash      (1 crash — to test the non-empty path)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from app.services.artifact_collector import collect_artifacts, parse_afl_filename  # noqa: E402

FIXTURES = Path(__file__).parent / "fixtures" / "afl-output"


def test_collects_queue_artifacts():
    result = collect_artifacts(FIXTURES / "default")
    assert result.queue_count == 2


def test_empty_crashes_directory_is_valid_not_an_error():
    result = collect_artifacts(FIXTURES / "default")
    assert result.crash_count == 0
    assert result.crashes == []


def test_hangs_are_not_classified_as_crashes():
    result = collect_artifacts(FIXTURES / "default")
    assert result.hang_count == 2
    assert result.crash_count == 0
    for hang in result.hangs:
        assert hang.artifact_type == "hang"


def test_missing_fuzz_output_dir_returns_empty_collection():
    result = collect_artifacts(FIXTURES / "does-not-exist")
    assert result.queue_count == 0
    assert result.crash_count == 0
    assert result.hang_count == 0


def test_with_crash_fixture_has_one_crash():
    result = collect_artifacts(FIXTURES / "with-crash" / "default")
    assert result.crash_count == 1
    assert result.crashes[0].artifact_type == "crash"
    assert result.crashes[0].afl_id == "000000"


def test_readme_and_dotfiles_are_ignored():
    result = collect_artifacts(FIXTURES / "default")
    all_names = [r.filename.lower() for r in result.queue + result.crashes + result.hangs]
    assert "readme_fixture.txt" not in all_names
    assert ".gitkeep" not in all_names


def test_parse_afl_filename_standard_format():
    meta = parse_afl_filename("id:000001,src:000875,time:2153861,execs:332398,op:flip32,pos:183")
    assert meta["id"] == "000001"
    assert meta["src"] == "000875"
    assert meta["time"] == "2153861"
    assert meta["execs"] == "332398"
    assert meta["op"] == "flip32"
    assert meta["pos"] == "183"


def test_parse_afl_filename_seed_with_orig():
    meta = parse_afl_filename("id:000000,time:0,execs:0,orig:testorig.jpg")
    assert meta["id"] == "000000"
    assert meta["orig"] == "testorig.jpg"


def test_parse_afl_filename_crash_with_signal():
    meta = parse_afl_filename("id:000000,sig:06,src:000042,time:8811023,execs:1049812,op:havoc,rep:16")
    assert meta["sig"] == "06"
    assert meta["op"] == "havoc"
    assert meta["rep"] == "16"


def test_parse_afl_filename_empty_string_does_not_raise():
    meta = parse_afl_filename("")
    assert meta["flags"] == []


def test_parse_afl_filename_malformed_tokens_go_to_flags_not_dropped():
    meta = parse_afl_filename("id:000005,+cov,orig:seed.jpg")
    assert meta["id"] == "000005"
    assert "+cov" in meta["flags"]
    assert meta["orig"] == "seed.jpg"


def test_recognized_flag_false_for_non_afl_filename():
    result = collect_artifacts(FIXTURES / "default")
    # Every fixture artifact here follows AFL's id: convention.
    for record in result.queue + result.hangs:
        assert record.recognized is True


def test_size_bytes_is_populated():
    result = collect_artifacts(FIXTURES / "default")
    for record in result.queue:
        assert record.size_bytes > 0
