"""Tests for encrypted consultation storage.

The load-bearing claims here are that stored records are genuinely unreadable
on disk and that the schema can migrate without losing them. Both are the kind
of thing that is quietly wrong until it matters.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

pytest.importorskip("cryptography")

from meeting_notes.crypto import Vault, VaultError  # noqa: E402
from meeting_notes.storage import (  # noqa: E402
    DB_FILENAME,
    KEY_FILENAME,
    SCHEMA_VERSION,
    MeetingRecord,
    MeetingStore,
    default_data_dir,
)

TRANSCRIPT = {
    "text": "Nag-reklamo siya ng chest pain. Binigyan ko ng amoxicillin.",
    "language": "tl",
    "language_name": "Filipino / Tagalog",
    "timed_transcript": "[00:00 – 00:05] Nag-reklamo siya ng chest pain.",
    "backend": "nemo-tagalog",
}

NOTES = {
    "title": "Chest pain, 3 days",
    "summary": "Patient reports intermittent chest pain.",
    "key_topics": ["chest pain"],
    "decisions": ["Likely musculoskeletal"],
    "action_items": ["Amoxicillin 500 mg twice a day"],
    "language": "tl",
}


@pytest.fixture
def store(tmp_path) -> MeetingStore:
    return MeetingStore(directory=tmp_path)


def make_record(**overrides) -> MeetingRecord:
    record = MeetingRecord.new(
        transcript=dict(TRANSCRIPT), notes=dict(NOTES), duration_seconds=90.0
    )
    for key, value in overrides.items():
        setattr(record, key, value)
    return record


# ---------------------------------------------------------------------------
# Vault
# ---------------------------------------------------------------------------


def test_vault_round_trips(tmp_path):
    vault = Vault(tmp_path / "k")
    assert vault.decrypt(vault.encrypt("lagnat at ubo")) == "lagnat at ubo"


def test_vault_creates_its_key_on_first_use(tmp_path):
    vault = Vault(tmp_path / "k")
    assert not vault.exists()
    vault.encrypt("x")
    assert vault.exists()


def test_vault_reuses_an_existing_key(tmp_path):
    token = Vault(tmp_path / "k").encrypt("secret")
    # A separate instance over the same key file must still read it.
    assert Vault(tmp_path / "k").decrypt(token) == "secret"


def test_vault_rejects_a_foreign_key(tmp_path):
    token = Vault(tmp_path / "a").encrypt("secret")
    with pytest.raises(VaultError, match="Could not decrypt"):
        Vault(tmp_path / "b").decrypt(token)


def test_vault_ciphertext_does_not_contain_the_plaintext(tmp_path):
    token = Vault(tmp_path / "k").encrypt("amoxicillin")
    assert b"amoxicillin" not in token


def test_passphrase_mode_round_trips(tmp_path):
    vault = Vault(tmp_path / "salt", passphrase="correct horse")
    assert vault.decrypt(vault.encrypt("lagnat")) == "lagnat"


def test_passphrase_mode_rejects_the_wrong_passphrase(tmp_path):
    token = Vault(tmp_path / "salt", passphrase="right").encrypt("secret")
    with pytest.raises(VaultError):
        Vault(tmp_path / "salt", passphrase="wrong").decrypt(token)


def test_passphrase_mode_stores_only_a_salt(tmp_path):
    path = tmp_path / "salt"
    Vault(path, passphrase="pw").encrypt("x")
    # A salt, not a key — 16 bytes, and useless without the passphrase.
    assert path.stat().st_size == 16


def test_vault_reports_its_mode(tmp_path):
    assert Vault(tmp_path / "k").mode == "key-file"
    assert Vault(tmp_path / "k", passphrase="pw").mode == "passphrase"


def test_key_material_is_written_byte_exact(tmp_path):
    """Regression: Windows text mode expanded 0x0A to 0x0D 0x0A.

    Key material is random binary, so roughly one salt in sixteen contained a
    newline byte and was silently corrupted on write. The bug was invisible
    from a round-trip test because reads saw the same corrupted bytes.
    """
    from meeting_notes.crypto import _write_private

    payload = bytes([0x00, 0x0A, 0x0D, 0x0A, 0x1A, 0xFF, 0x41])
    path = tmp_path / "binary"
    _write_private(path, payload)

    assert path.read_bytes() == payload
    assert path.stat().st_size == len(payload)


@pytest.mark.parametrize("run", range(12))
def test_passphrase_salt_is_always_exactly_16_bytes(tmp_path, run):
    # Repeated because the original failure only showed up when the random
    # salt happened to contain a newline byte.
    path = tmp_path / f"salt-{run}"
    Vault(path, passphrase="pw").encrypt("x")
    assert path.stat().st_size == 16


# ---------------------------------------------------------------------------
# Round-tripping
# ---------------------------------------------------------------------------


def test_save_and_get(store):
    record = make_record()
    store.save(record)

    loaded = store.get(record.id)
    assert loaded is not None
    assert loaded.notes["title"] == "Chest pain, 3 days"
    assert loaded.transcript["text"] == TRANSCRIPT["text"]
    assert loaded.duration_seconds == 90.0


def test_get_missing_returns_none(store):
    assert store.get("does-not-exist") is None


def test_save_is_idempotent_on_id(store):
    record = make_record()
    store.save(record)
    record.notes = {**NOTES, "title": "Revised title"}
    store.save(record)

    assert store.count() == 1
    loaded = store.get(record.id)
    assert loaded is not None and loaded.notes["title"] == "Revised title"


def test_records_get_distinct_ids():
    assert MeetingRecord.new({}, {}).id != MeetingRecord.new({}, {}).id


def test_record_exposes_metadata_from_payload():
    record = make_record()
    assert record.title == "Chest pain, 3 days"
    assert record.language == "tl"
    assert record.backend == "nemo-tagalog"


def test_record_title_falls_back_when_absent():
    assert MeetingRecord.new({}, {}).title == "Untitled consultation"


# ---------------------------------------------------------------------------
# Encryption at rest — the claim worth actually checking
# ---------------------------------------------------------------------------


def test_patient_content_is_not_readable_in_the_database_file(store, tmp_path):
    store.save(make_record())
    raw = (tmp_path / DB_FILENAME).read_bytes()

    for secret in (b"chest pain", b"amoxicillin", b"Nag-reklamo"):
        assert secret not in raw


def test_metadata_stays_queryable_in_the_clear(store, tmp_path):
    record = make_record()
    store.save(record)

    # Sorting and pagination must not require decrypting every row.
    with sqlite3.connect(tmp_path / DB_FILENAME) as conn:
        row = conn.execute(
            "SELECT language, backend, duration_seconds FROM meetings"
        ).fetchone()
    assert row == ("tl", "nemo-tagalog", 90.0)


def test_a_database_without_its_key_is_unreadable(tmp_path):
    store = MeetingStore(directory=tmp_path)
    record_id = store.save(make_record())

    (tmp_path / KEY_FILENAME).unlink()
    fresh = MeetingStore(directory=tmp_path)  # generates a new key

    with pytest.raises(VaultError):
        fresh.get(record_id)


# ---------------------------------------------------------------------------
# Listing and search
# ---------------------------------------------------------------------------


def test_list_is_newest_first(store):
    for index, stamp in enumerate(["2026-01-01", "2026-03-01", "2026-02-01"]):
        record = make_record()
        record.created_at = f"{stamp}T09:00:00+00:00"
        record.notes = {**NOTES, "title": f"Visit {index}"}
        store.save(record)

    titles = [s.title for s in store.list()]
    assert titles == ["Visit 1", "Visit 2", "Visit 0"]


def test_list_paginates(store):
    for index in range(5):
        record = make_record()
        record.created_at = f"2026-01-0{index + 1}T09:00:00+00:00"
        store.save(record)

    assert len(store.list(limit=2)) == 2
    assert len(store.list(limit=2, offset=4)) == 1


def test_list_summary_includes_a_preview(store):
    store.save(make_record())
    summary = store.list()[0]
    assert summary.preview.startswith("Patient reports")
    assert summary.title == "Chest pain, 3 days"


def test_search_matches_the_transcript(store):
    store.save(make_record())
    assert len(store.search("amoxicillin")) == 1


def test_search_matches_the_title(store):
    store.save(make_record())
    assert len(store.search("chest")) == 1


def test_search_is_case_insensitive(store):
    store.save(make_record())
    assert len(store.search("AMOXICILLIN")) == 1


def test_search_excludes_non_matches(store):
    store.save(make_record())
    assert store.search("metformin") == []


def test_empty_search_returns_everything(store):
    store.save(make_record())
    assert len(store.search("   ")) == 1


def test_search_respects_its_limit(store):
    for index in range(4):
        record = make_record()
        record.created_at = f"2026-01-0{index + 1}T09:00:00+00:00"
        store.save(record)
    assert len(store.search("chest", limit=2)) == 2


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------


def test_delete_removes_the_record(store):
    record_id = store.save(make_record())
    assert store.delete(record_id) is True
    assert store.get(record_id) is None
    assert store.count() == 0


def test_delete_reports_a_missing_record(store):
    assert store.delete("nope") is False


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


def test_schema_version_is_stamped(tmp_path):
    MeetingStore(directory=tmp_path)
    with sqlite3.connect(tmp_path / DB_FILENAME) as conn:
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION


def test_reopening_does_not_re_run_migrations_or_lose_data(tmp_path):
    record_id = MeetingStore(directory=tmp_path).save(make_record())

    reopened = MeetingStore(directory=tmp_path)
    assert reopened.get(record_id) is not None
    assert reopened.count() == 1


def test_data_dir_honours_the_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("MEETING_NOTES_DATA_DIR", str(tmp_path))
    assert default_data_dir() == tmp_path


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def test_record_survives_a_json_round_trip():
    record = make_record()
    restored = MeetingRecord.from_dict(json.loads(json.dumps(record.to_dict())))

    assert restored.id == record.id
    assert restored.transcript == record.transcript
    assert restored.notes == record.notes


def test_from_dict_tolerates_missing_optional_fields():
    record = MeetingRecord.from_dict(
        {"id": "x", "created_at": "2026-01-01T00:00:00+00:00"}
    )
    assert record.transcript == {}
    assert record.speaker_names == {}
    assert record.duration_seconds == 0.0


def test_searchable_text_covers_notes_and_transcript():
    text = make_record().searchable_text()
    assert "chest pain" in text
    assert "Amoxicillin 500 mg" in text
    assert "musculoskeletal" in text
