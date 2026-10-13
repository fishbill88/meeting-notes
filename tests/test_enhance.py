"""Enhance: a person's own notes, expanded from the transcript without being rewritten."""

from __future__ import annotations

import json
import time

import pytest

from meeting_notes import enhance
from meeting_notes.cancel import PipelineCancelled


def _transcript(*turns):
    """Segments in the stored shape (centiseconds): ``(start_s, end_s, speaker, text)``."""
    return {
        "language": "en",
        "text": " ".join(t[3] for t in turns),
        "segments": [
            {"text": text, "t0": start * 100, "t1": end * 100, "speaker": who} for start, end, who, text in turns
        ],
    }


MEETING = _transcript(
    (0, 20, "Person 1", "Welcome everyone, let's start with the release schedule."),
    (20, 60, "Person 2", "We ship the new build on Friday after the regression tests pass."),
    (60, 120, "Person 1", "On pricing, the enterprise plan goes up by ten percent next quarter."),
    (120, 160, "Person 2", "Marketing will announce the pricing change by email in March."),
    (900, 960, "Person 1", "Finally the office move happens in April, boxes arrive Monday."),
)


def _reply(*lines):
    """A model's JSON answer: ``(n, [(text, at), ...])`` per line."""
    return json.dumps({"lines": [
        {"n": n, "details": [{"text": text, "at": at} for text, at in details]} for n, details in lines
    ]})


def test_the_clock_round_trips():
    assert enhance.clock(760) == "12:40"
    assert enhance.clock(3725) == "1:02:05"
    assert enhance.parse_clock("12:40") == 760.0
    assert enhance.parse_clock("[1:02:05]") == 3725.0
    assert enhance.parse_clock("soon") is None
    assert enhance.parse_clock("12:99") is None
    assert enhance.parse_clock(None) is None


def test_turns_merge_consecutive_speech_and_use_seconds():
    turns = enhance.turns_from_transcript(_transcript(
        (1, 2, "Person 1", "Hello."), (2, 4, "Person 1", "Again."), (4, 5, "Person 2", "Hi.")))
    assert [(t.start, t.end, t.speaker, t.text) for t in turns] == [
        (1.0, 4.0, "Person 1", "Hello. Again."), (4.0, 5.0, "Person 2", "Hi.")]


def test_a_transcript_without_segments_still_gives_text():
    turns = enhance.turns_from_transcript({"text": "Only text here.", "segments": []})
    assert len(turns) == 1 and turns[0].text == "Only text here."


def test_lines_are_cleaned_but_never_reworded():
    lines = enhance.clean_lines([
        {"t": "12", "text": "  - pricing?  "}, {"text": "   "}, "plain string", {"t": "x", "text": "bad time"},
        {"t": -5, "text": "negative"}, 42,
    ])
    assert [(l["t"], l["text"]) for l in lines] == [
        (12.0, "  - pricing?"), (None, "plain string"), (None, "bad time"), (0.0, "negative")]


# ------------------------------------------------------------------ the invariant


def test_every_line_of_mine_comes_back_in_order_in_the_persons_words_whatever_the_model_says():
    mine = [{"t": 70, "text": "pricing?"}, {"t": None, "text": "  - ship date"}, {"t": 5, "text": "# Agenda"}]
    # A model that scrambles numbering, invents a line 9, repeats, and returns junk for one line.
    def hostile(prompt):
        return json.dumps({"lines": [
            {"n": 9, "details": [{"text": "invented", "at": "00:01"}]},
            {"n": 2, "details": "not a list"},
            {"n": "x"},
            {"n": 1, "details": [{"text": "Enterprise pricing goes up ten percent next quarter.", "at": "01:00"}]},
        ]})

    out = enhance.enhance(mine, MEETING, generate=hostile, generator="test-model")
    items = [i for i in out["items"] if i["kind"] == "mine"]
    assert [i["text"] for i in items] == ["pricing?", "  - ship date", "# Agenda"]
    assert items[0]["children"][0]["text"].startswith("Enterprise pricing")
    assert items[1]["children"] == [] and items[2]["children"] == []
    assert items[0]["t"] == 70.0 and "t" not in items[1]


def test_a_failing_model_falls_back_to_the_extractive_path_and_loses_nothing():
    def broken(prompt):
        raise RuntimeError("model fell over")

    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=broken, generator="llamacpp:x")
    assert out["generator"] == "extractive"
    item = out["items"][0]
    assert item["text"] == "pricing"
    assert item["children"] and all(c["source_t"] is not None for c in item["children"])


def test_unparseable_model_output_also_falls_back():
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=lambda p: "I cannot do that", generator="m")
    assert out["generator"] == "extractive"


def test_no_model_is_extractive():
    out = enhance.enhance([{"t": 25, "text": "ship build friday"}], MEETING)
    assert out["generator"] == "extractive"
    assert out["items"][0]["children"][0]["source_t"] == 20.0


# ------------------------------------------------------------------ provenance


def test_a_detail_the_transcript_does_not_support_is_dropped():
    reply = _reply((1, [("The CEO resigned and the company was sold to Acme.", "01:00"),
                        ("Enterprise plan goes up ten percent next quarter.", "01:00")]))
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=lambda p: reply, generator="m")
    texts = [c["text"] for c in out["items"][0]["children"]]
    assert texts == ["Enterprise plan goes up ten percent next quarter."]


def test_a_true_detail_with_the_wrong_minute_is_pinned_to_where_it_was_said():
    reply = _reply((1, [("Marketing announces the pricing change by email in March.", "02:30")]))
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=lambda p: reply, generator="m")
    child = out["items"][0]["children"][0]
    assert child["source_t"] == 120.0          # the turn it actually came from


def test_a_citation_far_from_the_real_place_is_not_trusted():
    # True sentence, but cited at 15:00 where the transcript is about the office move.
    reply = _reply((1, [("Enterprise plan goes up ten percent next quarter.", "15:00")]))
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=lambda p: reply, generator="m")
    assert out["items"][0]["children"] == []


def test_a_missing_or_garbled_time_is_checked_against_every_turn():
    reply = _reply((1, [("Enterprise plan goes up ten percent next quarter.", "whenever")]))
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=lambda p: reply, generator="m")
    assert out["items"][0]["children"][0]["source_t"] is not None   # no cited time: any turn may back it up


def test_details_are_capped_per_line():
    many = [("Enterprise plan goes up ten percent next quarter.", "01:00")] * 8
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING, generate=lambda p: _reply((1, many)), generator="m")
    assert len(out["items"][0]["children"]) == enhance.MAX_DETAILS_PER_LINE


# ------------------------------------------------------------------ extras and empties


def test_decisions_and_action_items_from_the_notes_are_added_as_sections():
    notes = {"decisions": ["Ship on Friday"], "action_items": ["Ana sends the build"]}
    out = enhance.enhance([{"t": None, "text": "hi"}], MEETING, notes)
    sections = [i for i in out["items"] if i["kind"] == "section"]
    assert [s["text"] for s in sections] == ["Decisions", "Action items"]
    assert sections[1]["children"][0] == {"kind": "added", "text": "Ana sends the build", "source_t": None, "children": []}


def test_an_empty_notepad_gives_only_the_sections():
    out = enhance.enhance([], MEETING, {"decisions": ["Ship on Friday"]})
    assert [i["kind"] for i in out["items"]] == ["section"]


def test_search_text_and_markdown_cover_everything():
    out = enhance.enhance([{"t": 70, "text": "pricing"}], MEETING,
                          {"decisions": ["Ship on Friday"]},
                          generate=lambda p: _reply((1, [("Enterprise plan goes up ten percent next quarter.", "01:00")])),
                          generator="m")
    text = enhance.flatten_text(out)
    assert "pricing" in text and "ten percent" in text and "Ship on Friday" in text
    md = enhance.to_markdown(out)
    assert md.startswith("pricing\n  - Enterprise plan goes up ten percent next quarter. (01:00)")
    assert "## Decisions\n- Ship on Friday" in md


# ------------------------------------------------------------------ long transcripts


def _long_transcript():
    turns = []
    for i in range(240):
        topic = "pricing" if i == 40 else "planning"
        turns.append((i * 30, i * 30 + 29, f"Person {i % 2 + 1}",
                      f"Number {i}: " + ("we talk about the {} item at length and in detail ".format(topic) * 6)))
    return _transcript(*turns)


def test_a_long_transcript_is_sent_in_batches_with_only_the_nearby_turns():
    transcript = _long_transcript()
    prompts = []

    def generate(prompt):
        prompts.append(prompt)
        return json.dumps({"lines": []})

    lines = [{"t": 60, "text": "early"}, {"t": 3600, "text": "middle"}, {"t": 7000, "text": "late"}]
    enhance.enhance(lines, transcript, generate=generate, generator="m")
    assert len(prompts) >= 2
    assert all(len(p) < enhance.BATCH_CHARS + 3500 for p in prompts)
    assert "[00:30]" in prompts[0] and "[1:56:30]" not in prompts[0]


def test_every_line_is_asked_about_in_some_batch():
    asked = set()

    def generate(prompt):
        import re
        asked.update(int(n) for n in re.findall(r"^(\d+)\. ", prompt, flags=re.M))
        return json.dumps({"lines": []})

    lines = [{"t": 100 * i, "text": f"line {i}"} for i in range(1, 30)] + [{"t": None, "text": "pricing"}]
    enhance.enhance(lines, _long_transcript(), generate=generate, generator="m")
    assert asked == set(range(1, 31))


def test_cancel_is_honoured_between_calls():
    calls = []

    def generate(prompt):
        calls.append(1)
        return json.dumps({"lines": []})

    lines = [{"t": 100 * i, "text": f"line {i}"} for i in range(1, 30)]
    with pytest.raises(PipelineCancelled):
        enhance.enhance(lines, _long_transcript(), generate=generate, generator="m",
                        should_cancel=lambda: len(calls) >= 1)
    assert len(calls) == 1


def test_progress_is_reported_per_batch():
    seen = []
    lines = [{"t": 100 * i, "text": f"line {i}"} for i in range(1, 30)]
    enhance.enhance(lines, _long_transcript(), generate=lambda p: json.dumps({"lines": []}), generator="m",
                    on_progress=lambda f, m: seen.append((f, m)))
    assert seen[0][0] == 0 and seen[-1][0] == 1.0
    assert "part 1 of" in seen[0][1]


# ------------------------------------------------------------------ storage and the routes

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from meeting_notes import server as server_module  # noqa: E402
from meeting_notes.server import TOKEN_ENV, app  # noqa: E402
from meeting_notes.storage import MeetingRecord, MeetingStore  # noqa: E402

TOKEN = "enhance-test-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
KEY = "rec-0123456789abcdef"


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv(TOKEN_ENV, TOKEN)
    monkeypatch.setenv("MEETING_NOTES_DATA_DIR", str(tmp_path))
    server_module.reset_store()
    server_module.reset_jobs()
    yield tmp_path
    server_module.reset_store()
    server_module.reset_jobs()


@pytest.fixture
def client(env):
    return TestClient(app)


def _save_meeting(client) -> str:
    response = client.post("/meetings", headers=AUTH, json={
        "transcript": MEETING, "notes": {"title": "Release sync", "summary": "s", "key_topics": [],
                                         "decisions": ["Ship on Friday"], "action_items": [], "language": "en"},
        "duration_seconds": 960})
    return response.json()["id"]


def _wait(client, job_id, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        body = client.get(f"/jobs/{job_id}", headers=AUTH).json()
        if body["state"] in ("done", "failed", "cancelled"):
            return body
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_health_advertises_the_features_without_changing_the_api_number(client):
    body = client.get("/health").json()
    assert body["api"] == 1
    assert {"notepad", "enhance"} <= set(body["features"])


def test_the_notepad_routes_need_the_token(client):
    assert client.put(f"/notepads/{KEY}", json={"lines": []}).status_code == 401
    assert client.get(f"/notepads/{KEY}").status_code == 401
    assert client.delete(f"/notepads/{KEY}").status_code == 401
    assert client.post("/jobs/enhance", json={"meeting_id": "a", "notepad_key": KEY}).status_code == 401


def test_a_notepad_saves_reads_back_and_deletes(client, env):
    assert client.get(f"/notepads/{KEY}", headers=AUTH).json() == {"lines": [], "updated_at": None}
    saved = client.put(f"/notepads/{KEY}", headers=AUTH,
                       json={"lines": [{"t": 12.5, "text": "pricing?"}, {"text": "  "}, {"text": "next steps"}]})
    assert saved.json()["count"] == 2
    got = client.get(f"/notepads/{KEY}", headers=AUTH).json()
    assert got["lines"] == [{"t": 12.5, "text": "pricing?"}, {"t": None, "text": "next steps"}]
    assert got["updated_at"]
    assert client.delete(f"/notepads/{KEY}", headers=AUTH).json() == {"deleted": True}
    assert client.get(f"/notepads/{KEY}", headers=AUTH).json()["lines"] == []


def test_the_notepad_is_encrypted_on_disk(client, env):
    client.put(f"/notepads/{KEY}", headers=AUTH, json={"lines": [{"text": "secret acquisition plan"}]})
    raw = b"".join(p.read_bytes() for p in env.iterdir() if p.suffix == ".db")
    assert b"secret acquisition plan" not in raw


@pytest.mark.parametrize("key", ["short", "has space here!", "x" * 200, "..%2f..%2fetc"])
def test_a_bad_notepad_key_is_refused(client, key):
    assert client.put(f"/notepads/{key}", headers=AUTH, json={"lines": []}).status_code in (400, 404, 422)


def test_a_database_from_before_notepads_is_migrated(tmp_path):
    import sqlite3

    db = tmp_path / "meetings.db"
    store = MeetingStore(directory=tmp_path)
    with sqlite3.connect(db) as conn:
        conn.execute("DROP TABLE notepads")
        conn.execute("PRAGMA user_version = 1")
    again = MeetingStore(directory=tmp_path)
    assert again.save_notepad(KEY, [{"t": None, "text": "ok"}])
    assert again.get_notepad(KEY)[0] == [{"t": None, "text": "ok"}]
    del store


def test_an_old_record_without_notes_fields_still_loads():
    record = MeetingRecord.from_dict({"id": "a", "created_at": "2026-01-01T00:00:00+00:00",
                                      "transcript": {}, "notes": {}})
    assert record.my_notes == [] and record.enhanced == {}


def test_enhance_end_to_end_stores_the_tree_on_the_meeting(client, monkeypatch):
    class Provider:
        name = "llamacpp:fake"

        def generate(self, prompt):
            return _reply((1, [("The enterprise plan goes up ten percent next quarter.", "01:00")]))

    from meeting_notes import llm
    monkeypatch.setattr(llm, "default_provider", lambda: Provider())

    meeting_id = _save_meeting(client)
    client.put(f"/notepads/{KEY}", headers=AUTH, json={"lines": [{"t": 70, "text": "pricing?"}, {"text": "- venue"}]})

    created = client.post("/jobs/enhance", headers=AUTH, json={"meeting_id": meeting_id, "notepad_key": KEY})
    assert created.status_code == 200
    assert created.json()["kind"] == "enhance"
    done = _wait(client, created.json()["id"])
    assert done["state"] == "done", done
    assert done["meeting_id"] == meeting_id

    meeting = client.get(f"/meetings/{meeting_id}", headers=AUTH).json()
    assert [l["text"] for l in meeting["my_notes"]] == ["pricing?", "- venue"]
    assert meeting["enhanced"]["generator"] == "llamacpp:fake"
    mine = [i for i in meeting["enhanced"]["items"] if i["kind"] == "mine"]
    assert [i["text"] for i in mine] == ["pricing?", "- venue"]
    assert mine[0]["children"][0]["source_t"] == 60.0
    # The original notes are untouched, and the search finds the person's words.
    assert meeting["notes"]["title"] == "Release sync"
    found = client.get("/meetings", headers=AUTH, params={"q": "venue"}).json()
    assert [m["id"] for m in found] == [meeting_id]


def test_enhance_without_a_model_still_works(client, monkeypatch):
    from meeting_notes import llm
    monkeypatch.setattr(llm, "default_provider", lambda: None)
    meeting_id = _save_meeting(client)
    client.put(f"/notepads/{KEY}", headers=AUTH, json={"lines": [{"t": 70, "text": "pricing enterprise plan"}]})
    job = client.post("/jobs/enhance", headers=AUTH, json={"meeting_id": meeting_id, "notepad_key": KEY}).json()
    assert _wait(client, job["id"])["state"] == "done"
    meeting = client.get(f"/meetings/{meeting_id}", headers=AUTH).json()
    assert meeting["enhanced"]["generator"] == "extractive"


def test_an_empty_notepad_fails_with_a_sentence(client):
    meeting_id = _save_meeting(client)
    job = client.post("/jobs/enhance", headers=AUTH, json={"meeting_id": meeting_id, "notepad_key": KEY}).json()
    done = _wait(client, job["id"])
    assert done["state"] == "failed" and "notepad" in done["error"]


def test_enhancing_notes_that_do_not_exist_is_404(client):
    assert client.post("/jobs/enhance", headers=AUTH,
                       json={"meeting_id": "nope", "notepad_key": KEY}).status_code == 404


def test_cancelling_an_enhance_job_saves_nothing(client, monkeypatch):
    import threading

    gate = threading.Event()

    class Provider:
        name = "llamacpp:fake"

        def generate(self, prompt):
            gate.wait(5)
            return _reply((1, [("The enterprise plan goes up ten percent next quarter.", "01:00")]))

    from meeting_notes import llm
    monkeypatch.setattr(llm, "default_provider", lambda: Provider())
    meeting_id = _save_meeting(client)
    client.put(f"/notepads/{KEY}", headers=AUTH, json={"lines": [{"t": 70, "text": "pricing?"}]})
    job = client.post("/jobs/enhance", headers=AUTH, json={"meeting_id": meeting_id, "notepad_key": KEY}).json()
    time.sleep(0.2)
    client.delete(f"/jobs/{job['id']}", headers=AUTH)
    gate.set()
    done = _wait(client, job["id"])
    assert done["state"] == "cancelled"
    assert client.get(f"/meetings/{meeting_id}", headers=AUTH).json()["enhanced"] == {}
