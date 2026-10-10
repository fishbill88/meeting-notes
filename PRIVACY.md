# Data flow and privacy — Meeting Notes AI (Desktop)

Written for the question a clinic's data protection officer will actually ask:
*where does the patient's voice go, and who else can hear it?*

Scope: the **desktop** application. The mobile app is documented separately.

---

## Short answer

Nothing leaves the computer. Audio is never written to permanent storage.
Consultation records are encrypted on disk.

---

## What is processed

| Data | Where it goes | How long it lives |
|---|---|---|
| Microphone / system audio | Browser memory in the app window → uploaded over loopback to the local transcription service | Deleted as soon as transcription finishes. **Never persisted.** |
| Transcript | Generated locally, held in the app, saved encrypted | Until the clinician deletes the consultation |
| Notes | Generated locally, saved encrypted | Until the clinician deletes the consultation |
| Metadata (date, duration, language code, engine name) | Stored unencrypted so the history list can sort and search | Until the consultation is deleted |

Metadata is deliberately not encrypted so the history list can paginate without
decrypting every record. It contains no patient identifier — a date, a length,
`tl`, and `nemo-tagalog`.

## What does not happen

- **No network transmission.** Speech recognition and note generation both run
  on the local machine. The application makes no outbound connections during a
  consultation.
- **No cloud AI service.** No OpenAI, Google, Azure, AWS or any other API is
  contacted with patient content.
- **No telemetry, analytics or crash reporting.**
- **No audio retention.** There is no "keep recordings" setting, because there
  is no stored recording to keep.

The only outbound network traffic the app ever makes is **model downloads**, on
first run or when a clinician selects a different model. Those requests contain
no patient data — they fetch published model files from Hugging Face.

## How the local service is protected

The transcription service runs as a separate local process (the "sidecar"):

- Bound to `127.0.0.1` only. It is not reachable from the clinic network, and
  this is not configurable.
- Listens on a **random port** chosen at launch.
- Requires a **bearer token** generated fresh each launch. The token is passed
  to the app window over a private channel and never written to disk.
- Terminates when the application window closes.

## Encryption at rest

Consultation content is encrypted with AES (Fernet: AES-128-CBC with HMAC
authentication) before it is written to the local database.

Two key modes, with different guarantees — read this before choosing:

**Key-file mode (default).** The key is stored beside the database with
owner-only permissions. This protects the database when it is read *somewhere
other than this Windows account*: a stolen backup, a folder synced to cloud
storage, a copy sent to IT support, another user on a shared machine. It does
**not** protect against malware or a person operating this user's account,
because they can read the key file too.

**Passphrase mode.** Set `MEETING_NOTES_PASSPHRASE`. The key is derived with
scrypt and only a salt is stored, so the database is unreadable without the
passphrase even with full access to the disk. **There is no recovery.** A
forgotten passphrase means the consultations are permanently unreadable.

Neither mode replaces full-disk encryption. A clinic handling sensitive records
should enable BitLocker regardless, and should consider passphrase mode.

⚠️ **Back up the key file.** In key-file mode, backing up `meetings.db` without
`meetings.key` produces a backup that cannot be restored. Both files live in the
application data directory. Use `meeting-notes vault` to inspect or export them.

## Deletion

Deleting a consultation removes its row from the database immediately. Note that
SQLite may leave the freed pages recoverable by forensic tools until they are
overwritten; for disposal of a machine, wipe the disk rather than relying on
record deletion.

## Philippine regulatory context

This is an engineering description, not legal advice — a clinic must make its
own assessment.

Relevant instruments: the **Data Privacy Act of 2012 (RA 10173)**, National
Privacy Commission issuances, and Department of Health guidance on health
information. Health data is **sensitive personal information** under the Act.

Points that follow from the design above:

- **No third-party processor.** Because transcription runs on-device, no
  outsourcing agreement or cross-border transfer assessment is needed for the
  transcription step. There is no vendor receiving patient audio.
- **No data residency question.** Patient data does not leave the machine, so
  it does not leave the Philippines.
- **Registration thresholds still apply to the clinic.** The Act's registration
  requirements attach to the organisation processing the data, not to this
  software. A clinic above the relevant thresholds registers its processing
  systems regardless of which tools it uses.
- **Consent and notice remain the clinician's responsibility.** This software
  records what it is pointed at. Informing the patient that a consultation is
  being recorded and transcribed, and obtaining consent, is a clinical and
  legal obligation this application does not discharge for you.

## Accuracy — a safety note, not a privacy one

Automatic transcription is imperfect, and materially more so for Tagalog and
code-switched Taglish speech than for English. The application shows which
engine produced each transcript and flags low-confidence passages, and the
transcript is editable for exactly this reason.

**Notes generated by this application must be reviewed by the clinician before
being entered into a patient's record.** They are a drafting aid, not a
substitute for clinical documentation.

---

*Last updated: 2026-07-30. Update this document whenever the data flow changes.*
