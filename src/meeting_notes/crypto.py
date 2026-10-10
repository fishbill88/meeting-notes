"""Encryption for stored consultations.

**Threat model — read this before relying on it.**

Two modes, with genuinely different guarantees:

*Key-file mode* (default) keeps the key in a file next to the database, with
owner-only permissions. It protects against the database being read somewhere
other than this account: a stolen backup, a synced folder, a file emailed to
support, another user on a shared machine. It does **not** protect against
malware or anyone running as this user, because they can read the key file too.

*Passphrase mode* derives the key with scrypt and stores only the salt, so the
database is unreadable without the passphrase even with full filesystem
access. The cost is that the clinician has to type it, and a forgotten
passphrase means the records are gone — there is no recovery path.

Neither is a substitute for full-disk encryption. Key-file mode is the default
because a scribe that loses a day of consultations to a forgotten passphrase is
worse than one with a merely-good-enough at-rest story, but a clinic handling
sensitive records should turn passphrase mode on.
"""

from __future__ import annotations

import base64
import os
import secrets
import stat
from pathlib import Path
from typing import Optional

_INSTALL_HINT = (
    "The `cryptography` package is required to store consultations.\n"
    "Install it with:\n"
    "    pip install -e \".[desktop]\""
)

#: scrypt parameters. n=2^15 costs roughly 100ms on a clinic laptop — slow
#: enough to matter for brute force, fast enough not to annoy at unlock.
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SALT_BYTES = 16


class VaultError(RuntimeError):
    """Raised when the vault cannot be opened or a payload cannot be read."""


def _require_cryptography():
    try:
        from cryptography.fernet import Fernet  # noqa: PLC0415
    except ImportError as exc:
        raise VaultError(_INSTALL_HINT) from exc
    return Fernet


class Vault:
    """Encrypts and decrypts record payloads.

    Args:
        key_path: Where the key (or, in passphrase mode, the salt) lives.
        passphrase: Enables passphrase mode when given.
    """

    def __init__(self, key_path: Path, passphrase: Optional[str] = None) -> None:
        self.key_path = Path(key_path)
        self._passphrase = passphrase
        self._fernet = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def mode(self) -> str:
        return "passphrase" if self._passphrase else "key-file"

    def encrypt(self, plaintext: str) -> bytes:
        """Encrypt UTF-8 text into an opaque token."""
        return self._cipher().encrypt(plaintext.encode("utf-8"))

    def decrypt(self, token: bytes) -> str:
        """Decrypt a token produced by :meth:`encrypt`.

        Raises:
            VaultError: If the token is corrupt or the key is wrong. These are
                not distinguishable, by design of the underlying construction.
        """
        try:
            from cryptography.fernet import InvalidToken  # noqa: PLC0415
        except ImportError as exc:
            raise VaultError(_INSTALL_HINT) from exc

        try:
            return self._cipher().decrypt(token).decode("utf-8")
        except InvalidToken as exc:
            raise VaultError(
                "Could not decrypt a stored consultation. The key does not "
                "match this database, or the record is corrupt."
            ) from exc

    def exists(self) -> bool:
        """Whether this vault has been initialised on disk."""
        return self.key_path.exists()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _cipher(self):
        if self._fernet is None:
            Fernet = _require_cryptography()
            self._fernet = Fernet(self._load_or_create_key())
        return self._fernet

    def _load_or_create_key(self) -> bytes:
        if self._passphrase:
            return self._derive_from_passphrase()
        return self._load_or_create_key_file()

    def _load_or_create_key_file(self) -> bytes:
        if self.key_path.exists():
            return self.key_path.read_bytes().strip()

        Fernet = _require_cryptography()
        key = Fernet.generate_key()
        _write_private(self.key_path, key)
        return key

    def _derive_from_passphrase(self) -> bytes:
        try:
            from cryptography.hazmat.primitives.kdf.scrypt import (  # noqa: PLC0415
                Scrypt,
            )
        except ImportError as exc:
            raise VaultError(_INSTALL_HINT) from exc

        if self.key_path.exists():
            salt = self.key_path.read_bytes()
        else:
            salt = secrets.token_bytes(SALT_BYTES)
            _write_private(self.key_path, salt)

        kdf = Scrypt(salt=salt, length=32, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P)
        derived = kdf.derive(self._passphrase.encode("utf-8"))
        # Fernet wants a url-safe base64 key, not raw bytes.
        return base64.urlsafe_b64encode(derived)


def _write_private(path: Path, data: bytes) -> None:
    """Write ``data`` with owner-only permissions where the OS supports it."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Create with restrictive permissions rather than relaxing them after the
    # fact — otherwise the key is world-readable for the moment in between.
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    # O_BINARY is mandatory on Windows: without it the handle is in text mode
    # and every 0x0A byte is silently written as 0x0D 0x0A. Key material is
    # random binary, so that corrupts roughly one salt in sixteen.
    flags |= getattr(os, "O_BINARY", 0)
    fd = os.open(path, flags, stat.S_IRUSR | stat.S_IWUSR)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)

    if os.name != "nt":
        os.chmod(path, stat.S_IRUSR | stat.S_IWUSR)
    # On Windows the file inherits the user profile's ACL, which already
    # excludes other non-administrator users. There is no portable chmod
    # equivalent worth faking here.
