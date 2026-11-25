"""JSON state files: atomic writes (temp file, fsync, rename), files 0600, directories 0700.

The helpers are synchronous. State files are small, so modules call them directly from async
code (one call never interleaves with another coroutine, which also keeps read-modify-write
sequences simple). Heavy scans, such as reading many transcripts, belong in `asyncio.to_thread`.

`StateModel` is the base for records stored in state files: Python attributes are snake_case, the
JSON keys are camelCase, and unknown keys survive a read-modify-write round trip.
"""

from __future__ import annotations

import fcntl
import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from typing import Any, NamedTuple

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


def read_json[T](path: str, fallback: T) -> Any | T:
    """Reads and parses a JSON file. Returns `fallback` when the file does not exist.

    Raises ValueError naming the file for invalid JSON; other OS errors propagate.
    """
    try:
        with open(path, encoding="utf-8") as handle:
            text = handle.read()
    except FileNotFoundError:
        return fallback
    try:
        return loads_json(text)
    except ValueError as err:
        raise ValueError(f"Invalid JSON in {path}: {err}") from err


def read_json_object(path: str, max_bytes: int) -> dict[str, Any] | None:
    """A JSON object file read defensively: None when it is missing, unreadable, larger than
    `max_bytes`, not UTF-8, invalid JSON (`loads_json`) or not an object. Never raises for these."""
    try:
        with open(path, "rb") as handle:
            raw = handle.read(max_bytes + 1)
        if len(raw) > max_bytes:
            return None
        data = loads_json(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, ValueError, RecursionError):
        return None
    return data if isinstance(data, dict) else None


def _reject_constant(name: str) -> Any:
    raise ValueError(f"invalid JSON constant {name}")


def loads_json(text: str, *, containers_only: bool = False) -> Any:
    """`json.loads`, but NaN, Infinity and -Infinity are errors (ValueError). With `containers_only`,
    the value must be an object or array, other top-level values are a ValueError too."""
    value = json.loads(text, parse_constant=_reject_constant)
    if containers_only and not isinstance(value, dict | list):
        raise ValueError("JSON text is not an object or array")
    return value


def dump_json(data: Any) -> str:
    """The exact text `write_json_atomic` writes: 2-space indented JSON, non-ASCII kept, plus a newline."""
    return json.dumps(data, indent=2, ensure_ascii=False) + "\n"


def write_json_atomic(path: str, data: Any) -> None:
    """Writes JSON atomically: a temp file (mode 0600) in the same directory, flushed to disk,
    then renamed over the target. Fails when the directory does not exist."""
    write_json_atomic_identity(path, data)


class FileIdentity(NamedTuple):
    """What identifies one version of a file: inode, mtime (ns) and size. Every atomic write creates a
    new inode, so a write by another process is detected even when it lands in the same timestamp
    tick (coarse mtimes, e.g. on some Linux file systems)."""

    ino: int
    mtime_ns: int
    size: int


def _identity(info: os.stat_result) -> FileIdentity:
    return FileIdentity(ino=info.st_ino, mtime_ns=info.st_mtime_ns, size=info.st_size)


def write_json_atomic_identity(path: str, data: Any) -> FileIdentity:
    """`write_json_atomic`, returning the `FileIdentity` of the file it wrote, taken from the written
    file's own descriptor (a later stat of `path` could already see another process's file)."""
    text = dump_json(data)
    directory = os.path.dirname(path) or "."
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{os.getpid()}.{secrets.token_hex(6)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        try:
            data_bytes = memoryview(text.encode("utf-8"))
            while data_bytes:
                data_bytes = data_bytes[os.write(fd, data_bytes) :]
            os.fsync(fd)
            written = _identity(os.fstat(fd))
        finally:
            os.close(fd)
        os.replace(tmp, path)
    except BaseException:
        with suppress(OSError):
            os.unlink(tmp)
        raise
    return written


def file_identity(path: str) -> FileIdentity | None:
    """The `FileIdentity` of `path`, or None when the file does not exist."""
    try:
        return _identity(os.stat(path))
    except FileNotFoundError:
        return None


def ensure_dir(path: str) -> None:
    """Creates a directory (and parents) with mode 0700 and tightens an existing one to 0700."""
    os.makedirs(path, mode=0o700, exist_ok=True)
    os.chmod(path, 0o700)


@contextmanager
def file_lock(path: str, *, ensure_parent: bool = True) -> Iterator[None]:
    """Exclusive `flock` on the lock file `path` (created with mode 0600), shared with every other
    process locking the same file. Blocks until the other holder is done. `ensure_parent` first
    creates (or tightens) the parent directory with `ensure_dir`."""
    if ensure_parent:
        ensure_dir(os.path.dirname(path) or ".")
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


class StateModel(BaseModel):
    """Base for JSON state records (task files, oauth.json entries, pairing.json, projects.json).

    - Attributes are snake_case; JSON keys are camelCase (`to_camel`).
    - `extra="allow"`: keys this version does not know are kept and written back.
    - Read with `Model.model_validate(data)`, write with `model.to_json()`.
    """

    model_config = ConfigDict(
        alias_generator=to_camel,
        validate_by_name=True,
        validate_by_alias=True,
        extra="allow",
    )

    def to_json(self, *, exclude_none_fields: frozenset[str] = frozenset()) -> dict[str, Any]:
        """JSON-ready dict with camelCase keys in field order. Fields named in
        `exclude_none_fields` are left out while None (for keys that are written only when set)."""
        data = self.model_dump(by_alias=True, mode="json")
        for name in exclude_none_fields:
            field = type(self).model_fields[name]
            key = field.alias or name
            if data.get(key) is None:
                data.pop(key, None)
        return data
