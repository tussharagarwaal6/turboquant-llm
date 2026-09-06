#!/usr/bin/env python3
"""Read architecture metadata from GGUF files and check draft/target compatibility.

An MTP / NextN draft head consumes the trunk's final hidden state, so it is only
valid against a trunk with the same embedding length and vocabulary. A mismatch
does not error at load time -- it silently produces near-zero draft acceptance,
which looks like "speculation just doesn't help" rather than a misconfiguration.

Usage:
    gguf_compat.py show TARGET.gguf
    gguf_compat.py check TARGET.gguf DRAFT.gguf

Exits non-zero on mismatch.
"""

from __future__ import annotations

import struct
import sys

# GGUF value type enum -> (struct format, byte size); None means variable-length.
_SCALARS: dict[int, tuple[str, int]] = {
    0: ("<B", 1),   # uint8
    1: ("<b", 1),   # int8
    2: ("<H", 2),   # uint16
    3: ("<h", 2),   # int16
    4: ("<I", 4),   # uint32
    5: ("<i", 4),   # int32
    6: ("<f", 4),   # float32
    7: ("<?", 1),   # bool
    10: ("<Q", 8),  # uint64
    11: ("<q", 8),  # int64
    12: ("<d", 8),  # float64
}
_STRING = 8
_ARRAY = 9


class GGUFError(Exception):
    pass


class _Reader:
    def __init__(self, fh):
        self.fh = fh

    def raw(self, n: int) -> bytes:
        b = self.fh.read(n)
        if len(b) != n:
            raise GGUFError("unexpected end of file")
        return b

    def scalar(self, vtype: int):
        fmt, size = _SCALARS[vtype]
        return struct.unpack(fmt, self.raw(size))[0]

    def string(self) -> str:
        (length,) = struct.unpack("<Q", self.raw(8))
        return self.raw(length).decode("utf-8", errors="replace")

    def value(self, vtype: int, want_array: bool):
        """Read one value. Arrays are skipped unless want_array, since token
        lists are hundreds of thousands of entries long."""
        if vtype == _STRING:
            return self.string()
        if vtype in _SCALARS:
            return self.scalar(vtype)
        if vtype == _ARRAY:
            (elem_type,) = struct.unpack("<I", self.raw(4))
            (count,) = struct.unpack("<Q", self.raw(8))
            if not want_array:
                # Fixed-size elements can be skipped by seeking; strings cannot.
                if elem_type in _SCALARS:
                    self.fh.seek(_SCALARS[elem_type][1] * count, 1)
                elif elem_type == _STRING:
                    for _ in range(count):
                        (length,) = struct.unpack("<Q", self.raw(8))
                        self.fh.seek(length, 1)
                else:
                    raise GGUFError(f"cannot skip array of type {elem_type}")
                return count
            return [self.value(elem_type, False) for _ in range(count)]
        raise GGUFError(f"unknown GGUF value type {vtype}")


def read_metadata(path: str) -> dict:
    """Parse the GGUF KV header. Array values are returned as their length."""
    with open(path, "rb") as fh:
        r = _Reader(fh)
        magic = r.raw(4)
        if magic != b"GGUF":
            raise GGUFError(f"{path}: not a GGUF file (magic {magic!r})")
        version, _n_tensors, n_kv = struct.unpack("<IQQ", r.raw(20))
        if version not in (2, 3):
            raise GGUFError(f"{path}: unsupported GGUF version {version}")

        kv = {}
        for _ in range(n_kv):
            key = r.string()
            (vtype,) = struct.unpack("<I", r.raw(4))
            kv[key] = r.value(vtype, want_array=False)
        return kv


def summarize(path: str) -> dict:
    kv = read_metadata(path)
    arch = kv.get("general.architecture", "unknown")
    return {
        "path": path,
        "arch": arch,
        "name": kv.get("general.name"),
        # Vocab size is authoritative as the token-list length; the explicit
        # vocab_size key is not always written.
        "vocab": kv.get("tokenizer.ggml.tokens") or kv.get(f"{arch}.vocab_size"),
        "hidden": kv.get(f"{arch}.embedding_length"),
        "layers": kv.get(f"{arch}.block_count"),
        "nextn": kv.get(f"{arch}.nextn_predict_layers"),
    }


def _fmt(info: dict) -> str:
    return (
        f"  {info['path']}\n"
        f"    arch={info['arch']} hidden={info['hidden']} "
        f"vocab={info['vocab']} layers={info['layers']}"
    )


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2

    mode = argv[1]

    if mode == "show":
        print(_fmt(summarize(argv[2])))
        return 0

    if mode == "check":
        if len(argv) < 4:
            print("check needs TARGET.gguf DRAFT.gguf", file=sys.stderr)
            return 2
        target = summarize(argv[2])
        draft = summarize(argv[3])

        problems = []
        if target["hidden"] != draft["hidden"]:
            problems.append(
                f"embedding_length differs: target={target['hidden']} draft={draft['hidden']}"
            )
        if target["vocab"] != draft["vocab"]:
            problems.append(
                f"vocab size differs: target={target['vocab']} draft={draft['vocab']}"
            )

        if problems:
            print("Draft head is INCOMPATIBLE with the target model:", file=sys.stderr)
            for p in problems:
                print(f"  - {p}", file=sys.stderr)
            print(_fmt(target), file=sys.stderr)
            print(_fmt(draft), file=sys.stderr)
            print(
                "\nAn MTP head consumes the trunk's hidden state, so both values must\n"
                "match exactly. Mismatched heads draft tokens that are always rejected.",
                file=sys.stderr,
            )
            return 1

        print(
            f"Draft head compatible: hidden={target['hidden']} vocab={target['vocab']} "
            f"(target arch={target['arch']}, draft arch={draft['arch']})"
        )
        if target["arch"] != draft["arch"]:
            print(
                f"  note: architecture strings differ "
                f"({target['arch']} vs {draft['arch']}); acceptance may be reduced."
            )
        return 0

    print(f"unknown mode: {mode}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except GGUFError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
