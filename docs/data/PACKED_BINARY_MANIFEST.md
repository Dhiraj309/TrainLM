# Packed binary shard manifest

TrainLM treats packed `.bin` files as untrusted byte streams until a versioned
manifest has validated their complete interpretation and content. A reader must
not infer dtype, byte order, header size, token count, or vocabulary bounds from
a filename or host platform.

## Manifest contract

`PackedBinaryShardManifest` records:

- a safe relative POSIX data path and stable shard ID;
- compatibility profile and header size;
- explicit token dtype and byte order;
- exact token count, observed minimum/maximum token IDs, and vocabulary size;
- exact file size and whole-file SHA-256; and
- document-boundary availability or a content-addressed `uint64` offset index.

The manifest is immutable, round-trips through deterministic JSON, and has a
language-neutral schema at
`schemas/data/packed_binary_shard_v1.schema.json`.

## Legacy TrainLM notebook compatibility

The `legacy_1024_uint16` profile describes the existing training shards:

```text
byte 0                                      byte 1024
+-------------------------------------------+-------------------------+
| opaque/reserved legacy header (1024 B)    | little-endian uint16... |
+-------------------------------------------+-------------------------+
```

The header remains opaque because the notebook never interpreted it. The
profile fixes the payload to little-endian `uint16`; relying on NumPy's native
endianness would make the same file platform-dependent. Legacy shards may set
document metadata to `unavailable` when boundaries were not retained.

`explicit_v1` supports zero or larger opaque prefixes and the declared
`uint16`, `uint32`, `int32`, or `int64` payload types. Signed payloads remain
subject to the non-negative token-ID check.

## Validation order

Call `validate_packed_binary_shard` before constructing a memmap or iterator.
It fails in this order:

1. manifest structure and cross-field invariants;
2. file existence and exact byte geometry;
3. whole-file SHA-256;
4. complete payload decoding and vocabulary-range validation;
5. exact observed token count and min/max agreement; and
6. optional document-index size, checksum, span, and monotonicity.

The validator scans every token. Sampling only an initial region is not an
integrity guarantee because corrupt or out-of-vocabulary values may occur later
in a shard. Shard hashing and token inspection share one bounded-memory
streaming pass; document indexes are also inspected in bounded chunks.

```python
from trainlm.data import (
    PackedBinaryShardManifest,
    validate_packed_binary_shard,
)

manifest = PackedBinaryShardManifest.from_json(manifest_text)
validated = validate_packed_binary_shard(
    manifest,
    "/local/cache/fineweb-edu/train-00000.bin",
)
```

The validator intentionally does not create tensors, memmaps, batches, or
workers. M3-F2 resolves remote shards; M3-F3 owns the high-throughput reader.
