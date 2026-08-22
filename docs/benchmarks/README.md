# Benchmark Contracts

TrainLM benchmarks use versioned manifests so performance comparisons cannot
silently change model geometry, numerical policy, batch size, data, or hardware.

## Locked LaughLM reference

- Manifest: [`laughlm_135m_v5e8_v1.json`](../../benchmarks/manifests/laughlm_135m_v5e8_v1.json)
- Lock: [`laughlm_135m_v5e8_v1.lock.json`](../../benchmarks/manifests/laughlm_135m_v5e8_v1.lock.json)
- Decision: [`0001-laughlm-135m-reference.md`](decisions/0001-laughlm-135m-reference.md)

The manifest is the matched workload for TrainLM's 135M TPU parity work. Its
reference metrics are evidence from LaughLM; they are not claims about current
TrainLM performance.

## Change rules

Never edit a locked manifest in place. A changed workload requires:

1. a new versioned manifest filename;
2. a new SHA-256 lock record;
3. a new decision record explaining every changed field;
4. a new known-lock entry in the contract test;
5. preservation of the previous version.

The `.gitattributes` rule forces manifest JSON to LF so the content digest is
stable across Windows and Linux checkouts.
