TrainLM agent handoff

Before changing this repository, read ""docs/IMPLEMENTATION_CONTEXT.md"" (docs/IMPLEMENTATION_CONTEXT.md) and the relevant sections of ""docs/ROADMAP.md"" (docs/ROADMAP.md).

Also inspect the current Git branch, working-tree status, and applicable repository instructions before implementing changes.

Working rules

Product architecture and scope

- TrainLM's product surface is HF-like: users provide model, datasets, tokenizer/processing class, and familiar training arguments. Worker launch, PJRT/rank setup, preflight, manifests, cache, and log parsing stay private.
- Preserve Hugging Face model implementations as the semantic source of truth.
- Add optimization through capability inspection, explicit adapters, and reversible plans; never rely on model-family name guesses.
- Current scope is dense decoder-only autoregressive models. MoE and DLLM are future extensions.
- Preserve backward compatibility unless an approved story explicitly requires a breaking change.
- Avoid unrelated refactoring or dependency upgrades.

Development and testing

- Use the existing repository checkout. Do not create additional Git worktrees unless explicitly requested.
- Preserve existing user changes and untracked files.
- Use the established Python environment and pinned dependencies.
- Run the relevant CPU tests after implementation.
- Do not run TPU training locally. TPU validation belongs to the owner's Kaggle/Cloud TPU environment.
- Do not claim TPU/CUDA certification based on CPU-only tests.
- If tests fail, investigate and report failures rather than hiding or bypassing them.

Git and GitHub operations

Codex Cloud is authorized to perform Git and GitHub operations for this repository, subject to the available credentials, repository permissions, and environment network policies.

Repository: "Dhiraj309/TrainLM"

Allowed operations:

- Inspect Git status, branch information, history, and diffs.
- Fetch the latest remote changes when required.
- Create dedicated feature or story branches.
- Stage and commit the files changed for the approved task.
- Push feature branches to "Dhiraj309/TrainLM" when authenticated write access is available.
- Create and update GitHub pull requests.
- Read GitHub issues, pull requests, and review feedback.
- Incorporate requested changes and push follow-up commits to the same feature branch.

Safety rules:

- Never change the repository's Git remote without explicit authorization.
- Never push directly to "main" or another protected branch.
- Never force-push, rewrite shared history, delete remote branches, or merge pull requests without explicit authorization.
- Never discard existing changes using destructive reset, checkout, clean, or rebase commands.
- Never overwrite unrelated files or commit unrelated modifications.
- Never embed GitHub tokens, passwords, or secrets in source code, scripts, logs, or commits.
- Do not modify managed HTTP/HTTPS proxy settings or bypass sandbox network restrictions.
- If GitHub authentication, network connectivity, or permission checks fail, report the exact sanitized error and stop the affected remote operation.
- Prefer Codex's supported GitHub branch publishing and pull-request workflow when available.
- Do not claim a successful push until the remote branch has been verified.

Branches and commits

- Use one cohesive outcome per PR in the sequential multi-PR roadmap; make multiple small, logical commits for implementation, tests, fixes, review changes and docs. Preserve traceable commits; do not squash by default.
- Create the approved roadmap PR branch only after the preceding PR merges to `main`, following the descriptive convention (e.g. `perf/03-chunked-loss-no-full-logits`).
- Use concise, descriptive commit messages following the repository's existing conventions.
- Run relevant tests before committing.
- Review the staged diff and commit only files associated with the task.
- Push the feature branch only when network access and authentication are working.
- Create a pull request against the appropriate base branch.
- Do not merge the pull request automatically.
- If committing or pushing is blocked, preserve the local implementation and report the recommended commit message and next action.

Documentation and reporting

In every implementation turn:

- Update both "docs/ROADMAP.md" and "docs/IMPLEMENTATION_CONTEXT.md" to reflect actual progress.
- Keep milestone statuses accurate; never mark unfinished work complete.
- Document meaningful architectural decisions, limitations, and validation results.
- Report the files changed, tests performed, and test outcomes.
- Report the branch name, commit SHA, and pull-request link when available.
- Provide a suggested commit message if a commit could not be created.

Do not invent test results, accelerator validation results, commit SHAs, or successful GitHub operations.

## Sequential roadmap PR and commit policy

The normative sequence and specific PR/commit plan are
`docs/ROADMAP.md#branch-and-pr-policy` and
`docs/PR_DELIVERY_PLAN.md`. **The roadmap contains multiple separate,
focused PRs, each with multiple logical commits, not a single PR for the
whole roadmap and not one commit per PR.**

- **Completed:** PR #18, including PR #16, merged into `main` on 2026-10-28.
  PR #19 established the T0.1 inventory; PR #20 merged the T0.2 mapping.
- **Current:** T1 environment feasibility. Start with single-device evidence
  on v5e-1; use v5e-8 for collectives, DP8, sharding, distributed checkpoints,
  and scaling. The 18-card XLA queue is historical, not an active queue.
- **Next:** Select the first implementation scope from verified T1 evidence.
  Current facts and labels are indexed in
  `docs/README.md`; the M0–M17 roadmap and implementation log are historical
  XLA records unless a page explicitly says otherwise.
- **Per PR:** Create `<type>/<NN>-<deliverable>` from the new, fetched
  `origin/main` only after the previous PR was merged and its remote
  `main` SHA verified. New planning IDs start at 01 only after T0 review and approval.
  A PR's branch name, PR title and body must match the full scope of
  its proposed commits, tests, fixes and docs.
- **During development:** Use small, coherent implementation/test/fix/
  regression/docs commits. Do not squash meaningful change history by
  default, invent a minimum commit count, or rewrite a reviewed branch.
  Commit in response to new evidence and follow-up reviews on the same PR.
- **Before merge:** Verify tests and docs, source/head SHAs, review/CI,
  HF semantic safety, security, user UX and evidence required by the
  PR's particular acceptance contract. TPU hardware verification is
  owner-run; unmeasured optimization paths stay guarded/off by default.
- **After merge:** Only an owner approves merge; Codex must not
  auto-merge. Verify actual merged `main` and unlock the next
  separately numbered PR. No speculative or stacked next branches.
- Request approved per-command network permission as needed for remote
  GitHub operations. Do not change managed proxy variables or tokens.

## Native TorchTPU architecture and approval policy

Read `docs/TORCHTPU_REBASELINE.md`. Native TorchTPU is the target backend;
existing PyTorch/XLA implementation and M0–M14 evidence are historical
salvage, not native certification. T0 inventories reusable, XLA-specific
and uncertain code before remapping the XLA-oriented 18-PR candidate queue.
T0.1 was read-only discovery; T0.2 is the owner-approved mapping merged by
PR #20. T1 feasibility is authorized on v5e-1 for single-device checks, with
v5e-8 for multi-device checks. No dependency pins or support claims change
until environment evidence is reviewed. For future implementation, propose
logical commit messages and obtain explicit owner approval before commit,
push, new PR or merge.
Maintain multiple small commits per approved PR, with branch/title/body
matching scope, and the next branch only after verified merge to `main`.
Use the requested trailer where appropriate:
`Co-authored-by: Dhiraj309 <patildhiraj309@outlook.com>`.
Do not add Codex/OpenAI/bot co-author trailers; preserve truthful Git
attribution. Never push to main, force-push, change managed proxy or
credentials, or auto-merge.

Current continuation point

PR #18 is merged to `main`; PR #19 completed T0.1 and PR #20 merged the T0.2
mapping. Native TorchTPU remains unverified. T1's v5e-1 probe is documented
in `docs/T1_V5E1_PROBE.md`; use v5e-8 for multi-device evidence. The local
checkout may be on a docs branch; before implementation, fetch and verify the
latest merged `main`. Do not delete legacy branches until T0 salvage review
determines whether they preserve needed source or evidence.

Start T1 on the approved Kaggle v5e-1 image and record install provenance,
device/autograd/optimizer/compile evidence. Use v5e-8 for collectives, DP8,
sharding, distributed checkpoints, and scale. Create an implementation PR only
after the T1 evidence identifies a concrete scope and its branch can start
from verified merged `main`.

Codex may read, test, commit and push to an approved PR branch, but must not
push to `main`, force-push, or merge without explicit owner authorization.
Retain the HF-like UX, reversible capability planning, CPU test gates, and
honest separation between historical XLA evidence and native TorchTPU claims.

Historical M8-F0 definition of done

The next M8-F0 story is ready for review when:

- The public training API remains concise and Hugging Face-like.
- TPU execution routes through the internal coordinator.
- Users do not need to construct subprocess commands, manually configure workers, or parse stage logs.
- Existing CPU/CUDA behavior remains compatible.
- Relevant tests pass.
- Documentation reflects the implementation and outstanding TPU validation.
- The implementation is available on a reviewable feature branch or pull request, if publishing is available.

Full M8-F0 completion additionally requires the TPU public path to be validated as specified in the roadmap. CPU-only tests must not be represented as TPU hardware validation.
