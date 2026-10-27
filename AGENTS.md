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

- Use one feature/story per branch and PR, with multiple small, logically coherent commits for implementation, tests, fixes, review changes, and documentation as appropriate. Do not squash by default; preserve traceable commits.
- Create the approved story branch only after its predecessor merges, using the new roadmap naming convention (e.g. `validation/001-public-tpu-lifecycle`).
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

## Sequential story and pull-request protocol

Read `docs/ROADMAP.md#branch-and-pr-policy`. The delivery contract is:
**one story → one branch → multiple small, reviewable commits → one PR →
tests and owner review → merge to `main` → verify `main` → next story**.

- The active legacy PR #18 (`milestone/m8-m9-optimization-core`)
  must be reviewed, validated, conflict-resolved and merged before starting
  the new delivery convention. Never start another story branch while
  #18 is unmerged.
- After that transition, start new branch names at `001`, using
  `<type>/<NNN>-<kebab-outcome>` (e.g.
  `validation/001-public-tpu-lifecycle`). Do not reuse
  obsolete `milestone/*` branches or carry M8-F0 names into new branch
  IDs; preserve old feature numbers only as technical lineage in the roadmap.
- Keep small, purposeful implementation, tests, fix, and docs commits
  within the **same** story PR. Do not produce one giant commit or
  squash meaningful audit history by default. Do not rewrite a branch
  under active review without owner approval.
- Every PR documents user-visible goal, scope, non-goals, tests,
  known failures, hardware evidence/limitations, rollback and
  documentation updates. CPU passes do not certify TPU kernels,
  HLO/HBM, throughput, or distributed resume.
- No automatic merging, protected-`main` pushes, force-pushes,
  parallel story PRs or speculative future branch creation.
  Only activate the next story once the previous PR is reported
  **merged** and its changes are verified on current `main`.
- For GitHub commands use the approved per-command network permission
  mechanism when needed; do not change managed proxy variables,
  credentials or sandbox policy.

Current continuation point

PR4 ("milestone/m8-m9-optimization-core") is active.

M8-F0's first public facade slice exists in "src/trainlm/api.py" and delegates to the existing backend-neutral engine on CPU/CUDA.

The next story is to hide TPU coordinator and worker orchestration behind "TrainLMTrainer.train()" while keeping the same concise user code.

Do not mark M8-F0 complete until the public training path reaches the TPU coordinator without requiring subprocess code or stage-log parsing in the user's workflow.

Next implementation priorities

1. Review the existing "TrainLMTrainer" API, TPU coordinator, and worker-launch architecture.
2. Design the smallest compatible integration that routes TPU execution through "TrainLMTrainer.train()".
3. Keep worker orchestration, distributed setup, manifests, caches, and log parsing internal.
4. Preserve the existing CPU/CUDA behavior.
5. Add focused CPU-side unit tests using mocks or controlled fixtures for the TPU orchestration boundary.
6. Run applicable CPU regression tests.
7. Update both required documentation files.
8. Commit and publish the implementation on an isolated feature branch when permitted.
9. Create a pull request for review without automatically merging it.

Definition of done

The next M8-F0 story is ready for review when:

- The public training API remains concise and Hugging Face-like.
- TPU execution routes through the internal coordinator.
- Users do not need to construct subprocess commands, manually configure workers, or parse stage logs.
- Existing CPU/CUDA behavior remains compatible.
- Relevant tests pass.
- Documentation reflects the implementation and outstanding TPU validation.
- The implementation is available on a reviewable feature branch or pull request, if publishing is available.

Full M8-F0 completion additionally requires the TPU public path to be validated as specified in the roadmap. CPU-only tests must not be represented as TPU hardware validation.
