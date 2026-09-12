# compactionrl-agent-rl

Source snapshots used to investigate and fix the CompactionRL agentic RL trajectory issue.

The repository contains two independently runnable AvaTrain trees:

- `compactionrl/AvaTrain`: the CompactionRL worktree.
- `swe-rl/AvaTrain`: the SWE-RL comparison worktree.

The original repositories used Git submodules. They are flattened here so the current
local Miles, SGLang, and Megatron-LM source—including uncommitted fixes—is versioned in
one repository. See `SOURCE_SNAPSHOT.md` for base commits and the exact local changes
captured from each source tree.

Large runtime files and private local configuration are intentionally excluded.
