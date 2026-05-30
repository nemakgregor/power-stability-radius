# Codex Project Memory

This directory is a compact working memory for Codex sessions. It is not a
replacement for the repository documentation in `docs/` or the contract in
`UNITS_CONTRACT.md`.

Read order at the start of a session:

1. `AGENTS.md`
2. `.codex/docs/project-memory.md`
3. `.codex/docs/session-log.md`
4. Only then open the specific source files or `docs/*.md` needed for the task.

Update rule after each meaningful conversation:

- Update `session-log.md` with a dated, short entry.
- Update `project-memory.md` only for durable facts, rules, decisions, known
  pitfalls, or result summaries that should survive future context compaction.
- Keep entries concise. Prefer replacing stale notes with a sharper summary over
  appending indefinitely.
- Do not paste terminal logs, large tables, generated CSV content, or full
  experiment outputs here. Store pointers and conclusions.

Size budget:

- Keep this directory under roughly 800 lines total unless the user explicitly
  asks for more.
- If it grows, consolidate old session-log entries into a short historical
  summary.
