# Project Skills

The backend loads Markdown skills from this folder and appends them to the GapGPT system prompt after `prompt.md`.

Imported UX/UI skills:
- `uxui-designer`
- `uxui-audit`
- `uxui-imagify`

Source: https://github.com/GlamgarOnDiscord/uxui-AI-Prompt/tree/main

Loading behavior:
- `SKILL.md` files are prioritized.
- Reference Markdown files are loaded after core skill files.
- `MAX_SKILL_CONTEXT_CHARS` in `.env` can limit the total skill context size. Default: `60000`.
