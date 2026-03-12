from __future__ import annotations

"""Entry-point script for prompt-to-model relay.

What it is:
- Runs `stability_radius.helpers.openapi.togpt.main`.
- Sends a prompt file to OpenAI or Anthropic and saves the generated response.

Aim and scope:
- Keep ad hoc LLM-assisted repository analysis reproducible from one place.
- Default generated outputs to `run_artifacts/openapi_togpt/...` with `debug.log`.

CLI options:
- Positional: `input_file`, optional `output_file`
- Flags: `--model`

Artifacts:
- default `<input_stem>_<model>.md`
- optional explicit output file
- `debug.log`
"""

from _support import run_entrypoint


if __name__ == "__main__":
    raise SystemExit(run_entrypoint("stability_radius.helpers.openapi.togpt", "main"))

