# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Essential Commands

```bash
# Activate venv
source .venv/bin/activate   # or: source venv/bin/activate

# Run tests (ALWAYS use the wrapper — never call pytest directly)
scripts/run_tests.sh                                    # full suite (CI parity)
scripts/run_tests.sh tests/gateway/                     # one directory
scripts/run_tests.sh tests/agent/test_foo.py::test_x    # single test
scripts/run_tests.sh -v --tb=long                       # pass-through pytest flags

# Start the CLI
./hermes                  # interactive CLI (auto-detects venv)
hermes gateway            # start messaging gateway
hermes --tui              # Ink (React) terminal UI

# Start gateway directly (for development)
python -m gateway.run

# TUI development
cd ui-tui && npm install && npm run dev
```

## Architecture

### Entry Points
- **`./hermes`** → `hermes_cli/main.py:main()` — CLI entry point
- **`hermes-agent`** → `run_agent.py:main()` — programmatic agent entry
- **`hermes-acp`** → `acp_adapter/entry:main()` — ACP server for VS Code/Zed/JetBrains

### Core Loop (`run_agent.py` — `AIAgent` class, ~12k LOC)
```
User message → run_conversation()
  ├── Build system prompt (agent/prompt_builder.py)
  ├── Call LLM (OpenAI-compatible API)
  ├── If tool_calls → handle_function_call() → registry dispatch → loop
  ├── If text → persist session (hermes_state.py SQLite) → return
  └── Context compression when near token limit
```

### Tool System
- **`tools/registry.py`** — self-registering pattern: each `tools/*.py` calls `registry.register()` at import time
- **`model_tools.py`** — orchestrates discovery, schema collection, dispatch, and `handle_function_call()`
- **`toolsets.py`** — groups tools into named sets (e.g., `hermes-cli`, `hermes-telegram`); `_HERMES_CORE_TOOLS` is the default for all platforms
- Tools auto-discovered: any `tools/*.py` with a top-level `registry.register()` call is imported automatically
- All tool handlers MUST return a JSON string

### CLI (`cli.py` — `HermesCLI` class, ~11k LOC)
- Rich for display, prompt_toolkit for input with autocomplete
- `process_command()` dispatches slash commands resolved via `resolve_command()` from central `COMMAND_REGISTRY` in `hermes_cli/commands.py`
- Skin engine (`hermes_cli/skin_engine.py`) — data-driven theming

### Gateway (`gateway/run.py` — `GatewayRunner`)
- Manages platform adapters (`gateway/platforms/` — telegram, discord, slack, whatsapp, signal, etc.)
- Session management in `gateway/session.py`
- Per-session agent cache with LRU eviction and idle TTL

### TUI (`ui-tui/` + `tui_gateway/`)
- `hermes --tui` spawns Node (Ink) process that talks JSON-RPC over stdio to Python backend
- TypeScript owns the screen; Python owns sessions, tools, model calls
- Dashboard embeds real TUI via PTY bridge (`hermes_cli/pty_bridge.py`)

### Plugins (`plugins/`)
- General plugins: `PluginManager` discovers from `~/.hermes/plugins/`, `./.hermes/plugins/`, and pip entry points
- Memory providers: `plugins/memory/<name>/` — pluggable backends (honcho, mem0, etc.) implementing `MemoryProvider` ABC
- Rule: plugins MUST NOT modify core files; expand the plugin surface instead

### Skills
- **`skills/`** — built-in, always available, organized by category
- **`optional-skills/`** — shipped but NOT active by default; installed via `hermes skills install official/<category>/<skill>`
- Skill slash commands injected as user messages (not system prompt) to preserve prompt caching

## Critical Rules

### Paths — use `get_hermes_home()`, never hardcode `~/.hermes`
```python
from hermes_constants import get_hermes_home, display_hermes_home
config_path = get_hermes_home() / "config.yaml"      # for code paths
print(f"Config: {display_hermes_home()}/config.yaml")  # for user-facing messages
```
Hardcoding `~/.hermes` breaks profiles (multi-instance support). Each profile has its own `HERMES_HOME`.

### Prompt caching must not break
Do NOT alter past context, change toolsets, or reload memories mid-conversation. The only time we alter context is during compression. Slash commands that mutate system-prompt state must be cache-aware: default to deferred invalidation (takes effect next session), with opt-in `--now` flag.

### Adding a slash command
1. Add `CommandDef` to `COMMAND_REGISTRY` in `hermes_cli/commands.py`
2. Add handler in `HermesCLI.process_command()` in `cli.py`
3. If gateway-available, add handler in `gateway/run.py`

### Adding a tool
1. Create `tools/your_tool.py` with `registry.register()` call
2. Add to `toolsets.py` (`_HERMES_CORE_TOOLS` or a new toolset)
No manual import list needed — auto-discovery picks it up.

### Adding config
- `config.yaml` keys → add to `DEFAULT_CONFIG` in `hermes_cli/config.py`
- `.env` secrets only (API keys, tokens) → add to `OPTIONAL_ENV_VARS` in `hermes_cli/config.py`
- Non-secret settings always go in config.yaml, not .env
- Three config loaders exist: `load_cli_config()` (CLI), `load_config()` (subcommands), direct YAML (gateway)

### Testing rules
- Use `scripts/run_tests.sh`, not raw `pytest` (ensures CI parity: 4 xdist workers, UTC, credentials unset)
- Tests must not write to `~/.hermes/` — `_isolate_hermes_home` autouse fixture redirects to temp dir
- Don't write change-detector tests (snapshots of model catalogs, config versions, enumeration counts)

## Key Files Reference

| File | Purpose |
|------|---------|
| `run_agent.py` | AIAgent class — core conversation loop |
| `model_tools.py` | Tool orchestration, `handle_function_call()` |
| `cli.py` | HermesCLI — interactive terminal UI |
| `hermes_state.py` | SessionDB — SQLite with FTS5 search |
| `hermes_cli/commands.py` | Central slash-command registry (CommandDef) |
| `hermes_cli/config.py` | Config defaults, migration, env var definitions |
| `tools/registry.py` | Self-registering tool registry |
| `toolsets.py` | Toolset definitions and presets |
| `gateway/run.py` | GatewayRunner — messaging platform lifecycle |
| `agent/prompt_builder.py` | System prompt assembly |

For comprehensive details, see `AGENTS.md`.
