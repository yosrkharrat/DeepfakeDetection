"""Wrapper around the multi-agent research system for claim verification."""

from __future__ import annotations

import importlib.util
import re
import sys
import time
import types
from pathlib import Path
from urllib.parse import urlparse

_AGENT_ROOT = Path(__file__).resolve().parent.parent.parent / "external" / "multi-agent-system"

# ---------------------------------------------------------------------------
# Bootstrap: load the multi-agent system without circular imports.
#
# Root cause: agents/__init__.py does `from src.agents.graph import build_graph`
# at import time, and src/agents/graph.py does `from agents.state import ...`
# at import time — circular.
#
# Strategy:
#  1. Back up + clear our project's src.* from sys.modules.
#  2. Pre-register stub packages for 'src', 'src.agents', and 'agents' so
#     neither __init__.py ever executes automatically.
#  3. Load each agents/*.py compatibility shim directly by file path —
#     their simple `from src.agents.X import Y` calls will find the agent's
#     src/ via the stub __path__, with no __init__ involvement.
#  4. Load src/agents/graph.py directly — by now all `agents.*` it needs
#     are already in sys.modules, so no circular import occurs.
#  5. Restore our project's src.* modules.
# ---------------------------------------------------------------------------

def _load_file(module_name: str, file_path: Path) -> types.ModuleType:
    """Load a module from an absolute path and register it in sys.modules."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)   # type: ignore[arg-type]
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)                   # type: ignore[union-attr]
    return mod


def _make_stub(name: str, path: Path) -> types.ModuleType:
    stub = types.ModuleType(name)
    stub.__path__ = [str(path)]      # type: ignore[attr-defined]
    stub.__package__ = name
    sys.modules[name] = stub
    return stub


# 1 — Backup & clear our src modules
_our_src_backup: dict = {k: v for k, v in sys.modules.items()
                         if k == "src" or k.startswith("src.")}
_build_graph = None
_AgentConfig = None
_agent_load_error: str = ""

try:
    for _k in list(_our_src_backup):
        del sys.modules[_k]

    if not _AGENT_ROOT.exists():
        raise FileNotFoundError(
            f"Multi-agent system not found at {_AGENT_ROOT}. "
            "Run: git submodule update --init --recursive"
        )

    # 2 — Add agent root to sys.path (needed for config, langgraph, etc.)
    if str(_AGENT_ROOT) not in sys.path:
        sys.path.insert(0, str(_AGENT_ROOT))

    # 3 — Register stub packages (no __init__.py will run)
    _make_stub("src",          _AGENT_ROOT / "src")
    _make_stub("src.agents",   _AGENT_ROOT / "src" / "agents")
    _make_stub("agents",       _AGENT_ROOT / "agents")

    # 4 — Load each compatibility shim directly (simple re-export files,
    #     each does `from src.agents.X import Y` — no circular deps)
    for _name in ("state", "planner", "researcher", "critic", "writer", "eval"):
        _shim_path = _AGENT_ROOT / "agents" / f"{_name}.py"
        if _shim_path.exists():
            _load_file(f"agents.{_name}", _shim_path)

    # 5 — Load the actual graph module by file path
    #     All `from agents.X import Y` inside it now resolve from step 4
    _graph_mod = _load_file("src.agents.graph", _AGENT_ROOT / "src" / "agents" / "graph.py")
    _build_graph = _graph_mod.build_graph  # type: ignore[attr-defined]

    # 6 — Load config (no conflicts)
    from config import PipelineConfig as _AgentConfig  # noqa: E402

except Exception as _agent_exc:
    _agent_load_error = str(_agent_exc)
    _build_graph = None
    _AgentConfig = None

finally:
    # 7 — Always restore our project's src.* modules (even on failure)
    sys.modules.update(_our_src_backup)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_ollama(base_url: str) -> bool:
    try:
        import httpx
        r = httpx.get(base_url, timeout=3)
        return r.status_code < 500
    except Exception:
        return False


def _extract_section(markdown: str, heading: str) -> str:
    pattern = rf"##\s+{re.escape(heading)}\s*\n(.*?)(?=\n##|\Z)"
    match = re.search(pattern, markdown, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.replace("www.", "")
    except Exception:
        return url


def _dedupe_sources(findings: list) -> list:
    seen: set = set()
    sources = []
    for finding in findings:
        conf = int(finding.get("confidence", 0))
        for src in finding.get("sources", []):
            if not isinstance(src, dict):
                continue
            url = str(src.get("url", "")).strip()
            if not url or url in seen:
                continue
            seen.add(url)
            title = str(src.get("title", "")).strip() or _domain(url)
            sources.append({
                "url": url,
                "title": title,
                "publisher": _domain(url),
                "confidence": conf,
            })
    return sources


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

class ClaimVerifier:
    """
    Wraps the LangGraph multi-agent pipeline as a simple .verify(claim) call.
    Requires Ollama running with mistral pulled.
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        max_iterations: int = 1,
    ) -> None:
        if _build_graph is None or _AgentConfig is None:
            raise RuntimeError(
                f"Multi-agent system unavailable: {_agent_load_error}. "
                "Clone the submodule with: git submodule update --init --recursive"
            )
        config = _AgentConfig(
            planner_model="mistral",
            researcher_model="mistral",
            critic_model="mistral",
            writer_model="llama3",
            max_iterations=max_iterations,
            ollama_base_url=ollama_base_url,
            planner_max_questions=3,
            researcher_max_questions=3,
            researcher_max_reasoning_steps=6,
            planner_num_predict=220,
            researcher_num_predict=260,
            critic_num_predict=220,
            writer_num_predict=900,
        )
        self.graph = _build_graph(config, enable_checkpointing=False)

    def verify(self, claim: str) -> dict:
        """Run the full multi-agent pipeline on a claim. Takes 1–3 minutes."""
        start = time.time()
        result = self.graph.invoke({
            "messages": [],
            "topic": claim,
            "plan": [],
            "findings": [],
            "critique": "",
            "report": "",
            "next_agent": "",
            "iteration": 0,
        })
        return self._normalize(claim, result, round(time.time() - start, 1))

    def _normalize(self, claim: str, state: dict, elapsed: float) -> dict:
        report: str = state.get("report", "")
        findings: list = state.get("findings", [])

        summary = _extract_section(report, "Executive Summary")
        if not summary:
            paragraphs = [p.strip() for p in report.split("\n\n") if p.strip()]
            summary = paragraphs[0] if paragraphs else report[:400]

        return {
            "claim": claim,
            "summary": summary,
            "report": report,
            "sources": _dedupe_sources(findings),
            "elapsed_seconds": elapsed,
        }


def is_ollama_available(base_url: str = "http://localhost:11434") -> bool:
    return _check_ollama(base_url)
