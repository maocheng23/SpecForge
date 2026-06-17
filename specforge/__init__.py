"""Top-level SpecForge exports.

Keep this module light. Runtime-only imports such as
``specforge.runtime.contracts`` should not import model backends, TensorFlow,
SGLang, or other optional/heavy dependencies just because Python initializes the
``specforge`` package. Public model/core symbols are loaded lazily to preserve
the existing ``from specforge import OnlineEagle3Model`` style.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_LAZY_EXPORTS = {
    "AutoDraftModelConfig": "specforge.modeling",
    "AutoEagle3DraftModel": "specforge.modeling",
    "CustomEagle3TargetModel": "specforge.modeling",
    "HFEagle3TargetModel": "specforge.modeling",
    "LlamaForCausalLMEagle3": "specforge.modeling",
    "OnlineDFlashModel": "specforge.core",
    "OnlineDominoModel": "specforge.core",
    "OnlineEagle3Model": "specforge.core",
    "QwenVLOnlineEagle3Model": "specforge.core",
    "SGLangEagle3TargetModel": "specforge.modeling",
    "get_eagle3_target_model": "specforge.modeling",
}

__all__ = sorted([*_LAZY_EXPORTS, "core", "modeling"])


def __getattr__(name: str) -> Any:
    if name in ("core", "modeling"):
        module = import_module(f"specforge.{name}")
        globals()[name] = module
        return module
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'specforge' has no attribute {name!r}")
    module = import_module(module_name)
    value = getattr(module, name)
    globals()[name] = value
    return value
