"""Compatibility shim for CrewAI >= 1.x + LiteLLM + Groq.

CrewAI tags its system/user messages with a non-standard ``cache_breakpoint`` key (used for
Anthropic prompt caching). Its native providers strip it, but the LiteLLM path used for Groq does
not, so Groq answers with:
    'messages.0': property 'cache_breakpoint' is unsupported
We wrap litellm.completion / acompletion so that key is removed from every message before the
request leaves the process.
"""
from __future__ import annotations

import functools

_STRIP_KEYS = ("cache_breakpoint",)
_PATCHED_FLAG = "_researchcrew_patched"


def _clean(messages):
    if not isinstance(messages, list):
        return messages
    out = []
    for m in messages:
        if isinstance(m, dict) and any(k in m for k in _STRIP_KEYS):
            m = {k: v for k, v in m.items() if k not in _STRIP_KEYS}
        out.append(m)
    return out


def _wrap(fn):
    if getattr(fn, _PATCHED_FLAG, False):
        return fn

    @functools.wraps(fn)
    def sync(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _clean(kwargs["messages"])
        return fn(*args, **kwargs)

    @functools.wraps(fn)
    async def asyn(*args, **kwargs):
        if "messages" in kwargs:
            kwargs["messages"] = _clean(kwargs["messages"])
        return await fn(*args, **kwargs)

    import asyncio
    wrapper = asyn if asyncio.iscoroutinefunction(fn) else sync
    setattr(wrapper, _PATCHED_FLAG, True)
    return wrapper


def patch_litellm() -> None:
    """Idempotent. Safe to call many times."""
    import litellm

    litellm.completion = _wrap(litellm.completion)
    litellm.acompletion = _wrap(litellm.acompletion)