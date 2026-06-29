"""tqdm helpers — terminal std.tqdm only (no ipywidgets / notebook widgets)."""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterable, Iterator, Sequence
from typing import Any, TypeVar

T = TypeVar("T")

_SAFE_BAR_FORMAT = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]"


def force_std_tqdm() -> None:
    """
    Prevent tqdm.autonotebook / sentence-transformers from using ipywidgets bars
    (breaks in some Jupyter builds with 'jupyter-ipywidget-renderer' errors).
    """
    try:
        from tqdm import tqdm as std_tqdm

        for mod_name in ("tqdm.autonotebook", "tqdm.auto"):
            try:
                import importlib

                mod = importlib.import_module(mod_name)
                mod.tqdm = std_tqdm
            except ImportError:
                pass
    except ImportError:
        pass


# Run once on import so ST encode and other libs pick up std tqdm
force_std_tqdm()


def _progress_disable() -> bool:
    """Only disable when stderr is not a TTY and we're not in IPython."""
    try:
        from IPython import get_ipython

        if get_ipython() is not None:
            return False
    except (ImportError, NameError, AttributeError):
        pass
    return not sys.stderr.isatty()


def _tqdm_cls():
    try:
        from tqdm import tqdm

        return tqdm
    except ImportError:
        return None


def _tqdm_kwargs(
    *,
    total: int | None,
    desc: str,
    leave: bool,
    position: int | None,
    unit: str | None,
    tqdm_cls: Any | None = None,  # ignored; kept for callers
) -> dict[str, Any]:
    kw: dict[str, Any] = {
        "total": total,
        "desc": desc,
        "leave": leave,
        "disable": _progress_disable(),
        "mininterval": 0.5,
        "file": sys.stderr,
        "bar_format": _SAFE_BAR_FORMAT,
        "dynamic_ncols": True,
    }
    if position is not None:
        kw["position"] = position
    if unit is not None:
        kw["unit"] = unit
    return kw


def iter_progress(
    iterable: Iterable[T],
    *,
    total: int | None = None,
    desc: str = "",
    enabled: bool = True,
    leave: bool = True,
    position: int | None = None,
    unit: str | None = None,
    force: bool = False,
) -> Iterator[T]:
    if not enabled:
        yield from iterable
        return
    tqdm = _tqdm_cls()
    if tqdm is None:
        yield from iterable
        return
    try:
        kw = _tqdm_kwargs(
            total=total,
            desc=desc,
            leave=leave,
            position=position,
            unit=unit,
        )
        if force:
            kw["disable"] = False
        yield from tqdm(iterable, **kw)
    except Exception:
        yield from iterable


def map_progress(
    func: Callable[[T], object],
    iterable: Sequence[T] | Iterable[T],
    *,
    desc: str = "",
    enabled: bool = True,
    total: int | None = None,
) -> list[object]:
    if not enabled:
        return [func(x) for x in iterable]
    if total is None and hasattr(iterable, "__len__"):
        total = len(iterable)  # type: ignore[arg-type]
    tqdm = _tqdm_cls()
    if tqdm is None:
        return [func(x) for x in iterable]
    try:
        out: list[object] = []
        with tqdm(
            iterable,
            **_tqdm_kwargs(total=total, desc=desc, leave=True, position=None, unit=None),
        ) as pbar:
            for x in pbar:
                out.append(func(x))
        return out
    except Exception:
        return [func(x) for x in iterable]


def progress_enabled_for_count(n: int, *, threshold: int = 500) -> bool:
    return n >= threshold


def _tqdm_factory():
    return _tqdm_cls()
