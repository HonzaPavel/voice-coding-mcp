"""Fixtures for the auth tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import pytest

from tests.auth.helpers import Ctx, setup

MakeCtx = Callable[..., Ctx]


@pytest.fixture
async def make_ctx(real_tmp: Path) -> AsyncIterator[MakeCtx]:
    """`make_ctx(oauth=..., http_client=..., rate_limit=...)` builds a Ctx; clients are closed afterwards."""
    made: list[Ctx] = []

    def make(**kwargs: Any) -> Ctx:
        ctx = setup(real_tmp, **kwargs)
        made.append(ctx)
        return ctx

    yield make
    for ctx in made:
        await ctx.http.aclose()
