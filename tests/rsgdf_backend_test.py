"""Backend selection must remain explicit and preserve existing input scripts."""
import argparse
from types import SimpleNamespace

import pytest

from green_mbtools.mint import common_utils as comm


@pytest.mark.parametrize("flags, expected", [
    ([], "ccgdf"), (["--df_backend", "ccgdf"], "ccgdf"),
    (["--df_backend", "rsgdf"], "rsgdf"),
])
def test_backend_cli(flags, expected):
    parser = argparse.ArgumentParser()
    comm.add_pbc_params(parser)
    args = parser.parse_args(["--a", "1 0 0\n0 1 0\n0 0 1", "--nk", "1"] + flags)
    assert args.df_backend == expected


def test_no_auto_backend():
    parser = argparse.ArgumentParser()
    comm.add_pbc_params(parser)
    with pytest.raises(SystemExit):
        parser.parse_args(["--a", "1 0 0\n0 1 0\n0 0 1", "--nk", "1",
                           "--df_backend", "auto"])


def arguments(**kwargs):
    return SimpleNamespace(auxbasis=None, beta=None, Nk=0, **kwargs)


@pytest.mark.parametrize("backend, prefer_ccdf", [(None, True), ("ccgdf", True), ("rsgdf", False)])
def test_backend_selector(monkeypatch, backend, prefer_ccdf):
    monkeypatch.setattr(comm.int_utils, "GreenGDF", lambda cell: SimpleNamespace(_prefer_ccdf=None))
    args = arguments(**({} if backend is None else {"df_backend": backend}))
    result = comm.construct_gdf(args, SimpleNamespace(omega=0))
    assert result._prefer_ccdf is prefer_ccdf


def test_invalid_programmatic_backend(monkeypatch):
    monkeypatch.setattr(comm.int_utils, "GreenGDF", lambda cell: SimpleNamespace(_prefer_ccdf=None))
    with pytest.raises(ValueError, match="Unsupported DF backend"):
        comm.construct_gdf(arguments(df_backend="invalid"), SimpleNamespace(omega=0))


def test_unsupported_adapter(monkeypatch):
    monkeypatch.setattr(comm.int_utils, "GreenGDF", lambda cell: SimpleNamespace())
    with pytest.raises(RuntimeError, match="needs a backend adapter"):
        comm.construct_gdf(arguments(), SimpleNamespace(omega=0))


def test_long_range_only_operator(monkeypatch):
    monkeypatch.setattr(comm.int_utils, "GreenGDF", lambda cell: SimpleNamespace(_prefer_ccdf=None))
    with pytest.raises(NotImplementedError, match="long-range-only"):
        comm.construct_gdf(arguments(df_backend="rsgdf"), SimpleNamespace(omega=0.2))
