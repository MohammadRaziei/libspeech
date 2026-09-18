"""
Model-weight downloading, done in Python.

Why this exists: the compiled extension can be built with
LIBSPEECH_ENABLE_HTTPP=OFF (the default for wheel builds -- see the option's
comment in CMakeLists.txt), in which case speech::utils::downloadFile has no
HTTP client compiled in at all and only ever succeeds if the target file is
*already on disk*. This module is what puts it there first: it mirrors
BaseModel's own bare-name -> GitHub-release-URL resolution and its default
cache directory convention exactly, so that by the time a compiled model
class's C++ constructor runs downloadFile() itself, the file is already at
the exact path it's about to look for and that call is a same-path
existence check, not a network request.

If the extension *was* built with LIBSPEECH_ENABLE_HTTPP=ON (a plain local
dev build, not a wheel), all of this is redundant but harmless: whichever
side gets there first downloads the file, the other one's a no-op.
"""

from __future__ import annotations

from pathlib import Path

# Must match BaseModel::BaseModel's hardcoded default in src/models/base_model.cpp.
_GITHUB_RELEASES_BASE = "https://github.com/MohammadRaziei/libspeech/releases/download/models/"


def resolve_model_url(name_or_url: str) -> str:
    """Mirrors BaseModel's constructor: a bare filename becomes a GitHub
    Releases URL; anything already starting with http(s):// is used as-is."""
    if name_or_url.startswith("http://") or name_or_url.startswith("https://"):
        return name_or_url
    return _GITHUB_RELEASES_BASE + name_or_url


def default_model_cache_dir() -> Path:
    """Mirrors speech::utils::getDefaultModelCacheDir()'s <home>/.libspeech
    convention. Path.home() already handles the Unix vs. Windows difference
    (it's what the C++ side's HOME/USERPROFILE/HOMEDRIVE+HOMEPATH fallback
    chain is doing manually), so no need to reimplement that chain here."""
    try:
        home = Path.home()
    except RuntimeError:
        # No home directory could be resolved (rare: e.g. no passwd entry) --
        # match the C++ side's own last-resort fallback.
        import tempfile

        home = Path(tempfile.gettempdir())
    return home / ".libspeech"


def ensure_model(name_or_url: str, *, force: bool = False, quiet: bool = False) -> str:
    """Ensures `name_or_url` is downloaded into the default model cache
    directory, downloading it with `requests` if it isn't there yet (or if
    force=True). Returns the *bare filename* (not a path) -- callers should
    pass that straight into the compiled model class's constructor, which
    resolves it the exact same way BaseModel always has, so it round-trips
    to the same cached file this function just ensured exists.

    Raises ImportError if the file needs downloading and `requests` isn't
    installed (`pip install libspeech[online]`), or requests.RequestException
    on an actual download failure.
    """
    url = resolve_model_url(name_or_url)
    filename = Path(url).name
    cache_dir = default_model_cache_dir()
    target = cache_dir / filename

    if not force and target.exists():
        return filename

    try:
        import requests
    except ImportError as e:
        raise ImportError(
            f"Model '{name_or_url}' is not cached locally at {target} and this build of "
            "libspeech has no download support compiled in, so it must be fetched in "
            "Python instead. Install the extra with `pip install libspeech[online]` "
            "(or just `pip install requests`) and try again."
        ) from e

    cache_dir.mkdir(parents=True, exist_ok=True)
    tmp_target = target.with_suffix(target.suffix + ".part")

    if not quiet:
        print(f"Downloading model from: {url}")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        written = 0
        with open(tmp_target, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                written += len(chunk)
                if not quiet and total:
                    pct = min(100, written * 100 // total)
                    print(f"\r[download {pct}% complete]", end="", flush=True)
    if not quiet:
        print(f"\nDownload completed: {target}")

    tmp_target.replace(target)
    return filename
