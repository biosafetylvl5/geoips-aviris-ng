"""Template repository demonstrating a basic GeoIPS plugin example."""

# NOTE: _version.py is generated automatically during build/install
try:
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback version when _version.py is not available (development mode)
    __version__ = "0.0.0-dev"
    __version_tuple__ = (0, 0, 0, "dev")

__all__ = ["__version__", "__version_tuple__"]
