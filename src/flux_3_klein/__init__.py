def main() -> None:
    """Entry point for `flux-3-klein` CLI."""
    # Lazy import to avoid circular import warning when run as `python -m`
    from flux_3_klein.generate import main as _main

    _main()
