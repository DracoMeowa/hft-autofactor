"""``python -m hft_autofactor.digest`` — same as the hftaf-digest CLI."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
