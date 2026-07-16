"""Enable `python -m s2t_bench ...` (bypasses the console-script shebang,
which is useful when the venv lives in a path containing spaces)."""
from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
