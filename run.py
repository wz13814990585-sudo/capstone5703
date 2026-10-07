"""Integration-only launcher. Original research scripts are executed unchanged."""
from integration.runner import main

if __name__ == "__main__":
    raise SystemExit(main())
