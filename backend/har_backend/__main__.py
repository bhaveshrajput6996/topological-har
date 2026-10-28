"""Module wrapper: ``python -m har_backend ...`` runs the :mod:`har.cli`."""

import sys

sys.path.insert(0, __file__.rsplit("har_backend", 1)[0])

from har.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
