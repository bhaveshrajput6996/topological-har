"""Dataset adapters.

Each adapter normalises a benchmark dataset into the common internal
representation (:class:`LoadedDataset` / :class:`SubjectSeries`) so the rest
of the pipeline never depends on dataset-specific quirks.
"""

from .base import (  # noqa: F401
    ADAPTERS,
    DatasetAdapter,
    LoadedDataset,
    SubjectSeries,
    get_adapter,
    register_adapter,
)

from . import uci_har  # noqa: F401,E402
from . import pamap2  # noqa: F401,E402
from . import opportunity  # noqa: F401,E402
from . import synthetic  # noqa: F401,E402
