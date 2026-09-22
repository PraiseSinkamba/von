"""Von Engine orchestrator.

Von supports its own decision backends only. The third-party encoders reachable
here exist purely so the benchmark suite can score Von against them on identical
inputs -- they are comparison baselines, not supported backends.
"""

import os
import warnings
import threading
from typing import Any, Dict, List, Optional, Union

from .backends import (
    BaseBackend,
    BertaBackend,
)
from .types import (
    Choice,
    ChoiceAnswer,
    Noul,
    NoulAnswer,
    Question,
    Score,
    ScoreAnswer,
    SystemOneResponse,
)


# Von ships exactly one model. Releases are identified by version number only --
# never by architecture name -- so callers never have to know or care how the
# current model is built. A model that changes enough to break parity gets the
# next version number.
VON_VERSION = "1.1"

# Aliases that resolve to the current model.
VON_CURRENT_ALIASES = ("von-1.1", "1.1", "von", "default", "latest", "von-latest")
SUPPORTED_BACKENDS = frozenset(VON_CURRENT_ALIASES)

# Superseded releases and third-party encoders. These exist only so the
# benchmark suite can score the current model against them on identical inputs.
# They are not supported, are not documented as options, and carry no
# compatibility guarantee.
BENCHMARK_BACKENDS = frozenset((
    "von-1.0", "1.0",
    "laya", "laya-421m", "convaiinnovations/laya",
    "berta", "berta-v3", "deberta", "deberta-v3",
))


class VonEngine:
    """System One inference engine orchestrator."""

    _instance: Optional["VonEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, backend_name: str = "von-1.1", device: Optional[str] = None):
        self.backend_name = backend_name.lower().strip()
        self.device = device or os.environ.get("VON_DEVICE")
        if self.backend_name in VON_CURRENT_ALIASES:
            from .backends.option_marker_backend import OptionMarkerBackend
            self.backend: BaseBackend = OptionMarkerBackend(device=self.device)
        elif self.backend_name in BENCHMARK_BACKENDS:
            # Third-party comparison baselines. These exist so the benchmark
            # suite can score Von against them on identical inputs; they are not
            # supported backends and carry no compatibility guarantee.
            warnings.warn(
                f"Backend '{self.backend_name}' is a superseded or third-party baseline kept "
                f"for benchmarking only. Von {VON_VERSION} is the supported model.",
                UserWarning,
                stacklevel=2,
            )
            try:
                if self.backend_name in ("laya", "laya-421m", "convaiinnovations/laya"):
                    from .local_backends.laya_backend import LayaBackend
                    self.backend = LayaBackend(device=self.device)
                elif self.backend_name in ("von-1.0", "1.0"):
                    self.backend = BertaBackend(variant="von-1.0", device=self.device)
                else:
                    self.backend = BertaBackend(variant="deberta-v3", device=self.device)
            except (ImportError, ModuleNotFoundError) as e:
                raise ValueError(
                    f"Benchmark baseline '{self.backend_name}' is not installed. "
                    f"Baselines ship only with the development checkout, not the public release."
                ) from e
        else:
            raise ValueError(
                f"Unknown model '{self.backend_name}'. "
                f"Von {VON_VERSION} is the current model; accepted aliases: "
                f"{', '.join(sorted(VON_CURRENT_ALIASES))}."
            )

    @classmethod
    def get_instance(cls, backend: Optional[str] = None, device: Optional[str] = None) -> "VonEngine":
        with cls._lock:
            if cls._instance is None:
                b = backend or os.environ.get("VON_BACKEND", f"von-{VON_VERSION}")
                d = device or os.environ.get("VON_DEVICE")
                cls._instance = cls(backend_name=b, device=d)
            return cls._instance

    @classmethod
    def set_backend(cls, backend: str, device: Optional[str] = None):
        """Switch active engine backend."""
        with cls._lock:
            d = device or os.environ.get("VON_DEVICE")
            cls._instance = cls(backend_name=backend, device=d)

    def evaluate_choice(self, *args, **kwargs) -> ChoiceAnswer:
        return self.backend.evaluate_choice(*args, **kwargs)

    def evaluate_score(self, *args, **kwargs) -> ScoreAnswer:
        return self.backend.evaluate_score(*args, **kwargs)

    def evaluate_noul(self, *args, **kwargs) -> NoulAnswer:
        return self.backend.evaluate_noul(*args, **kwargs)

    def evaluate(
        self,
        state: Any,
        questions: Dict[str, Union[Question, Dict[str, Any]]],
        model: Optional[str] = None,
    ) -> SystemOneResponse:
        # Von ships exactly one model, so the response is always stamped with the
        # version actually served. Echoing the caller's requested id back would
        # let a stale client (JS SDK 1.0.1 still asks for "von-1.0.0") receive a
        # response labelled as a model that no longer exists, served by a
        # different one. Old ids are accepted, never reflected.
        resolved_model = f"von-{VON_VERSION}.0"
        return self.backend.evaluate(state=state, questions=questions, model=resolved_model)
