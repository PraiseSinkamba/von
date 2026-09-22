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


# Von's own backends. These are the only supported, publicly documented options.
OPTION_MARKER_ALIASES = ("option-marker", "von-marker", "marker", "option_marker")
VON_1_0_ALIASES = ("von-1.0", "von", "modernbert", "default", "berta-modern", "modernbert-nli")
SUPPORTED_BACKENDS = frozenset(OPTION_MARKER_ALIASES + VON_1_0_ALIASES)

# Third-party baselines: benchmark-only, unsupported, no compatibility guarantee.
BENCHMARK_BACKENDS = frozenset((
    "laya", "laya-421m", "convaiinnovations/laya",
    "berta", "berta-v3", "deberta", "deberta-v3",
))


class VonEngine:
    """System One inference engine orchestrator."""

    _instance: Optional["VonEngine"] = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, backend_name: str = "von-1.0", device: Optional[str] = None):
        self.backend_name = backend_name.lower().strip()
        self.device = device or os.environ.get("VON_DEVICE")
        if self.backend_name in OPTION_MARKER_ALIASES:
            from .backends.option_marker_backend import OptionMarkerBackend
            self.backend = OptionMarkerBackend(device=self.device)
        elif self.backend_name in VON_1_0_ALIASES:
            self.backend: BaseBackend = BertaBackend(variant="von-1.0", device=self.device)
        elif self.backend_name in BENCHMARK_BACKENDS:
            # Third-party comparison baselines. These exist so the benchmark
            # suite can score Von against them on identical inputs; they are not
            # supported backends and carry no compatibility guarantee.
            warnings.warn(
                f"Backend '{self.backend_name}' is a benchmark comparison baseline, "
                f"not a supported Von backend. Use 'option-marker' or 'von-1.0' in production.",
                UserWarning,
                stacklevel=2,
            )
            try:
                if self.backend_name in ("laya", "laya-421m", "convaiinnovations/laya"):
                    from .local_backends.laya_backend import LayaBackend
                    self.backend = LayaBackend(device=self.device)
                else:
                    self.backend = BertaBackend(variant="deberta-v3", device=self.device)
            except (ImportError, ModuleNotFoundError) as e:
                raise ValueError(
                    f"Benchmark baseline '{self.backend_name}' is not installed. "
                    f"Baselines ship only with the development checkout, not the public release."
                ) from e
        else:
            raise ValueError(
                f"Unknown backend '{self.backend_name}'. "
                f"Supported Von backends: {', '.join(sorted(SUPPORTED_BACKENDS))}."
            )

    @classmethod
    def get_instance(cls, backend: Optional[str] = None, device: Optional[str] = None) -> "VonEngine":
        with cls._lock:
            if cls._instance is None:
                b = backend or os.environ.get("VON_BACKEND", "von-1.0")
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
        if model in ("von-latest", "von-preview", "jev-latest", "jev-preview", None):
            resolved_model = "von-1.0.0"
        else:
            resolved_model = model
        return self.backend.evaluate(state=state, questions=questions, model=resolved_model)
