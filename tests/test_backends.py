import warnings

import pytest

import von
from von.engine import VON_CURRENT_ALIASES, VON_VERSION, VonEngine


@pytest.fixture(autouse=True)
def reset_backend():
    yield
    VonEngine._instance = None


def test_default_model_loads_and_decides():
    """The zero-config path must work.

    This is the call every new user makes first. It previously broke -- the
    default checkpoint directory did not exist, so loading silently fell through
    to a Hub download and raised at runtime -- while the suite stayed green
    because every test pinned an explicit backend. Nothing is pinned here on
    purpose.
    """
    VonEngine._instance = None
    res = von.decide(
        "Customer requests refund for duplicate charge on invoice #100",
        choices={
            "billing": "Invoices, billing, duplicate charges, refunds",
            "technical": "Software bugs and technical issues",
        },
    )
    assert res.choice == "billing"
    assert "billing" in res.probabilities
    assert res.confidence > 0.0


def test_default_resolves_to_current_version():
    VonEngine._instance = None
    engine = VonEngine.get_instance()
    assert engine.backend_name in VON_CURRENT_ALIASES


@pytest.mark.parametrize("alias", sorted(VON_CURRENT_ALIASES))
def test_every_current_alias_loads_the_same_model(alias):
    engine = VonEngine(backend_name=alias)
    assert type(engine.backend).__name__ == "OptionMarkerBackend"


def test_current_model_decides():
    von.set_backend(f"von-{VON_VERSION}")
    res = von.decide(
        "Server CPU temperature reached 105 degrees Celsius",
        choices={
            "hardware_alert": "Hardware and temperature warnings",
            "billing": "Invoices and subscription payments",
        },
    )
    assert res.choice == "hardware_alert"
    assert res.confidence > 0.0


def test_superseded_release_warns_and_is_not_supported():
    """von-1.0 is kept for benchmarking only and must announce that."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            VonEngine(backend_name="von-1.0")
        except ValueError:
            pytest.skip("von-1.0 baseline weights not present in this checkout")
    assert any(issubclass(w.category, UserWarning) for w in caught)


def test_unknown_model_names_the_current_version():
    with pytest.raises(ValueError) as exc:
        VonEngine(backend_name="option-marker")
    message = str(exc.value)
    assert "option-marker" in message
    assert VON_VERSION in message
