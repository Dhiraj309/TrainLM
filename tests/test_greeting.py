import pytest

from trainlm import greet


def test_greet_uses_trainlm_by_default():
    assert greet() == "Hello, TrainLM!"


def test_greet_accepts_and_normalizes_a_name():
    assert greet("  Dhiraj  ") == "Hello, Dhiraj!"


@pytest.mark.parametrize("name", ["", "   "])
def test_greet_rejects_empty_names(name):
    with pytest.raises(ValueError, match="must not be empty"):
        greet(name)


def test_greet_rejects_non_string_names():
    with pytest.raises(TypeError, match="must be a string"):
        greet(123)
