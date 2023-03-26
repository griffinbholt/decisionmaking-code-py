import pytest

class TestVariable():
    variable = Variable("x", 2)

    def test_string_representation():
        assert str(variable) == "(x, 2)"