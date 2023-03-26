import pytest
import sys; sys.path.append('../../')

from ch02 import Variable

class TestVariable():
    variable = Variable("x", 2)

    def test_string_representation():
        assert str(variable) == "(x, 2)"