import pytest
import sys; sys.path.append('./code/'); sys.path.append('../../')

from ch02 import Variable

class TestVariable():
    variable = Variable("x", 2)

    def test_string_representation(self):
        assert str(self.variable) == "(x, 2)", "String representation of Variable is not correct"