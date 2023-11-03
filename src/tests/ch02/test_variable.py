import sys; sys.path.append('./src/'); sys.path.append('../../')

from ch02 import Variable


class TestVariable():
    variable = Variable("x", 2)

    def test_string_representation(self):
        assert str(self.variable) == "(x, 2)", "String representation of Variable is not correct"
