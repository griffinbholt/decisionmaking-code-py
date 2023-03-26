import pytest
import sys; sys.path.append('./code/'); sys.path.append('../../')

from ch02 import Variable, Assignment
from ch02 import assignments

class TestAssignment():
    a1 = Assignment({"x": 0, "y": 0, "z": 0})
    a2 = Assignment({"x": 0, "y": 1, "z": 0})

    def test_select(self):
        subassignment = self.a1.select(["x", "y"])
        assert subassignment == Assignment({"x": 0, "y": 0})

    def test_hash(self):
        try:
            assert hash(self.a1) != hash(self.a2)
        except Exception as exc:
            assert False, "{exc}"

    def test_copy(self):
        a1_copy = self.a1.copy()
        assert a1_copy == self.a1
        a1_copy["x"] = 1
        assert a1_copy != self.a1

def test_assignment_function():
    variables = [Variable("x", 2), Variable("y", 2), Variable("z", 2)]
    values = [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]]
    a = assignments(variables)
    assert len(a) == 8
    for i in range(len(a)):
        assert a[i] == Assignment({"x": values[i][0], "y": values[i][1], "z": values[i][2]})
