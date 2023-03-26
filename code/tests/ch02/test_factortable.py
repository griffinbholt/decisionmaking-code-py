import pytest
import sys; sys.path.append('./code/'); sys.path.append('../../')

from ch02 import Assignment, FactorTable

class TestFactorTable():
    table = FactorTable({
                Assignment({"x": 0, "y": 0, "z": 0}): 0.08, Assignment({"x": 0, "y": 0, "z": 1}): 0.31,
                Assignment({"x": 0, "y": 1, "z": 0}): 0.09, Assignment({"x": 0, "y": 1, "z": 1}): 0.37,
                Assignment({"x": 1, "y": 0, "z": 0}): 0.01, Assignment({"x": 1, "y": 0, "z": 1}): 0.05,
                Assignment({"x": 1, "y": 1, "z": 0}): 0.02, Assignment({"x": 1, "y": 1, "z": 1}): 0.07
            })

    def test_default_get_value(self):
        assert self.table.get(Assignment({"x": 0, "y": 0, "z": 0}), default_val=0.0) == 0.08
        assert self.table.get(Assignment({"x": 2, "y": 2, "z": 2}), default_val=0.0) == 0.0