#!/usr/bin/env python3
import numpy as np


def checkMin(k, expectedMinimum, tol):
  minimum = k["Solver"]["Best Ever Value"]
  assert np.isclose(expectedMinimum, minimum, atol = tol), "Minimum {0} "\
          "deviates from true min {1} by more than {2}".format(minimum, expectedMinimum, tol)

def checkInfeasible(k, expectedMinimum):
  minimum = k["Solver"]["Infeasible Sample Count"]
  assert np.less(expectedMinimum, minimum, ), "Minimum {0} "\
          "is not less than {1}".format(minimum, expectedMinimum)
          
def checkEvals(k, expected):
  val = k["Solver"]["Model Evaluation Count"]
  assert np.equal(expected, val, ), "Value {0} "\
          "is not equal to {1}".format(expected, val)
