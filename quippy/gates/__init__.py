# This file is part of QuIPPy.
#
# QuIPPy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# QuIPPy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with QuIPPy. If not, see <http://www.gnu.org/licenses/>.
'''
Created on Nov 4, 2019

@author: Dirk Toewe
'''

import numpy as np

H = [[1, 1],
     [1,-1]] / np.sqrt(2)

X = np.array([[0,1],
              [1,0]])

Z = np.array([[1, 0],
              [0,-1]])

CNOT = np.array([[1, 0, 0, 0],
                 [0, 1, 0, 0],
                 [0, 0, 0, 1],
                 [0, 0, 1, 0]])

CCNOT = np.eye(8)
CCNOT[-2:,-2:] = [[0,1],
                  [1,0]]

TOFFOLI = CCNOT

for mat in [H,X,Z,CNOT,CCNOT]:
  mat.flags.writeable = False
