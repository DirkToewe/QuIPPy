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

from quippy import QMem
from quippy.gates import H,CNOT

#              ┌───┐
# |1> =: in1 ──┤ H ├──●─── out1
#              └───┘  │
# |0> =: in0 ─────────⊕─── out0

in0 = 0
in1 = 1

qMem = QMem( bits = [in0,in1] )
qMem.apply_gate([1], H)
qMem.apply_gate([0,1], CNOT)

print('entangled state:', qMem.qstate.real)
