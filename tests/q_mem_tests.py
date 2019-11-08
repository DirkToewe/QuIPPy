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

import numpy as np, unittest
from quippy import QMem
from quippy.gates import H, X, Z, CNOT, CCNOT


# TODO:
#   * test with larger overall systems
#   * test complex-valued gates
class QMemTests(unittest.TestCase):
  def test_bell_epr_states(self):
    '''
    Tests if the the Bell/EPR states can be produced, see:

    "Quantum Computation and Quantum Information"
     10th Anniversary Edition
     Michael A. Nielsen & Isaac L. Chuang
     p. 25f
          ┌───┐
    in1 ──┤ H ├──●─── out1
          └───┘  │
    in0 ─────────⊕─── out0
    '''
    def bell( in0, in1 ):
      qMem = QMem( bits = [in0,in1] )
      qMem.apply_gate([1], H)
      qMem.apply_gate([0,1], CNOT)
      return qMem._qstate

    assert np.allclose( bell(0,0)[:4], [1, 0, 0, 1] / np.sqrt(2) )
    assert np.allclose( bell(1,0)[:4], [0, 1, 1, 0] / np.sqrt(2) )
    assert np.allclose( bell(0,1)[:4], [1, 0, 0,-1] / np.sqrt(2) )
    assert np.allclose( bell(1,1)[:4], [0, 1,-1, 0] / np.sqrt(2) )


  def test_quantum_teleportation(self):
    '''
    Tests if quantum teleportation example works:
    
    "Quantum Computation and Quantum Information"
     10th Anniversary Edition
     Michael A. Nielsen & Isaac L. Chuang
     p. 26ff
                               ┌───┐  ┌───┐
           in2 ─────────────●──┤ H ├──┤ ↗ ╞═════════●
                            │  └───┘  └───┘         ║
                 ┌───┐      │         ┌───┐         ║
    |0> =: in1 ──┤ H ├──●───⊕─────────┤ ↗ ╞══●      ║
                 └───┘  │             └───┘  ║      ║
                        │                  ┌─╨─┐  ┌─╨─┐         !
    |0> =: in0 ─────────⊕──────────────────┤ X ├──┤ Z ├── out0 = in2
                                           └───┘  └───┘
    '''
    for angle in np.linspace(-np.pi, +np.pi, 8*1024):
      in2 = (np.cos(angle),
             np.sin(angle))
      p_in = in2[1]**2

      qMem = QMem(qbits=[
        (1,0), # <- in0
        (1,0), # <- in1
        in2
      ])

      assert np.isclose( p_in, qMem.probe(2) )

      qMem.apply_gate([1], H)
      qMem.apply_gate([0,1], CNOT)

      qMem.apply_gate([1,2], CNOT)
      qMem.apply_gate([2], H)

      out1,out2 = qMem.measure([1,2])
      if out1: qMem.apply_gate([0], X)
      if out2: qMem.apply_gate([0], Z)

      p_out = qMem.probe(0)
      assert np.isclose(p_in, p_out)


  def test_toffoli(self):
    '''
    Tests if quantum teleportation example works:
    
    "Quantum Computation and Quantum Information"
     10th Anniversary Edition
     Michael A. Nielsen & Isaac L. Chuang
     p. 29f
            
    in2 ───●─── out2
           │
    in1 ───●─── out1
           │
    in0 ───⊕─── out0

    The toffoli gate should has the following transition table:
    
     in2 │ in1 │ in0 ┃ out2│ out1│ out0
    ━━━━━┿━━━━━┿━━━━━╋━━━━━┿━━━━━┿━━━━━
      0  │  0  │  0  ┃  0  |  0  |  0
      0  │  0  │  1  ┃  0  |  0  |  1
      0  │  1  │  0  ┃  0  |  1  |  0
      0  │  1  │  1  ┃  0  |  1  |  1
      1  │  0  │  0  ┃  1  |  0  |  0
      1  │  0  │  1  ┃  1  |  0  |  1
      1  │  1  │  0  ┃  1  |  1  |  1
      1  │  1  │  1  ┃  1  |  1  |  0
    '''
    for inp,out in (
      ( (0,0,0), (0,0,0) ),
      ( (0,0,1), (0,0,1) ),
      ( (0,1,0), (0,1,0) ),
      ( (0,1,1), (0,1,1) ),
      ( (1,0,0), (1,0,0) ),
      ( (1,0,1), (1,0,1) ),
      ( (1,1,0), (1,1,1) ),
      ( (1,1,1), (1,1,0) )
    ):
      inp = inp[::-1]
      out = out[::-1]
      qMem = QMem(bits=inp)
      qMem.apply_gate([0,1,2], CCNOT) # <- CCNOT a.k.a. TOFFOLI
      self.assertEqual( qMem.measure(), out )


  def test_deutsch(self):
    '''
    Makes sure that Deutsch's algorithm works, see:
    
    "Quantum Computation and Quantum Information"
     10th Anniversary Edition
     Michael A. Nielsen & Isaac L. Chuang
     p. 32ff

    For an unknown unary boolean function f: bool -> bool,
    we are given a blackbox quantum circuit that computes
    said function:

          ┌───────┐
    in1 ──┤       ├── in1
          │ Black │
          │ Box   │
    in0 ──┤       ├── in0 ⊕ f(in1)
          └───────┘

    Where `⊕` is the xor operator. The `in0 ⊕` is necessary
    for the operation to be revertible/unitary.

    Deutsch's algorithm can tell us whether `f` is a constant
    function (x -> True, x -> False) or a non-constant function
    (x -> x, x -> not x). Deutsch's algorithm can do this with
    a single (quantum) function evaluation. On a classical computer
    two function evaluations would be required. 

                ┌───┐ ┌───────┐ ┌───┐ ┌───┐
    |0> =: in1 ─┤ H ├─┤       ├─┤ H ├─┤ ↗ ╞═ out1
                └───┘ │ Black │ └───┘ └───┘
                ┌───┐ │ Box   │  
    |1> =: in0 ─┤ H ├─┤       ├───────────── out0
                └───┘ └───────┘

    Where `out1` is false if and only if f is a constant function.
    '''
    def is_const( blackbox ):
      '''
      Parameters
      ----------
      blackbox: int[4,4]
        A blackbox 2-qbit quantum gate that either represents
        a classical set-to-0, set-to-1, not or identity gate.
        The first qbit is the input to the blackbox and is passed
        through the blackbox unchanged. The second qbit is zero
        on the input side and contains the output value of the
        blackbox.
      '''
      qMem = QMem(bits=[1,0])
      qMem.apply_gate(0, H)
      qMem.apply_gate(1, H)
      qMem.apply_gate([1,0], blackbox) # <- reverse the qbit order order to adjust to the book's conventions 
      qMem.apply_gate(1, H)
      p = qMem.probe(1)
      assert np.isclose(p,0) or np.isclose(p,1)
      return not p

    ID  = np.array([[1,0,0,0],
                    [0,0,0,1],
                    [0,0,1,0],
                    [0,1,0,0]])
    NOT = np.array([[0,0,1,0],
                    [0,1,0,0],
                    [1,0,0,0],
                    [0,0,0,1]])
    CONST_0 = np.array([[1,0,0,0],
                        [0,1,0,0],
                        [0,0,1,0],
                        [0,0,0,1]])
    CONST_1 = np.array([[0,0,1,0],
                        [0,0,0,1],
                        [1,0,0,0],
                        [0,1,0,0]])
    for mat in [ID,NOT,CONST_0,CONST_1]:
      mat.flags.writeable = False

    # test that the gates are correct
    for bit0 in [0,1]:
      for bit1 in [0,1]:
        for gate,result in (
          (ID,      bit0),
          (NOT, not bit0),
          (CONST_0,    0),
          (CONST_1,    1)
        ):
          qMem = QMem(bits=[bit0,bit1])
          qMem.apply_gate([0,1], gate)
          self.assertEqual( qMem.measure(), (bit0, bit1 ^ result) )

    self.assertTrue ( is_const(CONST_0) )
    self.assertTrue ( is_const(CONST_1) )
    self.assertFalse( is_const(ID ) )
    self.assertFalse( is_const(NOT) )


def __generate_QMemTests_test_measure_entangledX():
  iTest = 0

  def generate_test( measure, qstate, result0,result1, p0,p1 ):
    assert np.isclose(1, p0+p1)

    def test_measure_entangled(self):
      qMem = QMem(bits=[0,0])
      n0,n1 = 0,0

      for _ in range(32*1024):
        qMem._qstate[:] = qstate
        result = measure(qMem)
        if   result0 == result: n0 += 1
        elif result1 == result: n1 += 1
        else: self.fail( 'Unexpected measurement: {}.'.format(result) )

      n = n0+n1
      assert abs(n0 - p0*n) <= n * 1e-2
      assert abs(n1 - p1*n) <= n * 1e-2

    test_measure_entangled.__name__ = 'test_measure_entangled{:02d}'.format(iTest)

    setattr(
      QMemTests,
      test_measure_entangled.__name__,
      test_measure_entangled
    )

  for inputs in ( # <- try out different entangled states
    ( [1,0,0,1] / np.sqrt(2),  (0,0),(1,1),  0.5,0.5 ),
    ( [1,0,0,2] / np.sqrt(5),  (0,0),(1,1),  0.2,0.8 ),
    ( [2,0,0,1] / np.sqrt(5),  (0,0),(1,1),  0.8,0.2 ),
    ( [0,1,1,0] / np.sqrt(2),  (1,0),(0,1),  0.5,0.5 ),
    ( [0,1,2,0] / np.sqrt(5),  (1,0),(0,1),  0.2,0.8 ),
    ( [0,2,1,0] / np.sqrt(5),  (1,0),(0,1),  0.8,0.2 )
  ):
    for measure in ( # <- try out different ways of measuring which must yield identical results
      lambda qMem:  qMem.measure(),
      lambda qMem:  qMem.measure([0,1]),
      lambda qMem:  qMem.measure([1,0])[::-1],
      lambda qMem: (qMem.measure(0), qMem.measure(1)),
      lambda qMem: (qMem.measure(1), qMem.measure(0))[::-1]
    ):
      iTest += 1
      generate_test(measure,*inputs)


__generate_QMemTests_test_measure_entangledX()


if __name__ == '__main__':
  unittest.main()
