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

import numpy as np, numpy.linalg as la


class QMem:
  '''
  A simulated representation of quantum memory, made up of a fixed (small) set of quantum bits.

  Parameters
  ----------
  bits: bool[:]
    [kwarg only] Initializes the quantum memory to a classical bit state.
    bits[0] represents the least significant and bits[-1] the most significant bit.

    Only either the `bits` or `qbits` kwarg may be used not both at once.
  qbits: complex[:,2]
    [kwarg only] Initializes the quantum memory to a non-entangled set of quantum bit states.
    qbits[0,:] represents the the least significant quantum bit and bits[-1] the most significant bit.

    Only either the `bits` or `qbits` kwarg may be used not both at once.

  Attributes
  ----------
  n_qbits: int
    The number of quantum bits that make up this quantum memory.
  _qstate: complex[2**n_qbits]
    The quantum state vector of this quantum memory. At all time the Euclidean
    norm of _qstate is supposed to be 1.
  '''

  def __init__(self, **kwargs):
    if 'bits' in kwargs:
      assert 'qbits' not in kwargs, 'QMem(*kwargs): kwargs must not contain both "bits" and "qbits".'

      # convert bits into qbits initializer
      bits = np.fromiter( kwargs['bits'], dtype=np.bool )
      qbits = np.zeros([len(bits), 2], dtype=np.float)
      qbits[ bits,1] = 1
      qbits[~bits,0] = 1

    else:
      assert 'qbits' in kwargs, 'QMem(*kwargs): kwargs must either contain "bits: bool[]" or "qbits: float[:,2]".'
      qbits = np.asarray(kwargs['qbits'])
      assert qbits.ndim     == 2, 'QMem(qbits=qbits): qbits.ndim must be 2.'
      assert qbits.shape[1] == 2, 'QMem(qbits=qbits): qbits.shape[1] must be 2.'

    assert qbits.ndim == 2
    assert qbits.shape[1] == 2
    # each bit must have norm 1
    assert np.allclose( 1, la.norm(qbits, axis=1) ), 'QMem(qbits=qbits): Every row in qbits must have a norm of 1.'

    # alloc quantum state
    self._qstate = np.zeros(
      1 << len(qbits),
      dtype=np.complex
    )
    self._qstate[0] = 1

    # init quantum state using tensor product for vectors
    n = 1
    for qbit in qbits:
      assert qbit.shape == (2,)
      
      self._qstate[n:n*2] = qbit[1] * self._qstate[:n]
      self._qstate[ :n]  *= qbit[0]
      
      n <<= 1

    assert n == len(self._qstate)
    # length of quantum state vector is supposed to be 1
    assert np.isclose(la.norm(self._qstate), 1)


  @property
  def n_qbits(self):
    '''
    Returns the number qbits in this quantum memory.

    Returns
    -------
    n_qbits: int
      The number of qbits in this quantum memory.
    '''
    n_qbits = np.log2( len(self._qstate) ).astype(np.int)
    assert 1<<n_qbits == len(self._qstate) # <- sanity check
    return n_qbits


  @property
  def qstate(self):
    '''
    Returns the quantum state vector representing this quantum memory.

    Returns
    -------
    qstate: complex[2**n_qbits]
      The quantum state vector representing this quantum memory.
    '''
    return self._qstate


  def apply_gate(self, qbits, gate):
    '''
    Applies a quantum gate to a subset of the qantum bits.

    Parameters
    ----------
    qbits: int[n]
      The indices of the quantum bit to which the the gate is applied.
    gate: int[2**n, 2**n]
      The matrix description of the applied gate.
    '''
    gate = np.asmatrix(gate, dtype=np.float)
    # make sure the gate is unitary
    assert np.allclose( gate   @ gate.H, np.eye(len(gate)) ), 'QMem.apply_gate(qbits, gate): gate must be unitary.'
    assert np.allclose( gate.H @ gate  , np.eye(len(gate)) ), 'QMem.apply_gate(qbits, gate): gate must be unitary.'

    N_qbits = self.n_qbits

    def normalize(q):
      '''
      Normalizes the id of a quantum bit where 0 is the least significant
      quantum bit and -1 is the most significant quantum bit.

      Parameters
      ----------
      q: int
        A positive of negative id of a quantum bit. As usual in Python,
        a negative id addresses from the opposite side, i.e. -1 is the most
        significant bit, -2 is the second to most significant bit and so on.

      Throws
      ------
      ae: AssertionError
        If the quantum bit id `q` is invalid.

      Returns
      -------
      q_norm: int
        The positive id of the quantum bit.
      '''
      if q < 0:
        q += N_qbits
      assert 0 <= q < N_qbits, 'QMem.apply_gate(qbits, gate): qbits contains invalid entries.'
      return q

    if isinstance(qbits, int):
      qbits = (qbits,)

    qbits = np.fromiter(
      (normalize(q) for q in qbits),
      dtype=np.int
    )
    del normalize
    n_qbits = len(qbits)

    assert  len(qbits)== len({*qbits}), 'QMem.apply_gate(qbits, gate): qbits must not contain duplicates.'
    assert 1<<n_qbits == len(gate), 'QMem.apply_gate(qbits, gate): gate shape must be compatible with number of qbits.'

    assert gate.ndim == 2,                'QMem.apply_gate(qbits, gate): gate must be square matrix.'
    assert gate.shape[0] == gate.shape[1],'QMem.apply_gate(qbits, gate): gate must be square matrix.'

    n_all = len(self._qstate) # <- no. of all quantum states
    n_rest = 1  <<  N_qbits - n_qbits # <- no. of quantum state entries independent of gate
    n_gate = len(gate) # <- no. of quantum state entries in gate

    assert n_rest*n_gate == n_all

    qstate = np.zeros_like(self._qstate)

    gate2mem = qbits
    rest2mem = sorted({*range(N_qbits)} - {*qbits})

    assert len({*gate2mem}) == len(gate2mem)
    assert len({*rest2mem}) == len(rest2mem) 
    assert {*gate2mem,*rest2mem}  ==  {*range(N_qbits)}

    def idx( iGate, iRest ):
      '''
      Computes a quantum state index inside of this quantum memory.

      Parameters
      ----------
      iGate: int
        Index into gate's quantum state.
      iRest: int
        Index into the quantum state that is independent from the gate.

      Returns
      -------
      idx: int
        The index into the quantum state of this quantum memory.
      '''
      idx = 0
      for k,l in enumerate(gate2mem): idx |= ( (iGate>>k & 1) << l )
      for k,l in enumerate(rest2mem): idx |= ( (iRest>>k & 1) << l )
      return idx

    for i in range(n_gate):
      for j in range(n_gate):
        if 0 != gate[i,j]: # <- sparse gate optimization
          for k in range(n_rest):
            qstate[idx(i,k)] += gate[i,j] * self._qstate[idx(j,k)]

    assert np.isclose( 1, la.norm(qstate) )
    self._qstate[:] = qstate


  def probe(self, qbit):
    '''
    Returns the probability of a quantum bit to collapse to 1 when measured.
    This method does not collapse the quantum state of the probed bit.

    Parameters
    ----------
    qbit: int
      The index of the quantum bit that is probed.

    Returns
    -------
    p: float
      The probability of the probed quantum bit to collapse to 1.
    '''
    assert isinstance(qbit, int), 'QMem.probe(qbit): qbit must be an integer for now. Probing multiple qbits is not (yet) supported.'
    n_qbits = self.n_qbits

    if qbit < 0:
      qbit += n_qbits
    assert 0 <= qbit < n_qbits, 'QMem.probe(qbit): qbit id out of bounds.'

    mask = np.bitwise_and(
      np.arange( len(self._qstate) ),
      1 << qbit
    ).astype(np.bool)

    p = np.square(np.abs( self._qstate[mask] )).sum()
    return p


  def measure(self, qbit=None):
    '''
    Measures one or more qbits, collapsing their quantum state.

    Parameters
    ----------
    qbit: int or int[]
      The bit indices that are to be measured.

    Returns
    -------
    bit: int or int[]
      The measured/collapsed state of the measured quantum bits.
    '''
    n_qbits = self.n_qbits

    if not isinstance(qbit, int):
      # if no parameter is given, measure all qbits
      if None is qbit:
        qbit = range(n_qbits)

      # for an iterable of qbits return a tuple of measured bits
      return tuple(
        self.measure( int(q) )
        for q in qbit
      )

    # negative quantum bit ids reference starting from the most significant bit
    # (-1: most significant, -2: second most significant, ...)
    if qbit < 0:
      qbit += n_qbits
    assert 0 <= qbit < n_qbits, 'QMem.measure(qbit): qbit id out of bounds.'

    # mask of all quantum state entries where qbit is set to 1
    mask = np.bitwise_and(
      np.arange( len(self._qstate) ),
      1 << qbit
    ).astype(np.bool)

    # compute the probability
    p = np.square(np.abs( self._qstate[mask] )).sum()

    # roll the dice for the classical bit state
    cbit = np.bool(np.random.rand() < p)

    # collapse part of the quantum state
    if cbit: mask = ~mask
    self._qstate[mask] = 0
    self._qstate /= np.sqrt( p if cbit else 1-p )

    assert np.isclose( 1, la.norm(self._qstate) )
    return cbit
