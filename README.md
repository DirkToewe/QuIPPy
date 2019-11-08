# Motivation
There is a wide variety of high-performance quantum circuit simulators available.
Due to their focus on performance, they are implemented in C++ and/or (GPU)
parallelized. This makes the source code hard to read and understand. The math
behind simulating quantum circuits however is very helpful in understanding
the power of quantum computing.

QuIPPy is a minimalistic Python implementation of a quantum circuit simulator.
The focus of QuIPPy is not on performance but on simplicity. Counting comments,
the key component of QuIPPy, the `QMem` class, is only 310 lines of code long.

QuIPPy was written to improve the authors understanding of quantum computing.
Its source code is released in the hopes that it helps others understand
quantum computing better as well.

# Usage
The following example brings two quantum bits into an entangled state using a
simple quantum circuit made up of a Hadamard gate followed by a controlled not
(CNOT) gate.

```py
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

out1,out0 = qMem.measure([1,0])
print( 'Measurement: |{:d}{:d}〉'.format(out1,out0) ) # <- either |00〉or |11〉

print('collapsed state:', qMem.qstate.real) # <- measurement collapses quantum state
```

In the [test cases](./tests/q_mem_tests.py), more usage examples can be found, most of them
based on examples from the book `Quantum Computation and Quantum Information`
by Michael A. Nielsen and Isaac L. Chuang.
