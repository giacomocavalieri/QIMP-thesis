import cv2 as cv
import numpy as np
from qiskit import *
from qiskit.extensions import UnitaryGate

def qc_to_edge(qc, mean_diff=True, backend=Aer.get_backend("qasm_simulator"),
               shots=100, intermediate_results=False):
    """ Retrieves the vertical edges from a quantum circuit encoding a single
        image.

        Args:
          -qc: the quantum circuit encoding the image
          -mean_diff: wether or not to use the more accurate mean finite
                      difference to compute the edges
          -backend: the backend used to execute the circuit. By default it is
                    executed using a local simulator provided by Aer
          -shots: the number of measurements to perform when recovering the
                  image
          -intermediate_results: wether or not to return the results obtained
                                 from intermediate circuits """

    imgs = []
    for even in [True, False]:
        edge_qc = _qc_edge_detect(qc, even_cols=even, mean_diff=mean_diff)
        edge_qc.measure_all()
        edge_res = execute(edge_qc, backend, shots=shots).result()
        imgs.append(_counts_to_edge(edge_res.get_counts(), even_cols=even,
                                    mean_diff=mean_diff))

    if intermediate_results:
        return (*imgs, cv.bitwise_or(*imgs))
    else:
        return cv.bitwise_or(*imgs)

def _qc_edge_detect(qc, even_cols=True, inplace=False, mean_diff=False):
    """Applies the edge detection gates to a quantum circuit.

       Args:
         -qc: the quantum circuit encoding the image
         -even_cols: wether to retrieve the even or odd columns
         -inplace: if False applies the gate to a copy of the circuit which is
                   then returned. Otherwise modifies directly |qc|
         -mean_diff: wether or not to use the more accurate mean finite
                     difference to compute the edges """

    if not inplace:
        qc = qc.copy()
    for reg in qc.qregs:
        if not even_cols:
            qc = _shift_amplitudes(qc, 1 if not mean_diff else 2,
                                   inplace=inplace)
        if mean_diff:
            qc.swap(reg[0], reg[1])
        qc.h(reg[0])
    return qc

def _counts_to_edge(counts, even_cols=True, mean_diff=True):
    """ Given the counts from a quantum circuit that performs edge detection it
        returns the corresponding image.

        Args:
          -counts: a dictionary {binaryString -> measurements} obtained from the
                   execution of a quantum circuit. It is interpreted as the
                   result of a quantum circuit performing edge detection on a
                   QPIE-encoded image
          -even_cols: wether or not the circuit performed edge detection on the
                      even or odd couples of differences
          -mean_diff: wether or not to use the more accurate mean finite
                      difference to compute the edges """

    edge = 2**(len(list(counts)[0])//2)
    img = np.zeros((edge, edge))
    for key in counts:
        # Only if the state is in the form |...1> (i.e. contains a difference)
        if key[-1] == '1':
            # Split the binary sequence in half to get the part encoding the row
            # and column
            row = int(key[:len(key)//2], 2)
            col = int(key[len(key)//2:], 2)

            # The following code performs a shift of the pixels as they are
            # recovered to get coherent final images.
            if not mean_diff:
                if even_cols:
                    col -= 1
            else:
                if even_cols:
                    if key[-2:] == "11":
                        col -= 1
                else:
                    if key[-2:] == "01":
                        col += 2
                    else:
                        col += 1
            if col < edge:
                img[row, col] = counts[key]
    img = np.sqrt(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

def _shift_amplitudes(qc, n, inplace=False):
    """ Apply an n-shift gate to a quantum circuit.

        Args:
          -qc: the quantum circuit
          -n: number of positions to shift (can be either positive or negative
              to obtain a right or left shift)
          -inplace: if False applies the gate to a copy of the circuit which is
                    then returned. Otherwise modifies directly |qc| """
    if not inplace:
        qc = qc.copy()
    for q_reg in qc.qregs:
        # Unitary gate representing the shift operation on n qubits
        shift_matrix = np.roll(np.eye(2**q_reg.size), n, axis=1)
        # Add the gate to the circuit
        qc.append(UnitaryGate(shift_matrix), q_reg)
    return qc
