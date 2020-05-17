import cv2 as cv
import math
import numpy as np
from qiskit import *

def img_to_qc(img):
    """ Given a grayscale image it returns a quantum circuit encoding it.

        Args:
          -img: grayscale image to encode in a quantum circuit """
    q_reg = QuantumRegister(int(math.log2(img.size)), "q_img")
    circuit = QuantumCircuit(q_reg)
    circuit.append(_img_to_initializer(img, "img"), q_reg)
    return circuit

def _img_to_initializer(img, name="img"):
    """ Returns a quantum circuit named |name| which, applied to an initial
        state |0...0> encodes the image passed as a parameter with the QPIE
        representation.

        Args:
          -img: an image to turn into the corresponding quantum circuit
                initializer
          -name: the name to give to the initializer (for displying purposes) """
    n = int(math.log2(img.size))
    qc = QuantumCircuit(n, name=name)
    qc.initialize(_img_to_state_vector(img), range(n))
    return qc

def _img_to_state_vector(img):
    """ Given a square grayscale image it returns a corresponding unitary vector
        representing the equivalent QPIE state.

        Args:
          -img: the nxn grayscale image to turn into a unitary vector

        Raises:
          -ValueError: if the image is not a square """
    if img.shape[0] != img.shape[1]:
        raise ValueError("Image must be square.")
    # The image is turned into a one dimensional vector and then normalized
    state_vector = np.reshape(img, (-1,1)) / np.sqrt(np.square(np.float64(img)).sum())
    return [complex(x) for x in state_vector]


def qc_to_img(qc, backend=Aer.get_backend("qasm_simulator"), shots=100):
    """ Returns an image retrieved from a quantum circuit which uses the QPIE
        representation.

        Args:
          -qc: the quantum circuit encoding the image
          -backend: the backend used to execute the circuit. By default it is
                    executed using a local simulator provided by Aer
          -shots: the number of measurements to perform when recovering the
                  image """
    # Performs measurement on a copy of the circuit and executes it
    out_qc = qc.copy()
    out_qc.measure_all()
    result = execute(out_qc, backend, shots = shots).result()
    return _counts_to_img(result.get_counts())

def _counts_to_img(counts):
    """ Returns an image from the counts resulting from the execution of a
        quantum circuit using the QPIE representation.

        Args:
          -counts: a dictionary {binaryString -> measurements} obtained from the
                   execution of a quantum circuit. It is interpreted as the
                   result of a quantum circuit encoding an image """

    edge = 2**(len(list(counts)[0])//2)
    img = np.zeros((edge, edge))
    for key in counts:
        # Split the binary sequence in half to get the part encoding the row and
        # column
        row = int(key[:len(key)//2], 2)
        col = int(key[len(key)//2:], 2)
        img[row, col] = counts[key]
    img = np.sqrt(img)
    return cv.normalize(img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
