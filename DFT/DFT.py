# For this part of the assignment, please implement your own code for all computations,
# Do not use inbuilt functions like fft from either numpy, opencv or other libraries

# Anh (Tony) Nguyen - 1596895

# Import libraries
import numpy as np

# Create class DFT
class DFT:

# Use static method, not non-stastic
# Helper function apply_to_matrix
    @staticmethod
    def apply_to_matrix(matrix, func, dtype):
        mat = np.zeros(matrix.shape, dtype=dtype)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i, j] = func(matrix, i, j)
        return mat

# Helper function fft
    @staticmethod
    def fft(matrix, x, y):
        (M, N) = matrix.shape
        total = 0
        for m in range(M):
            for n in range(N):
                period = (x * m / M) + (y * n / N)
                exponent = -2j * np.pi * period
                result = matrix[m, n] * np.exp(exponent)
                total += result
        return total

# Helper function ifft
    @staticmethod
    def ifft(matrix, x, y):
        (M, N) = matrix.shape
        total = 0
        for m in range(M):
            for n in range(N):
                period = (x * m / M) + (y * n / N)
                exponent = 2j * np.pi * period
                result = matrix[m, n] * np.exp(exponent)
                total += result
        return total / (M * N)

    def forward_transform(self, matrix):
        """Computes the forward Fourier transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a complex matrix representing fourier transform"""
        return DFT.apply_to_matrix(matrix, DFT.fft, complex)

        #return matrix

    def inverse_transform(self, matrix):
        """Computes the inverse Fourier transform of the input matrix
        matrix: a 2d matrix (DFT) usually complex
        takes as input:
        returns a complex matrix representing the inverse fourier transform"""
        return DFT.apply_to_matrix(matrix, DFT.ifft, complex)

        #return matrix


    def discrete_cosine_tranform(self, matrix):
        """Computes the discrete cosine transform of the input matrix
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing discrete cosine transform"""
        return DFT.apply_to_matrix(matrix, DFT.fft, complex).real

        #return matrix


    def magnitude(self, matrix):
        """Computes the magnitude of the DFT
        takes as input:
        matrix: a 2d matrix
        returns a matrix representing magnitude of the dft"""
        mag_func = lambda mat, i, j: (mat[i, j].real ** 2 + mat[i, j].imag ** 2) ** 0.5

        return DFT.apply_to_matrix(matrix, mag_func, np.float)

        #return matrix