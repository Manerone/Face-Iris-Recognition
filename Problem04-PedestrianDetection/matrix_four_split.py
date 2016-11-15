import numpy as np


class MatrixFourSplit:
    @staticmethod
    def split(matrix_original):
        matrix = np.array(matrix_original)
        result = []
        heigth, width = matrix.shape
        result.append(matrix[0:heigth/2,0:width/2])
        result.append(matrix[heigth/2:,0:width/2])
        result.append(matrix[0:heigth/2,width/2:])
        result.append(matrix[heigth/2:,width/2:])
        return result
