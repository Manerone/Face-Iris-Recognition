from blocknizer import Blocknizer
from matrix_four_split import MatrixFourSplit
import math


class HOG:
    def __init__(self):
        self.configurations = {
            'blockSize': (16, 16),
            'blockDisplacement': 8
        }
        self._buckets = None

    def calculate(self, orientations, magnitudes):
        hog = []
        blocks = self._blocknize_both(orientations, magnitudes)
        for block_ori, block_mag in blocks:
            for cell_ori, cell_mag in self._get_cells(block_ori, block_mag):
                hog += self._trilinear_interpolation(cell_ori, cell_mag)
        return hog

    def _blocknize_both(self, m1, m2):
        blocks_m1 = self._blocknize(m1)
        blocks_m2 = self._blocknize(m2)
        return zip(blocks_m1, blocks_m2)

    def _blocknize(self, matrix):
        return Blocknizer.call(
            matrix,
            self.configurations['blockSize'],
            self.configurations['blockDisplacement']
        )

    def _get_cells(self, orientations, magnitudes):
        cells_ori = self._split_matrix_in_four(orientations)
        cells_mag = self._split_matrix_in_four(magnitudes)
        return zip(cells_ori, cells_mag)

    def _split_matrix_in_four(self, matrix):
        return MatrixFourSplit.split(matrix)

    def _trilinear_interpolation(self, orientations, magnitudes):
        histogram = {i: 0 for i in xrange(10, 180, 20)}
        height, width = orientations.shape
        for line in xrange(height):
            for col in xrange(width):
                values = self._find_values(
                    orientations[line][col], magnitudes[line][col]
                )
                self._add_dict(histogram, values)
        return self._sort_dict(histogram)

    def _sort_dict(self, dictionary):
        sorted_dict = []
        for key in sorted(dictionary):
            sorted_dict.append(dictionary[key])
        return sorted_dict

    def _add_dict(self, dict1, dict2):
        for key in dict2.keys():
            dict1[key] += dict2[key]

    def _find_values(self, angle_rad, magnitude):
        angle = math.degrees(angle_rad)
        buckets = self._create_buckets()
        values = {}
        if angle >= 170:
            values[170] = magnitude
        elif angle <= 10:
            values[10] = magnitude
        else:
            for bucket in buckets:
                if bucket[0] <= angle <= bucket[1]:
                    dist = bucket[1] - bucket[0]
                    values[bucket[0]] = ((bucket[1] - angle)/dist) * magnitude
                    values[bucket[1]] = ((angle - bucket[0])/dist) * magnitude
                    break
        return values

    def _create_buckets(self):
        if self._buckets is None:
            self._buckets = zip(
                range(10, 180, 20),
                range(30, 180, 20)
            )
        return self._buckets
