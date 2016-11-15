class Blocknizer:
    @staticmethod
    def call(matrix, size=(16, 16), displacement=8):
        heigth, width = matrix.shape
        blockHeigth, blockWidth = size
        for i in xrange(0, heigth - blockHeigth + 1, displacement):
            for j in xrange(0, width - blockWidth + 1, displacement):
                yield matrix[i:i + blockHeigth, j: j + blockWidth]
