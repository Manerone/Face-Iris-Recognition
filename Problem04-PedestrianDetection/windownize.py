class Windownize:
    @staticmethod
    def call(image, windowSize=(128, 64), displacement=8):
        heigth, width = windowSize
        img_heigth, img_width = image.shape
        for i in xrange(0, img_heigth - heigth + 1, displacement):
            for j in xrange(0, img_width - width + 1, displacement):
                yield image[i:i + heigth, j:j + width]
