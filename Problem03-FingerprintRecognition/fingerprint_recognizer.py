from processed_image import ProcessedImage


class FingerprintRecognizer:
    def __init__(self, trainingImages, trainingLabels):
        self.trainingSet = []
        for image, label in zip(trainingImages, trainingLabels):
            self.trainingSet.append(
                ProcessedImage(image, label)
            )

    def predict(self, image):
        test_image = ProcessedImage(image)
        min_s_index = 0
        min_s = 0
        for index, train_image in enumerate(self.trainingSet):
            s = calculate_s(train_image, test_image)
            if min_s < s:
                min_s_index = index
                min_s = s
        label = self.trainingSet[min_s_index].label
        return label
