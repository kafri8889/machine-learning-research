import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
import math

# Eksperimen memprediksi apakah suatu warna yang diberikan gelap atau terang
# berdasarkan luminance dari nilai RGB

def calculateLuminance(rgb: list[int]) -> float:
    """
    get luminance from given rgb color

    :param rgb: rbg from 0 until 255
    :return: luminance from 0f (black) until 1f (white)
    """
    return (0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]) / 255


def generateTrainingData(
        n: int,
        testData: float = 0.25,
        normalize: bool = True
) -> (list[list[int]], list[int], list[list[int]], list[int]):
    """
    :return: list[list[int]], list[int], list[list[int]], list[int]

    x: list of rgb values
    y: boolean 1 or 0

    if normalized, rgb range will be from 0f until 1f
    otherwise rgb range will be from 0f until 255f

    example:

    ```
    xTrain, yTrain, xTest, yTest = generateTrainingData(100, 0.25)
    ```

    The code above will generate 75 data for training and 25 data for test

    """
    xTrain = []
    yTrain = []
    xTest = []
    yTest = []
    for i in range(n):
        rgb = [random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)]
        luminance = calculateLuminance(rgb)
        isLight = 0  # false

        if luminance >= 0.5:
            isLight = 1  # true

        if normalize:
            rgb = list(map(lambda x: x / 255, rgb))

        if i > n * testData:
            xTrain.append(rgb)
            yTrain.append(isLight)
        else:
            xTest.append(rgb)
            yTest.append(isLight)

    return xTrain, yTrain, xTest, yTest


def rgbToHex(rgb: list[int]) -> str:
    """
    rgb range from 0 until 255

    :param rgb:
    :return: hex string
    """
    return '#%02x%02x%02x' % (rgb[0], rgb[1], rgb[2])

class _NeuralNetwork:

    """
    1 Input layer
    1 Hidden layer
    1 Output layer

    Input layer node => 3
    Hidden layer node => 2
    Output layer node => 1
    """

    def __init__(self, learningRate: float = 0.001):
        self._INPUT_NODE = 3
        self._HIDDEN_NODE = 2
        self._OUTPUT_NODE = 1

        # Input layer with hidden layer
        self.weightHidden = []
        # Hidden layer with output layer
        self.weightOutput = []
        self.biasHidden = []
        self.biasOutput = 1
        self.learningRate = learningRate

        # Digunakan untuk menyimpan hasil dari output node di hidden layer dan output layer
        # panjang arraynya sesuai banyaknya node di setiap layer, misalnya di hidden layer ada 2 node,
        # berarti isi dari yHidden ada 2 elemen
        # nilai yang ada di dalam variabel output ini harus sudah di aktivasi menggunakan activation function
        # ini nantinya digunakan di backpropogation
        self.yHidden = []
        self.yOutput = 0

        # Init weightHidden with random value
        for i in range(self._HIDDEN_NODE):
            weight = []
            for j in range(self._INPUT_NODE):
                weight.append(random.random())
            self.weightHidden.append(weight)


        # Init weightOutput with random value
        for i in range(self._HIDDEN_NODE * self._OUTPUT_NODE):
            self.weightOutput.append(random.random())

        # Init biasHidden with 1
        for i in range(self._HIDDEN_NODE):
            self.biasHidden.append(1)


    def relu(self, x):
        return max(0, x)


    def leakyRelu(self, x, a=0.1):
        return max(a * x, x)


    def sigmoid(self, z):
        return 1 / (1 + math.e**(-z))


    def derivativeSigmoid(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))


    def fit(self, x: list[list[int]], y: list[int], epoch: int = 5000):
        """
        x[i] = list[int], where x[i][0] = Red, x[i][1] = Green, x[i][2] = Blue

        y[i] = 1 or 0 where 1 = True, 0 = False

        :param x: list of rgb
        :param y: list of answer
        """

        for e in range(epoch):
            for i in range(len(x)):
                # Feed forward
                yPred = self.predict(x[i])

                # Backward propagation
                error = y[i] - yPred
                errorDelta = error * self.derivativeSigmoid(yPred)

                # Update weight output
                for woi in range(self._HIDDEN_NODE * self._OUTPUT_NODE):
                    self.weightOutput[woi] += errorDelta * self.yHidden[woi] * self.learningRate

                # Update bias output
                self.biasOutput += errorDelta * self.learningRate

                # Update weight hidden
                for whi in range(self._HIDDEN_NODE):
                    derivativeYHidden = self.derivativeSigmoid(self.yHidden[whi])

                    for whj in range(self._INPUT_NODE):
                        inputs = x[i][whj]  # x[current data train index][rgb index]
                        self.weightHidden[whi][whj] += errorDelta * self.weightOutput[whi] * derivativeYHidden * inputs * self.learningRate

                    # Update bias hidden
                    self.biasHidden[whi] += errorDelta * self.weightOutput[whi] * derivativeYHidden * self.learningRate

                # print(self.weightHidden)


    def predict(self, x: list[int]) -> float:
        # Reset nilai dari yHidden
        self.yHidden = []

        # Input layer dengan hidden layer
        for i in range(self._HIDDEN_NODE):
            self.yHidden.append(0)

            # Kalkulasi nilai dari input dengan weight
            for j in range(self._INPUT_NODE):
                self.yHidden[i] += self.weightHidden[i][j] * float(x[j])

            # apply dengan fungsi aktivasi
            self.yHidden[i] = self.relu(self.yHidden[i] + self.biasHidden[i])

        # Hidden layer dengan output layer
        for i in range(self._HIDDEN_NODE * self._OUTPUT_NODE):
            self.yOutput = self.yHidden[i] * self.weightOutput[i]

        self.yOutput = self.sigmoid(self.yOutput + self.biasOutput)

        return self.yOutput


if __name__ == '__main__':
    nn = _NeuralNetwork()

    xTrain, yTrain, xTest, yTest = generateTrainingData(n=300, testData=0.25, normalize=True)

    print()
    print("Eksperimen neural network prediksi kecerahan berdasarkan luminance")

    print()
    print(f"X test: {xTest}")
    print(f"Y test: {yTest}")

    preds = []
    actuals = []
    trueCount = 0
    for i in range(len(xTest)):
        x = xTest[i]
        y = yTest[i]

        output = nn.predict(x)

        preds.append("terang" if output > 0.5 else "gelap")
        actuals.append("terang" if y > 0.5 else "gelap")

        if preds[i] == actuals[i]:
            trueCount += 1


    print()
    print("Hasil sebelum di latih")
    print(f"Prediksi: {preds}")
    print(f"Jawaban : {actuals}")
    print(f"Benar   : {trueCount} dari {len(xTest)}")

    nn.fit(xTrain, yTrain, epoch=5000)

    preds = []
    actuals = []
    trueCount = 0
    for i in range(len(xTest)):
        x = xTest[i]
        y = yTest[i]

        output = nn.predict(x)

        preds.append("terang" if output > 0.5 else "gelap")
        actuals.append("terang" if y > 0.5 else "gelap")

        if preds[i] == actuals[i]:
            trueCount += 1

    print()
    print("Hasil sesudah di latih")
    print(f"Prediksi: {preds}")
    print(f"Jawaban : {actuals}")
    print(f"Benar   : {trueCount} dari {len(xTest)}")

    # # Plotting
    #
    # hexColor = rgbToHex(list(map(lambda x: int(x * 255), x)))
    #
    # figure, (ax1) = plt.subplots(1, 1)
    #
    # ax1.add_patch(
    #     patches.Rectangle(
    #         (0,0),
    #         1,
    #         1,
    #         facecolor=hexColor
    #     )
    # )
    #
    # plt.title("Prediksi apakah warna yg diberikan gelap (0) atau terang (1)")
    # plt.text(0.5 / 2, 0.5, f"Prediksi: {outputStr}, Jawaban: {actualOutputStr}")
    # plt.show()

