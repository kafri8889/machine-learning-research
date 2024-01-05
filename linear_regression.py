import matplotlib.pyplot as plt
import numpy as np
import random

PLOT_WIDTH = 800
PLOT_HEIGHT = 800

class Point:

    def __init__(self):
        self.x = random.randint(0, PLOT_WIDTH)
        self.y = random.randint(0, PLOT_HEIGHT)
        self.label = 0

        if self.x > self.y:
            self.label = 1
        else:
            self.label = -1

    def __str__(self):
        return f"Point(x: {self.x}, y: {self.y}, label: {self.label})"


class Perceptron:
    '''
    2 input, 1 hidden layer
    '''

    def __init__(self):
        self.input = []
        self.weight = []
        self.learningRate = 0.001

        for i in range(2):
            self.weight.append(random.random())

    # Activation function
    def sign(self, n: int) -> int:
        if n >= 0:
            return 1
        else:
            return -1

    def fit(self, inputs: list[float], target: int):
        # Calculate error
        yPred = self.predict(inputs)
        err = target - yPred

        # Update weight
        for i in range(len(self.weight)):
            self.weight[i] += err * inputs[i] * self.learningRate

    def predict(self, inputs: list[float]) -> int:
        self.input = inputs

        sum = 0
        for i in range(len(self.weight)):
            sum += self.input[i] * self.weight[i]

        out = self.sign(sum)
        return out


if __name__ == '__main__':
    trainedPerceptron = Perceptron()
    untrainedPerceptron = Perceptron()

    # Color gray: warna abu2 tandanya belum diprediksi
    points = []
    # Color red: warna merah tandanya prediksinya salah
    pointsRedTrained = []
    pointsRedUntrained = []
    # Color green: warna ijo tandanya prediksinya bener
    pointsGreenTrained = []
    pointsGreenUntrained = []

    trainedPlot = plt.subplot(1,2,1)
    untrainedPlot = plt.subplot(1,2,2)

    trainedPlot.set_title("Trained perceptron")
    untrainedPlot.set_title("Untrained perceptron")

    # Draw line
    trainedPlot.axline((0, 0), (400, 400))
    untrainedPlot.axline((0, 0), (400, 400))

    plt.ion()
    plt.show()

    for i in range(200):
        points.append(Point())

    # Draw initial points
    trainedPlot.scatter(x=list(map(lambda point: point.x, points)), y=list(map(lambda point: point.y, points)), c="gray")
    untrainedPlot.scatter(x=list(map(lambda point: point.x, points)), y=list(map(lambda point: point.y, points)), c="gray")

    # Train neural network
    for point in points:
        x = [point.x, point.y]
        y = point.label

        # Comment code dibawah untuk melihat perbedaan
        trainedPerceptron.fit(x, y)

        yPredTrained = trainedPerceptron.predict(x)
        yPredUntrained = untrainedPerceptron.predict(x)

        print(f"Trained => Pred: {yPredTrained}, actual: {y}")
        print(f"Untrained => Pred: {yPredUntrained}, actual: {y}")

        if yPredTrained == y:
            pointsGreenTrained.append(point)
        else:
            pointsRedTrained.append(point)

        if yPredUntrained == y:
            pointsGreenUntrained.append(point)
        else:
            pointsRedUntrained.append(point)

        trainedPlot.scatter(x=list(map(lambda point: point.x, pointsGreenTrained)), y=list(map(lambda point: point.y, pointsGreenTrained)),c="g")
        trainedPlot.scatter(x=list(map(lambda point: point.x, pointsRedTrained)), y=list(map(lambda point: point.y, pointsRedTrained)), c="r")
        untrainedPlot.scatter(x=list(map(lambda point: point.x, pointsGreenUntrained)), y=list(map(lambda point: point.y, pointsGreenUntrained)),c="g")
        untrainedPlot.scatter(x=list(map(lambda point: point.x, pointsRedUntrained)), y=list(map(lambda point: point.y, pointsRedUntrained)), c="r")

        plt.draw()
        plt.pause(0.001)

    plt.show(block=True)


