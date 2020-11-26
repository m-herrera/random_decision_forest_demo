from PyQt5 import QtCore
import sys
import os
from sklearn.ensemble import RandomForestClassifier  # Random Forest algorithm
from tensorflow.keras.datasets import mnist
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import pyqtSlot
import numpy as np


# Class designed based on the tutorial available at
# https://www.learnpyqt.com/tutorials/bitmap-graphics/

class Window(QWidget):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.canvas_width = 280
        self.canvas_height = 280
        self.initUI()

    def initUI(self):
        self.resize(300, 300)
        self.setWindowTitle("Number recognizer")
        vbox = QVBoxLayout()

        self.canvas = QLabel()
        pixmap = QPixmap(self.canvas_width, self.canvas_height)
        pixmap.fill(QColor('black'))
        self.canvas.setPixmap(pixmap)
        vbox.addWidget(self.canvas)

        hbox = QHBoxLayout()

        predict_btn = QPushButton("Predict", self)
        predict_btn.clicked.connect(self.predict_number)
        hbox.addWidget(predict_btn)

        clear_btn = QPushButton("Clear", self)
        clear_btn.clicked.connect(self.clear)
        hbox.addWidget(clear_btn)

        vbox.addLayout(hbox)

        self.answer = QLabel("Answer: ")
        vbox.addWidget(self.answer)

        self.setLayout(vbox)
        self.show()

    def mouseMoveEvent(self, event):
        painter = QPainter(self.canvas.pixmap())
        pen = QPen()
        pen.setWidth(30)
        pen.setColor(QColor('white'))
        painter.setPen(pen)
        painter.drawPoint(event.x(), event.y())
        painter.end()
        self.update()

    @pyqtSlot()
    def predict_number(self):
        image = self.canvas.pixmap().toImage()
        s = image.bits().asstring(self.canvas_width * self.canvas_height * 4)
        full_res = np.fromstring(s, dtype=np.uint8).reshape((self.canvas_width, self.canvas_height, 4))[:, :, 0]
        low_res = np.array([full_res[::10, ::10].reshape(784)])
        prediction = self.model.predict(low_res)
        self.answer.setText("Answer: " + str(prediction))

    @pyqtSlot()
    def clear(self):
        painter = QPainter(self.canvas.pixmap())
        painter.eraseRect(0, 0, 280, 280)
        self.canvas.pixmap().fill(QColor('black'))
        painter.end()
        self.update()


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape matrices to 784-length vectors for training
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)


    model = RandomForestClassifier(n_estimators=100, n_jobs=10)
    model.fit(x_train, y_train)

    os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    window = Window(model)
    sys.exit(app.exec_())