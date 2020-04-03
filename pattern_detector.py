import dlib
import cv2

class Blink:
    def __init__(self, occurred, count):
        self._occured = occurred
        self._count = count

    def setOccurred(self, occurred):
        self._occured = occurred

    def getOccurred(self):
        return self._occured

    def incrementCount(self):
        self._count += 1

    def restartCount(self):
        self._count = 0

    def getCount(self):
        return self._count

class Wink_left:
    def __init__(self, occurred, count):
        self._occured = occurred
        self._count = count

    def setOccurred(self, occurred):
        self._occured = occurred

    def getOccurred(self):
        return self._occured

    def incrementCount(self):
        self._count += 1

    def restartCount(self):
        self._count = 0

    def getCount(self):
        return self._count

class Wink_right:
    def __init__(self, occurred, count):
        self._occured = occurred
        self._count = count

    def setOccurred(self, occurred):
        self._occured = occurred

    def getOccurred(self):
        return self._occured

    def incrementCount(self):
        self._count += 1

    def restartCount(self):
        self._count = 0

    def getCount(self):
        return self._count