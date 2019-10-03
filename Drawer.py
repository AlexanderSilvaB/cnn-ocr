import cv2
import numpy as np

class Drawer:
    def __init__(self, width = 200, height = 200):
        self.img = np.zeros((height, width,1), np.uint8)
        self.fn = None
        self.radius = 5

    def on_key(self, fn):
        self.fn = fn

    def clear(self):
        self.img = np.zeros(self.img.shape, np.uint8)

    def get(self, width = 0, height = 0):
        if width <= 0:
            width = self.img.shape[1]
        if height <= 0:
            height = self.img.shape[0]

        
        if width != self.img.shape[1] or height != self.img.shape[0]:
            dim = (width, height)
            img = cv2.resize(self.img, dim, interpolation = cv2.INTER_AREA)
            cv2.imshow('www', img)
        else:
            img = self.img

        return img

    def run(self, name = 'drawer'):
        self.drawing = False
        self.ix = -1
        self.iy = -1

        cv2.namedWindow(name)
        cv2.setMouseCallback(name, self.__on_mouse)
        while(1):
            cv2.imshow(name, self.img)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
            elif key != 0 and key != 255 and self.fn != None:
                self.fn(key)

        cv2.destroyAllWindows()

    def __on_mouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix = x
            self.iy = y

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing == True:
                cv2.circle(self.img ,(x,y), self.radius, (255,0,0), -1)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cv2.circle(self.img, (x,y), self.radius, (255,0,0), -1)