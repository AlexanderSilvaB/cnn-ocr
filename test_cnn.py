from CNN import *
from Drawer import *

cnn = CNN('models/mnist_cnn.h5')
drawer = Drawer()

def on_key(key):
    if key == 13:
        img = drawer.get(28, 28)
        number, pred = cnn.predict(img)
        print(number)
        print(pred)
        drawer.clear()


drawer.on_key(on_key)
drawer.run()