from NN import *
from Drawer import *

nn = NN('models/mnist_nn.npy')
drawer = Drawer()

def on_key(key):
    if key == 13:
        img = drawer.get(28, 28)
        number,pred = nn.predict(img)
        print(number)
        print(pred)
        drawer.clear()


drawer.on_key(on_key)
drawer.run()