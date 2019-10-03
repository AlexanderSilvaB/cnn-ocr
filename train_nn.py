from NN import *

nn = NN('models/mnist_nn.npy', retrain = False)
acc, elapsed = nn.train(epochs = 30, alpha=0.01)
nn.save()
print('Acuraccy: %f' % acc)
print('Total elapsed time: %fs' % elapsed)