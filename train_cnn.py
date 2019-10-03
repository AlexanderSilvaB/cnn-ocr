from CNN import *

cnn = CNN('models/mnist_cnn.h5', retrain = False)
acc, elapsed = cnn.train(epochs = 12, batch_size = 128)
cnn.save()
print('Acuraccy: %f' % acc)
print('Total elapsed time: %fs' % elapsed)