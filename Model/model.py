import numpy as np
import matplotlib.pyplot as plt
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from keras.models import Model
from keras.layers import Input
from keras.layers import Conv2D, LeakyReLU, Activation, Concatenate, Dropout, BatchNormalization, Conv2DTranspose
from keras.utils import plot_model
from os import listdir
from numpy import asarray, load
from numpy import vstack
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
from numpy import savez_compressed
from re import X

def discriminator_defn(img_size):
  wt = RandomNormal(stddev = 0.02)
  src_img = Input(shape = img_size)
  target_img = Input(shape = img_size)
  ''' concatenating channel wise'''
  conc_img = Concatenate()([src_img, target_img])

  ''' Discriminator architechture described in the paper :
    C64-C128-C256-C512
    After the last layer, a convolution is applied to map to
    a 1-dimensional output, followed by a Sigmoid function.
    As an exception to the above notation, BatchNorm is not
    applied to the first C64 layer. All ReLUs are leaky, with
    slope 0.2.'''


  l = Conv2D(64, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(conc_img)
  l = LeakyReLU(alpha = 0.2)(l)

  l = Conv2D(128, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(conc_img)
  l = BatchNormalization()(l)
  l = LeakyReLU(alpha = 0.2)(l)

  l = Conv2D(256, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(conc_img)
  l = BatchNormalization()(l)
  l = LeakyReLU(alpha = 0.2)(l)

  l = Conv2D(512, (4, 4), padding = 'same', kernel_initializer = wt)(l)
  l = BatchNormalization()(l)
  l = LeakyReLU(alpha = 0.2)(l)

  l = Conv2D(1, (4, 4), padding = 'same', kernel_initializer = wt)(l)
  output = Activation('sigmoid')(l)

  model = Model(inputs = [src_img, target_img], outputs = output)
  model.summary()
  model.compile(loss = 'binary_crossentropy', optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999), metrics = ['accuracy'])
  return model

disc_model = discriminator_defn((256, 256, 3))

''' encoder-decoder architechture as described in the paper:
    encoder:
    C64-C128-C256-C512-C512-C512-C512-C512
    decoder:
    CD512-CD512-CD512-C512-C256-C128-C64
    All ReLUs in the encoder are leaky, with slope 0.2, while
    ReLUs in the decoder are not leaky '''

def encoder(layer, n, batch_norm = True):
  wt = RandomNormal(stddev = 0.02)
  l = Conv2D(n, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(layer)
  if batch_norm:
    l = BatchNormalization()(l, training = True)
  l = LeakyReLU(alpha = 0.02)(l)
  return l

def decoder(layer, skip, n, dropout = True):
  wt = RandomNormal(stddev = 0.02)
  l = Conv2DTranspose(n, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(layer)
  l = BatchNormalization()(l, training = True)
  if dropout:
    l = Dropout(0.5)(l, training = True)
  l = Concatenate()([l , skip])
  l = Activation('relu')(l)
  return l

def generator(img_size):
  wt = RandomNormal(stddev = 0.02)
  src_img = Input(shape = img_size)

  l1 = encoder(src_img, 64, batch_norm = False)
  l2 = encoder(l1, 128)
  l3 = encoder(l2, 256)
  l4 = encoder(l3, 512)
  l5 = encoder(l4, 512)
  l6 = encoder(l5, 512)
  l7 = encoder(l6, 512)

  b = Conv2D(512, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(l7)
  b = Activation('relu')(b)

  d1 = decoder(b, l7, 512)
  d2 = decoder(d1, l6, 512)
  d3 = decoder(d2, l5, 512)
  d4 = decoder(d3, l4, 512, dropout = False)
  d5 = decoder(d4, l3, 256, dropout = False)
  d6 = decoder(d5, l2, 128, dropout = False)
  d7 = decoder(d6, l1, 64, dropout = False)

  l = Conv2DTranspose(3, (4, 4), strides = (2, 2), padding = 'same', kernel_initializer = wt)(d7)
  output = Activation('tanh')(l)

  model = Model(src_img, output)
  model.summary()
  return model
def GAN(gen, disc, img_size):
  for i in disc.layers:
    if not isinstance(i, BatchNormalization):
      i.trainable = False
  disc.trainable = False
  src_img = Input(shape = img_size)
  gen_output = gen(src_img)
  disc_output = disc([src_img, gen_output])
  model = Model(src_img, [disc_output, gen_output])
  optimizer = Adam(learning_rate = 0.0002, beta_1 = 0.5, beta_2 = 0.999)
  model.compile(loss = ['binary_crossentropy', 'mae'], optimizer = optimizer, loss_weights = [1, 100])
  return model

def load_real_imgs(data, n_samples, patch_size):
  trainA, trainB = data
  i = np.random.randint(0, trainA.shape[0], n_samples)
  X1, X2 = trainA[i], trainB[i]
  y = np.ones((n_samples, patch_size, patch_size, 1))
  return [X1, X2], y

def load_fake_imgs(gen, n_samples, patch_size):
  X = gen.predict(data)
  y = np.zeros((len(X), patch_size, patch_size, 0))
  return X, y
# plot_model(disc_model)

def summarize_performance(step, g_model, dataset, n_samples=3):

	[X_realA, X_realB], _ = load_real_imgs(dataset, n_samples, 1)

	X_fakeB, _ = load_fake_imgs(g_model, X_realA, 1)

	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0

	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + i)
		plt.axis('off')
		plt.imshow(X_realA[i])

	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples + i)
		plt.axis('off')
		plt.imshow(X_fakeB[i])

	for i in range(n_samples):
		plt.subplot(3, n_samples, 1 + n_samples*2 + i)
		plt.axis('off')
		plt.imshow(X_realB[i])

	filename1 = 'plot_%06d.png' % (step+1)
	plt.savefig(filename1)
	plt.close()
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

def train(disc, gen, gan, data, epochs = 100, n_batch = 1):
  n_patch = disc.output_shape[1]
  trainA, trainB = data
  batches_per_epoch = int(len(trainA)/n_batch)
  n_steps = batches_per_epoch * epochs
  for i in range(n_steps):
    [X_realA, X_realB], y_real = load_real_imgs(data, n_batch, n_patch)
    X_fakeB, y_fake = load_fake_imgs(gen, X_realA, n_patch)
    loss1 = disc.train_on_batch([X_realA, X_realB], y_real)
    loss2 = disc.train_on_batch([X_realA, X_fakeB], y_fake)
    gen_loss, _, _ = gan.train_on_batch(X_realA, [y_real, X_realB])
    print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, loss1, loss2, gen_loss))
    if (i+1) % (batches_per_epoch * 10) == 0:
		    summarize_performance(i, gen, data)
