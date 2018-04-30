from cifar10_models import deterministic_encoder_model, generator_model

encoder = deterministic_encoder_model()
encoder.summary()

generator = generator_model()
generator.summary()

from keras.utils.vis_utils import plot_model
graph1 = plot_model(encoder, to_file='cifar10_encoder.png', show_shapes=True)
graph2 = plot_model(generator, to_file='cifar10_generator.png', show_shapes=True)