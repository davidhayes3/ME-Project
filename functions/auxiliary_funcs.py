
# Function to save models
def save_models(path, encoder=None, generator=None):
    if encoder is not None:
        encoder.save_weights(path + '_encoder.h5')

    if generator is not None:
        generator.save_weights(path + '_generator.h5')