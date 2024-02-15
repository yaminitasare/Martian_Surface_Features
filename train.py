from keras.preprocessing.image import ImageDataGenerator

def train_model(model, train_images, train_labels, batch_size, num_epochs):
    generator = ImageDataGenerator(rotation_range=0, zoom_range=0, width_shift_range=0,
                                   height_shift_range=0, shear_range=0, horizontal_flip=False,
                                   fill_mode="nearest")
    
    model.fit_generator(generator.flow(train_images, train_labels, batch_size=batch_size), epochs=num_epochs)
