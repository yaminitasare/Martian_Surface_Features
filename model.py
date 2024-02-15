import tensorflow as tf

def create_model():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(64,(3,3),padding = 'same', activation = tf.nn.relu, input_shape = (227,227,1)),
            tf.keras.layers.MaxPooling2D((2,2),strides = 2),
            tf.keras.layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D((2, 2), strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(8, activation='softmax')
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model
