import tensorflow as tf


def create_model(lr):
    dp = 0.2 # define droppout parametre
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1024,kernel_initializer='normal', activation=tf.nn.relu, input_shape=(1280,)),
        # # tf.keras.layers.Dense(4096,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(1024,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(2048,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),

        # tf.keras.layers.Dense(1024,kernel_initializer='normal', activation=tf.nn.relu),
        # # tf.keras.layers.Dense(1024,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(512,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(512,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(256,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(256,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(128,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(128,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(64,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(64,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(32,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(64,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(16,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dense(64,kernel_initializer='normal', activation=tf.nn.relu),
        tf.keras.layers.Dropout(dp),
        
        # tf.keras.layers.Dense(8,kernel_initializer='normal', activation=tf.nn.relu),
        # # tf.keras.layers.Dense(64,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dropout(dp),

        # tf.keras.layers.Dense(4,kernel_initializer='normal', activation=tf.nn.relu),
        # # tf.keras.layers.Dense(64,kernel_initializer='normal', activation=tf.nn.relu),
        # tf.keras.layers.Dropout(dp),

        tf.keras.layers.Dense(2,kernel_initializer='normal', activation=tf.nn.softmax)
    ])
    opt = tf.keras.optimizers.Adam(lr)
    # opt = tf.keras.optimizers.SGD(lr=lr, momentum=0.2)

    model.compile(optimizer= opt,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

