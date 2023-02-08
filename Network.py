from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Embedding, GRU, Dropout
import tensorflow as tf
import datetime
from tensorflow.keras import backend as K


class Net:

    def __init__(self, learning_rate=0.001,
                 epochs=2,
                 embed_vector_size=26,
                 embedding_trainable=True,
                 seq_len=20):

        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embed_vector_size = embed_vector_size
        self.embedding_trainable = embedding_trainable
        self.seq_len = seq_len
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = None

    def create_bidirectional_model(self):
        model = tf.keras.Sequential()
        model.add(Bidirectional(LSTM(512, return_sequences=True, activation='relu'),
                                input_shape=(self.seq_len, self.embed_vector_size)))
        model.add(Dropout(0.2))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.embed_vector_size, activation='sigmoid'))

        # opt = tf.keras.optimizers.Adam(self.learning_rate)
        # loss = CustomNonPaddingTokenLoss()
        # model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])

        self.model = model

        return model

    def train_model(self, x, y):
        log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
        path_checkpoint = f"Models/final_model.h5"
        es_callback = tf.keras.callbacks.EarlyStopping(monitor="val_loss", min_delta=0, patience=50)
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=2, min_lr=0.0001)

        modelckpt_callback = tf.keras.callbacks.ModelCheckpoint(
            monitor="val_loss",
            filepath=path_checkpoint,
            verbose=1,
            save_best_only=True,
            save_weights_only=True)

        self.model.fit(x, y,
                       batch_size=512,
                       validation_split=0.05,
                       epochs=self.epochs,
                       callbacks=[modelckpt_callback, es_callback, tensorboard_callback, reduce_lr])

        return self.model

    def grad(self, model, inputs, targets):
        with tf.GradientTape() as tape:
            loss_value = self.loss(inputs, targets, training=True)
        return loss_value, tape.gradient(loss_value, model.trainable_variables)

    def loss(self, x, y, training):
        # training=training is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        y_ = self.model(x, training=training)
        loss_object = CustomNonPaddingTokenLoss()
        return loss_object(y_true=y, y_pred=y_)


class CustomNonPaddingTokenLoss(tf.keras.losses.Loss):
    def __init__(self, name="custom_ner_loss"):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss_fn = tf.keras.losses.BinaryCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        loss = loss_fn(y_true, y_pred)
        mask1 = tf.cast((y_true[:, :, -1] != 1), dtype=tf.float32)
        mask2 = tf.cast((y_true[:, :, -4] != 1), dtype=tf.float32) * 1e+1
        loss = loss * mask1
        loss += loss * mask2
        return tf.reduce_sum(loss) / tf.reduce_sum(mask1)

