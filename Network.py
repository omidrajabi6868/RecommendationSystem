from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Embedding, Dropout, Activation, Input, Concatenate
from tensorflow.keras import Model
import tensorflow as tf
import datetime
from tensorflow.keras import backend as K


class Net:

    def __init__(self, learning_rate=0.001,
                 epochs=2,
                 embed_vector_size=26,
                 embedding_matrix=None,
                 embedding_trainable=True,
                 vocab_size=None,
                 en_len=20, de_len=20):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.embed_vector_size = embed_vector_size
        self.embedding_trainable = embedding_trainable
        self.en_len = en_len
        self.de_len = de_len
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.model = None
        self.embedding_matrix = embedding_matrix
        self.vocab_size = vocab_size

    def create_bidirectional_model(self, n_units):
        embedding_layer = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.embed_vector_size,
            embeddings_initializer=tf.keras.initializers.Constant(self.embedding_matrix),
            trainable=False
        )
        en_inp = Input(shape=(None,))
        en_type = Input(shape=(None, 4))
        embedded_sequence = embedding_layer(en_inp)
        en_concat = Concatenate()
        encoder = Bidirectional(LSTM(n_units, activation='tanh', return_state=True))
        x_en, state_h_f, state_h_b, state_c_f, state_c_b = encoder(en_concat([embedded_sequence, en_type]))
        state_h_f = Dropout(0.2)(state_h_f)
        state_h_b = Dropout(0.2)(state_h_b)
        state_c_f = Dropout(0.2)(state_c_f)
        state_c_b = Dropout(0.2)(state_c_b)
        encoder_states = [state_h_f, state_h_b, state_c_f, state_c_b]

        de_inp = Input(shape=(None,))
        de_type = Input(shape=(None, 4))
        embedded_sequence = embedding_layer(de_inp)
        de_concat = Concatenate()
        decoder = Bidirectional(LSTM(n_units, activation='tanh', return_sequences=True,
                                     return_state=True))
        x_de, _, _, _, _ = decoder(de_concat([embedded_sequence, de_type]), initial_state=encoder_states)
        decoder_dense = Dense(self.embed_vector_size + 4, activation='linear')
        x_de = decoder_dense(x_de)
        softmax_act = Activation('softmax')
        x1 = x_de[:, :, :-4]
        x2 = softmax_act(x_de[:, :, -4:])
        output = Concatenate()([x1, x2])
        self.model = Model(inputs=[en_inp, en_type, de_inp, de_type], outputs=output)

        # define inference encoder
        encoder_model = Model([en_inp, en_type], encoder_states)
        # define inference decoder
        decoder_state_input_h_f = Input(shape=(n_units,))
        decoder_state_input_h_b = Input(shape=(n_units,))
        decoder_state_input_c_f = Input(shape=(n_units,))
        decoder_state_input_c_b = Input(shape=(n_units,))
        decoder_states_inputs = [decoder_state_input_h_f, decoder_state_input_h_b,
                                 decoder_state_input_c_f, decoder_state_input_c_b]
        embedded_sequence = embedding_layer(de_inp)
        decoder_outputs, state_h_f, state_h_b, state_c_f, state_c_b = decoder(de_concat([embedded_sequence, de_type]),
                                                                              initial_state=decoder_states_inputs)
        decoder_states = [state_h_f, state_h_b, state_c_f, state_c_b]
        decoder_outputs = decoder_dense(decoder_outputs)
        x1 = decoder_outputs[:, :, :-4]
        x2 = softmax_act(decoder_outputs[:, :, -4:])
        decoder_outputs = Concatenate()([x1, x2])
        decoder_model = Model([de_inp, de_type] + decoder_states_inputs, [decoder_outputs] + decoder_states)
        # return all models

        return self.model, encoder_model, decoder_model

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

    def xor_loss(self, y_true, y_pred):
        y_true = tf.convert_to_tensor(y_true, dtype=tf.bool)
        y_pred = tf.convert_to_tensor(y_pred > .5, dtype=tf.bool)
        xor = tf.math.logical_xor(y_true, y_pred)
        xor = tf.cast(xor, dtype=tf.float32)
        error = tf.math.reduce_sum(xor, axis=-1)

        return error

    def call(self, y_true, y_pred):
        cosine_loss_fn = tf.keras.losses.CosineSimilarity()
        mse_loss_fn = tf.keras.losses.MeanSquaredError()
        categorical_loss_fn = tf.keras.losses.CategoricalCrossentropy(
            from_logits=False, reduction=tf.keras.losses.Reduction.NONE
        )
        cosine_loss = cosine_loss_fn(tf.cast(y_true[:, :, :-4], dtype=tf.float64), tf.cast(y_pred[:, :, :-4], tf.float64))
        cat_loss = categorical_loss_fn(y_true[:, :, -4:], y_pred[:, :, -4:])
        mse_loss = mse_loss_fn(y_true[:, :, :-4], y_pred[:, :, :-4])

        loss = (cosine_loss + 1) + tf.cast(mse_loss, dtype=tf.float64) + tf.cast(cat_loss, dtype=tf.float64)

        mask = tf.cast(y_true[:, :, -4] != 1, dtype=tf.float64)
        loss_non_click = loss * mask

        loss_click = loss * (1 - mask)

        loss = loss_click + loss_non_click*10

        mask = tf.cast(y_true[:, :, -2] != 1, dtype=tf.float64)
        loss_non_order = loss * mask

        loss_order = loss * (1 - mask)

        loss = loss_order*(2) + loss_non_order

        mask = tf.cast(y_true[:, :, -1] != 1, dtype=tf.float64)
        loss = loss * mask

        return (tf.reduce_sum(loss)/tf.reduce_sum(mask))

