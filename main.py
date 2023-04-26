import pickle as pickle

import numpy as np
import polars as pl
from Network import Net
from utils import *
from gensim.models import Word2Vec


test_mode = False
encoder_len = 50
decoder_len = 20
chunks = pd.read_json('out/train_sessions.jsonl', lines=True, chunksize=1)

word2vec = Word2Vec.load("Models/word2vec.model")

net = Net(en_len=encoder_len, de_len=decoder_len,
          embed_vector_size=128,
          embedding_matrix=word2vec.wv.vectors,
          vocab_size=len(word2vec.wv.index_to_key))

model, encoder_model, decoder_model = net.create_bidirectional_model(2)
model.summary()

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
recall_results = []
precision_results = []
# Include the epoch in the file name (uses `str.format`)
checkpoint_path_model = "Models/model-{epoch:03d}.ckpt"

checkpoint_dir_model = os.path.dirname(checkpoint_path_model)

if not test_mode:
    num_epochs = 100
    for epoch in range(num_epochs):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
        epoch_recall = tf.keras.metrics.Recall()
        epoch_precision = tf.keras.metrics.Precision()
        i = 0
        for chunk in chunks:
            i += 1
            if i > 100:
                break

            # print(chunk.head(10))
            session_AIDs = chunk.apply((lambda x: [row['aid'] for row in x['events']]), axis=1)
            # print(session_AIDs.head(10))
            session_types = chunk.apply((lambda x: [row['type'] for row in x['events']]), axis=1)
            # print(session_types.head(10))
            # feature_engineering(session_AIDs, session_types)

            X_en, Type_en, X_de, Type_x_de, Y_de, Type_y_de = create_word2vec_training_data(session_AIDs, session_types, word2vec,
                                                                                            interval_size=[encoder_len, decoder_len])

            for j in range(30):
                # Optimize the model
                loss_value, grads = net.grad(model, [X_en, Type_en, X_de, Type_x_de],
                                             np.concatenate([Y_de[..., np.newaxis], Type_y_de], axis=-1))
                net.optimizer.apply_gradients(zip(grads, model.trainable_variables))

            # Track progress
            epoch_loss_avg.update_state(loss_value)  # Add current batch loss
            # Compare predicted label to actual label
            # training=True is needed only if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            # epoch_accuracy.update_state(y_train, model([x_train_en, x_train_de], training=True))
            # epoch_recall.update_state(y_train, model([x_train_en, x_train_de], training=True))
            # epoch_precision.update_state(y_train, model([x_train_en, x_train_de], training=True))

        # End epoch
        train_loss_results.append(epoch_loss_avg.result())
        # train_accuracy_results.append(epoch_accuracy.result())
        # recall_results.append(epoch_recall.result())
        # precision_results.append(epoch_precision.result())

        if epoch % 1 == 0:
            print("\n Epoch {:03d}: Loss: {:.3f}".format(epoch, epoch_loss_avg.result()))

        # if epoch % 1 == 0:
        #     # Save the weights using the `checkpoint_path` format
        #     model.save_weights(checkpoint_path_model.format(epoch=epoch))

if checkpoint_dir_model:
    # latest_model = tf.train.latest_checkpoint(checkpoint_dir_model)
    # net.model.get_layer('embedding').set_weights([word2vec.wv.vectors])
    # net.model.load_weights(latest_model)
    prediction('out/test_sessions.jsonl',
               net.model,
               encoder_model,
               decoder_model,
               word2vec,
               interval_size=[encoder_len, decoder_len])
