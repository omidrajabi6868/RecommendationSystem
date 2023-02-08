from Network import Net
from utils import *

chunks = pd.read_json('out/train_sessions.jsonl', lines=True, chunksize=128)

net = Net()
model = net.create_bidirectional_model()
model.summary()

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []
recall_results = []
precision_results = []
# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "Models/cp-{epoch:03d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

num_epochs = 2
for epoch in range(num_epochs):
    epoch_loss_avg = tf.keras.metrics.Mean()
    epoch_accuracy = tf.keras.metrics.BinaryAccuracy()
    epoch_recall = tf.keras.metrics.Recall()
    epoch_precision = tf.keras.metrics.Precision()
    i = 0
    for chunk in chunks:
        # print(i, end=" ")
        i += 1
        if i > 10:
            break
        # print(chunk.head(10))
        session_AIDs = chunk.apply((lambda x: [row['aid'] for row in x['events']]), axis=1)
        # print(session_AIDs.head(10))
        session_types = chunk.apply((lambda x: [row['type'] for row in x['events']]), axis=1)
        # print(session_types.head(10))
        x_train, y_train = create_trainig_data(session_AIDs, session_types, interval_size=20)
        # Optimize the model
        loss_value, grads = net.grad(model, x_train, y_train)
        net.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track progress
        epoch_loss_avg.update_state(loss_value)  # Add current batch loss
        # Compare predicted label to actual label
        # training=True is needed only if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        epoch_accuracy.update_state(y_train, model(x_train, training=True))
        epoch_recall.update_state(y_train, model(x_train, training=True))
        epoch_precision.update_state(y_train, model(x_train, training=True))

    # End epoch
    train_loss_results.append(epoch_loss_avg.result())
    train_accuracy_results.append(epoch_accuracy.result())
    recall_results.append(epoch_recall.result())
    precision_results.append(epoch_precision.result())

    if epoch % 1 == 0:
        print("\n Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}, Recall: {:.3%}, Precision: {:.3%}".format(epoch,
                                                                                                          epoch_loss_avg.result(),
                                                                                                          epoch_accuracy.result(),
                                                                                                          epoch_recall.result(),
                                                                                                          epoch_precision.result()))

    if epoch % 10 == 0:
        # Save the weights using the `checkpoint_path` format
        model.save_weights(checkpoint_path.format(epoch=epoch))

latest = tf.train.latest_checkpoint(checkpoint_dir)
net.model.load_weights(latest)
prediction('out/test_sessions.jsonl', net.model)
