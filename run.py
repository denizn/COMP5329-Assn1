import time
import h5py
from mlp import *
import numpy as np

with h5py.File('input/train_128.h5','r') as H:
    data = np.copy(H['data'])
with h5py.File('input/train_label.h5','r') as H:
    label = np.copy(H['label'])

# INITIALIZE MLP WITH BEST PERFORMING ARCHITECTURE
mlp = MLP([128, 256, 256, 10],activation=[None, 'ReLU', 'ReLU','softmax'], dropout=[0.3, 0.3, 0.0, 0])

# START TIMING
start = time.time()

# CHECKPOINT MODEL AND KEEP RECORD OF 
losses_desc256, accuracies_train_desc256, accuracies_test_desc256 = mlp.model_checkpointer(data, label, batch_size=32, momentum=0.9, learning_rate=0.0001,epochs=80)

end = time.time()
# END TIMING

print('Time taken to train and predict: {:.2f} seconds'.format(end-start))
print('Best accuracy achieved: {:.3f} accuracy'.format(mlp.best_accuracy))

plt.plot(accuracies_train_desc256, label='train')
plt.plot(accuracies_test_desc256, label='validation')
plt.tight_layout()
plt.legend()
plt.savefig('accuracy_epoch.png')

# READ TEST DATA
with h5py.File('input/test_128.h5','r') as H:
    test_data = np.copy(H['data'])
    
scaler = StandardScaler().fit(data)
scaled_test_data = scaler.transform(test_data)
test_predictions = mlp.best_model.predict(scaled_test_data)

best_accuracy_desc256 = mlp.best_accuracy

# WRITE PREDICTIONS
with h5py.File('output/Predicted_labels.h5','w') as hdf:
    hdf.create_dataset('labels', data = test_predictions)