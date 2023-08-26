import pickle
from matplotlib import pyplot as plt

with open('trainHistoryDict', "rb") as file_pi:
    history = pickle.load(file_pi)


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.plot(history['f1_score'])
plt.plot(history['val_f1_score'])
plt.show()
