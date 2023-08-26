import matplotlib
import numpy as np

from alalyser import get_key_array, get_align_dist
import seaborn as sns


if __name__ == '__main__':
    BASE_PATH = 'D:\\VolodymyrRushchak\\projects\\pianocamera\\dataset'
    arr = get_key_array(BASE_PATH)
    freqs = np.unique(arr, return_counts=True)[1]
    print(freqs)
    sns.displot(data=arr, bins=51)

    matplotlib.pyplot.show()
