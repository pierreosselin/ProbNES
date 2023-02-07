import numpy as np
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
    x = np.loadtxt('data/airfoil_self_noise.dat')
    scale= StandardScaler()
    scaled_data = scale.fit_transform(x)
    np.save("./data/airfoil_self_noise.npy", scaled_data)