import dataset as ds 
import numpy as np

x_data,y_data = ds.load_data()
print(x_data.shape)
print(y_data.shape)
# print(x_test.shape)
# print(y_test.shape)

np.save('../data/x_data.npy', x_data)
np.save('../data/y_data.npy', y_data)
# np.save('x_test.npy', x_test)
# np.save('y_test.npy', y_test)
