import data
import numpy as np

images=data.load_train_images()
print(images.shape)
images_ep = np.expand_dims(images, axis=1) / 255
print(images_ep.shape)

class LeNet:
    def __init__(self,lr=0.1):
        self.lr=lr
        self.fc1=np.zeros(256,200)
    #input_data al-1,wl
    def full_connect(self,self.input_data,fc,front_delta=None,deriv=False):
        if(deriv==False):
            N=input_data.shape[0]
            output_data=np.dot(input_data.reshape(N,-1),fc)
            return output_data
        else:
            back_delta=np.dot(front_delta,fc.T)
            fc+=self.lr*np.dot(front_delta,input_data.T)
            return back_delta,fc
