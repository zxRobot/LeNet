# -*-  encoding:utf-8   -*-
import data
import numpy as np
from scipy.signal import convolve2d
from skimage.measure import block_reduce

class LeNet():
    def __init__(self,lr=0.1):
        self.lr=lr
        self.conv1=self.xavier_init(6,1,5,5)
        self.pool1=[2,2]
        self.conv2=self.xavier_init(16,6,5,5)
        self.pool2=[2,2]
        self.fc1=self.xavier_init(256,200,is_fc=True)
        self.fc2=self.xavier_init(200,10,is_fc=True)
    def forward_prop(self,input_data):
        self.l0=np.expand_dims(input_data,axis=1)/255
        self.l1=self.convolution(self.l0,conv1)
        self.l2=self.meanpool(self.l1,self.pool1)
        self.l3=self.convolution(self.l2,conv2)
        self.l4=self.meanpool(self.l3,pool2)
        self.l5=self.fully_connect(self.l4,self.fc1)
        self.l6=self.relu(self.l5)
        self.l7=self.fully_connect(self.l6,self.fc2)
        self.l8=self.relu(self.l7)
        self.l9=self.softmax(self.l8)
        return self.l9
    def backward_prop(self,softmax_out,out_labels):
        l8_delta             = (output_label - softmax_output) / softmax_output.shape[0]
        l7_delta             = self.relu(self.l8, l8_delta, deriv=True)                     
        l6_delta, self.fc2   = self.fully_connect(self.l6, self.fc2, l7_delta, deriv=True)  
        l5_delta             = self.relu(self.l6, l6_delta, deriv=True)                     
        l4_delta, self.fc1   = self.fully_connect(self.l4, self.fc1, l5_delta, deriv=True)  
        l3_delta             = self.mean_pool(self.l3, self.pool2, l4_delta, deriv=True)    
        l2_delta, self.conv2 = self.convolution(self.l2, self.conv2, l3_delta, deriv=True)  
        l1_delta             = self.mean_pool(self.l1, self.pool1, l2_delta, deriv=True)    
        l0_delta, self.conv1 = self.convolution(self.l0, self.conv1, l1_delta, deriv=True)  
    def softmax(self,x):
        y=list()
        for t in x:
            e_t=np.exp(t-np.max(t))
            y.append(e_t/e_t.sum())
        return np.append(y)
    def relu(self,x,front_delta=None,deriv=False):
        if(deriv==False):
            return x*(x>0)
        else:
            return front_delta*1.0*(x>0)
    def fully_connect(self,input_data,kernel,front_delta=None,deriv=False):
        in_nums=input_data.shape[0]
        if(deriv==False):
            return np.dot(input_data.reshape(in_nums,-1),kernel)
        else:
            back_delta=np.dot(front_delta,kernel.T).reshape(in_nums,-1)
            kernel+=self.lr*np.dot(input_data.reshape(in_nums,-1),front_delta)
            return back_delta,kernel

    def meanpool(self,input_data,pool,front_delta=None,deriv=False):
        in_nums,in_dims,in_rows,in_cols=input_data.shape
        p_rows,p_cols=tuple(pool)
        if(deriv==False):
            output_data=np.zeros((in_nums,in_dims,in_rows/p_rows,in_cols/p_cols))
            output_data=block_reduce(input_data,tuple(1,1,p_rows,p_cols),func=mean)
            return output_data
        else:
            back_delta=np.zeros((in_nums,in_dims,in_rows,in_cols))
            back_delta=front_delta.repeat(p_rows,axis=2).repeat(p_cols,axis=3)/(p_rows*p_cols)
            return back_delta
    def convolution(self,input_data,kernel,front_delta=None,deriv=False):
        in_nums,in_dims,in_rows,in_cols=input_data.shape
        ke_nums,ke_dims,ke_rows,ke_cols=kernel.shape
        if(deriv==False):
            output_data=np.zeros((in_nums,ke_nums,in_rows-ke_rows+,in_cols-ke_cols+1))
            for in_nums_index in range(in_nums):
                for ke_nums_index in range(ke_nums):
                    for in_dims_index in range(in_dims):
                        output_data+=convolve2d(input_data[in_nums_index][in_dims_index]\
                            ,kernel[ke_nums_index][in_dims_index],mode='valid')
            return output_data
        else:
            back_delta=np.zeros((in_nums,in_dims,in_rows,in_cols))
            kernel_gradient=np.zeros((ke_nums,ke_dims,ke_rows,ke_cols))
            pad_front_delta=np.pad(front_delta,[(0,0),(0,0),(ke_rows-1,ke_cols-1),(ke_rows-1,ke_cols-1)],mode='constant',constant_values=0)
            for in_nums_index in range(in_nums):
                for ke_nums_index in range(ke_nums):
                    for in_dims_index in range(in_dims):
                        back_delta+=convolve2d(pad_front_delta[in_nums_index][in_dims_index]\
                            ,kernel[ke_nums_index,in_dims_index,::-1,::-1]mode='valid')
                        kernel_gradient=convolve2d(input_data[in_nums_index][in_dims_index],
                        kernel[ke_nums_index][in_dims_index],mode='valid')
            kernel+=self.lr*kernel_gradient
            return back_delta,kernel

    def xavier_init(self,nums,dims,rows=1,cols=1,is_fc=False):
        ratio=np.sqrt(6.0/(nums*rows*cols+dims*rows*cols))
        params=ration*2.0*np.random.random((nums,dims,rows,cols)-1.0)
        if(is_fc==False):
            return params
        else:
            return params.reshape((nums,dims))