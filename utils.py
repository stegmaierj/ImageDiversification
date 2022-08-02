# -*- coding: utf-8 -*-
"""
@article{eschweiler2022diversify,
  title={Probabilistic Image Diversification to Improve Segmentation in 3D Microscopy Image Data},
  author={Dennis Eschweiler and Justus Schock and Johannes Stegmaier},
  journal={MICCAI International Workshop on Simulation and Synthesis in Medical Imaging (SASHIMI)},
  year={2022}
}
"""

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.ndimage import generic_filter



# Auxiliarry print function with additional time stamp
def print_timestamp(msg, args=None):
    
    print('[{0:02.0f}:{1:02.0f}:{2:02.0f}] '.format(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec) +\
          msg.format(*args if not args is None else ''))
        
        
        
# Axuliarry plot function
def plot_image(img_list, title=['',]):
    
    fig, ax = plt.subplots(1,len(img_list), figsize=[5*len(img_list),5])
    for n,img in enumerate(img_list):
        ax[n].imshow(img[img.shape[0]//2,...], cmap='gray', vmin=0, vmax=1)
        ax[n].set_axis_off()
        ax[n].set_title(title[n])
    plt.show()
        
    
    
def draw_weights(strategy='random', num_samples=4, w_i_range=[0.2,1.8], w_v_range=[0,8]):
        
    # Start with neutral element
    mean_weights = [1,]
    variance_weights = [0,]
    
    # Add further weights depending on the chosen parameters
    if strategy == 'structured':
        if num_samples == 1:
            mean_weights = mean_weights + [1]
            variance_weights = variance_weights + [1]
        elif num_samples > 1:
            mean_weights = mean_weights +\
                list(np.linspace(w_i_range[1], w_i_range[0], num_samples))
            variance_weights = variance_weights +\
                list(np.linspace(w_v_range[0], w_v_range[1], num_samples))
    elif strategy == 'random':
        if num_samples > 0:
            mean_weights = mean_weights +\
                list(np.random.uniform(w_i_range[0], w_i_range[1], num_samples))
            variance_weights = variance_weights +\
                list(np.random.uniform(w_v_range[0], w_v_range[1], num_samples))
    elif strategy == 'singleAug':
        if w_i_range[0]==1 and w_i_range[1]==1:
                mean_weights = [1,]*num_samples
        else:
            mean_weights = mean_weights + list(truncnorm.rvs(-3*(1-w_i_range[0])/np.maximum(1-w_i_range[0],w_i_range[1]-1)-1e-5,\
                                                    3*(w_i_range[1]-1)/np.maximum(1-w_i_range[0],w_i_range[1]-1)+1e-5,\
                                                    size=num_samples)/3*np.maximum(1-w_i_range[0],w_i_range[1]-1)+1)
        if w_v_range[0]==0 and w_v_range[1]==0:
                variance_weights = [0,]*num_samples
        else:
            variance_weights = variance_weights + list(truncnorm.rvs(-3*(1-w_v_range[0])/np.maximum(1-w_v_range[0],w_v_range[1]-1)-1e-5,\
                                                    3*(w_v_range[1]-1)/np.maximum(1-w_v_range[0],w_v_range[1]-1)+1e-5,\
                                                    size=num_samples)/3*np.maximum(1-w_v_range[0],w_v_range[1]-1)+1)
    else:
        print('Did not recognize strategy "{0}"... Only neutral element will be retuned.'.format(strategy))
                
    return mean_weights, variance_weights
            

        
# Function for local variance computation
def get_variance(img, downscale=1, variance_window=(5,5,5)):
    
    # create downscales image to prevent computantially intensive processing
    small_img = img[::downscale,::downscale,::downscale]
    
    # create variance image
    std_img = generic_filter(small_img, np.std, size=variance_window)
    
    # rescale variance image
    std_img = np.repeat(std_img, downscale, axis=0)
    std_img = np.repeat(std_img, downscale, axis=1)
    std_img = np.repeat(std_img, downscale, axis=2)
    dim_missmatch = np.array(img.shape)-np.array(std_img.shape)
    if dim_missmatch[0]<0: std_img = std_img[:dim_missmatch[0],...]
    if dim_missmatch[1]<0: std_img = std_img[:,:dim_missmatch[1],:]
    if dim_missmatch[2]<0: std_img = std_img[...,:dim_missmatch[2]]
    
    return std_img.astype(np.float32)



# Reparametrization function for sample generation
def reparametrize(patch, var, mean_weight=1, var_weight=1):
    assert patch.shape==var.shape, 'Patch size and variance size must be the same.'
    
    patch = mean_weight*patch + var_weight*np.random.randn(*var.shape)*var
    patch = np.clip(patch, 0, 1)
    
    return patch    


# Torch implementation of the reparametrization function
def reparametrize_pytorch(patch, var, mean_weight=1, var_weight=1):
    assert patch.shape==var.shape, 'Patch size and variance size must be the same.'
    
    patch = mean_weight*patch + var_weight*torch.randn_like(var)*var
    patch = torch.clamp(patch, 0, 1)
    
    return patch   