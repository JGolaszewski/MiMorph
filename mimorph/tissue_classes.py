import numpy as np
from collections import UserList
from skimage.color import hed2rgb
import matplotlib.pyplot as plt

class Tissue:
    def __init__(self, he, rgb, mask):
        self.he = he.copy()
        self.rgb = rgb.copy()
        self.mask = mask.copy()

    @property
    def hem(self):
        return self.he[...,0]
    
    @hem.setter
    def hem(self, v):
        self.he[..., 0] = v
    
    @property
    def eos(self):
        return self.he[...,1]
    
    @eos.setter
    def eos(self, v):
        self.he[...,0] = v

    def he_vis(self):
        null = np.zeros_like(self.he[:, :, 0])
        hem_rgb = hed2rgb(np.stack((self.he[:, :, 0], null, null), axis=-1))
        eos_rgb = hed2rgb(np.stack((null, self.he[:, :, 1], null), axis=-1))

        return hem_rgb, eos_rgb
    
    def quick_plot(self):
        plt.figure(figsize=(12, 4))
        plt.subplot(1,4,1)
        plt.axis('off')
        plt.imshow(self.rgb)
        plt.title('Original')
        plt.subplot(1,4,2)
        plt.axis('off')
        plt.imshow(self.mask, cmap='gray')
        plt.title('Mask')
        hem_rgb, eos_rgb = self.he_vis()
        plt.subplot(1,4,3)
        plt.axis('off')
        plt.imshow(hem_rgb)
        plt.title('Hematoxylin')
        plt.subplot(1,4,4)
        plt.axis('off')
        plt.imshow(eos_rgb)
        plt.title('Eosine')

        pass
    

class Slide(UserList):
    def __init__(self, lod, filepath, thumbnail=None, tissues=None):
        self.lod = lod
        self.thumbnail = thumbnail.copy()
        self.filepath = filepath
        
        if tissues is None:
            tissues = []
        super().__init__(tissues)