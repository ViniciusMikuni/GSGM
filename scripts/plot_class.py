import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick


class PlottingConfig():
    def __init__(self,name,idx,is_big=False,one_class=False):
        self.name=name
        self.idx=idx
        self.big=is_big
        self.one_class=one_class
        self.binning = self.get_binning()
        self.logy=self.get_logy()
        self.var = self.get_name()
        self.max_y = self.get_y()


    def get_name(self):
        if self.name == 'jet':
            name_translate = [
                r'Jet p$_T$ [GeV]',
                r'Jet $\eta$',
                r'Jet mass [GeV]',
                r'Jet particle multiplicity',
            ]
        else:
            name_translate = [
                r'All particles $\eta_{rel}$',
                r'All particles $\phi_{rel}$',
                r'All particles p$_{Trel}$',
            ]

        return name_translate[self.idx]
    
    def get_binning(self):
        if self.name == 'jet':
            binning_dict = {
                0 : np.linspace(600,1800,45),
                1 : np.linspace(-2.,2.,45),
                2 : np.linspace(50,200,30),
                3 : np.linspace(10,30,20),                
            }
            if self.big:
                binning_dict[3] = np.linspace(5,150,50)
        else:
            binning_dict = {
                0 : np.linspace(-0.4,0.4,40),
                1 : np.linspace(-0.4,0.4,40),
                2 : np.linspace(0,0.8,40),
            }
            
        return binning_dict[self.idx]

    def get_logy(self):
        if self.name == 'jet':
            binning_dict = {
                0 : True,
                1 : False,
                2 : False,
                3 : True,                
            }

        else:
            binning_dict = {
                0 : False,
                1 : False,
                2 : True,
            }
            
        return binning_dict[self.idx]



    def get_y(self):
        if self.name == 'jet':
            binning_dict = {
                0 : 0.1,
                1 : 0.7,
                2 : 0.12,
                3 : 5.0,                
            }
            if self.big:
                binning_dict[3] = 0.5
            if self.one_class:
                binning_dict[2] = 0.05

        else:
            binning_dict = {
                0 : 18,
                1 : 18,
                2 : 140,
            }
            if self.one_class:
                binning_dict[0] = 6
                binning_dict[1] = 6
            
        return binning_dict[self.idx]
