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
                0 : np.linspace(700,1800,15),
                1 : np.linspace(-2.,2.,15),
                2 : np.linspace(50,200,15),
                3 : np.linspace(10,30,15),                
            }
            if self.big:
                binning_dict[3] = np.linspace(5,150,20)
        else:
            binning_dict = {
                0 : np.linspace(-0.5,0.5,20),
                1 : np.linspace(-0.5,0.5,20),
                2 : np.linspace(0,0.8,20),
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
                2 : 0.07,
                3 : 5.0,                
            }
            if self.big:
                binning_dict[3] = 0.5
            if self.one_class:
                binning_dict[2] = 0.03

        else:
            binning_dict = {
                0 : 12,
                1 : 12,
                2 : 120,
            }
            
        return binning_dict[self.idx]
