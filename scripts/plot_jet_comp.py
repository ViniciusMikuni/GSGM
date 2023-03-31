import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM import GSGM
from GSGM_distill import GSGM_distill
import time
import gc
import sys
sys.path.append("JetNet")
from jetnet.evaluation import w1p,w1m,w1efp,cov_mmd,fpnd
from scipy.stats import wasserstein_distance
from plot_class import PlottingConfig


def plot(jet1,jet2,flav1,flav2,nplots,title,plot_folder,is_big):
    for ivar in range(nplots):
        config = PlottingConfig(title,ivar,is_big)
        
        for i,unique in enumerate(np.unique(np.argmax(flavour,-1))):
            mask1 = np.argmax(flav1,-1)== unique
            mask2 = np.argmax(flav2,-1)== unique        
            
            name = utils.names[unique]
            feed_dict = {
                '{}_truth'.format(name):jet1[:,ivar][mask1],
                '{}_gen'.format(name):  jet2[:,ivar][mask2]
            }
            
            if i == 0:                            
                fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                             binning=config.binning,
                                             plot_ratio=False,
                                             reference_name='{}_truth'.format(name),
                                             ylabel= 'Normalized entries',logy=config.logy)
            else:
                fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                             reference_name='{}_truth'.format(name),
                                             plot_ratio=False,
                                             fig=fig,gs=gs,binning=config.binning,
                                             ylabel= 'Normalized entries',logy=config.logy)
        ax0 = plt.subplot(gs[0])     
        ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)

        if not os.path.exists(flags.plot_folder):
            os.makedirs(flags.plot_folder)
        fig.savefig('{}/GSGM_{}_{}.pdf'.format(flags.plot_folder,title,ivar),bbox_inches='tight')


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/GSGM', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='../plots', help='Folder to save results')
    parser.add_argument('--config', default='config_jet.json', help='Training parameters')
    
    parser.add_argument('--model', default='GSGM', help='Type of generative model to load')
    parser.add_argument('--big', action='store_true', default=False,help='Use bigger dataset (150 particles) as opposed to 30 particles')



    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)

    if flags.big:
        labels = utils.labels150
        npart=150
    else:
        labels=utils.labels30
        npart=30

    particles,jets,flavour = utils.DataLoader(flags.data_folder,
                                              labels=labels,
                                              npart=npart,
                                              make_tf_data=False)

        
    model_name = config['MODEL_NAME']
    if flags.big:
        model_name+='_big'

    sample_name = model_name
    idx = 2
    jet_dict = {}
    part_dict = {}
    distill_list = [64,256]
    sample_names = [sample_name + '_d{}'.format(factor) for factor in distill_list]
    sample_names.append(model_name)
    particles,jets= utils.ReversePrep(particles,jets,npart=npart)
    mask = np.argmax(flavour,-1)== idx
    jet_dict['t_truth'] = jets[mask]
    part_dict['t_truth'] = particles[mask].reshape((-1,3))
    
    for isamp,sample_name in enumerate(sample_names):
        with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
            jets_gen = h5f['jet_features'][:]
            particles_gen = h5f['particle_features'][:]
            flavour_gen = jets_gen[:,-1]
            jets_gen = jets_gen[:,:-1]
            
        mask = flavour_gen == idx
        part_dict[sample_name.replace(model_name,'t_gen')] = particles_gen[mask].reshape((-1,3))
        jet_dict[sample_name.replace(model_name,'t_gen')] = jets_gen[mask]
            


        
    for ivar in range(4):
        config = PlottingConfig('jet',ivar,flags.big,one_class=True)
        feed_dict = {}
        for key in jet_dict:
            feed_dict[key] = jet_dict[key][:,ivar]
        

        reference_name='t_truth',
        plot_ratio=True,
        binning=config.binning,

        fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                     binning=config.binning,
                                     plot_ratio=True,
                                     reference_name='t_truth',
                                     ylabel= 'Normalized entries',logy=config.logy)
        
        ax0 = plt.subplot(gs[0])     
        ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)


        if not os.path.exists(flags.plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/GSGM_jet_comp_{}.pdf'.format(flags.plot_folder,ivar),bbox_inches='tight')


    for ivar in range(3):
        config = PlottingConfig('particle',ivar,flags.big,one_class=True)
        feed_dict = {}
        for key in part_dict:
            mask = part_dict[key][:,2]>0.
            feed_dict[key] = part_dict[key][:,ivar][mask]
        

        reference_name='t_truth',
        plot_ratio=True,
        binning=config.binning,

        fig,gs,_ = utils.HistRoutine(feed_dict,xlabel=config.var,
                                     binning=config.binning,
                                     plot_ratio=True,
                                     reference_name='t_truth',
                                     ylabel= 'Normalized entries',logy=config.logy)
        
        ax0 = plt.subplot(gs[0])     
        ax0.set_ylim(top=config.max_y)
        if config.logy == False:
            yScalarFormatter = utils.ScalarFormatterClass(useMathText=True)
            yScalarFormatter.set_powerlimits((100,0))
            ax0.yaxis.set_major_formatter(yScalarFormatter)


        if not os.path.exists(flags.plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/GSGM_part_comp_{}.pdf'.format(flags.plot_folder,ivar),bbox_inches='tight')
