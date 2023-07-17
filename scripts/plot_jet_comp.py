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
import matplotlib.colors as colors

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

def _center(particles):
    #Find the most energetic particle in the event and move it to (0,0)
    particles[:,:,0]-=particles[:,:1,0]
    particles[:,:,1]-=particles[:,:1,1]
    return particles

def _rotate(particles):
    x = particles[:, :, 0]
    y = particles[:, :, 1]
    
    # Convert to polar coordinates
    r = np.sqrt(x**2 + y**2)
    phi =  np.arctan2(y, x)

    theta  = np.pi/2. - np.arctan2(particles[:, 1, 1], particles[:, 1, 0])
    phi += np.expand_dims(theta,-1)
    # Convert back to Cartesian coordinates
    x_rotated = r * np.cos(phi)
    y_rotated = r * np.sin(phi)
    
    # Combine the rotated coordinates back into the array
    rotated_array = np.stack((x_rotated, y_rotated), axis=2)
    rotated_array = np.concatenate((rotated_array, particles[:, :, 2:]), axis=2)    
    
    return rotated_array

def _flip(particles):
    mask = particles[:,2,0]<0
    particles[mask,:,0] = -particles[mask,:,0]
    return particles
def plot2D(feed_dict):
    eta_binning = np.linspace(-0.7,0.7,50)
    phi_binning = np.linspace(-0.7,0.7,50)
    eta_x = 0.5*(eta_binning[:-1] + eta_binning[1:])
    phi_x = 0.5*(phi_binning[:-1] + phi_binning[1:])

    plots = {}
    cmap = plt.get_cmap('RdBu')
    fig,gs = utils.SetGrid(False,figsize=(12,3),npanels = len(feed_dict.keys()),horizontal=True)
    
    for ikey, key in enumerate(feed_dict):        
        amax = np.argsort(-feed_dict[key][:,:,2],1)
        particles = np.take_along_axis(feed_dict[key],np.expand_dims(amax,-1), axis=1)
        particles = _center(particles)
        particles = _rotate(particles)
        particles = _flip(particles)
        
        plots[key],_,_ = np.histogram2d(particles[:,:,1].flatten(), particles[:,:,0].flatten(),
                                    weights=(particles[:,:,2]).flatten(), bins=(eta_binning, phi_binning))
        ax = plt.subplot(gs[ikey])
        Z = plots[key]/particles.shape[0]
        im = ax.pcolor(phi_x, eta_x, Z, cmap=cmap,
                       norm=colors.LogNorm(vmin=0.00001, vmax=Z.max()))
        bar = ax.set_title(utils.name_translate[key],fontsize=12)
        
        #if ikey == len(feed_dict) -1: fig.colorbar(im, ax=ax,label=r'Average energy fraction')

        plt.xticks(fontsize=0)
        plt.yticks(fontsize=0)            
        # if ikey > 0:
        #     plt.xticks(fontsize=0)
        #     plt.yticks(fontsize=0)            
        # else:
        #     ax.set_xlabel(r'$\eta$',fontsize=20)
        #     ax.set_ylabel(r'$\phi$',fontsize=20)
    fig.savefig('../plots/plot_2D.pdf',bbox_inches='tight')
        
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

    particles,jets,flavour,_ = utils.DataLoader(flags.data_folder,
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
    distill_list = [64,512]
    sample_names = [sample_name + '_d{}'.format(factor) for factor in distill_list]
    sample_names = [model_name] + sample_names
    particles,jets= utils.ReversePrep(particles,jets,npart=npart,flavour=np.argmax(flavour,-1))
    mask = np.argmax(flavour,-1)== idx
    jet_dict['t_truth'] = jets[mask]
    part_dict['t_truth'] = particles[mask]
    
    for isamp,sample_name in enumerate(sample_names):
        with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
            jets_gen = h5f['jet_features'][:]
            particles_gen = h5f['particle_features'][:]
            flavour_gen = jets_gen[:,-1]
            jets_gen = jets_gen[:,:-1]
            
        mask = flavour_gen == idx
        part_dict[sample_name.replace(model_name,'t_gen')] = particles_gen[mask]
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
            mask = part_dict[key][:,:,2]>0.
            feed_dict[key] = part_dict[key][:,:,ivar][mask].reshape((-1))
            

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


    
    plot2D(part_dict)
