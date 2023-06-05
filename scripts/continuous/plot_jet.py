import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib import gridspec
import argparse
import h5py as h5
import os
import utils
import tensorflow as tf
from GSGM_uniform import GSGM
import time
import gc
import sys
sys.path.append("JetNet")
from jetnet.evaluation import w1p,w1m,w1efp,cov_mmd,fpnd
from scipy.stats import wasserstein_distance
from plot_class import PlottingConfig

def W1(
        jet1,
        jet2,
        num_batches = 10,
        return_std = True,
        num_eval=50000,
):

    w1s = []
    
    for j in range(num_batches):
        rand1 = np.random.choice(len(jet1), size=num_eval,replace=True)
        rand2 = np.random.choice(len(jet2), size=num_eval,replace=True)

        rand_sample1 = jet1[rand1]
        rand_sample2 = jet2[rand2]

        w1 = [wasserstein_distance(rand_sample1, rand_sample2)]
        w1s.append(w1)
        
    means = np.mean(w1s, axis=0)
    stds = np.std(w1s, axis=0)
    return means, stds

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

        if not os.path.exists(plot_folder):
            os.makedirs(plot_folder)
        fig.savefig('{}/GSGM_{}_{}.pdf'.format(plot_folder,title,ivar),bbox_inches='tight')


if __name__ == "__main__":
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


    utils.SetStyle()


    parser = argparse.ArgumentParser()

    parser.add_argument('--data_folder', default='/global/cfs/cdirs/m3929/GSGM', help='Folder containing data and MC files')
    parser.add_argument('--plot_folder', default='plots', help='Folder to save results')
    parser.add_argument('--config', default='config_jet.json', help='Training parameters')
    
    parser.add_argument('--model', default='GSGM', help='Type of generative model to load')
    parser.add_argument('--test', action='store_true', default=False,help='Test if inverse transform returns original data')
    parser.add_argument('--big', action='store_true', default=False,help='Use bigger dataset (150 particles) as opposed to 30 particles')
    parser.add_argument('--sample', action='store_true', default=False,help='Sample from the generative model')


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

    if flags.test:
        particles_gen,jets_gen,flavour_gen = utils.SimpleLoader(flags.data_folder,labels=labels)
    else:
        
        model_name = config['MODEL_NAME']
        if flags.big:
            model_name+='_big'

        sample_name = model_name



        if flags.sample:            
            model = GSGM(config=config,npart=npart)
            checkpoint_folder = 'checkpoints_{}/checkpoint'.format(model_name)
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()

            particles_gen = []
            jets_gen = []

            nsplit = 5
            split_part = np.array_split(jets,nsplit)
            for i,split in enumerate(np.array_split(flavour,nsplit)):
                # if i>0:break
                #,split_part[i]
                p,j = model.generate(split,split_part[i])
                particles_gen.append(p)
                jets_gen.append(j)
    
            particles_gen = np.concatenate(particles_gen)
            jets_gen = np.concatenate(jets_gen)
            
            particles_gen,jets_gen= utils.ReversePrep(particles_gen,
                                                      jets_gen,
                                                      npart=npart,

            )
            jets_gen = np.concatenate([jets_gen,np.expand_dims(np.argmax(flavour,-1),-1)[:jets_gen.shape[0]]],-1)

            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"w") as h5f:
                dset = h5f.create_dataset("particle_features", data=particles_gen)
                dset = h5f.create_dataset("jet_features", data=jets_gen)
                
        else:
            with h5.File(os.path.join(flags.data_folder,sample_name+'.h5'),"r") as h5f:
                jets_gen = h5f['jet_features'][:]
                particles_gen = h5f['particle_features'][:]
                
        flavour_gen = jets_gen[:,-1]
        assert np.all(flavour_gen == np.argmax(flavour,-1)), 'The order between the particles dont match'
        jets_gen = jets_gen[:,:-1]
            
    particles,jets= utils.ReversePrep(particles,jets,npart=npart)
    plot(jets,jets_gen,flavour,flavour,title='jet',
         nplots=4,plot_folder=flags.plot_folder,is_big=flags.big)
    
    print("Calculating metrics")

    with open(sample_name+'.txt','w') as f:
        for unique in np.unique(np.argmax(flavour,-1)):
            mask = np.argmax(flavour,-1)== unique
            print(utils.names[unique])
            f.write(utils.names[unique])
            f.write("\n")
            mean_mass,std_mass = w1m(particles[mask], particles_gen[mask])
            print("W1M",mean_mass,std_mass)
            f.write("{:.2f} $\pm$ {:.2f} & ".format(1e3*mean_mass,1e3*std_mass))            
            mean,std = w1p(particles[mask], particles_gen[mask])
            print("W1P: ",np.mean(mean),mean,np.mean(std))
            f.write("{:.2f} $\pm$ {:.2f} & ".format(1e3*np.mean(mean),1e3*np.mean(std)))
            mean_efp,std_efp = w1efp(particles[mask], particles_gen[mask])
            print("W1EFP",np.mean(mean_efp),np.mean(std_efp))
            f.write("{:.2f} $\pm$ {:.2f} & ".format(1e5*np.mean(mean_efp),1e5*np.mean(std_efp)))
            if flags.big or 'w' in utils.names[unique] or 'z' in utils.names[unique]:
                #FPND only defined for 30 particles and not calculated for W and Z
                pass
            else:
                fpnd_score = fpnd(particles_gen[mask], jet_type=utils.names[unique])
                print("FPND", fpnd_score)
                f.write("{:.2f} & ".format(fpnd_score))
                
            cov,mmd = cov_mmd(particles[mask],particles_gen[mask],num_eval_samples=1000)
            print("COV,MMD",cov,mmd)
            f.write("{:.2f} & {:.2f} \\\\".format(cov,mmd))
            f.write("\n")
            

        for unique in np.unique(np.argmax(flavour,-1)):
            mask = np.argmax(flavour,-1)== unique
            
            print("Jet "+utils.names[unique])
            f.write("Jet "+utils.names[unique])
            f.write("\n")
            for i in range(jets_gen.shape[-1]):
                mean,std=W1(jets_gen[:,i],jets[:,i])
                print("W1J {:.2f}: {:.2f}".format(i,mean[0],std[0]))
                f.write("{:.3f} $\pm$ {:.3f} & ".format(np.mean(mean),np.mean(std)))
            f.write("\\ \n")
        
    flavour = np.tile(np.expand_dims(flavour,1),(1,particles_gen.shape[1],1)).reshape((-1,flavour.shape[-1]))

    particles_gen=particles_gen.reshape((-1,3))
    mask_gen = particles_gen[:,2]>0.
    particles_gen=particles_gen[mask_gen]
    particles=particles.reshape((-1,3))
    mask = particles[:,2]>0.
    particles=particles[mask]
    
    flavour_gen = flavour[mask_gen]
    flavour = flavour[mask]


    plot(particles,particles_gen,
         flavour,flavour_gen,
         title='part',
         nplots=3,
         plot_folder=flags.plot_folder,
         is_big=flags.big)


