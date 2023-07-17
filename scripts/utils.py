import json, yaml
import os
import h5py as h5
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mtick
from sklearn.utils import shuffle
import tensorflow as tf
from keras.utils.np_utils import to_categorical
#import energyflow as ef

np.random.seed(0) #fix the seed to keep track of validation split

line_style = {
    'true':'dotted',
    'gen':'-',
    'Geant':'dotted',
    'GSGM':'-',
    'q_truth':'-',
    'q_gen':'dotted',
    'g_truth':'-',
    'g_gen':'dotted',
    't_truth':'-',
    't_gen':'dotted',
    'w_truth':'-',
    'w_gen':'dotted',
    'z_truth':'-',
    'z_gen':'dotted',

    't_gen_d64':'dotted',
    't_gen_d256':'dotted',
    't_gen_d512':'dotted',

    'toy_truth':'dotted',
    'toy_gen':'-',
    
}

colors = {
    'true':'black',
    'gen':'#7570b3',
    'Geant':'black',
    'GSGM':'#7570b3',

    'q_truth':'#7570b3',
    'q_gen':'#7570b3',
    'g_truth':'#d95f02',
    'g_gen':'#d95f02',
    't_truth':'#1b9e77',
    't_gen':'#1b9e77',
    'w_truth':'#e7298a',
    'w_gen':'#e7298a',
    'z_truth':'black',
    'z_gen':'black',

    't_gen_d64':'red',
    't_gen_d256':'blue',
    't_gen_d512':'blue',
    'toy_truth':'red',
    'toy_gen':'blue',
}

name_translate={
    'true':'True distribution',
    'gen':'Generated distribution',
    'Geant':'Geant 4',
    'GSGM':'Graph Diffusion',

    'q_truth':'Sim.: q',
    'q_gen':'FPCD: q',
    'g_truth':'Sim.: g',
    'g_gen':'FPCD: g',
    't_truth':'Sim.: top',
    't_gen':'FPCD: top',
    'w_truth':'Sim.: W',
    'w_gen':'FPCD: W',
    'z_truth':'Sim.: Z',
    'z_gen':'FPCD: Z',

    't_gen_d64':'FPCD: top 8 steps',
    't_gen_d256':'FPCD: top 2 steps',
    't_gen_d512':'FPCD: top 1 step',

    'toy_truth':'toy true',
    'toy_gen':'toy gen',
}

names = ['g','q','t','w','z']

labels30 = {
    'g.hdf5':0,
    'q.hdf5':1,
    't.hdf5':2,
    'w.hdf5':3,
    'z.hdf5':4,
}

labels150 = {
    'g150.hdf5':0,
    'q150.hdf5':1,
    't150.hdf5':2,
    'w150.hdf5':3,
    'z150.hdf5':4,
}


nevts = -1
num_classes = 5
num_classes_eval = 5


def SetStyle():
    from matplotlib import rc
    rc('text', usetex=True)

    import matplotlib as mpl
    rc('font', family='serif')
    rc('font', size=22)
    rc('xtick', labelsize=15)
    rc('ytick', labelsize=15)
    rc('legend', fontsize=15)

    # #
    mpl.rcParams.update({'font.size': 19})
    #mpl.rcParams.update({'legend.fontsize': 18})
    mpl.rcParams['text.usetex'] = False
    mpl.rcParams.update({'xtick.labelsize': 18}) 
    mpl.rcParams.update({'ytick.labelsize': 18}) 
    mpl.rcParams.update({'axes.labelsize': 18}) 
    mpl.rcParams.update({'legend.frameon': False}) 
    mpl.rcParams.update({'lines.linewidth': 2})
    
    import matplotlib.pyplot as plt
    import mplhep as hep
    hep.set_style(hep.style.CMS)
    
    # hep.style.use("CMS") 

def SetGrid(ratio=True,figsize=(9, 9),horizontal=False,npanels = 3):
    fig = plt.figure(figsize=figsize)
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
    elif horizontal:
        gs = gridspec.GridSpec(1, npanels) 
        gs.update(wspace=0.0, hspace=0.025)
    else:
        gs = gridspec.GridSpec(1, 1)
    return fig,gs



        
def PlotRoutine(feed_dict,xlabel='',ylabel='',reference_name='gen'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"
    
    fig,gs = SetGrid() 
    ax0 = plt.subplot(gs[0])
    plt.xticks(fontsize=0)
    ax1 = plt.subplot(gs[1],sharex=ax0)

    for ip,plot in enumerate(feed_dict.keys()):
        if 'steps' in plot or 'r=' in plot:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,marker=line_style[plot],color=colors[plot],lw=0)
        else:
            ax0.plot(np.mean(feed_dict[plot],0),label=plot,linestyle=line_style[plot],color=colors[plot])
        if reference_name!=plot:
            ratio = 100*np.divide(np.mean(feed_dict[reference_name],0)-np.mean(feed_dict[plot],0),np.mean(feed_dict[reference_name],0))
            #ax1.plot(ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
            if 'steps' in plot or 'r=' in plot:
                ax1.plot(ratio,color=colors[plot],markeredgewidth=1,marker=line_style[plot],lw=0)
            else:
                ax1.plot(ratio,color=colors[plot],linewidth=2,linestyle=line_style[plot])
                
        
    FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
    ax0.legend(loc='best',fontsize=16,ncol=1)

    plt.ylabel('Difference. (%)')
    plt.xlabel(xlabel)
    plt.axhline(y=0.0, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
    plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
    plt.ylim([-100,100])

    return fig,ax0

class ScalarFormatterClass(mtick.ScalarFormatter):
    #https://www.tutorialspoint.com/show-decimal-places-and-scientific-notation-on-the-axis-of-a-matplotlib-plot
    def _set_format(self):
        self.format = "%1.1f"


def FormatFig(xlabel,ylabel,ax0):
    #Limit number of digits in ticks
    # y_loc, _ = plt.yticks()
    # y_update = ['%.1f' % y for y in y_loc]
    # plt.yticks(y_loc, y_update) 
    ax0.set_xlabel(xlabel,fontsize=20)
    ax0.set_ylabel(ylabel)
        

def WriteText(xpos,ypos,text,ax0):

    plt.text(xpos, ypos,text,
             horizontalalignment='center',
             verticalalignment='center',
             transform = ax0.transAxes, fontsize=25, fontweight='bold')


def HistRoutine(feed_dict,
                xlabel='',ylabel='',
                reference_name='Geant',
                logy=False,binning=None,
                fig = None, gs = None,
                plot_ratio= True,
                idx = None,
                label_loc='best'):
    assert reference_name in feed_dict.keys(), "ERROR: Don't know the reference distribution"

    if fig is None:
        fig,gs = SetGrid(plot_ratio) 
    ax0 = plt.subplot(gs[0])
    if plot_ratio:
        plt.xticks(fontsize=0)
        ax1 = plt.subplot(gs[1],sharex=ax0)
        
    
    if binning is None:
        binning = np.linspace(np.quantile(feed_dict[reference_name],0.0),np.quantile(feed_dict[reference_name],1),20)
        
    xaxis = [(binning[i] + binning[i+1])/2.0 for i in range(len(binning)-1)]
    reference_hist,_ = np.histogram(feed_dict[reference_name],bins=binning,density=True)
    maxy = np.max(reference_hist)
    
    for ip,plot in enumerate(feed_dict.keys()):
        dist,_,_=ax0.hist(feed_dict[plot],bins=binning,label=name_translate[plot],linestyle=line_style[plot],color=colors[plot],density=True,histtype="step")
        if plot_ratio:
            if reference_name!=plot:
                ratio = 100*np.divide(reference_hist-dist,reference_hist)
                ax1.plot(xaxis,ratio,color=colors[plot],marker='o',ms=10,lw=0,markerfacecolor='none',markeredgewidth=3)
        
    ax0.legend(loc=label_loc,fontsize=12,ncol=5)

    if logy:
        ax0.set_yscale('log')



    if plot_ratio:
        FormatFig(xlabel = "", ylabel = ylabel,ax0=ax0)
        plt.ylabel('Difference. (%)')
        plt.xlabel(xlabel)
        plt.axhline(y=0.0, color='r', linestyle='-',linewidth=1)
        # plt.axhline(y=10, color='r', linestyle='--',linewidth=1)
        # plt.axhline(y=-10, color='r', linestyle='--',linewidth=1)
        plt.ylim([-100,100])
    else:
        FormatFig(xlabel = xlabel, ylabel = ylabel,ax0=ax0)
    
    return fig,gs, binning


def LoadJson(file_name):
    import json,yaml
    JSONPATH = os.path.join(file_name)
    return yaml.safe_load(open(JSONPATH))

def SaveJson(save_file,data):
    with open(save_file,'w') as f:
        json.dump(data, f)


def revert_npart(npart,max_npart,flavour):

    #Revert the preprocessing to recover the particle multiplicity
    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(max_npart))

    x = np.copy(npart)
    for unique in flavour.astype(int):
        mask = unique==flavour
        x[mask] = npart[mask]*data_dict['std_jet_{}'.format(unique)][-1] + data_dict['mean_jet_{}'.format(unique)][-1]
    return np.round(x).astype(np.int32)

def Recenter(particles):
    
    px = particles[:,:,2]*np.cos(particles[:,:,1])
    py = particles[:,:,2]*np.sin(particles[:,:,1])
    pz = particles[:,:,2]*np.sinh(particles[:,:,0])

    jet_px = np.sum(px,1)
    jet_py = np.sum(py,1)
    jet_pz = np.sum(pz,1)
    
    jet_pt = np.sqrt(jet_px**2 + jet_py**2)
    jet_phi = np.ma.arctan2(jet_py,jet_px).filled(0)
    jet_eta = np.ma.arcsinh(np.ma.divide(jet_pz,jet_pt).filled(0))

    particles[:,:,0]-= np.expand_dims(jet_eta,1)
    particles[:,:,1]-= np.expand_dims(jet_phi,1)


    return particles


def ReversePrep(particles,jets,npart,flavour):

    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(npart))
    num_part = particles.shape[1]
    particles=particles.reshape(-1,particles.shape[-1])
    mask=np.expand_dims(particles[:,2]!=0,-1)
    def _revert(x,name='jet'):
        for unique in np.unique(flavour).astype(int):
            mask_class = unique == flavour
            if 'particle' in name:
                mask_class = np.tile(mask_class.reshape(-1,1),(1,num_part)).reshape(-1)
                
            x[mask_class] = x[mask_class]*data_dict['std_{}_{}'.format(name,unique)] + data_dict['mean_{}_{}'.format(name,unique)]
        return x
        
    particles = _revert(particles,'particle')
    jets = _revert(jets,'jet')

    particles[:,2] = 1.0 - np.exp(particles[:,2])
    particles[:,2] = np.clip(particles[:,2],0.00013,1.0) #apply min pt cut
    particles[:,0] = np.clip(particles[:,0],-0.5,0.5)
    particles[:,1] = np.clip(particles[:,1],-0.5,0.5)

    jets[:,3] = np.round(jets[:,3])
    jets[:,3] = np.clip(jets[:,3],5,npart)

    
    particles = Recenter((particles*mask).reshape(jets.shape[0],num_part,-1))
    return particles,jets


def SimpleLoader(data_path,labels):
    particles = []
    jets = []

    for label in labels:
        #if 'w' in label or 'z' in label: continue #no evaluation for w and z
        with h5.File(os.path.join(data_path,label),"r") as h5f:
            ntotal = h5f['jet_features'][:].shape[0]
            particle = h5f['particle_features'][int(0.7*ntotal):].astype(np.float32)
            jet = h5f['jet_features'][int(0.7*ntotal):].astype(np.float32)
            jet = np.concatenate([jet,labels[label]*np.ones(shape=(jet.shape[0],1),dtype=np.float32)],-1)

            particles.append(particle)
            jets.append(jet)

    particles = np.concatenate(particles)
    jets = np.concatenate(jets)
    particles,jets = shuffle(particles,jets, random_state=0)
    particles = Recenter(particles)

    #cond = to_categorical(jets[:nevts,-1], num_classes=num_classes)
    cond = jets[:nevts,-1]
    mask = np.expand_dims(particles[:nevts,:,-1],-1)
    
    return particles[:nevts,:,:-1]*mask,jets[:nevts,:-1],cond

    
def DataLoader(data_path,labels,
               npart,
               rank=0,size=1,
               batch_size=64,make_tf_data=True):
    particles = []
    jets = []

    def _preprocessing(particles,jets,save_json=False):
        num_part = particles.shape[1]
        particles=particles.reshape(-1,particles.shape[-1]) #flatten

        def _logit(x):                            
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*x
            return np.ma.log(x/(1-x)).filled(0)

        #Transformations
        particles[:,2] = np.ma.log(1.0 - particles[:,2]).filled(0)
        if save_json:
            data_dict = {}

            for unique in np.unique(jets[:,-1]).astype(int):
                mask_class = jets[:,-1] == unique
                mask_class_part = np.tile(mask_class.reshape(-1,1),(1,num_part)).reshape(-1)
                mask = particles[mask_class_part,-1]
                mean_particle = np.average(particles[mask_class_part,:-1],axis=0,weights=mask)
                data_dict['mean_jet_{}'.format(unique)]=np.mean(jets[mask_class,:-1],0).tolist()
                data_dict['std_jet_{}'.format(unique)]=np.std(jets[mask_class,:-1],0).tolist()
                data_dict['mean_particle_{}'.format(unique)]=mean_particle.tolist()
                data_dict['std_particle_{}'.format(unique)]=np.sqrt(np.average((particles[mask_class_part,:-1] - mean_particle)**2,axis=0,weights=mask)).tolist()
            
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)
        else:
            data_dict = LoadJson('preprocessing_{}.json'.format(npart))

        for unique in np.unique(jets[:,-1]).astype(int):
            mask_class = jets[:,-1] == unique
            mask_class_part = np.tile(mask_class.reshape(-1,1),(1,num_part)).reshape(-1)            
            jets[mask_class,:-1] = np.ma.divide(jets[mask_class,:-1]-data_dict['mean_jet_{}'.format(unique)],data_dict['std_jet_{}'.format(unique)]).filled(0)
            particles[mask_class_part,:-1]= np.ma.divide(particles[mask_class_part,:-1]-data_dict['mean_particle_{}'.format(unique)],data_dict['std_particle_{}'.format(unique)]).filled(0)
        
        particles = particles.reshape(jets.shape[0],num_part,-1)
        return particles.astype(np.float32),jets.astype(np.float32)
            
            
    for label in labels:
        
        with h5.File(os.path.join(data_path,label),"r") as h5f:
            ntotal = h5f['jet_features'][:].shape[0]

            if make_tf_data:
                particle = h5f['particle_features'][rank:int(0.7*ntotal):size].astype(np.float32)
                jet = h5f['jet_features'][rank:int(0.7*ntotal):size].astype(np.float32)
                jet = np.concatenate([jet,labels[label]*np.ones(shape=(jet.shape[0],1),dtype=np.float32)],-1)
            else:
                #load evaluation data
                #if 'w' in label or 'z' in label: continue #no evaluation for w and z
                particle = h5f['particle_features'][int(0.7*ntotal):].astype(np.float32)
                jet = h5f['jet_features'][int(0.7*ntotal):].astype(np.float32)
                jet = np.concatenate([jet,labels[label]*np.ones(shape=(jet.shape[0],1),dtype=np.float32)],-1)           

            particles.append(particle)
            jets.append(jet)

    particles = np.concatenate(particles)
    jets = np.concatenate(jets)
    particles,jets = shuffle(particles,jets, random_state=0)
    particles = Recenter(particles)


    

    particles,jets = _preprocessing(particles,jets)        
    data_size = jets.shape[0]
    
    if make_tf_data:
        train_particles = particles[:int(0.9*data_size)]
        train_jets = jets[:int(0.9*data_size)]
        
        test_particles = particles[int(0.9*data_size):]
        test_jets = jets[int(0.9*data_size):]
        
    
        def _prepare_batches(particles,jets):
            
            nevts = jets.shape[0]
            tf_jet = tf.data.Dataset.from_tensor_slices(jets[:,:-1])
            cond = to_categorical(jets[:,-1], num_classes=num_classes) 
            tf_cond = tf.data.Dataset.from_tensor_slices(cond)
            mask = np.expand_dims(particles[:,:,-1],-1)
            masked = particles[:,:,:-1]*mask
            tf_part = tf.data.Dataset.from_tensor_slices(masked)
            tf_mask = tf.data.Dataset.from_tensor_slices(mask)
            tf_zip = tf.data.Dataset.zip((tf_part, tf_jet,tf_cond,tf_mask))
            return tf_zip.shuffle(nevts).repeat().batch(batch_size)
    
        train_data = _prepare_batches(train_particles,train_jets)
        test_data  = _prepare_batches(test_particles,test_jets)    
        return data_size, train_data,test_data
    
    else:
        cond = to_categorical(jets[:nevts,-1], num_classes=num_classes_eval)
        cond = np.concatenate([cond,np.zeros(shape=(cond.shape[0],num_classes-num_classes_eval),
                                             dtype=np.float32)], -1)
        mask = np.expand_dims(particles[:nevts,:,-1],-1)
        return particles[:nevts,:,:-1]*mask,jets[:nevts,:-1],cond, mask
