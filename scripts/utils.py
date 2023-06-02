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

    'toy_truth':'toy true',
    'toy_gen':'toy gen',
}

names = ['g','q','t','w','z']


# labels30 = {
#     'g.hdf5':0,
#     't.hdf5':1,
# }


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


# labels150 = {
#     'g150.hdf5':0,
#     't150.hdf5':1,
# }

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

def SetGrid(ratio=True):
    fig = plt.figure(figsize=(9, 9))
    if ratio:
        gs = gridspec.GridSpec(2, 1, height_ratios=[3,1]) 
        gs.update(wspace=0.025, hspace=0.1)
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
        

    # xposition = 0.9
    # yposition=1.03
    # text = 'H1'
    # WriteText(xposition,yposition,text,ax0)


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


def revert_npart(npart,max_npart):

    #Revert the preprocessing to recover the particle multiplicity
    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(max_npart))
    x = npart*data_dict['std_jet'][-1] + data_dict['mean_jet'][-1]
    x = revert_logit(x)
    x = x * (data_dict['max_jet'][-1]-data_dict['min_jet'][-1]) + data_dict['min_jet'][-1]
    #x = np.exp(x)
    return np.round(x).astype(np.int32)
     
def revert_logit(x):
    alpha = 1e-6
    exp = np.exp(x)
    x = exp/(1+exp)
    return (x-alpha)/(1 - 2*alpha)                

def ReversePrep(particles,jets,npart):

    alpha = 1e-6
    data_dict = LoadJson('preprocessing_{}.json'.format(npart))
    num_part = particles.shape[1]    
    particles=particles.reshape(-1,particles.shape[-1])
    mask=np.expand_dims(particles[:,2]!=0,-1)
    def _revert(x,name='jet'):    
        x = x*data_dict['std_{}'.format(name)] + data_dict['mean_{}'.format(name)]
        x = revert_logit(x)
        #print(data_dict['max_{}'.format(name)],data_dict['min_{}'.format(name)])
        x = x * (np.array(data_dict['max_{}'.format(name)]) -data_dict['min_{}'.format(name)]) + data_dict['min_{}'.format(name)]
        return x
        
    particles = _revert(particles,'particle')
    jets = _revert(jets,'jet')
    jets[:,3] = np.round(jets[:,3])
    particles[:,2] = 1.0 - particles[:,2]
    return (particles*mask).reshape(jets.shape[0],num_part,-1),jets


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
    

    mask = np.sqrt(particles[:,:,0]**2 + particles[:,:,1]**2) < 0.8 #eta looks off
    particles*=np.expand_dims(mask,-1)

    cond = to_categorical(jets[:nevts,-1], num_classes=num_classes)
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
        mask = np.sqrt(particles[:,:,0]**2 + particles[:,:,1]**2) < 0.8 #eta looks weird
        particles*=np.expand_dims(mask,-1)

        particles=particles.reshape(-1,particles.shape[-1]) #flatten

        def _logit(x):                            
            alpha = 1e-6
            x = alpha + (1 - 2*alpha)*x
            return np.ma.log(x/(1-x)).filled(0)

        #Transformations
        particles[:,2] = 1.0 - particles[:,2]

        if save_json:
            data_dict = {
                'max_jet':np.max(jets[:,:-1],0).tolist(),
                'min_jet':np.min(jets[:,:-1],0).tolist(),
                'max_particle':np.max(particles[:,:-1],0).tolist(),
                'min_particle':np.min(particles[:,:-1],0).tolist(),
            }                
            
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)
        else:
            data_dict = LoadJson('preprocessing_{}.json'.format(npart))

        #normalize
        jets[:,:-1] = np.ma.divide(jets[:,:-1]-data_dict['min_jet'],np.array(data_dict['max_jet'])- data_dict['min_jet']).filled(0)        
        particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['min_particle'],np.array(data_dict['max_particle'])- data_dict['min_particle']).filled(0)

        jets[:,:-1] = _logit(jets[:,:-1])
        particles[:,:-1] = _logit(particles[:,:-1])
        if save_json:
            mask = particles[:,-1]
            mean_particle = np.average(particles[:,:-1],axis=0,weights=mask)
            data_dict['mean_jet']=np.mean(jets[:,:-1],0).tolist()
            data_dict['std_jet']=np.std(jets[:,:-1],0).tolist()
            data_dict['mean_particle']=mean_particle.tolist()
            data_dict['std_particle']=np.sqrt(np.average((particles[:,:-1] - mean_particle)**2,axis=0,weights=mask)).tolist()                        
            SaveJson('preprocessing_{}.json'.format(npart),data_dict)
        
            
        jets[:,:-1] = np.ma.divide(jets[:,:-1]-data_dict['mean_jet'],data_dict['std_jet']).filled(0)
        particles[:,:-1]= np.ma.divide(particles[:,:-1]-data_dict['mean_particle'],data_dict['std_particle']).filled(0)
        
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
    
    data_size = jets.shape[0]
    particles,jets = _preprocessing(particles,jets)    
    
    
    #if rank==0:print("Training events: {}, Test Events: {} Validation Events: {}".format(train_jets.shape[0],test_jets.shape[0],val_jets.shape[0]))
    # print(np.max(train_jets,0),np.min(train_jets,0))
    # print(np.mean(train_jets,0),np.std(train_jets,0))
    # print(np.sum(train_jets[:,0]>5.)/train_jets.shape[0])

    # print(np.max(train_particles,0),np.min(train_particles,0))
    # print(np.sum(train_particles[:,:,0]>6.)/(150*train_jets.shape[0]))
    # print(np.mean(train_particles,0),np.std(train_particles,0))
    # print(np.max(test_particles,0),np.min(test_particles,0))
    # input()
    if make_tf_data:
        train_particles = particles[:int(0.8*data_size)]
        train_jets = jets[:int(0.8*data_size)]
        
        test_particles = particles[int(0.8*data_size):]
        test_jets = jets[int(0.8*data_size):]
        
    
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
