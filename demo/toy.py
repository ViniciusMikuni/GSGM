import numpy as np
import os,re
import tensorflow as tf
from tensorflow import keras
import time
from tensorflow.keras.callbacks import ReduceLROnPlateau,EarlyStopping
#import horovod.tensorflow.keras as hvd
import argparse
import utils
from GSGM import GSGM
from tensorflow.keras.callbacks import ModelCheckpoint
# import tensorflow_addons as tfa

tf.random.set_seed(1233)


if __name__ == "__main__":

    #hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    # if gpus:
    #     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='config.json', help='Config file with training parameters')
    parser.add_argument('--load', action='store_true', default=False, help='Load trained model only')

    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)


    nevts = int(1e5)

    num_feat = config['NUM_FEAT']

    #Let's generate sets containing 2 particles using gaussians
    part1 = np.random.normal(-2.0,1.0,size=(nevts,1,num_feat))
    part2 = np.random.normal(2.0,1.0,size=(nevts,1,num_feat))
    data = np.concatenate([part1[:nevts//2],part2[:nevts//2]],1)

    model = GSGM(config=config)
    checkpoint_folder = '../checkpoints_toy/checkpoint'

    if flags.load:
        model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
    else:        
        opt = tf.optimizers.Adam(learning_rate=config['LR'])    
        model.compile(optimizer=opt)

        callbacks = []
        checkpoint = ModelCheckpoint(checkpoint_folder,mode='auto',
                                     period=1,save_weights_only=True)

        callbacks.append(checkpoint)

        history = model.fit(
            data,
            epochs=config['MAXEPOCH'],
            batch_size=config['BATCH'],
            callbacks=callbacks
            #steps_per_epoch=1,
        )


    #Run the reverse diffusion to generate new point clouds. Try out generating different number of particles, even though the model is never trained to do so!

    generated = model.PCSampler(nevts,2).numpy()

    if not os.path.exists('../plots'):
        os.mkdir('../plots')

    #Let's do some plotting
    feed_dict = {
        'gen':np.max(generated[:,:,0],1),
        'true':np.max(data[:,:,0],1),
    }

    binning = np.linspace(0,4,20)
    fig,ax0 =utils.HistRoutine(feed_dict,xlabel='max feat', ylabel= 'Normalized entries',
                               logy=False,binning=binning)
    fig.savefig('../plots/toy_max.pdf')


    feed_dict = {
        'gen':generated[:,0,0],
        'true':data[:,0,0],
    }

    binning = np.linspace(-4,4,20)
    fig,ax0 =utils.HistRoutine(feed_dict,xlabel='feat 0', ylabel= 'Normalized entries',
                               logy=False,binning=binning)
    fig.savefig('../plots/toy.pdf')


    feed_dict = {
        'gen':generated[:,0,0]+generated[:,1,0],
        'true':data[:,0,0]+data[:,1,0],
    }

    binning = np.linspace(-5.0,5.0,10)
    fig,ax0 =utils.HistRoutine(feed_dict,xlabel='diff feat', ylabel= 'Normalized entries',
                               logy=False,binning=binning)
    fig.savefig('../plots/toy_diff.pdf')
