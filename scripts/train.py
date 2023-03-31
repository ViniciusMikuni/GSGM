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
from GSGM_distill import GSGM_distill
from tensorflow.keras.callbacks import ModelCheckpoint
import tensorflow_addons as tfa
import horovod.tensorflow.keras as hvd

tf.random.set_seed(1233)
#tf.keras.backend.set_floatx('float64')
if __name__ == "__main__":
    hvd.init()
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


    parser = argparse.ArgumentParser()    
    parser.add_argument('--config', default='config_jet.json', help='Config file with training parameters')
    parser.add_argument('--data_path', default='/global/cfs/cdirs/m3929/GSGM', help='Path containing the training files')
    parser.add_argument('--distill', action='store_true', default=False,help='Use the distillation model')
    parser.add_argument('--big', action='store_true', default=False,help='Use bigger dataset (150 particles) as opposed to 30 particles')
    parser.add_argument('--factor', type=int,default=1, help='Step reduction for distillation model')


    flags = parser.parse_args()
    config = utils.LoadJson(flags.config)

    assert flags.factor%2==0 or flags.factor==1, "Distillation reduction steps needs to be even"

    if flags.big:
        labels = utils.labels150
        npart=150
    else:
        labels=utils.labels30
        npart=30
    
    data_size,training_data,test_data = utils.DataLoader(flags.data_path,
                                                         labels,
                                                         npart,
                                                         hvd.rank(),hvd.size(),
                                                         config['BATCH'])

    model = GSGM(config=config,npart=npart)

    model_name = config['MODEL_NAME']
    if flags.big:
        model_name+='_big'
    checkpoint_folder = '../checkpoints_{}/checkpoint'.format(model_name)
    if flags.distill:
        if flags.factor>2:
            checkpoint_folder = '../checkpoints_{}_d{}/checkpoint'.format(model_name,flags.factor//2)
            model = GSGM_distill(model.ema_jet,model.ema_part,factor=flags.factor//2,config=config)
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
            #previous student, now teacher
            model = GSGM_distill(model.ema_jet,model.ema_part,factor=flags.factor,
                                 config=config,npart=npart)
        else:
            model.load_weights('{}'.format(checkpoint_folder)).expect_partial()
            model = GSGM_distill(model.ema_jet,model.ema_part,factor=flags.factor,
                                 config=config,npart=npart)

        if hvd.rank()==0:print("Loading Teacher from: {}".format(checkpoint_folder))
        checkpoint_folder = '../checkpoints_{}_d{}/checkpoint'.format(model_name,flags.factor)
        
        
    lr_schedule = tf.keras.experimental.CosineDecay(
        initial_learning_rate=config['LR']*hvd.size(),
        decay_steps=config['MAXEPOCH']*int(data_size*0.8/config['BATCH'])
    )
    opt = tf.keras.optimizers.Adamax(learning_rate=lr_schedule)
    opt = hvd.DistributedOptimizer(
        opt, average_aggregated_gradients=True)

        
    model.compile(            
        optimizer=opt,
        #run_eagerly=True,
        experimental_run_tf_function=False,
        weighted_metrics=[])
    
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
        hvd.callbacks.MetricAverageCallback(),
        EarlyStopping(patience=100,restore_best_weights=True),
    ]

        
    if hvd.rank()==0:
        checkpoint = ModelCheckpoint(checkpoint_folder,mode='auto',
                                     period=1,save_weights_only=True)
        callbacks.append(checkpoint)
        
    
    history = model.fit(
        training_data,
        epochs=config['MAXEPOCH'],
        callbacks=callbacks,
        steps_per_epoch=int(data_size*0.8/config['BATCH']),
        validation_data=test_data,
        validation_steps=int(data_size*0.1/config['BATCH']),
        verbose=1 if hvd.rank()==0 else 0,
        #steps_per_epoch=1,
    )

    
