import numpy as np
import tensorflow as tf
from tensorflow import keras
import time

import utils

# tf and friends
tf.random.set_seed(1234)

class GSGM_distill(keras.Model):
    """Score based generative model distill"""
    def __init__(self, teacher_jet,teacher_part,factor,npart=30,config=None):
        super(GSGM_distill, self).__init__()
        self.config = config
        if config is None:
            raise ValueError("Config file not given")
        
        self.factor = factor
        self.activation = 'swish'
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.shape = (-1,1,1)
        self.ema=0.999
        self.verbose=False
        
        self.num_steps = self.config['MAX_STEPS']//self.factor

        
        self.betas,self.alphas_cumprod,self.alphas = self.get_alpha_beta(self.num_steps)
        self.teacher_betas,self.teacher_alphas_cumprod,_ = self.get_alpha_beta(2*self.num_steps)
        
        
        alphas_cumprod_prev = tf.concat((tf.ones(1, dtype=tf.float32), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * tf.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * tf.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        self.loss_tracker = keras.metrics.Mean(name="loss")


        self.teacher_jet = teacher_jet
        self.teacher_part = teacher_part
        
        self.model_jet = keras.models.clone_model(teacher_jet)
        self.model_part = keras.models.clone_model(teacher_part)
        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_part = keras.models.clone_model(self.model_part)

        if self.verbose:
            print(self.model_part.summary())
        self.teacher_jet.trainable = False    
        self.teacher_part.trainable = False    
            
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]

    def get_alpha_beta(self,num_steps):
        timesteps =tf.range(start=0,limit=num_steps + 1, dtype=tf.float32) / num_steps + 8e-3 
        alphas = timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = tf.math.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = tf.clip_by_value(betas, clip_value_min =0, clip_value_max=0.999)
        alphas = 1 - betas
        alphas_cumprod = tf.math.cumprod(alphas, 0)
        return betas,alphas_cumprod,alphas


    def get_alpha_sigma(self,t,use_teacher=False,shape=None):
        if use_teacher:
            alphas_cumprod = self.teacher_alphas_cumprod
        else:
            alphas_cumprod = self.alphas_cumprod
        alpha = tf.gather(tf.sqrt(alphas_cumprod),t)
        sigma = tf.gather(tf.sqrt(1-alphas_cumprod),t)
        if shape is not None:
            alpha = tf.reshape(alpha,shape)
            sigma = tf.reshape(sigma,shape)
        return alpha,sigma

    @tf.function
    def train_step(self, inputs):
        part,jet,cond,mask = inputs

        random_t = 2*tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        eps = tf.random.normal((tf.shape(part)),dtype=tf.float32)

        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True,shape=self.shape)
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,shape=self.shape)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True,shape=self.shape)
        
            
        #part                        
        z = alpha*part + eps * sigma
        score = self.teacher_part([z, random_t+1,jet,cond,mask],training=False)

        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)

        score_1 = self.teacher_part([z_1, random_t,jet,cond,mask],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2
        w = (1. + alpha_s / sigma_s)**0.3
        with tf.GradientTape() as tape:
            score = self.model_part([z, random_t//2,jet,cond,mask])
            loss_part = tf.square(score - target)*mask
            loss_part = tf.reduce_mean(loss_part)      

            
        g = tape.gradient(loss_part, self.model_part.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.model_part.trainable_variables))
        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)




        #jet
        eps = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True)
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True)
        
            

        z = alpha*jet + eps * sigma
        score = self.teacher_jet([z, random_t+1,cond],training=False)
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_jet([z_1, random_t,cond],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2
        w = (1. + alpha_s / sigma_s)**0.3
        with tf.GradientTape() as tape:
            score = self.model_jet([z, random_t//2,cond])
            loss_jet = tf.square(score - target)
            loss_jet = tf.reduce_mean(loss_jet)
            
            
        g = tape.gradient(loss_jet, self.model_jet.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.model_jet.trainable_variables))
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        
        
        self.loss_tracker.update_state(loss_part)

        return {
            "loss": self.loss_tracker.result(),
            "loss_part":loss_part,
            "loss_jet":loss_jet,
        }

    @tf.function
    def test_step(self, inputs):
        part,jet,cond,mask = inputs

        random_t = 2*tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        eps = tf.random.normal((tf.shape(part)),dtype=tf.float32)

        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True,shape=self.shape) 
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,shape=self.shape)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True,shape=self.shape)
        
            
        #part                        
        z = alpha*part + eps * sigma
        score = self.teacher_part([z, random_t+1,jet,cond,mask])
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_part([z_1, random_t,jet,cond,mask])

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2


        score = self.model_part([z, random_t//2,jet,cond,mask])
        w = (1. + alpha_s / sigma_s)**0.3
        loss_part = tf.square(score - target)*mask
        loss_part = tf.reduce_mean(loss_part)
            

        #jet
        eps = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        alpha,sigma = self.get_alpha_sigma(random_t+1,use_teacher=True)
        alpha_s, sigma_s = self.get_alpha_sigma(random_t//2,)
        alpha_1, sigma_1 = self.get_alpha_sigma(random_t,use_teacher=True)
        

        z = alpha*jet + eps * sigma
        score = self.teacher_jet([z, random_t+1,cond])
        rec = (alpha * z - sigma * score)
        z_1 = alpha_1 * rec + (sigma_1 / sigma) * (z - alpha * rec)
        
        score_1 = self.teacher_jet([z_1, random_t,cond],training=False)

        x_2 = (alpha_1 * z_1 - sigma_1 * score_1)
        eps_2 = (z - alpha_s * x_2) / sigma_s
        target = alpha_s * eps_2 - sigma_s * x_2

        score = self.model_jet([z, random_t//2,cond])
        w = (1. + alpha_s / sigma_s)**0.3
        loss_jet = tf.square(score - target)
        loss_jet = tf.reduce_mean(loss_jet)
            
        
        self.loss_tracker.update_state(loss_part)

        return {
            "loss": self.loss_tracker.result(),
            "loss_part":loss_part,
            "loss_jet":loss_jet,
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def generate(self,cond,jet_info):
        start = time.time()
        batch_size = cond.shape[0]
        #cond = tf.data.Dataset.from_tensor_slices(cond)
        jet_info = self.DDPMSampler(cond,self.ema_jet,
                                    batch_size = batch_size,
                                    data_shape=[self.num_jet],
                                    const_shape = [-1,1]).numpy()
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(batch_size,end - start))

        nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part),
                                        0,self.max_part),-1)
        #print(np.unique(nparts))
        mask = np.expand_dims(
            np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
        assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of particles'

        start = time.time()
        parts = self.DDPMSampler(cond,
                                 self.ema_part,
                                 batch_size = batch_size,
                                 data_shape=[self.max_part,self.num_feat],
                                 jet=tf.convert_to_tensor(jet_info, dtype=tf.float32),
                                 const_shape = self.shape,
                                 mask=tf.convert_to_tensor(mask, dtype=tf.float32)).numpy()
        
        # parts = np.ones(shape=(cond.shape[0],self.max_part,3))
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(batch_size,end - start))
        return parts*mask,jet_info


    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    batch_size,
                    data_shape=None,
                    const_shape=None,
                    jet=None,
                    mask=None):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        cond: Conditional input
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        

        data_shape = np.concatenate(([batch_size],data_shape))
        init_x = tf.random.normal(data_shape,dtype=tf.float32)
        if jet is not None:
            init_x *= mask 

        x = init_x
        for  time_step in tf.range(self.num_steps, 0, delta=-1):
            batch_time_step = tf.ones((batch_size,1),dtype=tf.int32) * time_step
            z = tf.random.normal(x.shape,dtype=tf.float32)

            alpha = tf.gather(tf.sqrt(self.alphas_cumprod),batch_time_step)
            sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),batch_time_step)
            
            if jet is None:
                score = model([x, batch_time_step,cond],
                              training=False)
            else:
                score = model([x, batch_time_step,jet,cond,mask],
                              training=False)
                
                alpha = tf.reshape(alpha,self.shape)
                sigma = tf.reshape(sigma,self.shape)
            
            # #print(np.max(score),np.min(score))
            x_recon = alpha * x - sigma * score

            p1 = tf.reshape(tf.gather(self.posterior_mean_coef1,batch_time_step),const_shape)
            p2 = tf.reshape(tf.gather(self.posterior_mean_coef2,batch_time_step),const_shape)
            mean = p1*x_recon + p2*x
           
            log_var = tf.reshape(tf.gather(tf.math.log(self.posterior_variance),batch_time_step),const_shape)

            x = mean + tf.exp(0.5 * log_var) * z
            
        # The last step does not include any noise
        return mean
