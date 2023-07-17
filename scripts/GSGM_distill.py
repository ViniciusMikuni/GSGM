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
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']

        self.max_part = npart
        self.shape = (-1,1,1)
        self.ema=0.999
        self.verbose=False
        
        self.num_steps = self.config['MAX_STEPS']//self.factor

        
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="loss_jet")
        self.loss_jet_tracker = keras.metrics.Mean(name="loss_part")


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
        return [self.loss_tracker,self.loss_part_tracker,self.loss_jet_tracker]
    @tf.function
    def get_logsnr_alpha_sigma(self,time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        return logsnr, alpha, sigma
    
    @tf.function
    def train_step(self, inputs):
        part,jet,cond,mask = inputs

        #Define the sigma and alphas for the different time steps used in the interpolation
        i = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)

        u = (i+1) / self.num_steps
        u_mid = u - 0.5/self.num_steps
        u_s = u - 1./self.num_steps

        
        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(u)
        logsnr_mid, alpha_mid, sigma_mid = self.get_logsnr_alpha_sigma(u_mid)
        logsnr_s, alpha_s, sigma_s = self.get_logsnr_alpha_sigma(u_s)
        
        sigma_frac = tf.exp(
            0.5 * (tf.math.softplus(logsnr) - tf.math.softplus(logsnr_s)))
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)

        alpha_mid_reshape = tf.reshape(alpha_mid,self.shape)
        sigma_mid_reshape = tf.reshape(sigma_mid,self.shape)
        
        alpha_s_reshape = tf.reshape(alpha_s,self.shape)
        sigma_s_reshape = tf.reshape(sigma_s,self.shape)

        sigma_frac_reshape = tf.reshape(sigma_frac,self.shape)
        


        
        #part
        eps = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        z = alpha_reshape*part + eps * sigma_reshape
        score_teacher = self.teacher_part([z*mask, u, jet,cond,mask],training=False)

        mean_part = alpha_reshape * z - sigma_reshape * score_teacher
        eps_part = (z - alpha_reshape * mean_part) / sigma_reshape
        
        z_mid = alpha_mid_reshape * mean_part + sigma_mid_reshape * eps_part
        score_teacher_mid = self.teacher_part([z_mid*mask, u_mid, jet,cond,mask],training=False)
        
        mean_part = alpha_mid_reshape * z_mid - sigma_mid_reshape * score_teacher_mid
        eps_part = (z_mid - alpha_mid_reshape * mean_part) / sigma_mid_reshape
        
        z_teacher = alpha_s_reshape * mean_part + sigma_s_reshape * eps_part

                
        x_target = (z_teacher - sigma_frac_reshape * z) / (alpha_s_reshape - sigma_frac_reshape * alpha_reshape)
        x_target = tf.where(tf.expand_dims(i,-1) == 0, mean_part, x_target)
        eps_target = (z - alpha_reshape * x_target) / sigma_reshape
        
        with tf.GradientTape() as tape:
            v_target = alpha_reshape * eps_target - sigma_reshape * x_target
            v = self.model_part([z*mask, u,jet,cond,mask])
            
            losses = tf.square(v - v_target)*mask
            loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_part.trainable_variables
        g = tape.gradient(loss_part, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]
        self.optimizer.apply_gradients(zip(g, trainable_variables)) 
        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

            
        #jet
        eps = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        z = alpha*jet + eps * sigma
        score_teacher = self.teacher_jet([z, u,cond],training=False)

        mean_jet = alpha * z - sigma * score_teacher
        eps_jet = (z - alpha * mean_jet) / sigma
        
        z_mid = alpha_mid * mean_jet + sigma_mid * eps_jet
        score_teacher_mid = self.teacher_jet([z_mid, u_mid,cond],training=False)
        
        mean_jet = alpha_mid * z_mid - sigma_mid * score_teacher_mid
        eps_jet = (z_mid - alpha_mid * mean_jet) / sigma_mid
        
        z_teacher = alpha_s * mean_jet + sigma_s * eps_jet

                
        x_target = (z_teacher - sigma_frac * z) / (alpha_s - sigma_frac * alpha)
        x_target = tf.where(i == 0, mean_jet, x_target)
        eps_target = (z - alpha * x_target) / sigma
        
        with tf.GradientTape() as tape:
            v_target = alpha * eps_target - sigma * x_target
            v = self.model_jet([z, u,cond])
            
            losses = tf.square(v - v_target)
            loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_jet.trainable_variables
        g = tape.gradient(loss_jet, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]
        self.optimizer.apply_gradients(zip(g, trainable_variables)) 
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)

        
        
        self.loss_tracker.update_state(loss_part+loss_jet)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)

        return {
            "loss": self.loss_tracker.result(),
            "loss_part":self.loss_part_tracker.result(),
            "loss_jet":self.loss_jet_tracker.result(),
        }

    @tf.function
    def test_step(self, inputs):
        part,jet,cond,mask = inputs

        #Define the sigma and alphas for the different time steps used in the interpolation
        i = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)

        u = (i+1) / self.num_steps
        u_mid = u - 0.5/self.num_steps
        u_s = u - 1./self.num_steps

        logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(u)
        logsnr_mid, alpha_mid, sigma_mid = self.get_logsnr_alpha_sigma(u_mid)
        logsnr_s, alpha_s, sigma_s = self.get_logsnr_alpha_sigma(u_s)
        
        sigma_frac = tf.exp(
            0.5 * (tf.math.softplus(logsnr) - tf.math.softplus(logsnr_s)))
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)

        alpha_mid_reshape = tf.reshape(alpha_mid,self.shape)
        sigma_mid_reshape = tf.reshape(sigma_mid,self.shape)
        
        alpha_s_reshape = tf.reshape(alpha_s,self.shape)
        sigma_s_reshape = tf.reshape(sigma_s,self.shape)

        sigma_frac_reshape = tf.reshape(sigma_frac,self.shape)
        


        
        #part
        eps = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        z = alpha_reshape*part + eps * sigma_reshape
        score_teacher = self.teacher_part([z*mask, u, jet,cond,mask],training=False)

        mean_part = alpha_reshape * z - sigma_reshape * score_teacher
        eps_part = (z - alpha_reshape * mean_part) / sigma_reshape
        
        z_mid = alpha_mid_reshape * mean_part + sigma_mid_reshape * eps_part
        score_teacher_mid = self.teacher_part([z_mid*mask, u_mid, jet,cond,mask],training=False)
        
        mean_part = alpha_mid_reshape * z_mid - sigma_mid_reshape * score_teacher_mid
        eps_part = (z_mid - alpha_mid_reshape * mean_part) / sigma_mid_reshape
        
        z_teacher = alpha_s_reshape * mean_part + sigma_s_reshape * eps_part

                
        x_target = (z_teacher - sigma_frac_reshape * z) / (alpha_s_reshape - sigma_frac_reshape * alpha_reshape)
        x_target = tf.where(tf.expand_dims(i,-1) == 0, mean_part, x_target)
        eps_target = (z - alpha_reshape * x_target) / sigma_reshape
        
        
        v_target = alpha_reshape * eps_target - sigma_reshape * x_target
        v = self.model_part([z*mask, u,jet,cond,mask])
        
        losses = tf.square(v - v_target)*mask
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                        
        #jet
        eps = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        z = alpha*jet + eps * sigma
        score_teacher = self.teacher_jet([z, u,cond],training=False)

        mean_jet = alpha * z - sigma * score_teacher
        eps_jet = (z - alpha * mean_jet) / sigma
        
        z_mid = alpha_mid * mean_jet + sigma_mid * eps_jet
        score_teacher_mid = self.teacher_jet([z_mid, u_mid,cond],training=False)
        
        mean_jet = alpha_mid * z_mid - sigma_mid * score_teacher_mid
        eps_jet = (z_mid - alpha_mid * mean_jet) / sigma_mid
        
        z_teacher = alpha_s * mean_jet + sigma_s * eps_jet

                
        x_target = (z_teacher - sigma_frac * z) / (alpha_s - sigma_frac * alpha)
        x_target = tf.where(i == 0, mean_jet, x_target)                
        eps_target = (z - alpha * x_target) / sigma
        

        v_target = alpha * eps_target - sigma * x_target
        v = self.model_jet([z, u,cond])
        
        losses = tf.square(v - v_target)
        loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
        
        self.loss_tracker.update_state(loss_part+loss_jet)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)

        return {
            "loss": self.loss_tracker.result(),
            "loss_part":self.loss_part_tracker.result(),
            "loss_jet":self.loss_jet_tracker.result(),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)


    def generate(self,cond,jet_info):
        start = time.time()
        jet_info = self.DDPMSampler(cond,self.ema_jet,
                                    data_shape=[self.num_jet],
                                    const_shape = [-1,1]).numpy()
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part,np.argmax(cond,-1)),
                                        5,self.max_part),-1)
        #print(np.unique(nparts))
        mask = np.expand_dims(
            np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
        assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of particles'

        start = time.time()
        parts = self.DDPMSampler(tf.convert_to_tensor(cond,dtype=tf.float32),
                                 self.ema_part,
                                 data_shape=[self.max_part,self.num_feat],
                                 jet=tf.convert_to_tensor(jet_info, dtype=tf.float32),
                                 const_shape = self.shape,
                                 mask=tf.convert_to_tensor(mask, dtype=tf.float32)).numpy()
        
        # parts = np.ones(shape=(cond.shape[0],self.max_part,3))
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        return parts*mask,jet_info


    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))
    
    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    
    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
                    data_shape=None,
                    const_shape=None,
                    jet=None,
                    mask=None):
        """Generate samples from score-based models with DDPM method.
        
        Args:
        cond: Conditional input
        model: Trained score model to use
        data_shape: Format of the data
        const_shape: Format for constants, should match the data_shape in dimensions
        jet: input jet conditional information if used
        mask: particle mask if used

        Returns: 
        Samples.
        """
        
        batch_size = cond.shape[0]
        data_shape = np.concatenate(([batch_size],data_shape))
        x = self.prior_sde(data_shape)

        for time_step in tf.range(self.num_steps, 0, delta=-1):
            random_t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps)

            if jet is None:
                score = model([x, random_t, cond], training=False)
            else:
                x = x * mask
                score = model([x, random_t, jet, cond, mask], training=False) * mask
                alpha = tf.reshape(alpha, self.shape)
                sigma = tf.reshape(sigma, self.shape)
                alpha_ = tf.reshape(alpha_, self.shape)
                sigma_ = tf.reshape(sigma_, self.shape)

            mean = alpha * x - sigma * score
            eps = (x - alpha * mean) / sigma
            x = alpha_ * mean + sigma_ * eps
        
        return mean
