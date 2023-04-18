import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
#import horovod.tensorflow.keras as hvd
import utils
from deepsets import DeepSetsAtt

# tf and friends
tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self, name='SGM',sde_type='VPSDE',config=None):
        super(GSGM, self).__init__()
        self.sde_type=sde_type
        self.config = config
        self.num_embed = self.config['EMBED']
        self.activation = self.config['ACT']
        #self.num_part = self.config['NUM_PART']
        self.num_feat = self.config['NUM_FEAT']

        if config is None:
            raise ValueError("Config file not given")
        if self.sde_type not in ['VESDE','VPSDE','subVPSDE']:
            raise ValueError("SDE strategy not implemented")
        if self.sde_type== 'VESDE':
            self.sigma_0 = 0.01
            self.sigma_1 = 50.0
        else:
            self.beta_0 = 0.1
            self.beta_1 = 20.0

        self.verbose = 1 

        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        time_projection = inputs_time*self.projection*2*np.pi
        time_embed = tf.concat([tf.math.sin(time_projection),tf.math.cos(time_projection)],-1)
        time_embed = layers.Dense(self.num_embed,activation=self.activation)(time_embed)
        
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            #num_part=self.num_part,
            num_feat=self.num_feat,
            time_embedding=time_embed)


        self.model = keras.Model(inputs=[inputs,inputs_time],outputs=outputs)

        if self.verbose:
            print(self.model.summary())



    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]


    def GaussianFourierProjection(self,scale = 30):
        # Randomly sample weights during initialization. These weights are fixed 
        # during optimization and are not trainable.
        return tf.constant(tf.random.normal(shape=(1,self.num_embed//2),seed=100))*scale



    def marginal_prob(self,x,t,sigma=25):
        if self.sde_type == 'VESDE':
            mean = x
            std = self.sigma_0*(self.sigma_1/self.sigma_0)**t
            # std = self.sigma_1*t
            std = tf.reshape(std,self.shape)
        else:
            log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
            log_mean_coeff = tf.reshape(log_mean_coeff,self.shape)
            mean = tf.where(tf.abs(log_mean_coeff) <= 1e-3, (1 + log_mean_coeff), tf.exp(log_mean_coeff))*x
            #mean = tf.exp(log_mean_coeff)*x
            if self.sde_type == 'VPSDE':
                # std = tf.math.sqrt(1 - tf.exp(2. * log_mean_coeff))
                std = tf.where(tf.abs(log_mean_coeff) <= 1e-3, tf.math.sqrt(-2. * log_mean_coeff),
                               tf.math.sqrt(1 - tf.exp(2. * log_mean_coeff)))
            elif self.sde_type == 'subVPSDE':
                #std = 1 - tf.exp(2. * log_mean_coeff)
                std = tf.where(tf.abs(log_mean_coeff) <= 1e-3, -2. * log_mean_coeff,
                               1 - tf.exp(2. * log_mean_coeff))
        return mean, std


    def prior_sde(self,dimensions):
        if self.sde_type == 'VESDE':
            return tf.random.normal(dimensions)*self.sigma_1
        else:
            return tf.random.normal(dimensions)

    def sde(self, x, t):
        if self.sde_type == 'VESDE':
            drift = tf.zeros_like(x,dtype=tf.float32)
            sigma = self.sigma_0 * (self.sigma_1 / self.sigma_0) ** t
            diffusion = sigma * tf.math.sqrt(2 * (tf.math.log(self.sigma_1) - tf.math.log(self.sigma_0)))
            diffusion =tf.reshape(diffusion,self.shape)
        else:
            beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
            beta_t = tf.reshape(beta_t,self.shape)
            drift = -0.5 * beta_t* x
            if self.sde_type == 'VPSDE':            
                diffusion = tf.math.sqrt(beta_t)
            elif self.sde_type == 'subVPSDE':
                exponent = -2 * self.beta_0 * t - (self.beta_1 - self.beta_0) * t ** 2
                discount = 1. - tf.exp(exponent)
                discount = tf.where(tf.abs(exponent) <= 1e-3, -exponent, discount)                
                discount = tf.reshape(discount,self.shape)
                diffusion = tf.math.sqrt(beta_t * discount)
        return drift, diffusion


    @tf.function
    def train_step(self, inputs):
        eps=1e-5

        data = inputs
        init_shape = tf.shape(data)
        with tf.GradientTape() as tape:
            random_t = tf.random.uniform((tf.shape(data)[0],1))*(1-eps) + eps
            z = tf.random.normal((tf.shape(data)))

            mean,std = self.marginal_prob(data,random_t)
            perturbed_x = mean + z * std
            score = self.model([perturbed_x, random_t])
            losses = tf.square(score*std + z)
            losses = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)), axis=-1)

            loss = tf.reduce_mean(losses)

        g = tape.gradient(loss, self.trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, self.trainable_variables))        
        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}

    @tf.function
    def test_step(self, inputs):
        eps=1e-5
        data = inputs
        random_t = tf.random.uniform((tf.shape(data)[0],1))*(1-eps) + eps
        z = tf.random.normal((tf.shape(data)),seed=345)

        mean,std = self.marginal_prob(data,random_t)
        perturbed_x = mean + z * std            
        score = self.model([perturbed_x, random_t])

        losses = tf.square(score*std + z)
        losses = tf.reduce_mean(tf.reshape(losses,(losses.shape[0], -1)), axis=-1)
        loss = tf.reduce_mean(losses)

        self.loss_tracker.update_state(loss)

        return {"loss": self.loss_tracker.result()}


    @tf.function
    def call(self,x):        
        return self.model(x)


    def PCSampler(self,
                  nevts,
                  num_part,
                  # num_steps=900, 
                  # snr=0.165,
                  #num_steps=2000,
                  num_steps=200, 
                  #snr=0.23,
                  snr=0.1,
                  ncorrections=0, #Corrector currently turned off
                  eps=1e-3):
        """Generate samples from score-based models with Predictor-Corrector method.
        
        Args:
        nevts: Number of events to sample
        num_part: Number of particles to create in a single batch
        num_steps: The number of sampling steps. 
        Equivalent to the number of discretized time steps.    
        eps: The smallest time step for numerical stability.
        
        Returns: 
        Samples.
        """
        import time
        batch_size = nevts
        t = tf.ones((batch_size,1))
        data_shape = [batch_size,num_part,self.num_feat]
        const_shape = np.concatenate(([batch_size],self.shape[1:]))

        init_x = self.prior_sde(data_shape)
        time_steps = np.linspace(1., eps, num_steps)
        step_size = time_steps[0] - time_steps[1]        
        x = init_x

        #For VPSDE-like corrector
        discrete_betas = np.linspace(self.beta_0 / num_steps, self.beta_1 / num_steps, num_steps)
        alphas = 1. - discrete_betas

        start = time.time()
        for istep,time_step in enumerate(time_steps):      
            batch_time_step = tf.ones((batch_size,1)) * time_step
            time_idx = num_steps - istep -1
            z = tf.random.normal(x.shape)
            score = self.model([x, batch_time_step])

            if self.sde_type == 'VESDE':
                alpha = tf.ones(const_shape)
            else:
                alpha = tf.ones(const_shape) *alphas[time_idx]



            for _ in range(ncorrections):
                # Corrector step (Langevin MCMC)
                grad = score
                noise = tf.random.normal(x.shape)

                grad_norm = tf.reduce_mean(tf.norm(tf.reshape(grad,(grad.shape[0],-1)),axis=-1,keepdims =True),-1)
                grad_norm = tf.reduce_mean(grad_norm)

                noise_norm = tf.reduce_mean(tf.norm(tf.reshape(noise,(noise.shape[0],-1)),axis=-1,keepdims =True),-1)
                noise_norm = tf.reduce_mean(noise_norm)

                langevin_step_size = alpha*2 * (snr * noise_norm / grad_norm)**2
                langevin_step_size = tf.reshape(langevin_step_size,self.shape)
                x_mean = x + langevin_step_size * grad
                x =  x_mean + tf.math.sqrt(2 * langevin_step_size) * noise


            # Predictor step (Euler-Maruyama)

            drift,diffusion = self.sde(x,batch_time_step)
            drift = drift - (diffusion**2) * score     
            x_mean = x - drift * step_size            
            x = x_mean + tf.math.sqrt(diffusion**2*step_size) * z

        end = time.time()
        print("Time for sampling {} events is {} seconds".format(batch_size,end - start))
        # The last step does not include any noise
        return x_mean
