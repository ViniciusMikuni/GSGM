import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
#import horovod.tensorflow.keras as hvd
import utils

from deepsets import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu


# tf and friends
tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM',npart=30,config=None,factor=1):
        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")

        self.activation = layers.LeakyReLU(alpha=0.01)
        
        self.beta_0 = 0.1
        self.beta_1 = 20.0
        
        self.sigma2_0 = 3e-5
        self.sigma2_1 = 0.999

        
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.ema=0.999

        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank

        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects
        

        graph_conditional = self.Embedding(inputs_time,self.projection)
        jet_conditional = self.Embedding(inputs_time,self.projection)


        
        graph_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [graph_conditional,inputs_jet,inputs_cond],-1))
        graph_conditional=self.activation(graph_conditional)
        
        jet_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [jet_conditional,inputs_cond],-1))
        jet_conditional=self.activation(jet_conditional)

        
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=1,
            num_transformer = 8,
            projection_dim = 64,
            mask = inputs_mask,
        )
        

        self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_jet,inputs_cond,inputs_mask],
                                      outputs=outputs)
        
        outputs = Resnet(
            inputs_jet,
            self.num_jet,
            jet_conditional,
            num_embed=self.num_embed,
            num_layer = 5,
            mlp_dim= 512,
        )
        
        self.model_jet = keras.Model(inputs=[inputs_jet,inputs_time,inputs_cond],
                                     outputs=outputs)

            
        self.ema_jet = keras.models.clone_model(self.model_jet)
        self.ema_part = keras.models.clone_model(self.model_part)
        
        
    @property
    def metrics(self):
        """List of the model's metrics.
        We make sure the loss tracker is listed as part of `model.metrics`
        so that `fit()` and `evaluate()` are able to `reset()` the loss tracker
        at the start of each epoch and at the start of an `evaluate()` call.
        """
        return [self.loss_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        #half_dim = 16
        half_dim = self.num_embed // 4
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq


    def Embedding(self,inputs,projection):
        angle = inputs*projection
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)

    def marginal_prob(self,t,shape=None):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        if shape is None:
            shape=self.shape
        log_mean_coeff = tf.reshape(log_mean_coeff,shape)
        mean = tf.exp(log_mean_coeff)
        std = tf.math.sqrt(1 - tf.exp(2. * log_mean_coeff))
        return mean, std

    def sde(self,t,shape=None):
        
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        if shape is None:
            shape=self.shape
        beta_t = tf.reshape(beta_t,shape)
        drift = -0.5 * beta_t
        diffusion = tf.math.sqrt(beta_t)
         
        return tf.cast(drift,tf.float32), tf.cast(diffusion,tf.float32)


    def inv_var(self,var):
        #Return inverse variance for importance sampling

        c = tf.math.log(1 - var)
        a = self.beta_1 - self.beta_0
        t = (-self.beta_0 + tf.sqrt(tf.square(self.beta_0) - 2 * a * c)) /a 
        return t


    
    @tf.function
    def train_step(self, inputs):
        eps=1e-5        
        part,jet,cond,mask = inputs
        
        random_t = tf.random.uniform((tf.shape(cond)[0],1))*(1-eps) + eps
        
        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
            mean,std = self.marginal_prob(random_t)
            
            perturbed_x = mean*part + z * std 
            score = self.model_part([perturbed_x, random_t,jet,cond,mask])

            
            losses = tf.square(score*std + z)*mask
            loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
            
        trainable_variables = self.model_part.trainable_variables
        g = tape.gradient(loss_part, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))

        for weight, ema_weight in zip(self.model_part.weights, self.ema_part.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)
        
        with tf.GradientTape() as tape:
            #jet
            z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
            mean,std = self.marginal_prob(random_t,shape=(-1,1))
            
            perturbed_x = mean*jet + z * std
            score = self.model_jet([perturbed_x, random_t,cond])
            losses = tf.square(score*std + z)
            loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))

        trainable_variables = self.model_jet.trainable_variables
        g = tape.gradient(loss_jet, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))        
        self.loss_tracker.update_state(loss_jet + loss_part)

            
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }


    @tf.function
    def test_step(self, inputs):
        eps = 1e-5
        part,jet,cond,mask = inputs

        random_t = tf.random.uniform((tf.shape(cond)[0],1))*(1-eps) + eps

        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        mean,std = self.marginal_prob(random_t)
            
        perturbed_x = mean*part + z * std 
        score = self.model_part([perturbed_x, random_t,jet,cond,mask]) 
        losses = tf.square(score*std + z)*mask
                        
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                    
        #jet
        z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        mean,std = self.marginal_prob(random_t,shape=(-1,1))
        
        perturbed_x = mean*jet + z * std
        score = self.model_jet([perturbed_x, random_t,cond])
        losses = tf.square(score*std + z)
        loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
        self.loss_tracker.update_state(loss_jet + loss_part)
        
        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":tf.reduce_mean(loss_part),
            "loss_jet":tf.reduce_mean(loss_jet),
        }

            
    @tf.function
    def call(self,x):        
        return self.model(x)

    def generate(self,cond,jet_info):
        start = time.time()
        jet_info = self.ODESampler(cond,self.ema_jet,
                                   data_shape=[self.num_jet]).numpy()
        
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))

        nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part),
                                        0,self.max_part),-1)
        
    
        mask = np.expand_dims(
            np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)

        start = time.time()
        parts = self.ODESampler(tf.convert_to_tensor(cond,dtype=tf.float32),
                                self.ema_part,
                                data_shape=[self.max_part,self.num_feat],
                                jet=tf.convert_to_tensor(jet_info, dtype=tf.float32),
                                mask=tf.convert_to_tensor(mask, dtype=tf.float32)).numpy()

        # parts = self.DDPMSampler(tf.convert_to_tensor(cond,dtype=tf.float32),
        #                         self.ema_part,
        #                         data_shape=[self.max_part,self.num_feat],
        #                         jet=tf.convert_to_tensor(jet_info, dtype=tf.float32),
        #                         mask=tf.convert_to_tensor(mask, dtype=tf.float32)).numpy()

        # parts = np.ones(shape=(cond.shape[0],self.max_part,3))
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        return parts*mask,jet_info




    def ODESampler(self,cond,model,
                   data_shape=None,
                   jet=None,
                   mask=None,
                   atol=1e-5,eps=1e-5):

        from scipy import integrate
        batch_size = cond.shape[0]

        t = np.ones((batch_size,1))
        data_shape = np.concatenate(([batch_size],data_shape))
        init_x = self.prior_sde(data_shape)
        
        shape = init_x.shape
        
        @tf.function
        def score_eval_wrapper(sample, time_steps,cond,jet=None,mask=None):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = tf.cast(tf.reshape(sample,shape),tf.float32)
            time_steps = tf.reshape(time_steps,(sample.shape[0], 1))
            if jet is None:
                score = model([sample, time_steps,cond])
            else:
                score = model([sample*mask, time_steps,jet,cond,mask])*mask
            return tf.reshape(score,[-1])



        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((shape[0],)) * t    
            f,g = self.sde(t,shape= (-1))
            return  f*x -0.5 * (g**2) * score_eval_wrapper(x, time_steps,cond,jet,mask).numpy()
        
        res = integrate.solve_ivp(
            ode_func, (1.0, 1e-5), tf.reshape(init_x,[-1]).numpy(),
            rtol=atol, atol=atol, method='RK45')  
        print(f"Number of function evaluations: {res.nfev}")
        sample = tf.reshape(res.y[:, -1],shape)
        return sample


