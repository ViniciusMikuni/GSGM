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
        #self.activation = swish
        # self.activation = relu
        self.factor=factor
        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.num_steps = self.config['MAX_STEPS']//self.factor
        self.ema=0.999

        self.timesteps =tf.range(start=0,limit=self.num_steps + 1, dtype=tf.float32) / self.num_steps + 8e-3 
        alphas = self.timesteps / (1 + 8e-3) * np.pi / 2.0
        alphas = tf.math.cos(alphas)**2
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        self.betas = tf.clip_by_value(betas, clip_value_min =0, clip_value_max=0.999)
        alphas = 1 - self.betas
        self.alphas_cumprod = tf.math.cumprod(alphas, 0)
        alphas_cumprod_prev = tf.concat((tf.ones(1, dtype=tf.float32), self.alphas_cumprod[:-1]), 0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef1 = (self.betas * tf.sqrt(alphas_cumprod_prev) / (1. - self.alphas_cumprod))
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * tf.sqrt(alphas) / (1. - self.alphas_cumprod)
        

        
                
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
    

    @tf.function
    def train_step(self, inputs):
        part,jet,cond,mask = inputs


        random_t = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
        
        #random_t = tf.cast(random_t,tf.float32)
        alpha = tf.gather(tf.sqrt(self.alphas_cumprod),random_t)
        sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),random_t)
        sigma = tf.clip_by_value(sigma, clip_value_min = 1e-3, clip_value_max=0.999)

        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            
        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
            perturbed_x = alpha_reshape*part + z * sigma_reshape
            score = self.model_part([perturbed_x, random_t,jet,cond,mask])
            
            v = alpha_reshape * z - sigma_reshape * part
            losses = tf.square(score - v)*mask
            
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
            perturbed_x = alpha*jet + z * sigma            
            score = self.model_jet([perturbed_x, random_t,cond])
            v = alpha * z - sigma * jet
            losses = tf.square(score - v)
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
        part,jet,cond,mask = inputs


        random_t = tf.random.uniform(
            (tf.shape(cond)[0],1),
            minval=0,maxval=self.num_steps,
            dtype=tf.int32)
            
        #random_t = tf.cast(random_t,tf.float32)
        alpha = tf.gather(tf.sqrt(self.alphas_cumprod),random_t)
        sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),random_t)
        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            

        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        perturbed_x = alpha_reshape*part + z * sigma_reshape



        score = self.model_part([perturbed_x, random_t,jet,cond,mask])
        v = alpha_reshape * z - sigma_reshape * part
        losses = tf.square(score - v)*mask
            
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                    
        #jet
        z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        perturbed_x = alpha*jet + z * sigma            
        score = self.model_jet([perturbed_x, random_t,cond])
        v = alpha * z - sigma * jet
        losses = tf.square(score - v)
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

    def generate_jet(self,cond):
        start = time.time()
        jet_info = self.DDPMSampler(cond,self.ema_jet,
                                    data_shape=[self.num_jet],
                                    const_shape = [-1,1]).numpy()
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        return jet_info


    def generate(self,cond,jet_info):
        start = time.time()
        jet_info = self.DDPMSampler(cond,self.ema_jet,
                                    data_shape=[self.num_jet],
                                    const_shape = [-1,1]).numpy()
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))

        nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part),
                                        0,self.max_part),-1)
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



    @tf.function
    def DDPMSampler(self,
                    cond,
                    model,
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
        
        batch_size = cond.shape[0]
        t = tf.ones((batch_size,1))
        data_shape = np.concatenate(([batch_size],data_shape))
        cond = tf.convert_to_tensor(cond, dtype=tf.float32)
        init_x = self.prior_sde(data_shape)
        if jet is not None:
            init_x *= mask 

        x = init_x
        
        
        for  time_step in tf.range(self.num_steps, 0, delta=-1):
            batch_time_step = tf.ones((batch_size,1),dtype=tf.int32) * time_step
            z = tf.random.normal(x.shape,dtype=tf.float32)

            alpha = tf.gather(tf.sqrt(self.alphas_cumprod),batch_time_step)
            sigma = tf.gather(tf.sqrt(1-self.alphas_cumprod),batch_time_step)
            
            if jet is None:
                score = model([x, batch_time_step,cond],training=False)
            else:
                score = model([x, batch_time_step,jet,cond,mask],training=False)
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
