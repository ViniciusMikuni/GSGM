import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Input
import time
#import horovod.tensorflow.keras as hvd
import utils
from deepsets import DeepSetsAtt, Resnet
from tensorflow.keras.activations import swish, relu
from scipy import integrate
        
        
# tf and friends
#tf.random.set_seed(1234)

class GSGM(keras.Model):
    """Score based generative model"""
    def __init__(self,name='SGM',npart=30,config=None,factor=1):
        super(GSGM, self).__init__()

        self.config = config
        if config is None:
            raise ValueError("Config file not given")



        #self.activation = layers.LeakyReLU(alpha=0.01)
        self.activation = swish
        self.factor=factor

        self.num_feat = self.config['NUM_FEAT']
        self.num_jet = self.config['NUM_JET']
        self.num_cond = self.config['NUM_COND']
        self.num_embed = self.config['EMBED']
        self.max_part = npart
        self.num_steps = self.config['MAX_STEPS']//self.factor
        self.ema=0.999
        
        #self.verbose = 1 if hvd.rank() == 0 else 0 #show progress only for first rank

        
        self.projection = self.GaussianFourierProjection(scale = 16)
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.loss_part_tracker = keras.metrics.Mean(name="loss_jet")
        self.loss_jet_tracker = keras.metrics.Mean(name="loss_part")


        #Transformation applied to conditional inputs
        inputs_time = Input((1))
        inputs_cond = Input((self.num_cond))
        inputs_jet = Input((self.num_jet))
        inputs_mask = Input((None,1)) #mask to identify zero-padded objects
        

        graph_conditional = self.Embedding(inputs_time,self.projection)
        jet_conditional = self.Embedding(inputs_time,self.projection)

        #ff_jet = self.FF(inputs_jet)
        dense_jet = layers.Dense(self.num_embed,activation=None)(inputs_jet) 
        dense_jet = self.activation(dense_jet)     
        
        
        graph_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [graph_conditional,dense_jet,inputs_cond],-1))
        graph_conditional=self.activation(graph_conditional)
        
        jet_conditional = layers.Dense(self.num_embed,activation=None)(tf.concat(
            [jet_conditional,inputs_cond],-1))
        jet_conditional=self.activation(jet_conditional)

        
        self.shape = (-1,1,1)
        inputs,outputs = DeepSetsAtt(
            num_feat=self.num_feat,
            time_embedding=graph_conditional,
            num_heads=2,
            num_transformer = 6,
            projection_dim = 128,
            mask = inputs_mask,
        )
        

        self.model_part = keras.Model(inputs=[inputs,inputs_time,inputs_jet,inputs_cond,inputs_mask],
                                      outputs=outputs)
        
        outputs = Resnet(
            inputs_jet,
            self.num_jet,
            jet_conditional,
            num_embed=self.num_embed,
            num_layer = 3,
            mlp_dim= 256,
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
        return [self.loss_tracker,self.loss_part_tracker,self.loss_jet_tracker]
    

    def GaussianFourierProjection(self,scale = 30):
        #half_dim = 16
        half_dim = self.num_embed // 2
        emb = tf.math.log(10000.0) / (half_dim - 1)
        emb = tf.cast(emb,tf.float32)
        freq = tf.exp(-emb* tf.range(start=0, limit=half_dim, dtype=tf.float32))
        return freq

    def FF(self,features):
        #Gaussian features to the inputs
        max_proj = 8
        min_proj = 6
        freq = tf.range(start=min_proj, limit=max_proj, dtype=tf.float32)
        freq = 2.**(freq) * 2 * np.pi        

        x = layers.Dense(self.num_jet,activation='tanh')(features)   #normalize to the range [-1,1]
        #x = features
        freq = tf.tile(freq[None, :], ( 1, tf.shape(x)[-1]))  
        h = tf.repeat(x, max_proj-min_proj, axis=-1)
        angle = h*freq
        h = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        return tf.concat([features,h],-1)

    def Embedding(self,inputs,projection):
        angle = inputs*projection*1000.0
        embedding = tf.concat([tf.math.sin(angle),tf.math.cos(angle)],-1)
        embedding = layers.Dense(2*self.num_embed,activation=None)(embedding)
        embedding = self.activation(embedding)
        embedding = layers.Dense(self.num_embed)(embedding)
        return embedding

    def prior_sde(self,dimensions):
        return tf.random.normal(dimensions,dtype=tf.float32)
    

    #@tf.function
    def train_step(self, inputs):
        part,jet,cond,mask = inputs
        part = part*mask
        
        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
        

        
        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            
        with tf.GradientTape() as tape:
            #part
            z = tf.random.normal((tf.shape(part)),dtype=tf.float32)*mask
            perturbed_x = alpha_reshape*part + z * sigma_reshape
            pred = self.model_part([perturbed_x*mask, random_t,jet,cond,mask])
            v = alpha_reshape * z - sigma_reshape * part
            losses = tf.square(pred - v)*mask

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
            pred = self.model_jet([perturbed_x, random_t,cond])
            v = alpha * z - sigma * jet
            losses = tf.square(pred - v)
            
            loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))

        trainable_variables = self.model_jet.trainable_variables
        g = tape.gradient(loss_jet, trainable_variables)
        g = [tf.clip_by_norm(grad, 1)
             for grad in g]

        self.optimizer.apply_gradients(zip(g, trainable_variables))        
        self.loss_tracker.update_state(loss_jet + loss_part)
        self.loss_part_tracker.update_state(loss_part)
        self.loss_jet_tracker.update_state(loss_jet)
            
        for weight, ema_weight in zip(self.model_jet.weights, self.ema_jet.weights):
            ema_weight.assign(self.ema * ema_weight + (1 - self.ema) * weight)


        return {
            "loss": self.loss_tracker.result(), 
            "loss_part":self.loss_part_tracker.result(),
            "loss_jet":self.loss_jet_tracker.result(),
        }


    @tf.function
    def test_step(self, inputs):
        part,jet,cond,mask = inputs

        random_t = tf.random.uniform((tf.shape(cond)[0],1))        
        
        _, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)

        alpha_reshape = tf.reshape(alpha,self.shape)
        sigma_reshape = tf.reshape(sigma,self.shape)
            

        #part
        z = tf.random.normal((tf.shape(part)),dtype=tf.float32)
        perturbed_x = alpha_reshape*part + z * sigma_reshape

        pred = self.model_part([perturbed_x*mask, random_t,jet,cond,mask])*mask
        
        v = alpha_reshape * z - sigma_reshape * part
        losses = tf.square(pred - v)*mask        
        loss_part = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
                    
        #jet
        z = tf.random.normal((tf.shape(jet)),dtype=tf.float32)
        perturbed_x = alpha*jet + z * sigma            
        pred = self.model_jet([perturbed_x, random_t,cond])
        
        v = alpha * z - sigma * jet
        losses = tf.square(pred - v)
        
        loss_jet = tf.reduce_mean(tf.reshape(losses,(tf.shape(losses)[0], -1)))
        self.loss_tracker.update_state(loss_jet + loss_part)
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
                                    data_shape=[cond.shape[0],self.num_jet],
                                    const_shape = [-1,1]).numpy()
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))

        nparts = np.expand_dims(np.clip(utils.revert_npart(jet_info[:,-1],self.max_part,np.argmax(cond,-1)),
                                        5,self.max_part),-1) #5 is the minimum in the datasets used for training
        #print(np.unique(nparts))
        mask = np.expand_dims(
            np.tile(np.arange(self.max_part),(nparts.shape[0],1)) < np.tile(nparts,(1,self.max_part)),-1)
        
        assert np.sum(np.sum(mask.reshape(mask.shape[0],-1),-1,keepdims=True)-nparts)==0, 'ERROR: Particle mask does not match the expected number of particles'

        start = time.time()
        parts = self.DDPMSampler(cond,
                                 self.ema_part,
                                 data_shape=[cond.shape[0],self.max_part,self.num_feat],
                                 jet=jet_info,
                                 const_shape = self.shape,
                                 mask=mask.astype(np.float32)).numpy()
        
        # parts = np.ones(shape=(cond.shape[0],self.max_part,3))
        end = time.time()
        print("Time for sampling {} events is {} seconds".format(cond.shape[0],end - start))
        return parts*mask,jet_info
    
    @tf.function
    def logsnr_schedule_cosine(self,t, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return -2. * tf.math.log(tf.math.tan(a * tf.cast(t,tf.float32) + b))

    @tf.function
    def inv_logsnr_schedule_cosine(self,logsnr, logsnr_min=-20., logsnr_max=20.):
        b = tf.math.atan(tf.exp(-0.5 * logsnr_max))
        a = tf.math.atan(tf.exp(-0.5 * logsnr_min)) - b
        return tf.math.atan(tf.exp(-0.5 * tf.cast(logsnr,tf.float32)))/a -b/a

    
    @tf.function
    def get_logsnr_alpha_sigma(self,time):
        logsnr = self.logsnr_schedule_cosine(time)
        alpha = tf.sqrt(tf.math.sigmoid(logsnr))
        sigma = tf.sqrt(tf.math.sigmoid(-logsnr))
        
        return logsnr, alpha, sigma

    def get_sde(self,time,shape=None):

        with tf.GradientTape(persistent=True,
                             watch_accessed_variables=False) as tape:
            tape.watch(time)
            _,alpha,sigma = self.get_logsnr_alpha_sigma(time)
            logsnr = tf.math.log(alpha/sigma)
            logalpha= tf.math.log(alpha)
            
        f = tape.gradient(logalpha, time)
        g2 = -2*sigma**2*tape.gradient(logsnr, time)
        
        if shape is None:
            shape=self.shape
        f = tf.reshape(f,shape)
        g2 = tf.reshape(g2,shape)
        return tf.cast(f,tf.float32), tf.cast(g2,tf.float32)
        
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
        x = self.prior_sde(data_shape)

        for time_step in tf.range(self.num_steps, 0, delta=-1):
            random_t = tf.ones((batch_size, 1), dtype=tf.int32) * time_step / self.num_steps
            logsnr, alpha, sigma = self.get_logsnr_alpha_sigma(random_t)
            logsnr_, alpha_, sigma_ = self.get_logsnr_alpha_sigma(tf.ones((batch_size, 1), dtype=tf.int32) * (time_step - 1) / self.num_steps)

            if jet is None:
                score = model([x, random_t, cond], training=False)
            else:
                x *= mask
                score = model([x, random_t, jet, cond, mask], training=False) * mask
                alpha = tf.reshape(alpha, self.shape)
                sigma = tf.reshape(sigma, self.shape)
                alpha_ = tf.reshape(alpha_, self.shape)
                sigma_ = tf.reshape(sigma_, self.shape)

            mean = alpha * x - sigma * score
            eps = (x - alpha * mean) / sigma
            x = alpha_ * mean + sigma_ * eps
        
        return mean

    
    def ODESampler(self,cond,model,
                   data_shape=None,
                   jet=None,
                   mask=None,
                   const_shape=None,
                   atol=1e-5,eps=1e-5):

        from scipy import integrate
        batch_size = cond.shape[0]

        t = np.ones((batch_size,1))
        init_x = self.prior_sde(data_shape)
        

        
        @tf.function
        def score_eval_wrapper(sample, time_steps,cond,jet=None,mask=None):
            """A wrapper of the score-based model for use by the ODE solver."""
            sample = tf.cast(tf.reshape(sample,data_shape),tf.float32)

            time_steps = tf.reshape(time_steps,(sample.shape[0], 1))
            
            # logsnr_steps = tf.reshape(time_steps,(sample.shape[0], 1))
            # time_steps = self.inv_logsnr_schedule_cosine(40*logsnr_steps -20.) 
            
            logsnr_steps, alpha, sigma = self.get_logsnr_alpha_sigma(time_steps)
            
            if jet is None:
                score = model([sample, (logsnr_steps+20.)/40.,cond])
            else:
                sample*=mask
                score = model([sample, (logsnr_steps+20.)/40.,jet,cond,mask])*mask
                alpha = tf.reshape(alpha, self.shape)
                sigma = tf.reshape(sigma, self.shape)
            

            
            f,g2 = self.get_sde(time_steps,shape = const_shape)
            drift = f*sample +0.5 * g2 *score/sigma

            return tf.reshape(drift,[-1])



        def ode_func(t, x):        
            """The ODE function for use by the ODE solver."""
            time_steps = np.ones((data_shape[0]),dtype=np.float32) * t                
            return  score_eval_wrapper(x, time_steps,cond,jet,mask).numpy()
        
        res = integrate.solve_ivp(
            ode_func, (1.0-eps, eps), tf.reshape(init_x,[-1]).numpy(),
            rtol=atol, atol=atol, method='RK45')  
        print(f"Number of function evaluations: {res.nfev}")
        sample = tf.reshape(res.y[:, -1],data_shape)
        return sample


