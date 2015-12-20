import os
import sys
import timeit
import numpy
import pdb
import cv2
import theano
import cPickle
import gzip
from theano.ifelse import ifelse
from collections import OrderedDict
import theano.tensor as T
from layers import layer
from utils import tile_raster_images
try:
    import PIL.Image as Image
except ImportError:
    import Image
	
       
def load_images(type):
    print '... loading data'
    if type == 'train':
        root = './Train_complete' 
    elif type == 'test':
        root = './Test_complete'
    else:    
        root = './ValImages'
    
    imgname = os.listdir(root)
    img_height = 32
    img_width = 32

    count = 0
    data =  numpy.zeros((len(imgname),img_height * img_width))
    for name in imgname:
        tempRGB = cv2.imread(root +'/'+ name)     
        tempgray  = rgb2gray(tempRGB)
        data[count,] = numpy.reshape(tempgray,[1, img_height * img_width])
        count = count + 1
    
    shared_x = theano.shared(numpy.asarray(data,
                                                dtype=theano.config.floatX),
                                    borrow = True)    
    return shared_x   


def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray / gray.max()
    
    	
class encoder(object):
    def __init__(
                self,
                init_params,
                flagy = True,
    ):  
          
        self.n_train_batches   = init_params [ "n_train_batches" ]
        self.batch_size        = init_params [ "batch_size" ]
        self.img_ht            = init_params [ "img_ht" ]
        self.img_wdt           = init_params [ "img_wdt" ]
        self.tile_ht           = init_params [ "tile_ht" ]
        self.tile_wdt          = init_params [ "tile_wdt" ]
        self.output_folder     = init_params [ "output_folder" ]
        self.disp_flag         = init_params [ "disp_flag" ]
    
        self.numpy_rng = numpy.random.RandomState(123)   
       
    def buildit(
                self,
                build_params, 
                flagy  = True, 
    	):
            
        self.learning_rate   = build_params [ "learning_rate" ]
        self.n_hidden_enc    = build_params [ "n_hidden_enc" ]
        self.n_hidden_dec    = build_params [ "n_hidden_dec" ]
        self.cost_fun        = build_params [ "cost_fun" ]
        self.activation      = build_params [ "activation" ]
        self.tied_weights    = build_params [ "tied_weights" ]   
        self.LR_decay        = build_params [ "LR_decay" ]	                
        self.begin_mom       = build_params [ "begin_mom" ]	
        self.end_mom         = build_params [ "end_mom" ]		
        self.mom_thrs        = build_params [ "mom_thrs" ]		
        
        print "...building the network"
        
        #allocate symbolic variables for data
        x = T.matrix('x')  
        
        next_layer_input = x
        layers = []
        enc_layers = []
        dec_layers = []
        curr_in = self.img_ht * self.img_wdt
               
        self.params = []
        
        for i in xrange(len(self.n_hidden_enc)):
            if flagy is True:
                print "..Hey! we are creating an encoding layer " + str(i)
             
            curr_out = self.n_hidden_enc[i]  
            enc_layers.append(layer(input = next_layer_input,
                                    n_in = curr_in,
                                    n_out = curr_out,
                                    activation = self.activation[i],
                                    numpy_rng = self.numpy_rng,
                                    W = None,
                                    flagy = flagy,
                                     ))
            self.params.extend(enc_layers[-1].params)                       
            next_layer_input =  enc_layers[-1].output
            curr_in = curr_out
             
        self.n_hidden_dec.append(self.img_ht * self.img_wdt)
        for i in xrange(len(self.n_hidden_dec)-1):
            if flagy is True:
                print "...Hey again! we are creating a decoding layer"  +  str(i)
            curr_in  = self.n_hidden_dec[i]
            curr_out = self.n_hidden_dec[i+1]
            dec_layers.append(layer(input = next_layer_input,
                                    n_in = curr_in,
                                    n_out = curr_out,
                                    activation = self.activation[i],
                                    numpy_rng = self.numpy_rng,
                                    W = None if self.tied_weights is False else enc_layers[len(self.n_hidden_enc)-i-1].params[0].T,
                                    flagy = flagy,
                                     ))
            next_layer_input = dec_layers[-1].output
            
            if self.tied_weights is False:
                self.params.extend(dec_layers[-1].params)
            
        z = next_layer_input
        self.z = z
            
        L = - T.sum(x * T.log(z) + (1 - x) * T.log(1 - z), axis=1)
        cce = T.mean(L)
        L2 = T.sum((x - z) ** 2,axis=1 )
        rmse = T.sqrt(T.mean(L2))
        
        if self.cost_fun == 'rmse' :
            self.cost = rmse
        elif self.cost_fun == 'cce' :
            self.cost = cce
        else :
            print " Enter a known cost function"
            
        if flagy is True:
            print "... estimating gradients"
        gradients = []      
        for param in self.params: 
            gradient = T.grad( self.cost ,param)
            gradients.append ( gradient )
        velocities = []
        for param in self.params:
            velocity = theano.shared(numpy.zeros(param.get_value(borrow=True).shape,dtype=theano.config.floatX))
            velocities.append(velocity)
         
        epoch = T.scalar()
        #pdb.set_trace();   
        self.mom = ifelse(epoch <= self.mom_thrs,
            self.begin_mom*(1.0 - epoch/self.mom_thrs) + self.end_mom*(epoch/self.mom_thrs),
            self.end_mom) 
        self.eta = theano.shared(numpy.asarray(self.learning_rate,dtype=theano.config.floatX))
        updates = OrderedDict()                           
        for param, gparam,velocity in zip(self.params, gradients,velocities):
            updates[velocity] = self.mom * velocity - (1.-self.mom) * self.eta * gparam  
            updates[param] = param + updates[velocity]
        self.get_mom = theano.function(
            inputs = [epoch],
            outputs = self.mom
        )
      
               
        index = T.lscalar() 
        
        self.train_set_x = load_images ('train')
        self.test_set_x  = load_images ('test')
        self.valid_set_x = load_images ('valid')
        
        self.train_ae = theano.function(
            inputs = [index,epoch],
            outputs = self.cost,
            updates=updates,
            givens={
                x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size] #slicing g
            }
        )
       
        self.test_ae = theano.function(
            inputs = [index],
            outputs = self.cost,
           givens={
                x: self.test_set_x[index * self.batch_size: (index + 1) * self.batch_size] #slicing g
            }
        )
        
        self.recon_op = theano.function(
            inputs = [index],
            outputs = self.z,
            givens = {
                x: self.train_set_x[index * self.batch_size: (index + 1) * self.batch_size]
            }
        )       
   
        self.decay_learning_rate = theano.function(
               inputs=[],          # Just updates the learning rates. 
               updates={self.eta: self.eta -  self.eta * self.LR_decay }
                )
  
    def visualize(
                self,
                epoch,
                batch_index,
                disp_flag = True
                ):
        newpath = './momentum_full_op_images/'+str(self.output_folder)+'/epoch_' + str(epoch) 
        if not os.path.exists(newpath):
                os.makedirs(newpath)
        ans = self.recon_op(batch_index)
        image = Image.fromarray(tile_raster_images(
        X=ans,
        img_shape=(self.img_ht, self.img_wdt), tile_shape=(self.tile_ht, self.tile_wdt),
        tile_spacing=(1, 1)))
           
        finalpath = newpath + '/batch_' + str(batch_index) + '.png'
        image.save(finalpath)
        
            
    def train(
                self,
                train_params,
                flagy = True,
    ):
        
        self.training_epochs = train_params [ "training_epochs" ]
        
        print "...training"
        start_time = timeit.default_timer()
        # go through training epochs
        for epoch in xrange(self.training_epochs):
            c = []
            for batch_index in xrange(self.n_train_batches):
                c.append(self.train_ae(batch_index,epoch))
                self.visualize(epoch = epoch, batch_index = batch_index, disp_flag = self.disp_flag)                
            print 'Training epoch %d, RMSE ' % epoch, numpy.mean(c)
            #print 'Training epoch %d, RMSE ' % epoch, numpy.mean(c),self.eta.get_value(borrow = True),self.get_mom(epoch)
            s = str(numpy.mean(c))
            s = s + " "
            wrfi = open('./momentum_full_op_images/rmsefolder/rmse_2048.txt','a')
            wrfi.write(s)
            wrfi.close()
            self.decay_learning_rate()

        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print >> sys.stderr, ('The code for file ran for %.2fm' % ((training_time) / 60.))
        ct = []
        for batch_index in xrange(self.n_train_batches):
                ct.append(self.test_ae(batch_index))
        print 'Testing RMSE ', numpy.mean(ct)                       
            
        os.chdir('../')    
