# autoencoders
This was my project on Implementing Autoencoders in neural networks using a python based library called Theano. 
Autoencoder is an artificial neural network used for dimensionality reduction. Autoencoders fall under the family of unsupervised neural networks where input units are encoded to form a hierarchical hidden representation through self learning which uses onlinear function to inturn decode back to reconstruct the input. 

In this project, we try to reconstruct the original 32x32 image using minimum number of nodes. The criterion for evaluation will be the same i.e. RMSE between original and reconstructed test image.

Theory behind it:
The number of hidden units in the hidden layers is much less than number of visible (input/output) ones, when the data passes through the network, it first compresses (encodes) input vector to fit in a smaller representation, and then tries to reconstruct (decode) it back. As the number of hidden layers increase, the autoencoder learns multiple levels of abstraction of increasing complexity. The task of training is to  minimize an error or reconstruction, i.e. find the most efficient compact representation (encoding) for input data. As the number of hidden layers increase, the autoencoder learns multiple levels of abstraction of increasing complexity. Learning is basically how the extraction of features happens in a hierarchical fashion, say the first hidden layers learns the edges and the next hidden layer learns the combination of edges viz, features and as the the number of hidden layers increase, network learns more and more prime features like contours of eyes and noses and so on and so forth.


ReadMe:
There are four python files:
1) initialize.py
2) bulid.py
3) layers.py
4) utils.py

Run the initialize.py by setting the input params(activation func name, cost func name, number of epochs, learning rate, momentum, batch size,number of train batches,etc) as required. (Note that the encoding layers should be equal to the decoding layers)
The code requires TrainImages, ValImages , TestImages to run.
For demo purposes, the given run file(initialize.py) has few parameters pre initialized. There are three demo cases available in the code, demo 1 is the default case but you and uncomment the other demos and check out how the experiment varies! :)


Dataset:
This project works on ANY dataset. The datasets used in this project are: MNIST and human faces. The size of these images is 28x28 pixels and have been normalized and stored in a pickled format. The other database used is the customised dataset of 13134(+100) images of human faces of size 32x32 pixels. Each image was normalized based on the maximum grayscale value of its original image. The training set consisted of 6567 images and the testing image set of 100 were considered.
(Dataset dropbox link will be sent on request)

