# import necessary packages 
import os
import numpy as np

import scipy
from scipy.spatial.distance import squareform
from scipy.spatial.distance import pdist

from sklearn.cluster import SpectralClustering
from sklearn.utils.linear_assignment_ import linear_assignment

import tensorflow as tf
from keras import backend as K
from keras.models import Model
from munkres import Munkres
import sklearn.metrics as ms



def getClusterLabelsFromIndexes (indexes):
    numClusters = len(indexes)
    clusterLabels = np.zeros(numClusters)
    for i in range(numClusters):
        clusterLabels[i] = indexes[i][1]
    return clusterLabels 

def calcCostMatrix(C, numClusters):
    costMat = np.zeros((numClusters, numClusters))
    # costMat[i,j] will be the cost of assigning cluster i to label j
    for j in range(numClusters):
        s = np.sum(C[:,j]) # number of examples in cluster i
        for i in range(numClusters):
            t = C[i,j]
            costMat[j,i] = s-t
    return(costMat) 
def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def _to_tensor(x, dtype):
    """Convert the input `x` to a tensor of type `dtype`.
    # Arguments
        x: An object to be converted (numpy array, list, tensors).
        dtype: The destination type.
    # Returns
        A tensor.
    """
    return tf.convert_to_tensor(x, dtype=dtype)

def squared_distance(X):
    '''
    Calculates the squared Euclidean distance matrix.

    X:              an n-by-p matrix, which includes n samples in dimension p

    returns:        n x n pairwise squared Euclidean distance matrix
    '''

    r = tf.reduce_sum(X*X, 1)
    r = tf.reshape(r, [-1, 1])
    D = r - 2*tf.matmul(X, X, transpose_b=True) + tf.transpose(r)
    return D


def full_affinity_knn(X, knn=2,fac=0.6):
    '''
    Calculates the symmetrized full Gaussian affinity matrix, the used kernel width is the median over the 
    k-nearst neighbor of all the given points in the given dataset times an input scale factor.

    X:              an n-by-p matrix, which includes n samples in dimension p
    knn:            the k in the k-nearest neighbor that will be used in order to determin the kernel width
    fac:            the scale factor of the 

    returns:        n x n affinity matrix
    '''

    Dx = squared_distance(X)    
    nn = tf.nn.top_k(-Dx, knn, sorted=True)
    knn_distances = -nn[0][:, knn - 1]
    mu=tf.contrib.distributions.percentile(knn_distances,50.,interpolation='higher')
    ml = tf.contrib.distributions.percentile(knn_distances,50.,interpolation='lower')
    sigma=(mu+ml)/2.
    sigma=tf.cond(tf.less(sigma,1e-8),lambda:1.,lambda:sigma)
    W = K.exp(-Dx/ (fac*sigma) )
    Dsum=K.sum(W,axis=1)
    Dminus=K.pow(Dsum,-1)
    Dminus=tf.linalg.diag(Dminus)
    P=tf.matmul(Dminus,W)
    return P,Dsum,W

class DataSet:
    """Base data set class
    """

    def __init__(self, shuffle=True, labeled=True, **data_dict):
        assert '_data' in data_dict
        if labeled:
            assert '_labels' in data_dict
            assert data_dict['_data'].shape[0] == data_dict['_labels'].shape[0]
        self._labeled = labeled
        self._shuffle = shuffle
        self.__dict__.update(data_dict)
        self._num_samples = self._data.shape[0]
        self._index_in_epoch = 0
        if self._shuffle:
            self._shuffle_data()

    def __len__(self):
        return len(self._data)

    @property
    def index_in_epoch(self):
        return self._index_in_epoch

    @property
    def num_samples(self):
        return self._num_samples

    @property
    def data(self):
        return self._data

    @property
    def labels(self):
        return self._labels

    @property
    def labeled(self):
        return self._labeled

    @property
    def valid_data(self):
        return self._valid_data

    @property
    def valid_labels(self):
        return self._valid_labels

    @property
    def test_data(self):
        return self._test_data

    @property
    def test_labels(self):
        return self._test_labels

    @classmethod
    def load(cls, filename):
        data_dict = np.load(filename)
        return cls(**data_dict)

    def save(self, filename):
        data_dict = self.__dict__
        np.savez_compressed(filename, **data_dict)

    def _shuffle_data(self):
        shuffled_idx = np.arange(self._num_samples)
        np.random.shuffle(shuffled_idx)
        self._data = self._data[shuffled_idx]
        if self._labeled:
            self._labels = self._labels[shuffled_idx]
    
    def get_amuont_batchs(self,batch_size):
        return int(np.ceil(self._num_samples/batch_size))
    
    def next_batch(self, batch_size):
        assert batch_size <= self._num_samples
        start = self._index_in_epoch
        if start + batch_size > self._num_samples:
            data_batch = self._data[start:]
            if self._labeled:
                labels_batch = self._labels[start:]
            remaining = batch_size - (self._num_samples - start)
            if self._shuffle:
                self._shuffle_data()
            start = 0
            data_batch = np.concatenate([data_batch, self._data[:remaining]],
                                        axis=0)
            if self._labeled:
                labels_batch = np.concatenate([labels_batch,
                                               self._labels[:remaining]],
                                              axis=0)
            self._index_in_epoch = remaining
        else:
            data_batch = self._data[start:start + batch_size]
            if self._labeled:
                labels_batch = self._labels[start:start + batch_size]
            self._index_in_epoch = start + batch_size
        batch = (data_batch, labels_batch) if self._labeled else data_batch
        return batch

    
class Model(object): 
     def __init__(self,
            input_dim,
            seed=1,
            lam=0.1,
            fac=1,
            knn=2,
            is_param_free_loss=False
        ):
        # Register hyperparameters for feature selection
        self.fac = fac
        self.knn=knn
        self.sigma = 0.5
        self.lam = lam
        self.input_dim=input_dim
        self.is_param_free_loss= is_param_free_loss


        G = tf.Graph()
        with G.as_default():
            self.sess = tf.Session(graph=G)
            # tf Graph Input
            X = tf.placeholder(tf.float32, [None, input_dim]) # i.e. mnist data image of shape 28*28=784            
            
            self.learning_rate= tf.placeholder(tf.float32, (), name='learning_rate')
            
            self.nnweights = []
            masked_input = X
            with tf.variable_scope('concrete', reuse=tf.AUTO_REUSE):
                self.alpha = tf.get_variable('alpha', [input_dim,],
                                          initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
                masked_input = self.feature_selector(masked_input)

                
            Pn,D,W=full_affinity_knn(masked_input,knn=self.knn,fac=self.fac)            
            
            ## gates regularization
            input2cdf = self.alpha
            reg = 0.5 - 0.5*tf.erf((-1/2 - input2cdf)/(self.sigma*np.sqrt(2)))
            reg_gates = tf.reduce_mean(reg)+tf.constant(1e-6)
            
            Pn=tf.matmul(Pn,Pn)
            laplacian_score= -tf.reduce_mean(tf.matmul(Pn,masked_input)*masked_input)
            
            if self.is_param_free_loss:  # Equivalent to equation 4 
                loss = laplacian_score/reg_gates
                
            else: # Is equivalent to equation 3 up to a normalization over the laplacian 
                  # score of the dimension of the data
                loss = laplacian_score +  reg_gates*self.lam 
        
 
            self.reg_gates = reg_gates # for debugging
  
            # Gradient Descent
            train_step = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(loss)
            

            accuracy=tf.Variable([0],tf.float32)
            # Initialize the variables (i.e. assign their default value)
            init_op = tf.global_variables_initializer()

            self.saver = tf.train.Saver()
            
        # Save into class members
        self.X = X

        self.loss = loss
        self.laplacian_score = laplacian_score
        self.kern=Pn
        self.train_step = train_step
        # set random state
        tf.set_random_seed(seed)
        # Initialize all global variables
        self.sess.run(init_op)
        
     def _to_tensor(self, x, dtype):
        """Convert the input `x` to a tensor of type `dtype`.
        # Arguments
            x: An object to be converted (numpy array, list, tensors).
            dtype: The destination type.
        # Returns
            A tensor.
        """
        return tf.convert_to_tensor(x, dtype=dtype)

     def hard_sigmoid(self, x):
        """Segment-wise linear approximation of sigmoid.
        Faster than sigmoid.
        Returns `0.` if `x < -0.5`, `1.` if `x > 0.5`.
        In `-0.5 <= x <= 0.5`, returns `x + 0.5`.
        # Arguments
            x: A tensor or variable.
        # Returns
            A tensor.
        """
        x = x + 0.5
        zero = _to_tensor(0., x.dtype.base_dtype)
        one = _to_tensor(1., x.dtype.base_dtype)
        x = tf.clip_by_value(x, zero, one)
        return x
    

     def feature_selector(self, x):
        '''
        feature selector - used at training time (gradients can be propagated)
        :param x - input. shape==[batch_size, feature_num]
        :return: gated input
        '''
        base_noise = tf.random_normal(shape=tf.shape(x), mean=0., stddev=1.)
        z = tf.expand_dims(self.alpha, axis=0) + self.sigma * base_noise 
        stochastic_gate = self.hard_sigmoid(z)
        masked_x = x * stochastic_gate
        return masked_x

    
    
     def get_raw_alpha(self):
        """
        evaluate the learned dropout rate
        """
        dp_alpha = self.sess.run(self.alpha)
        return dp_alpha

     def get_prob_alpha(self):
        """
        convert the raw alpha into the actual probability
        """
        dp_alpha = self.get_raw_alpha()
        prob_gate = self.compute_learned_prob(dp_alpha)
        return prob_gate
    
     def hard_sigmoid_np(self, x):
        return np.minimum(1, np.maximum(0,x+0.5))
    
     def compute_learned_prob(self, alpha):
        return self.hard_sigmoid_np(alpha)

     def load(self, model_path=None):
        if model_path == None:
            raise Exception()
        self.saver.restore(self.sess, model_path)

     def save(self, step, model_dir=None):
        if model_dir == None:
            raise Exception()
        try:
            os.mkdir(model_dir)
        except:
            pass
        model_file = model_dir + "/model"
        self.saver.save(self.sess, model_file, global_step=step)
 
     def train(self, dataset, learning_rate,batch_size,display_step=100, num_epoch=100, labeled=False):
        losses = []
        LS = []
        reg_arr=[]
        precision_arr=[]
        recall_arr=[]
        Spectral_Kmeans_acc_arr=[]
        self.display_step=display_step
        self.batch_size=batch_size
        print("num_samples : {}".format(dataset.num_samples))
        for epoch in range(num_epoch):
            avg_loss = 0. 
            avg_score=0.  
            reg_loss=0.

            # Loop over all batches
            amount_batchs= dataset.get_amuont_batchs(self.batch_size)
            for i in range(amount_batchs):
                if labeled:
                    batch_xs ,batch_ys= dataset.next_batch(self.batch_size)
                else:
                     batch_xs = dataset.next_batch(self.batch_size)
                _, loss, laplacian_score, reg_fs = self.sess.run([self.train_step, self.loss, self.laplacian_score, self.reg_gates], \
                                                          feed_dict={self.X: batch_xs, self.learning_rate:learning_rate})
                
                avg_loss += loss / amount_batchs
                avg_score += laplacian_score / amount_batchs
                reg_loss += reg_fs / amount_batchs
            
            alpha_p= self.get_prob_alpha()
            precision=np.sum(alpha_p[:2])/np.sum(alpha_p[:])
            recall=np.sum(alpha_p[:2])/2
            losses.append(avg_loss)
            LS.append(avg_score)
            reg_arr.append(reg_loss)
            recall_arr.append(recall)
            precision_arr.append(precision)
           
            if (epoch+1) % self.display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "loss=", "{:.9f}".format(avg_loss), "score=", "{:.9f}".format(avg_score)\
                      ,"reg=", "{:.9f}".format(reg_fs) )
                
                if labeled:
                    
                    indices=np.where(alpha_p>0)[0]
                    XS=batch_xs[:,indices]
                    if XS.size==0:
                        XS= np.zeros((batch_xs.shape[0],1))
                        Spectral_Kmeans_acc_arr.append(cluster_acc(batch_ys,batch_ys*0 ))
                        
                    else:

                        Dist= squareform(pdist(XS))
                        fac=5/np.max(np.min(Dist+np.eye(XS.shape[0])*1e5,axis=1))
                        clustering = SpectralClustering(n_clusters=2,affinity='rbf',gamma=fac,
                             assign_labels="discretize",
                             random_state=0).fit(XS)

                        
                        
                        netConfusion = ms.confusion_matrix(batch_ys, clustering.labels_, labels=None)
                        netCostMat = calcCostMatrix(netConfusion, 2)
                        m = Munkres()
                        indexes = m.compute(netCostMat)
                        clusterLabels = getClusterLabelsFromIndexes(indexes)
                        netSolClassVectorOrdered = clusterLabels[clustering.labels_].astype('int')
                        accuracy_ls = np.mean(netSolClassVectorOrdered == np.array(batch_ys))
                        Spectral_Kmeans_acc_arr.append(accuracy_ls)


        print("Optimization Finished!")

        return LS,losses,reg_arr,precision_arr,recall_arr,Spectral_Kmeans_acc_arr

