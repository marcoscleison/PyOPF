"""
Copyright 2019 PyOPF Contributors

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""
import pyopf_native as opfn
import numpy as np
from sklearn.preprocessing import LabelEncoder



class OPFClassifier(object):
    def __init__(self, copy=False, distance=None, precomputed=False):
        '''
        OPF classifier main class constructor.
        :param distance: python function or string informing the distance. ['euclidean','cosine']
        :param precomputed: True if the data input is the precomputed pairwise distance.
        :param algorithm: Opf algorithm name. ['supervised']
        '''
        
        self.precomputed = precomputed
        self.distance = distance
        if self.distance is None:
            self.distance = 'euclidean'
        self.copy = copy

        # Instanciate the machine
        self.opf = opfn.SupervisedFloatOpf.SupervisedOpfFloatFactory(self.precomputed, self.distance, self.copy)

        # Handle non-integer label types
        self.non_int_label = False
        self.label_encoder = None

    def fit(self, X, y):
        '''
        Trains the Opf classifier using.
        :param X: Input features matrix.
        :param y: int vector representing labels (for supervised algorithm).
        :return: None
        '''
        if len(y) == 0:
            raise ValueError("Data size must be higher than 0.")
        if X.dtype != np.float32:
            raise ValueError("OPF fit: Data values must be float.")
        # Check label type
        el = y[0]

        if not isinstance(el, (np.int64, np.int32, int)):
            self.non_int_label = True
            self.label_encoder = LabelEncoder()
            y_ = self.label_encoder.fit_transform(y)
        else:
            y_ = y #.astype(np.int32)
            
        self.opf.fit(X, y_)

    def predict(self, X):
        """
        Classifies X
        :param X: Input matrix features.
        :return: Int vector with labels (for supervised algorithm).
        """

        if X.dtype != np.float32:
            raise ValueError("OPF predict: Data values must be float.")

        preds = self.opf.predict(X)

        if self.non_int_label:
            preds = self.label_encoder.inverse_transform(preds)

        return preds

    def get_params(self, deep=True):
        return {}
    def set_params(self, **params):
        pass

    def save_weights(self, driver=None):
        pass

    def load_weights(self, driver):
        pass


class OPFClustering(object):
    def __init__(self, k=5, anomaly=False, thresh=.1, copy=False, distance=None, precomputed=False):
        '''
        OPF clustering main class constructor.
        :param distance: python function or string informing the distance. ['euclidean','cosine']
        :param precomputed: True if the data input is the precomputed pairwise distance.
        :param algorithm: Opf algorithm name. ['supervised']
        '''
        
        self.k = k
        self.anomaly = anomaly
        self.thresh = thresh
        self.n_clusters = 0
        self.precomputed = precomputed
        self.distance = distance
        self.copy = copy
        if self.distance is None:
            self.distance = 'euclidean'

        # Instanciate the machine
        self.opf = opfn.UnsupervisedFloatOpf.UnsupervisedOpfFloatFactory(self.k, self.anomaly, self.thresh, self.precomputed, self.distance, self.copy)


    def fit(self, X):
        '''
        Trains the Opf clustering using.
        :param X: Input features matrix.
        :return: None
        '''
        if X.dtype != np.float32:
            raise ValueError("OPF fit: Data values must be float.")
        # Check label type
            
        self.opf.fit(X)
        self.n_clusters = self.opf.get_n_clusters()
    
    def fit_predict(self, X):
        '''
        Trains the Opf clustering using.
        :param X: Input features matrix.
        :return: None
        '''
        if X.dtype != np.float32:
            raise ValueError("OPF fit: Data values must be float.")
        # Check label type
            
        preds = self.opf.fit_predict(X)
        self.n_clusters = self.opf.get_n_clusters()
        return preds

    def predict(self, X):
        """
        Classifies X
        :param X: Input matrix features.
        :return: Int vector with labels (for supervised algorithm).
        """

        if X.dtype != np.float32:
            raise ValueError("OPF predict: Data values must be float.")

        return self.opf.predict(X)
    
    def find_best_k(self, train_data, kmin=2, kmax=202, step=5):
        if train_data.dtype != np.float32:
            raise ValueError("OPF find_best_k: Data values must be float.")
        self.opf.find_best_k(train_data, kmin, kmax, step)
        self.k = self.opf.get_k()
        self.n_clusters = self.opf.get_n_clusters()

    @staticmethod
    def unserialize(data):
        opf = OPFClustering()
        c_opf = opfn.UnsupervisedFloatOpf.unserialize(data)

        opf.k = c_opf.get_k()
        opf.anomaly = c_opf.get_anomaly()
        opf.thresh = c_opf.get_thresh()
        opf.n_clusters = c_opf.get_n_clusters()
        opf.precomputed = c_opf.get_precomputed()
        opf.distance = 'euclidean'

        opf.opf = c_opf

        return opf

    def __reduce__(self, flags=0):
        data = self.opf.serialize(flags)
        
        return (OPFClustering.unserialize, (data,))

    def get_params(self, deep=True):
        return {}
    
    def set_params(self, **params):
        pass

    def save_weights(self, driver=None):
        pass

    def load_weights(self, driver):
        pass
