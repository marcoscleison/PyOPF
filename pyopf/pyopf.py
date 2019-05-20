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
from sklearn.preprocessing import LabelEncoder

# def raise_(ex):
#     '''
#     Helper to raise exception from lambda function.
#     :param ex: Exception Object.
#     :return: None.
#     '''
#     raise ex


# class OpfAlgorithmNotFound(Exception):

#     def __init__(self, msg):
#         '''
#         Excpetion class for not found Opf Algorithm.
#         :param msg: Exception message.
#         '''
#         super(OpfAlgorithmNotFound, self).__init__(msg)


class OPFClassifier(object):
    def __init__(self, distance=None, precomputed=False):
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

        # Instanciate the machine
        self.opf = opfn.SupervisedFloatOpf.SupervisedOpfFloatFactory(self.precomputed, self.distance)

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
        # Check label type
        el = y[0]
        if not isinstance(el, int):
            self.non_int_label = True
            self.label_encoder = LabelEncoder()
            y_ = self.label_encoder.fit_transform(y)
        else:
            y_ = y
            
        self.opf.fit(X, y_)

    def predict(self, X):
        """
        Classifies X
        :param X: Input matrix features.
        :return: Int vector with labels (for supervised algorithm).
        """
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
