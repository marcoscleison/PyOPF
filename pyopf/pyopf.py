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


def raise_(ex):
    '''
    Helper to raise exeption form lambda function.
    :param ex: Exception Object.
    :return: None.
    '''
    raise ex


class OpfAlgorithmNotFound(Exception):

    def __init__(self, msg):
        '''
        Excpetion class for not found Opf Algorithm.
        :param msg: Exception message.
        '''
        super(OpfAlgorithmNotFound, self).__init__(msg)


class OPF(object):
    def __init__(self, distance=None, precomputed=False, algorithm="supervised"):
        '''
        OPF classifier main class constructor.
        :param distance: python function or string informing the distance. ['euclidean','cos']
        :param precomputed: True if the data input is the precomputed pairwise distance.
        :param algorithm: Opf algorithm name. ['supervised']
        '''
        self.distance = distance
        self.precomputed = precomputed
        self.algorithm = algorithm

        self.algorithm_driver = {
            "supervised": self._create_supervised_opf,
            "unsupervised": lambda: raise_(NotImplementedError("Unsupervised OPF is not implemented yet."))
        }

        if self.algorithm in self.algorithm_driver.keys():
            self.opf = self.algorithm_driver[self.algorithm]()
        else:
            raise OpfAlgorithmNotFound("Opf algorithm not found.")

    def _create_supervised_opf(self):

        if self.distance is not None:
            opf = opfn.SupervisedOpfFloatProxy.SupervisedOpfFloatFactory(self.precomputed, self.distance)
        else:
            opf = opfn.SupervisedOpfFloatProxy.SupervisedOpfFloatFactory(self.precomputed, 'euclidean')
        return opf

    def fit(self, X, y):
        '''
        Trains the Opf classifier using.
        :param X: Input features matrix.
        :param y: int vector representing labels (for supervised algorithm).
        :return: None
        '''
        self.opf.fit(X, y)

    def predict(self, X):
        """
        Classifies X
        :param X: Input matrix features.
        :return: Int vector with labels (for supervised algorithm).
        """
        return self.opf.predict(X)

    def save_weights(self, driver=None):
        pass

    def load_weights(self, driver):
        pass
