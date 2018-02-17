""" 
This module implements gradient boosting regression for all scikit-learn 
estimators or user-defined algorithms. User-defined algorithm should implement
method fit() and method predict() in scikit-learn fashion.
 
Example:
       gb = GypsyBoost(loss_function, DecisionTreeRegressor())
       for loss in gb.grow_ensemble(10, X_train, y_train):
           if abs(prev_loss - loss) < stop_criterion:
               break
       y_pred = GBLR.predict(X_test)
 
"""


import autograd.numpy as np
from autograd import elementwise_grad
from copy import copy
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
 
class GypsyBoost:
    
    def __init__(self, loss_function, estimator):
 
        if callable(loss_function) and hasattr(estimator, 'fit'):
            self.grad_loss = elementwise_grad(loss_function)
            self.loss = loss_function
            self.estimator = estimator
            self.X_train = None
            self.y_train = None
            self.ens_pred = None
            self.ensemble = []
            self.gammas = []
        else:
            raise AssertionError('wrong arguments has been passed')
            
    def grow_ensemble(self, n_estimators, X, y, validation=0.1, shuffle=True):
        
        """ this method build ensemble of estimators 
        passed in class constructor
        
        Args:
            n_estimators (int): number of estimators in ensemble
            X (numpy array): array (n*m) with features
            y (numpy array): array (n,) with target
        """
        self.X_train, X_val, self.y_train, y_val  = train_test_split(X, y, test_size=validation, shuffle=shuffle)
        r = self.y_train.copy()
        for i in range(n_estimators):
            regressor = self.estimator
            regressor.fit(self.X_train, r)
            if len(self.ensemble) != 0:
                gamma = float(minimize(self.__loss_wrap, 1, method='L-BFGS-B').x)
                self.gammas.append(gamma)
            else:
                self.gammas.append(1.0)
            self.ensemble.append(copy(regressor))
            self.ens_pred = sum(gamma*estimator.predict(self.X_train) for estimator, gamma in zip(self.ensemble, self.gammas))
            r = -1*self.grad_loss(self.y_train, self.ens_pred)
            yield np.mean(self.loss(y_val, self.predict(X_val)))
            
    def __loss_wrap(self, gamma):
        f_step = self.estimator.predict(self.X_train)
        return np.mean(self.loss(self.y_train, self.ens_pred+gamma*f_step))
 
    def predict(self, X):
        
        """ this method returns prediction using ensemble
        built by calling grow_ensemble()
        
        Args:
            X (numpy array): array (n*m) with features
        
        Returns:
            prediction (numpy array): array with predictions (n,)
        """
        
        prediction = np.zeros(X.shape[0])
        
        for estimator, gamma in zip(self.ensemble, self.gammas):
            prediction += gamma*estimator.predict(X)
            
        return prediction





