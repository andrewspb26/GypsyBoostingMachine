
class boosting:
    
    def define_algorithm(self, name, loss_function, estimator):
        
        if name = 'plain':
            return plain_boosting(loss_function, estimator)
        elif name = 'ordering':
            return ordering_boosting(loss_function, estimator)
        else:
            return dart(loss_function, estimator)
            
    def save_model(self, path_to_storage):
        pass
    
    def load_model(self, path_to_model):
        pass
    