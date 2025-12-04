import numpy as np
class SGD:
   def __init__(self,lr=0.01): "initialize learning rate with a value "
       self.lr = lr
       
    def step(self, layers):
        for layer in layers:               # visit every layer 
            for (weights, grad) in layer.params():   # checks is the layer has weights and gradient
                weights = weights - self.lr * grad
       
   "   ===============================        =============================== "


       
    """
        looping is done inside other class 
        outer loop for epochs and shuffling data to and get batches from them
        inner loop for training every batch of the dataset 
        ===============================        ===============================
def train(self, X_train, y_train, optimizer, n_epochs, batch_size=32):
    m = len(X_train)  " size of dataset "
    n_batches = m // batch_size 
    
    for epoch in range(n_epochs):
        shuffled_indices = np.random.permutation(m)
        X_shuffled = X_train[shuffled_indices] " shuffles list of indices (Raandom dataa)"
        y_shuffled = y_train[shuffled_indices] " shuffles list of indices (Raandom dataa)"
        
        for i in range(n_batches):
            start_index = i * batch_size
            end_index = (i + 1) * batch_size
            
            X_batch = X_shuffled[start_index:end_index] " slices data into random batches
            y_batch = y_shuffled[start_index:end_index] " slices data into random batches
            "Training "
            prediction = self.forward(X_batch)
            loss_gradient = self.loss_function.backward(y_batch, prediction)
            
            self.backward(loss_gradient)
            
            for layer in self.layers:
                if hasattr(layer, 'W'):
                    optimizer.step(layer)
        ===============================         ===============================

       
    """