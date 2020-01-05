import re
ratio = 0.25 # Randomely sets 25% of the weights to zero

for layer in model.layers: 
    if(re.match(r"layer name", layer.name)): # Replace "layer name" with the name of the ablated layer
        myweights = layer.get_weights()
        X = np.array(myweights)
        N = X.shape[4]
        print(N)

        random_w = np.random.permutation(N)
        for i in range(int(ratio*N)):
            m = random_w[i]
        
            for w in myweights:
              for x in range(X.shape[1]):
                for y in range(X.shape[1]):
                  for z in range(X.shape[1]):
                      w[x][y][z][m] = 0
                        
        layer.set_weights(myweights)
        layer.trainable = False
    
    
