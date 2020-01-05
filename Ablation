import re
ratio = 0.25
for layer in model.layers: 
    if(re.match(r"conv2d_158", layer.name)):
        print(layer.name)    
        myweights = layer.get_weights()
        X = np.array(myweights)
        #N = X.shape[1]*X.shape[2]*X.shape[3]
        N = X.shape[4]
        print(N)

        random_w = np.random.permutation(N)
        for i in range(int(ratio*N)):
            m = random_w[i]
            print(m)
            '''z = int((m - m % (X.shape[1]*X.shape[2])) / (X.shape[1]*X.shape[2]))
            temp = m % (X.shape[1]*X.shape[2])
            y = int((temp - temp % X.shape[1]) / X.shape[1])
            x = temp % X.shape[1]
            print(x,y,z)
            for w in myweights:
                w[x][y][z] = 0'''
            for w in myweights:
              for x in range(X.shape[1]):
                for y in range(X.shape[1]):
                  for z in range(X.shape[1]):
                      w[x][y][z][m] = 0
        layer.set_weights(myweights)
        layer.trainable = False
    
