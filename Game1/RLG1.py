'''
Created on Jul 24, 2017

@author: David
'''
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import sgd
from g1qlearn import Game, Memory
import numpy as np


def model1(input_shape,output_dim):
    model=Sequential()
    model.add(Dense(100,input_shape=input_shape, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(output_dim))
    #model.save(filepath, overwrite)
    print(model.summary())
    return model

def training(model ,epoch, optimizer='RMSprop', loss='mse', batch_size=128):
    ep=0.1
    
    g=Game(canvasdim=(10,10),objnum=1,life=1)
    m=Memory(memory=200)
    
    #model=model1((300), 3)
    model.compile(optimizer=optimizer, loss=loss)
    
    #TRAIN
    tt_rewards=0
    
    for i in range(epoch):
        g.reset()
        loss=0
        game_over=False
        
        while not game_over:
            #observe
            canvas0=g.observe()
            canvas0=canvas0.reshape(1,-1)
            
            #choose action:
            if np.random.rand()<=ep:
                action=np.random.randint(3,size=1)[0]
            else:
                action=np.argmax(model.predict(canvas0),axis=-1)
            
            #take action
            canvas1,reward,game_over=g.take_action(action)
            #print('reward',reward)
            if reward>0:
                tt_rewards+=reward
            canvas1=canvas1.reshape(1,-1)
            
            #add to memory
            m.add([canvas0,action,reward,canvas1,game_over])
            
            x,y=m.get_training_batch(model, batch_size=32)
            
            loss+=model.train_on_batch(x,y)
        print("Epoch {:d}/{:d} | Loss {:.4f} | Win count {}".format(i, epoch, loss, tt_rewards))
    
    return model

trained_model=training(model1((100,),3), epoch=1000)
trained_model.save('C:/Users/David/Desktop/Thesis/RL/model1')

