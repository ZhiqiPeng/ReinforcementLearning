'''
Created on Jul 25, 2017

@author: David
'''
from keras.models import Sequential
from keras.models import load_model
from keras.layers.core import Dense
from keras.layers import Convolution1D, Flatten
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
def model2(input_shape,output_dim):
    model=Sequential()
    model.add(Convolution1D(32,5,input_shape=input_shape,activation='relu'))
    model.add(Convolution1D(32,5,activation='relu'))
    model.add(Flatten())
    model.add(Dense(output_dim))
    print(model.summary())
    return model
    
def training(model ,epoch, game, memory, optimizer='RMSprop', loss='mse', batch_size=128, epsilon=0.1):
    
    model.compile(loss=loss,optimizer=optimizer)
    input_shape=filter(None,model.input_shape)
    #input_shape=(1,)+input_shape
    #print input_shape
    num_actions=model.output_shape[-1]
    tt_rewards=0
    
    for i in range(epoch):
        game.reset()
        loss=0
        game_over=False
        
        while not game_over:
            #observe
            canvas0=game.observe()
            canvas0=canvas0.reshape(input_shape)
            
            #choose action:
            if np.random.rand()<=epsilon:
                action=np.random.randint(num_actions,size=1)[0]
            else:
                action=np.argmax(model.predict(canvas0[np.newaxis]),axis=-1)
            
            #take action
            canvas1,reward,game_over=game.take_action(action)
            #print('reward',reward)
            if reward>0:
                #tt_rewards+=reward
                tt_rewards+=1#if there is a penalty in reward and tt_rewards means win count.
            canvas1=canvas1.reshape(input_shape)
            
            #add to memory
            memory.add([canvas0,action,reward,canvas1,game_over])
            
            x,y=memory.get_training_batch(model, batch_size=batch_size)
            
            loss+=model.train_on_batch(x,y)
        print("Epoch {:d}/{:d} | Loss {:.4f} | Win count {}".format(i, epoch, loss, tt_rewards))
    
    return model
def retrain(model_path, epoch, game, memory, optimizer='RMSprop', loss='mse', batch_size=128, epsilon=0.1):   
    model=load_model(model_path)
    print(model.summary())
    model=training(model,epoch, game, memory, optimizer=optimizer, loss=loss, batch_size=batch_size, epsilon=epsilon)
    return model

if __name__=='__main__':
    
    g=Game(canvasdim=(30,10),objnum=3,life=1)
    m=Memory(memory=1500)
    
    #model=model1((300,), 3)
    #model=model2((30,10), 3)
    #model=training(model, epoch=1000, game=g, memory=m)
    model=retrain('C:/Users/David/Desktop/Thesis/RL/model1_1_1', epoch=1000, game=g, memory=m,batch_size=512)
    model.save('C:/Users/David/Desktop/Thesis/RL/model1_1_1')