'''
Created on Jul 24, 2017

@author: David
'''
from keras.models import load_model
from g1qlearn import Game
import matplotlib.pyplot as plt
import numpy as np


'''
path='C:/Users/David/Desktop/Thesis/RL/model1'

model=load_model(path)

g=Game(canvasdim=(10,10),objnum=1,life=1)
canvas0=g.observe()
game_over=False


count=0
tt_rewards=0
while not game_over:

    canvas_input=canvas0.reshape(1,-1)
    action=np.argmax(model.predict(canvas_input)[0])
    canvas1,reward,game_over=g.take_action(action)
    canvas0=canvas1
    if reward>0:
        tt_rewards+=reward
    #print canvas1
    plt.imshow(canvas1, cmap='gray')
    plt.savefig('C:/Users/David/Desktop/Thesis/RL/img/'+"%03d.png" % count)
    count+=1

print 'Game Over! Overall rewards is: ',tt_rewards
'''
def save_imgs(imglist,save_path):
    count=0
    for img in imglist:
        plt.imshow(img,cmap='gray')
        plt.savefig(save_path+"%04d.png" % count)
        count+=1
        
def play(model_path,game,save_img=True,save_img_path=None):
    game.reset()
    model=load_model(model_path)
    canvas0=game.observe()
    game_over=False
    input_shape=filter(None,model.input_shape)
    
    tt_rewards=0
    img_cache=[]
    
    while not game_over:
        
        canvas_input=canvas0.reshape(input_shape)
        action=np.argmax(model.predict(canvas_input[np.newaxis])[0])
        canvas1,reward,game_over=game.take_action(action)
        canvas0=canvas1
        
        if reward>0:
            #tt_rewards+=reward
            tt_rewards+=1
            print'win count',tt_rewards
        img_cache.append(canvas1)   
    
    print 'Game Over! Overall rewards is: ',tt_rewards
    
    #show last frame
    plt.imshow(img_cache[-2],cmap='gray')
    plt.show()
    #show last prediction
    lastlist=[]
    lastlist.append(img_cache[-3].reshape(input_shape))
    lastlist.append(img_cache[-2].reshape(input_shape))
    lastlist=np.array(lastlist)
    print 'reward prediction',model.predict(lastlist,batch_size=2)
            
    if save_img:
        save_imgs(img_cache,save_img_path)              
        
if __name__=="__main__":
    path='C:/Users/David/Desktop/Thesis/RL/model2_1'
    save_img_path='C:/Users/David/Desktop/Thesis/RL/img2/'
    g=Game(canvasdim=(30,10),objnum=3,life=1)
    
    for i in range(10):
        play(path, g, save_img=False,save_img_path=save_img_path)