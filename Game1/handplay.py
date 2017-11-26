'''
Created on Jul 24, 2017

@author: David
'''
from g1qlearn import Game
import matplotlib.pyplot as plt
#init
g=Game(life=2)
game_over=False
canvas=g.observe()
print canvas

count=0
while not game_over:
    action=input('Enter action:')
    print action
    canvas,rewards,game_over=g.take_action(action)
    print 'rewards',rewards
    print canvas
    plt.imshow(canvas, cmap='gray')
    plt.savefig('C:/Users/David/Desktop/Thesis/RL/img/'+"%03d.png" % count)
    count+=1

print 'Game Over! Overall rewards is: ',rewards