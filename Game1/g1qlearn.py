'''
Created on Jul 21, 2017

@author: David
'''
import numpy as np


class Game(object):
    def __init__(self,canvasdim=(30,10),objnum=3,life=1):
        self.canvas_dim=canvasdim
        self.obj_num=objnum
        self.life=life
        self.org_life=life
        #self.rewards=0
        self.reset()
    def _draw_state(self):
        img_size=self.canvas_dim
        canvas=np.zeros(img_size)
        
        canvas[-1,self.status[1]-1:self.status[1]+2]=1#draw basket
        
        for r in self.status[0]:#draw obj
            canvas[r[0],r[1]]=1
        
        return canvas
    def _add_obj(self):
        st=self.status
        if len(st[0])<self.obj_num:
            if len(st[0])==0:
                st[0].append([0,np.random.randint(0,self.canvas_dim[1]-1,size=1)[0]])                
            elif st[0][-1][0]>self.canvas_dim[0]/self.obj_num:
                st[0].append([0,np.random.randint(0,self.canvas_dim[1]-1,size=1)[0]])
        
    def _update_status(self,action):
        state = self.status
        if action == 0:  # left
            action = -1
        elif action == 1:  # stay
            action = 0
        else:
            action = 1  # right        
        
        for r in state[0]:#update objs position
            r[0]+=1
        
        self._add_obj()#add new obj
        
        state[1]=min(max(1,state[1]+action),self.canvas_dim[1]-2)#update basket
        
    def _get_reward(self):
        state=self.status
        if state[0][0][0]==self.canvas_dim[0]-1:
            if abs(state[0][0][1]-state[1])<=1:
                del state[0][0]

                return 1
            else:
                self.life-=1
                del state[0][0]
                return -1*self.obj_num#lower this value will make model more focus on future rewards
        else:
            return 0
        
    def _is_over(self):
        if self.life==0:
            return True
        else:
            return False                  
    def reset(self):
        self.life=self.org_life
        #self.rewards=0
        self.status=[]#[[t_location],p_location]
        t_location=[]
        
        t=np.random.randint(0,self.canvas_dim[1]-1,size=1)[0]
        t_location.append([0,t])
        p=np.random.randint(1,self.canvas_dim[1]-2,size=1)[0]
        self.status=[t_location,p]
        
    def observe(self):
        canvas=self._draw_state()
        return canvas
    def take_action(self,action):
        self._update_status(action)
        
        #add movement panelty
        if action<>1:
            pen=0.05
        else:
            pen=0
        
        reward=self._get_reward()-pen
        #self.rewards+=reward
        is_over=self._is_over()
        return self.observe(),reward,is_over   
    

class Memory(object):
    def __init__(self,memory=200,discount=0.9):
        self.memory=memory
        self.mem_list=[]
        self.discount=discount
    
    def add(self,interaction):#[canvas0,action,reward,canvas1,game_over]
        self.mem_list.append(interaction)
        if len(self.mem_list)>self.memory:
            del self.mem_list[0]
    
    def get_training_batch(self,model,batch_size=32):
        interactions=self.mem_list
        num_actions=model.output_shape[-1]
        env_dim=interactions[0][0].shape
        #print env_dim
        int_dim=((min(len(interactions), batch_size)),)+env_dim
        #inputs=np.zeros((min(len(interactions), batch_size), env_dim))#flattened canvas only for now
        inputs=np.zeros(int_dim)#problem solved
        #print(inputs.shape)
        outputs=np.zeros((inputs.shape[0],num_actions))
        
        for i,idx in enumerate(np.random.randint(0,len(self.mem_list),size=inputs.shape[0])):           
            canvas0,action,reward,canvas1,game_over=interactions[idx]
            
            inputs[i]=canvas0
            outputs[i]=model.predict(canvas0[np.newaxis])[0]
            Q_sa=np.max(model.predict(canvas1[np.newaxis])[0])
            
            if game_over:
                outputs[i,action]=reward
            else:
                outputs[i,action]=reward+self.discount*Q_sa
        
        return inputs,outputs


        
