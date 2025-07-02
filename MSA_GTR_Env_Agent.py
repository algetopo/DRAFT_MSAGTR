#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  1 14:29:09 2022

@author: sergio


"""

# This script aligns protein sequences randomly and converts them into
# reinforcement learning sequences.

#This python script aligns protein sequences
# randomly and converts them  into 
# reinforcement learning sequences


import math
import pdb
import numpy as np
import gym
from gym import spaces
from perfilV5 import  profalgn
import tensorflow as tf
import pickle
import collections
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
from Bio.Seq import Seq
import os


#from stable_baselines.common.env_checker import check_env


class protEnv(gym.Env):
    
    def __init__(self,sequences):
        super(protEnv,self).__init__()
        
        
        self.sequences=sequences
        self.alignment=[]
        n=len(self.sequences)
        self.state=np.zeros(n, dtype=int)
        self.observation_space = gym.spaces.Box(low=0, high =n, shape=(1, n) ,dtype=int)
        self.action_space=spaces.Discrete(len(sequences))
        self.step_index=0 
        self.reward=0
        
        
        
    def reset(self):
        
        
        n=len(self.sequences)
        self.alignment=[]
        self.state=np.zeros(n, dtype=int)
        self.step_index=0
        self.reward=0
        return self.state
    
    
    def S(self):
        print( self.observations_space.sample())
       
        
        
    def take_action(self, action):
       
       
        if self.step_index==0:
            self.alignment=[self.sequences[action]]
           
           
          
        else:
            if self.step==1:
                
                # Align two sequences on second step
                
                s_aux=[self.alignment[0],self.sequences[action]]
                self.alignment=profalgn(s_aux,[])
               
            else:
                
                # Make an alignment betwwen the already aligned sequences and
                # the new provided sequence
                
                # you could also modify this line with biopython
                # but maybe alignmnet should be 
                # biopython MSA records not just some strings
                
                
                self.alignment,self.reward=profalgn(self.alignment,self.sequences[action])
       
         
        
        
        
    def repeated(self, action): 
        
        #L=self.state.tolist()
        
        states_set=set()
        
        if self.step_index==1:
            L=[self.state[0]]
            states_set=set(L)
        
        ru_there=action in states_set
        
        if ru_there:
            return True
        else:
            states_set.add(action)
            return False
            
    
        
        
       
        return action in states_set   
        
    def get_step_index(self):
        return self.step_index
        
    def step(self,action):
        
        info={}
        
        if self.step_index==0:
            
            self.take_action(action)
            self.state[self.step_index]=action
            self.step_index=self.step_index+1 
            reward=0
            
            
        else:
            r=self.repeated(action)
            
            
            if r:
                
                self.state[self.step_index]=action
                reward=float('-inf')
                
            else: 
                
                self.take_action(action)
                self.state[self.step_index]=action
                self.step_index=self.step_index+1
                
                
                reward=self.reward
        
        
        
        done=False
        
        # if self.step_index==len(self.sequences)-1:
        #     done=True
        # else:
        #     done=False
        
        return self.state,reward,done,info
        
    
    



def reset_data():
    return {'observations': [],
            'actions': [],
            'terminals': [],
            'timeouts': [],
            'rewards': [],
            'infos/goal': [],
            }

def append_data(data, s, a, r, tgt, done, timeout):
    
    # tgt is target goal
    
    data['observations'].append(s)
    data['actions'].append(a)
    data['rewards'].append(r)
    data['terminals'].append(done)
    data['timeouts'].append(timeout)
    data['infos/goal'].append(tgt)   
    
    # def get_seqs(self):
    #     return self.alignment
        
   
def qlearning_dataset(env, dataset=None, terminate_on_end=False):
    """
    Returns datasets formatted for use by standard Q-learning algorithms,
    with observations, actions, next_observations, rewards, and a terminal
    flag.
    Args:
        env: An OfflineEnv object.
        dataset: An optional dataset to pass in for processing. If None,
            the dataset will default to env.get_dataset()
        terminate_on_end (bool): Set done=True on the last timestep
            in a trajectory. Default is False, and will discard the
            last timestep in each trajectory.
        **kwargs: Arguments to pass to env.get_dataset().
    Returns:
        A dictionary containing keys:
            observations: An N x dim_obs array of observations.
            actions: An N x dim_action array of actions.
            next_observations: An N x dim_obs array of next observations.
            rewards: An N-dim float array of rewards.
            terminals: An N-dim boolean array of "done" or episode termination flags.
    """
   

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
    
        use_timeouts = True
    
    
    episode_step = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not terminate_on_end) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step = 0
            continue  
        if done_bool or final_timestep:
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }        



def saveTrajectories(dataset,name):
    
    N = dataset['rewards'].shape[0] 
    data_=collections.defaultdict(list)
    
    use_timeouts = False
    
    if 'timeouts' in dataset:
        use_timeouts = True
        
    episode_step = 0    
    paths = []
    
    for i in range(N):
        
       
        done_bool = bool(dataset['terminals'][i])
        
        if use_timeouts:
            final_timestep = dataset['timeouts'][i] 
            
        else:
            final_timestep = (episode_step == 1000-1)
        for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
            data_[k].append(dataset[k][i])
            
            if done_bool or final_timestep:
                
                episode_step = 0
                episode_data = {} 
                
                for k in data_:
                    episode_data[k] = np.array(data_[k]) 
                    #print(np.array(data_[k]))
                    paths.append(episode_data)
                    
                data_ = collections.defaultdict(list)
            
            episode_step += 1
    
    # returns = np.array([np.sum(p['rewards']) for p in paths]) 
    # num_samples = np.sum([p['rewards'].shape[0] for p in paths])
    # print(f'Number of samples collected: {num_samples}')
    # print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
    
    
    
    name=name+'.pkl'
    with open(name, 'wb') as f: 
        pickle.dump(paths, f)


def protFile2String(input_file):
    
    # Load the .fasta file
    #input_file = "trimed_dataset.fasta"
    #input_file = "BB11002"
    
    #print(input_file)
    
    #pdb.set_trace() 
    with open(input_file) as handle:
        records = list(SeqIO.parse(handle, 'fasta'))
    
    


    # Extract the protein sequences as a list of strings
    protein_sequences = []
    for record in records:
        protein_sequences.append(str(record.seq))


    return protein_sequences

    

def npify(data):
    for k in data:
        
        if k == 'terminals' :
            dtype = np.bool_
        else:
            if k == 'timeouts' :
                dtype = np.bool_
            else:
                dtype = np.float32

        data[k] = np.array(data[k], dtype=dtype)
    


        

    data[k] = np.array(data[k], dtype=dtype)    
  
                

# if __name__ == "__main__":






def printFastaFile(input_file):
    
    with open(input_file) as fasta_file:
        records = list(SeqIO.parse(fasta_file, "fasta"))
        for record in records:
            # Print the ID and sequence for each record
            print("ID:", record.id)
            print("Sequence:", record.seq)
            print("Sequence:", record.seq)


def ProtSeq2RLseqs(file_name,input_path="",output_path=""):
    
    if input_path=="ninguno":
        
        output_file=file_name+"_" + "out"+".npy"
        
    else:
        
        input_file=input_path+file_name
        output_file=output_path+file_name+"_" + "out"+".npy"  
        
        
         
    
    secuencias= protFile2String(input_file)
    
    
    
    env=protEnv(secuencias)  
    
    Num_actions=env.action_space.n
    

    
    
    Num_samples=int(((Num_actions*(Num_actions+1))/2))
    
    #print(Num_samples)
    
    
    samples = np.empty(shape=(Num_samples,Num_actions), dtype='object')
    
    np.random.seed(42)
    
    # We will only extract  n out of  2 samples 
    # n from  all the n factorial permutations that exist
    
    
    
    for row in range(Num_samples):
        samples[row,]=np.random.choice(Num_actions,Num_actions, replace=False)
        
    samples=samples.flatten()
    
    
    ts=0
    max_epi_steps=Num_actions
    some_info=[]
    
    
    data=reset_data()
    timeout=False
    done=False
    num_episodes=0 
    
    
    
    
    s_num=0
    
    
    #while s_num<len(samples):
    while s_num<len(samples):    
        
        state_aux=env.reset()
        action=samples[s_num]
        s_num+=1
        
        # if s_num==1:
        #     s=state_aux.copy()
        #     append_data(data,s, action, 0,0, done, False)
        
        
        ns,reward, done, info = env.step(action)
        
      
            
        
       
        
        state_aux=ns
        s=state_aux.copy()
        
        
        # You know what set range from 1 to max_epi_steps
        
        for i in range(1,max_epi_steps):
            
            if i!=max_epi_steps-1:
                
                action=samples[s_num]
                s_num+=1
                
                ns,reward, done, info = env.step(action)
                
                
                append_data(data,s, action, reward,0, done, False)
                
                state_aux=ns
                s=state_aux.copy()
                
                
            else:
                
                action=samples[s_num]
                s_num+=1
                
                ns,reward, done, info = env.step(action)
                #print(reward)
                append_data(data,s, action, reward,0, True, True)
                
                append_data(data,ns, 0, 0,0, False, True)
                    
    
    
    npify(data)
    
    
    np.save(output_file, data)
    
    
    #print("\n" * 5)
    #print(data)
    
    read_d = np.load(output_file, allow_pickle='TRUE').item()

def createMircea():
    
    secuencias=['GAFTGEVSLPILKDFGVNWIVLGHSERRAYYGNEIVADKVAAAV',
    'GLASLKDFGVNWIVLGHSERRWYYGEVADKVAAAV',
    'GAFTGENSVDQIKDVGAKWVILGHSERRSEDDKFIADKTKFAL',
    'QAYTVSPVMLKDLGVTYVILGHSERRQMFAETDETVNKKVLAAF',
    'GSHTGHVLPEAVKEAGAVGTLLNHSENRMILADLEAAIRRAE' ]
    
    seq_records = []
    
 
    
    for index,protein_seq in enumerate(secuencias):
        seq_record = SeqRecord(Seq(protein_seq), id="sequence_id"+str(index), description="description")
        seq_records.append(seq_record)

    SeqIO.write(seq_records, "Mircea.fasta", "fasta")
    
    
    
def file_strings():
  # Open the text file for reading
 #file_path = '/content/drive/MyDrive/Experi_v1/unprocessed_sorted.txt'
 file_path = '/home/sergio/Documents/MSA_GTR_V1/unprocessed_sorted.txt'

 
 #file_path = '/content/drive/MyDrive/Experi_v1/sorted_sabre.txt'

# Read the lines from the file and store them in an array
 with open(file_path, 'r') as file:
     lines = file.readlines()

# Initialize an empty array to store rows from the text file
 file_names = []


# Fill the array with rows from the text file
 for line in lines:
    # Remove trailing newline characters and split the line into elements
     row = line.strip()
    # Convert elements to desired data type if necessary
    # For example, if you want integers: row = list(map(int, row))
     file_names.append(row)

 return file_names





def main():
    


    
    #folder_path = '/content/drive/MyDrive/Experi_v1/Bali20'
    #folder_path = '/content/drive/MyDrive/Experi_v1/Test1'

    folder_path = '/home/sergio/Documents/MSA_GTR_V1/unprocessed'
    #folder_path = '/content/drive/MyDrive/Experi_v1/sabre_inputs'
    #/home/sergio/Documents/MSA_GTR_V1/

    
    folder_path=folder_path+"/"

    files = file_strings()

    # Latest settings for Balibase:   range(145,150)

    for i in range(1,5):
      protein_file=folder_path+files[i]

      ProtSeq2RLseqs(protein_file)
      



    
    
    
    
    
    

    
if __name__ == '__main__':
    main() 



