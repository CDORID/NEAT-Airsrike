# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 16:47:54 2019

@author: Romain
"""

import retro
import numpy as np
import cv2
import neat
import pickle


####################"

check = 'neat-checkpoint-20'




try : 
    env.close()
except : 
    pass


limit_time = 20
#env.close()
env = retro.make('Airstriker-Genesis','Level1')

imgarray = []

xpos_end = 0

def eval_genomes(genomes, config):
    for genome_id, genome in genomes : 
        ob = env.reset()
        ac = env.action_space.sample()
        
        inx, iny, inc = env.observation_space.shape
        
        inx = int(inx/8)
        iny = int(iny/8)
        
        
        net = neat.nn.recurrent.RecurrentNetwork.create(genome,config)
        
        current_max_fitness = 0
        fitness_current = 0
        frame = 0
        counter = 0
        score = 0
        score_max = 0
        lives = 0
        
        done = False
        
        while not done : 
            env.render()
            frame += 1
            
            ob = cv2.resize(ob, (inx,iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx,iny))
            
            for x in ob:
                for y in x:
                    imgarray.append(y)
                    
            nnOutput =  net.activate(imgarray)
            
            ob, rew, done, info = env.step(nnOutput)
            
            imgarray.clear()
            
            score = info['score']
            lives =  info['lives']
            
            
            
            if lives == 2 or frame == limit_time*60:
                done = True
                print(genome_id, score)
                
            
            genome.fitness = score
            
            
        
        
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                    neat.DefaultSpeciesSet, neat.DefaultStagnation,
                    'config-feedforward')


if check != None : 
    p = neat.Checkpointer.restore_checkpoint(check)
else :
    p = neat.Population(config)
    

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

winner = p.run(eval_genomes)

env.close()

with open ('winner.pkl','wb') as output:
    pickle.dump(winner,output,1)