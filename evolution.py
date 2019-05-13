import gym
from collections import defaultdict
import random
import numpy as np

class Robot:
    def __init__(self):
        self.actions = []
        self.fitness_value = 0

    def get_fitness(self):
        return self.fitness_value

def gen_population(env, steps):
    robot = Robot()
    action = env.action_space.sample()
    i = 0
    done = False
    r = 1
    reward = 0
    env.reset()
    while i < steps and not done and r != reward:
        #env.render()
        r = reward
        robot.actions.append(action)
        _, reward, done, _ = env.step(action)
        robot.fitness_value += reward
        i += 1
    return robot
    
def mutation(action):
    t = random.randint(0,4)
    for _ in range(t):
        pos = random.randint(0,3)
        if action[pos] > 0:
            action[pos] = random.uniform(0,1)
        elif action[pos] < 0:
            action[pos] = random.uniform(-1,0)
        else:
            action[pos] = random.uniform(-1,1)
    return action

def select_parent(population):
    weights = []
    for p in population:
        weights.append(p.get_fitness() + abs(population[len(population)-1].get_fitness()) + 1)
    w = sum(weights)
    weights = weights/w
    p = list(range(0,len(population)))
    r1, r2 = random.choices(population=p, weights=weights, k = 2)
    return population[r1], population[r2]

def cross(p1, p2):
    child = Robot()
    size = random.randint(1, len(p1.actions))
    for i in range(size):
        child.actions.append(mutation(p1.actions[i]))
    size = random.randint(1, len(p2.actions))
    for i in range(size):
        child.actions.append(mutation(p2.actions[i]))
    return child

def breeding(env, population, population_size, steps):
    children = []
    for _ in range(population_size):
        p1, p2 = select_parent(population)
        child = cross(p1, p2)
        env.reset()
        r = 1
        reward = 0
        done = False
        i = 0
        j = 0
        while i < 500 and not done and r != reward:
            #env.render()
            r = reward
            if j == len(child.actions):
                j = 0
            action = child.actions[j]
            _, reward, done, _ = env.step(action)
            child.fitness_value += reward
            i += 1
            j += 1
        children.append(child)
    return children + population

def main():
    env = gym.make('BipedalWalker-v2')
    env.reset()

    p_size = [100]
    steps = [4, 8, 16,32,64]
    gens = [100]
    for population_size in p_size:
        for step in steps:
            for gen in gens:
                for i in range(1):
                    death_size = 10/100
                    population = sorted([gen_population(env, step) for _ in range(population_size)], key=Robot.get_fitness, reverse = True)[:int(population_size*death_size)]
                    nome_arquivo = str(population_size) + "_" + str(step) + "_" + str(gen) + "_" + str(i)
                    arquivo = open("500/" + nome_arquivo, 'w+')
                    for i in range(gen):
                        death_size = random.randint(10,30)/100
                        population = sorted(breeding(env, population, population_size, step), key=Robot.get_fitness, reverse = True)[:int(population_size*death_size)]
                        print("{} - {}".format(i, population[0].fitness_value))
                        rew = 0
                        for j in range(5):
                            rew += population[j].fitness_value    
                        arquivo.writelines("{} - {} - {} \n".format(i, population[0].fitness_value, rew))
                    arquivo.close()        
    env.close()

if __name__ == '__main__':
    main()