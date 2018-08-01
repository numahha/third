#!/usr/bin/env python
# This code is to check parameters of the hopper task.
# Compare the original hopper task by openAI and a slightly modified one.
import mujoco_py # Mujoco must come before other imports. https://openai.slack.com/archives/C1H6P3R7B/p1492828680631850
import gym
import my_env
import numpy



#env_id = "Hopper-v2"   # the original hopper
env_id = "MyHopper-v1"  # a modified hopper



print("\n##### 0 #####")
e = gym.make(env_id)
print("env id is ",e.env.spec.id)



print("\n##### 1 #####")
print("e.env.model.body_mass")
print(e.env.model.body_mass)
#print("torso mass divided by pi is ",e.env.model.body_mass[1]/numpy.pi)
#print(e.env.model.body_mass[1]/(numpy.pi*2.5)) # size*size*density = 0.05*0.05*1000 = 2.5
if env_id is not "Hopper-v2":
    def temp_torso_mass_dist():
        return 4.0
    e.env.torso_mass_dist = temp_torso_mass_dist
    e.reset()
    print("modified body_mass")
    print(e.env.model.body_mass)



print("\n##### 2 #####")
print("e.env.model.geom_friction")
print(e.env.model.geom_friction)
if env_id is not "Hopper-v2":
    def temp_friction_dist():
        return 3.0
    e.env.friction_dist = temp_friction_dist
    e.reset()
    print("modified friction")
    print(e.env.model.geom_friction)




e.close()

