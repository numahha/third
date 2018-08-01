import os

import numpy as np
from gym import utils
from gym.envs.mujoco import MujocoEnv, HopperEnv


class MyHopperEnv1(HopperEnv):
    def __init__(self, xml_filename="hopper.xml"):
        print("\n[  Start print in __init___ in MyHopperEnv1  ]")
        utils.EzPickle.__init__(self)
        assets_path = os.path.join(os.path.dirname(__file__), "assets")
        xml_path = os.path.join(assets_path, xml_filename)
        MujocoEnv.__init__(self, xml_path, 2)
        print("xml_filename is ",xml_filename) # add
        print("body_mass is ",self.sim.model.body_mass) # add
        print("geom_friction is ",self.sim.model.geom_friction) # add
        print("[  End print in __init___ in MyHopperEnv1  ]\n") # add
        self.default_body_mass=self.sim.model.body_mass # add
        self.default_geom_friction=self.sim.model.geom_friction # add
        self.torso_mass_dist=None # add
        self.friction_dist=None # add

        
    def reset_model(self):
        if self.torso_mass_dist is not None: # add
            self.sim.model.body_mass[1] = self.torso_mass_dist() # add
        if self.friction_dist is not None: # add
            self.sim.model.geom_friction[4,0] = self.friction_dist() # add
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

