#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:00:43 2022

@author: domesova
"""

import os
import json
import ruamel.yaml as yaml

json_dir = "/home/domesova/GIT/Endorse-2Dtest-Bayes/surrDAMH/examples/"
json_name = "Darcy.json"

# json_dir = "/home/domesova/GIT/Endorse-2Dtest-Bayes/"
# json_name = "config_mcmc_bayes.json"

json_path = os.path.join(json_dir, json_name)
name, fext = os.path.splitext(os.path.basename(json_path))
yaml_path = json_dir + name + ".yaml"

## load json, dump to yaml

# with open(json_path) as f:
#     d = json.load(f)

# with open(yaml_path, 'w') as f:
#     yaml.dump(d, f, default_flow_style=False)

# with open(yaml_path, "r") as f:
#     config_dict = yaml.safe_load(f)


## load and dump yaml with comments:

with open(json_dir + "Darcy_comment.yaml") as f:
    text = f.read()

Y = yaml.YAML()
d = Y.load(text)
d["new_key"] = "new_value"

with open(yaml_path, 'w') as f:
    Y.dump(d, f)

with open(yaml_path, "r") as f:
    config_dict = yaml.safe_load(f)