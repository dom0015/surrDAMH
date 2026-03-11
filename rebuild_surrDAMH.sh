#!/bin/bash

cd /workspaces/surrDAMH/

rm -rf build
rm -rf surrDAMH.egg-info

pip install -e .

cd toy_examples/

