#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path
