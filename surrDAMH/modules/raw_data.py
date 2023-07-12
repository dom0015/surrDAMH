#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 13:10 2023

@author: Pavel Exner
"""

import numpy as np
import pandas as pd
import os
from os import listdir
from os.path import isfile, join


class Sample:
    def __init__(self, raw_data, idx):
        self.raw_data = raw_data
        self.idx = idx
        self.type_names = ["accepted", "prerejected", "rejected"]

    def type(self):
        return self.raw_data.types[self.idx]

    def type_name(self):
        return self.type_names[self.raw_data.types[self.idx]]

    def stage(self):
        return self.raw_data.stages[self.idx]

    def chain(self):
        return self.raw_data.chains[self.idx]

    def weight(self):
        return self.raw_data.weights[self.idx]

    def parameters(self):
        return self.raw_data.parameters[self.idx, :]

    def observations(self):
        return self.raw_data.observations[self.idx, :]


class RawData:
    def __init__(self):
        self.types = None
        self.stages = None
        self.chains = None
        self.tags = None
        self.parameters = None
        self.observations = None
        self.weights = None

        self.no_chains = 0
        self.no_stages = 0
        self.no_samples = 0

    def load(self, folder_samples, no_parameters, no_observations):
        folder_samples = os.path.join(folder_samples, 'raw_data')
        file_samples = [f for f in listdir(folder_samples) if isfile(join(folder_samples, f))]
        file_samples.sort()
        N = len(file_samples)

        self.types = np.empty((0, 1), dtype=np.int8)
        self.stages = np.empty((0, 1), dtype=np.int8)
        self.chains = np.empty((0, 1), dtype=np.int8)
        self.tags = np.empty((0, 1), dtype=np.int8)
        self.parameters = np.empty((0, no_parameters))
        self.observations = np.empty((0, no_observations))
        self.weights = np.empty((0, 1), dtype=np.int)

        for i in range(N):
            path_samples = folder_samples + "/" + file_samples[i]
            df_samples = pd.read_csv(path_samples, header=None)

            types = df_samples.iloc[:, 0]
            idx = np.zeros(len(types), dtype=np.int8)
            # idx[types == "accepted"] = 0
            idx[types == "prerejected"] = 1
            idx[types == "rejected"] = 2
            # print(np.shape(self.types))
            # print(np.shape(idx))

            stg = int(file_samples[i][3])
            self.no_stages = max(self.no_stages, stg+1)
            stages = stg * np.ones(len(types), dtype=np.int8)

            chain = int(file_samples[i][file_samples[i].find("rank")+4:file_samples[i].find(".")])
            self.no_chains = max(self.no_chains, chain+1)
            chains = chain * np.ones(len(types), dtype=np.int8)

            parameters = np.array(df_samples.iloc[:, 1:1 + no_parameters])
            tags = np.array(df_samples.iloc[:, 1 + no_parameters])
            observation = np.array(df_samples.iloc[:, 2 + no_parameters:])

            self.types = np.append(self.types, idx)
            self.stages = np.append(self.stages, stages)
            self.chains = np.append(self.chains, chains)
            self.tags = np.append(self.tags, tags)
            self.parameters = np.vstack((self.parameters, parameters))
            self.observations = np.vstack((self.observations, observation))

            # compute weights
            # prerejected and rejected have weight=0
            widx = np.ones(len(types), dtype=bool)
            widx[types == "prerejected"] = False
            widx[types == "rejected"] = False
            # not considering first sample
            # TODO get rid of last value
            temp = np.arange(len(types))[widx]
            temp = np.append(temp, len(types))
            temp = np.diff(temp)
            weights = np.zeros(len(types))
            weights[widx] = temp
            # if sum(widx) > 0:
            weights = weights.reshape((-1, 1))
            self.weights = np.vstack((self.weights, weights)).astype(int)

        self.no_samples = np.shape(self.types)[0]
        print("raw data: no_stages", self.no_stages)
        print("raw data: no_chains", self.no_chains)
        print("raw data: no_samples", self.no_samples)
        print("raw data: no_nonconverging", np.shape(self.types[self.tags < 0])[0])

        print("raw data: p", np.shape(self.parameters))
        print("raw data: w", np.shape(self.weights))
        # print(self.weights[:20])
        # print(self.weights[-20:])
        # print(self.weights[self.weights>0][:])
        print("raw_data: np.sum(weights):", np.sum(self.weights))

    def len(self):
        return len(self.types)

    def no_parameters(self):
        return np.shape(self.parameters)[1]

    def no_observations(self):
        return np.shape(self.observations)[1]

    def filter(self, types, stages):
        idx = np.zeros(len(self.types), dtype=bool)

        for t in types:
            for s in stages:
                idx[(self.types == t)*(self.stages == s)] = 1

        raw_data = RawData()
        raw_data.types = self.types[idx]
        raw_data.stages = self.stages[idx]
        raw_data.chains = self.chains[idx]
        raw_data.parameters = self.parameters[idx]
        raw_data.observations = self.observations[idx]
        raw_data.weights = self.weights[idx]

        raw_data.no_samples = np.shape(raw_data.types)[0]
        raw_data.no_stages = len(stages)

        print("filter raw data: p", np.shape(raw_data.parameters))
        return raw_data