from __future__ import division

import sys
import os
import time
import copy
import argparse
import importlib
import numpy as np
import os, sys
current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)


import torch
import torch.nn as nn
from torch.autograd import Variable
import data_util
from evaluator import SeqAgent, Calculator, interactive_sample, match_rate
from model import GRUNetwork, RNNNetwork, LSTMNetwork
from lesion import lesion_rnn
def main(task_configurations):
    # pass parameters, for the entire network
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", help="if cuda acceleration is enabled",
                        default=1,
                        type=bool)
    args = parser.parse_args()

    cuda_enabled = args.cuda

    model_parameters = {
        "nhid": 128,
        "nlayers": 1,
        "input_size": task_configurations[0]["input_size"],
        "batch_size": task_configurations[0]["batch_size"],
        "clip": clip_setting,
        "lr": task_configurations[0]["lr"]
    }
    print("Model Parameters: {}".format(model_parameters))
    # 20 tasks in total, for each task pass parameters
    for nth, task_setting in enumerate(task_configurations):
        model_parameters["lr"] = task_setting["lr"]
        if task_setting['model_mode'] == 'GRU':
            rnn_model = GRUNetwork(model_parameters["input_size"],
                                   model_parameters["nhid"],
                                   model_parameters["batch_size"],
                                   model_parameters["nlayers"],
                                   model_parameters["lr"],
                                   # cuda_enabled=cuda_enabled,
                                   cuda_enabled=False,
                                   )
        elif task_setting['model_mode'] == 'LSTM':
            print('LSTM')
            rnn_model = LSTMNetwork(model_parameters["input_size"],
                                    model_parameters["nhid"],
                                    model_parameters["batch_size"],
                                    model_parameters["nlayers"],
                                    model_parameters["lr"],
                                    cuda_enabled=cuda_enabled,
                                    )
        else:
            raise Exception('unknown model mode')
        if len(task_setting["load_model"]) > 0:
            rnn_model.load_state_dict(torch.load(task_setting["load_model"]))
        print("{}th repeat:".format(nth + 1))
        # lesion rnn may include several models? (if lesion and anlys_file != '')
        rnn_model_lesion = lesion_rnn(rnn_model, task_setting['lesion'], task_setting['anlys_file'])
        # feed data for each model in rnn_model_lesion
        for label, model in rnn_model_lesion.items(): #TODO: [notes by Jiaqi] for each model of lesion models; for now, only one model
            # global settings
            log_path = task_setting["log_path"]  # "../log_m/"

            data_name = time.strftime("%Y%m%d_%H%M", time.localtime())
            if label:
                data_name = label + '-' + data_name
            print(data_name)
            dp = data_util.DataProcessor(log_name=data_name)
            training_data_path = task_setting["data_file"]
            validation_data_path = task_setting["validation_data_file"]

            data_attr = task_setting["data_attr"]
            data_brief_attr = task_setting["data_brief"]

            # mat[data_attr].shape = (750000,1,26,10)
            # mat[data_brief_attr][0,0] is data_ST_Brief in training data generator
            # training_set shape: (750000, 26, 10)
            training_set, training_conditions = dp.load_data_v2(
                training_data_path, data_attr, data_brief_attr)

            validation_set, validation_conditions = dp.load_data_v2(
                validation_data_path, data_attr, data_brief_attr)

            training_data = {
                "training_set": training_set,
                "training_conditions": training_conditions
            }
            validation_data = {
                "validation_set": validation_set,
                "validation_conditions": validation_conditions
            }
            if task_setting["record"]:
                dp.create_log(log_path, task_setting["name"])  # create log first
                dp.save_condition(model_parameters, task_setting, training_data_path, validation_data_path)

            task = importlib.import_module('tasks.' + task_setting["name"])
            # sqa = SeqAgent(model, batch_size = 1, cuda=cuda_enabled) # test the trial with the batch_size being 1
            sqa = SeqAgent(model, batch_size=1, cuda=False)

            if task_setting["need_train"] > 0:
                print('-' * 89)
                print("START TRAINING: {}".format(task_setting["name"]))
                print('-' * 89)
                # for i in range(task_setting["need_train"]):
                # train_stage(
                #             dp, model, sqa, task_setting["model_path"], training_data, validation_data, log_path,
                #             task, record=task_setting["record"], train_truncate=task_setting["train_num"],
                #             batch_size=model_parameters["batch_size"], clip=model_parameters["clip"], cuda_enabled=cuda_enabled
                #             )
                # train_stage(
                #     dp, model, sqa, task_setting["model_path"], training_data, validation_data, log_path,
                #     task, record=task_setting["record"], train_truncate=task_setting["train_num"],
                #     batch_size=model_parameters["batch_size"], clip=model_parameters["clip"],
                #     cuda_enabled=False
                # )
            model.load_state_dict(torch.load('../save_m/ST/model-sp_ts_250000-20190702_0553-2.pt'))
            model.eval()
            if task_setting["need_validate"] > 0:
                print('-' * 89)
                print("START VALIDATION: {}".format(task_setting["name"]))
                print('-' * 89)
                wining_rate, completed_rate = validate(dp, model, sqa, validation_data, task, task_setting["record"],
                                                       truncate=5e3)
                print("wining_rate :{}|completed_rate :{}".format(wining_rate, completed_rate))


if __name__ == '__main__':
    os.chdir("C:/Users/76774/Desktop/SS/SeqCode/seqrnn_multitask")
    config = importlib.import_module('config_st')

    #    config = importlib.import_module('config_sure')
    #    config = importlib.import_module('config_mi')
    #    config = importlib.import_module('config_st')
    #
    task_configurations = config.task_configurations
    #    task_configurations = config.task_configurations_lesion

    reset_hidden = task_configurations[0]['reset_hidden']
    main(task_configurations)
