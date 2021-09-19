''' Moudle utils.py
This module contains important functionalities related to device, plotting the scores of the agent.
@author: Rohith Banka.

'''
import torch

def get_device():
    '''To set the device CPU/GPU for training/inference'''

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device
