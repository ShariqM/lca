from enum import Enum

RunType = Enum('RunType', 'learning rt_learning rt_reconstruct')

def get_eta(t,batch_size):
    start = 1000
    if t < start:
        return 6.0/batch_size
    if t < start+1000:
        return 3.0/batch_size
    if t < start+2000:
        return 1.0/batch_size
    if t < start+3000:
        return 0.5/batch_size
    if t < start+4000:
        return 0.25/batch_size
    if t < start+5000:
        return 0.10/batch_size
    return 0.10/batch_size


def get_RunType_name(rt):
    if rt == RunType.learning:
        return 'Learning'
    if rt == RunType.rt_learning:
        return 'RT_Learning'
    if rt == RunType.rt_reconstruct:
        return 'RT_Reconstruct'
