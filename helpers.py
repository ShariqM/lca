class RunType():
    Learning = 1
    vLearning = 2
    vReconstruct = 3

def get_eta(t, runtype, batch_size):
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

def get_eta_mess(t, runtype, batch_size):
    if runtype == RunType.Learning: # Hack
        start = 1000
        if t < start:
            return 0.5/batch_size
        if t < start+1000:
            return 0.25/batch_size
        return 0.10/batch_size
    if runtype == RunType.Learning:
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
    elif True: # hack
        start = 1000
        if t < start:
            return 1.00/batch_size
        if t < start+1000:
            return 0.50/batch_size
        if t < start+2000:
            return 0.25/batch_size
        return 0.10/batch_size
    else:
        start = 6000
        if t < start:
            return 6.0/batch_size
        if t < start+6000:
            return 3.0/batch_size
        if t < start+12000:
            return 1.0/batch_size
        if t < start+18000:
            return 0.5/batch_size
        if t < start+24000:
            return 0.25/batch_size
        if t < start+30000:
            return 0.10/batch_size
        return 0.10/batch_size



def get_RunType_name(rt):
    if rt == RunType.Learning:
        return 'Learning'
    if rt == RunType.vLearning:
        return 'vLearning'
    if rt == RunType.vReconstruct:
        return 'vReconstruct'
