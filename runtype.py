class RunType():
    Learning     = 1
    vLearning    = 2
    vmLearning   = 3
    vReconstruct = 4

def get_RunType_name(rt):
    if rt == RunType.Learning:
        return 'Learning'
    if rt == RunType.vLearning:
        return 'vLearning'
    if rt == RunType.vmLearning:
        return 'vmLearning'
    if rt == RunType.vReconstruct:
        return 'vReconstruct'
