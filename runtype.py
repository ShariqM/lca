class RunType():
    Learning     = 1
    vLearning    = 2
    vmLearning   = 3
    vgLearning   = 4
    vReconstruct = 5
    vPredict     = 6
    vDynamics    = 7

def get_RunType_name(rt):
    if rt == RunType.Learning:
        return 'Learning'
    if rt == RunType.vLearning:
        return 'vLearning'
    if rt == RunType.vmLearning:
        return 'vmLearning'
    if rt == RunType.vgLearning:
        return 'vgLearning'
    if rt == RunType.vReconstruct:
        return 'vReconstruct'
    if rt == RunType.vPredict:
        return 'vPredict'
    if rt == RunType.vDynamics:
        return 'vDynamics'
    raise Exception("Unknown run type")
