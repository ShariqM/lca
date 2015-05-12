class RunType():
    Learning     = 1
    vLearning    = 2
    vmLearning   = 3
    vReconstruct = 4
    vPredict     = 5
    vDynamics    = 6

def get_RunType_name(rt):
    if rt == RunType.Learning:
        return 'Learning'
    if rt == RunType.vLearning:
        return 'vLearning'
    if rt == RunType.vmLearning:
        return 'vmLearning'
    if rt == RunType.vReconstruct:
        return 'vReconstruct'
    if rt == RunType.vPredict:
        return 'vPredict'
    if rt == RunType.vDynamics:
        return 'vDynamics'
    raise Exception("Unknown run type")
