class RunType():
    Learning = 1
    vLearning = 2
    vReconstruct = 3

def get_RunType_name(rt):
    if rt == RunType.Learning:
        return 'Learning'
    if rt == RunType.vLearning:
        return 'vLearning'
    if rt == RunType.vReconstruct:
        return 'vReconstruct'
