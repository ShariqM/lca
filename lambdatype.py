class LambdaType():
    Fixed = 1
    Decay = 2
    LSM   = 3

def get_LambdaType_name(lt):
    if lt == LambdaType.Fixed:
        return 'Fixed'
    if lt == LambdaType.Decay:
        return 'Decay'
    if lt == LambdaType.LSM:
        return 'LSM'
