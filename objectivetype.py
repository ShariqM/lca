class ObjectiveType():
    SC     = 1
    VSC    = 2
    GSC    = 3
    PSC    = 4
    PPSC   = 5

def get_ObjectiveType_name(ot):
    if ot == ObjectiveType.SC:
        return 'Sparse Coding'
    if ot == ObjectiveType.VSC:
        return 'Video Sparse Coding'
    if ot == ObjectiveType.GSC:
        return 'Group Sparse Coding'
    if ot == ObjectiveType.PSC:
        return 'Prediction Sparse Coding'
    if ot == ObjectiveType.PPSC:
        return 'Prime Prediction Sparse Coding'
