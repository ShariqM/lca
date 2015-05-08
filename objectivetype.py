class ObjectiveType():
    BSC    = 1
    SC     = 2
    VSC    = 3
    GSC    = 4
    PSC    = 5
    PPSC   = 6

def get_ObjectiveType_name(ot):
    if ot == ObjectiveType.BSC: # E^a
        return 'Bruno Sparse Coding'
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
