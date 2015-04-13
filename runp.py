class RunP():
    def __init__(self, initP, iters, lambdav):
        self.initP   = initP # Initialize coeff to previous frame
        self.iters   = iters # Number of iterations per frame
        self.lambdav = lambdav

def get_labels(run_p):
        labels = []
        for rp in run_p:
            if rp.initP == True:
                labels += ["InitP (%d)" % rp.iters]
            else:
                labels += ["Init0 (%d)" % rp.iters]
        return labels
