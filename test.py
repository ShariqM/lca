import numpy as np
import pdb
neurons = 2
timepoints = 3
time_batch_size = 4
patch_dim = 5

a = np.zeros((neurons, time_batch_size))
x = 0
for j in range(time_batch_size):
    for i in range(neurons):
        a[i,j] = x
        x += 1
pdb.set_trace()


'''
Phi = np.random.randn(patch_dim, neurons)
Psi = np.random.randn(neurons, cells, timepoints)
e   = np.random.randn(patch_dim, batch_size, time_batch_size)


ups = np.tensordot(Phi, Psi, [[1], [0]]) # pct
ups = np.swapaxes(ups, 0, 1)
ups = np.reshape(ups, (cells, patch_dim * timepoints))
ec = np.copy(e)
ec = np.swapaxes(ec, 0,1)
error2 = np.zeros((batch_size, patch_dim * timepoints, time_batch_size))
for t in range(time_batch_size - timepoints):
    size = min(timepoints, time_batch_size - t)
    error2[:,:,t] = np.reshape(ec[:,:,t:t+size], (batch_size, patch_dim * timepoints))

result = np.tensordot(ups, error2, [[1], [1]])
profile_print("dA Calc", start)
'''

def get_reconstruction(self, Psi, Phi, a):
        start = dt.now()
        Ups = self.compute_Ups(Psi, Phi)

        ac = np.copy(a)
        ac = np.swapaxes(ac, 0, 1)
        ahat = np.zeros((self.batch_size, self.cells * self.timepoints,
                         self.time_batch_size))
        for t in range(self.time_batch_size):
            act = np.zeros((self.batch_size, self.cells, self.timepoints))
            size = min(self.timepoints - 1, t)
            act[:,:,0:size] = ac[:,:,t::-1][:,:,0:size+1]
            ahat[:,:,t] = np.reshape(act, (self.batch_size,
                                           self.cells * self.timepoints))
            #r[:,:,t] = ten3dot2(Phi, Psi[:,:,0:size+1],

        r = np.tensordot(Ups, ahat, [[1], [1]])
        self.profile_print("get_reconstruction Calc", start)

        return r

'''
for i in range(3):
            ac = np.copy(a)
            ac = np.swapaxes(ac, 0, 1)

            if i == 0:
                start = dt.now()
                ahat = np.zeros((self.batch_size, self.cells * self.timepoints,
                                 self.time_batch_size))
                ac = ac[:,:,-1::-1] # Reverse
                ac = np.reshape(ac, (self.batch_size, self.cells * self.time_batch_size))
                #dac = np.zeros((self.batch_size, self.cells * self.time_batch_size))
                #x = 0
                #for r in range(self.cells):
                    #for q in range(self.time_batch_size):
                        #dac[:,x] = ac[:,r,q]
                        #x += 1
                #ac = np.copy(dac)
                x = 0
                for t in range(self.time_batch_size):
                    x = x + 1
                print x
                self.profile_print("get_reconstruction iter1 Calc", start)

                for t in range(self.timepoints):
                    act = np.zeros((self.batch_size, self.cells * self.timepoints))
                    idx = self.time_batch_size-t-1
                    act[:,0:self.cells*(t+1)] = ac[:,self.cells * idx : self.cells * (idx+t+1)]
                    ahat[:,:,t] = act

                for t in range(self.timepoints, self.time_batch_size):
                    act = np.zeros((self.batch_size, self.cells * self.timepoints))
                    size = self.timepoints - 1

                    idx = self.time_batch_size-t-1
                    act[:,0:self.cells*(size+1)] = ac[:,self.cells * idx : self.cells * (idx+size+1)]
                    ahat[:,:,t] = act
                    #if t == 0:
                        #pdb.set_trace()
                self.profile_print("get_reconstruction loop3 1 Calc", start)
            elif i == 1:
                start = dt.now()
                ahat2 = np.zeros((self.batch_size, self.cells * self.timepoints,
                                 self.time_batch_size))
                ac = ac[:,:,-1::-1] # Reverse
                for t in range(self.time_batch_size):
                    act = np.zeros((self.batch_size, self.cells, self.timepoints))
                    size = min(self.timepoints - 1, t)
                    idx = self.time_batch_size-t-1
                    act[:,:,0:size+1] = ac[:,:,idx:idx+size+1]
                    ahat2[:,:,t] = np.reshape(act, (self.batch_size,
                                                   self.cells * self.timepoints))
                    #if t == 0:
                        #pdb.set_trace()
                self.profile_print("get_reconstruction loop3 2 Calc", start)
            else:
                start = dt.now()
                ahat3 = np.zeros((self.batch_size, self.cells * self.timepoints,
                                 self.time_batch_size))

                for t in range(self.time_batch_size):
                    act = np.zeros((self.batch_size, self.cells, self.timepoints))
                    size = min(self.timepoints - 1, t)
                    act[:,:,0:size+1] = ac[:,:,t::-1][:,:,0:size+1]
                    ahat3[:,:,t] = np.reshape(act, (self.batch_size,
                                                   self.cells * self.timepoints))
                self.profile_print("get_reconstruction loop3 3 Calc", start)

                print '1,2', np.allclose(ahat, ahat2, atol=1e-4)
                print '2,3', np.allclose(ahat2, ahat3, atol=1e-4)
                pdb.set_trace()

'''

