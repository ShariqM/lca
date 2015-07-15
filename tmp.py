#print 'b', b[t,:,:]
#print 'Gu', Gu
#print 'v', v[t,:,:]
#print 'a', a[t,:,:]
#pdb.set_trace()
if c == 5:
    for m in range(self.neurons):
        tot = 0
        tot_a = 0
        #for p in range(self.patch_dim):
            #tot += Phi[p,m] * Phi[p,m] * a[t,m,0]
        for p in range(self.patch_dim):
            for l in range(self.neurons):
                tot += Phi[p,m] * Phi[p,l]
                #tollit += Phi[p,m] * Phi[p,l] * a[t,m,0]
        print tot
        pdb.set_trace()


