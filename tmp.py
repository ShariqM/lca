    def msparsify(self, I, nI, Phi, Z, u_pred=None, num_iterations=80):
        'Run the LCA coefficient dynamics'

        b = tdot(Phi.T, I)
        G = tdot(Phi.T, Phi) - np.eye(self.neurons)

        ZPhi = tdot(Phi, Z)
        Zb = tdot(ZPhi.T, nI)
        ZG = tdot(ZPhi.T, ZPhi) - np.eye(self.neurons)

        if self.lambda_type == LambdaType.Fixed:
            l = np.ones(self.batch_size)
            l *= self.lambdav
        else:
            raise Exception("Fix lambda type")
            l = 0.5 * np.max(np.abs(b), axis = 0)

        u = u_pred if u_pred is not None else np.zeros((self.neurons, self.batch_size))
        a = self.thresh(u, l)

        showme = True # set to false if you just want to save the images
        if self.coeff_visualizer:
            if not self.graphics_initialized:
                self.graphics_initialized = True
                fg, self.ax = plt.subplots(3,3, figsize=(10,10))
                #fg.set_size_inches(08.0,8.0)


                self.ax[1,2].set_title('Coefficients')
                self.ax[1,2].set_xlabel('Coefficient Index')
                self.ax[1,2].set_ylabel('Activity')

                self.coeffs = self.ax[1,2].bar(range(self.neurons), np.abs(u), color='r', lw=0)
                self.lthresh = self.ax[1,2].plot(range(self.neurons+1), list(l) * (self.neurons+1), color='g')

                axis_height = 1.05 if self.runtype == RunType.Learning else self.lambdav * 5
                self.ax[1,2].axis([0, self.neurons, 0, axis_height])

                # Present
                recon = tdot(Phi, a)
                self.ax[1,1].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,1].set_title('Iter=%d\nReconstruct (t)' % 0)

                self.ax[0,1].set_title('Reconstruction Error (t)')
                self.ax[0,1].set_xlabel('Time (steps)')
                self.ax[0,1].set_ylabel('SNR (dB)')
                self.ax[0,1].axis([0, self.num_frames * num_iterations, 0.0, 22])

                # Prediction
                p_recon = tdot(ZPhi, a)
                self.ax[1,0].imshow(np.reshape(p_recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,0].set_title('Iter=%d\nReconstruct (t+1)' % 0)

                self.ax[0,0].set_title('Reconstruction Error (t+1)')
                self.ax[0,0].set_xlabel('Time (steps)')
                self.ax[0,0].set_ylabel('SNR (dB)')
                self.ax[0,0].axis([0, self.num_frames * num_iterations, 0.0, 22])


                # The subplots move around if I don't do this lol...
                for i in range(6):
                    plt.savefig('animation/junk.png')
                if self.save_cgraphs:
                    plt.savefig('animation/%d.jpeg' % self.iter_idx)
                self.iter_idx += 1

            self.ax[2,1].imshow(np.reshape(I[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[2,1].set_title('Image (t)')

            self.ax[2,0].imshow(np.reshape(nI[:,0], (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
            self.ax[2,0].set_title('Image (t+1)')

            if showme:
                plt.draw()
                plt.show()

        for t in range(num_iterations):
            u = self.coeff_eta * (b - tdot(G,a)) + (1 - self.coeff_eta) * u
            a = self.thresh(u, l)

            explode = check_activity_m(b, Zb, tdot(G, a), tdot(ZG, a), u)
            if explode:
                ahat_c = np.copy(a)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)
                print 'Coefficients Active=%.2f%%' % (100 * float(ac)/(self.neurons * self.batch_size))


            l = self.lambda_decay * l
            l[l < self.lambdav] = self.lambdav

            if self.coeff_visualizer:
                ahat_c = np.copy(a)
                ahat_c[np.abs(ahat_c) > self.lambdav/1000.0] = 1
                ac = np.sum(ahat_c)
                self.ax[1,2].set_title('Coefficients Active=%.2f%%' % (100 * float(ac)/self.neurons))

                for coeff, i in zip(self.coeffs, range(self.neurons)):
                    coeff.set_height(abs(u[i]))  # Update the potentials
                self.lthresh[0].set_data(range(self.neurons+1), list(l) * (self.neurons+1))


                # Update Reconstruction
                recon = tdot(Phi, a)
                self.ax[1,1].imshow(np.reshape(recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,1].set_title('Iter=%d\nReconstruct (t)' % self.iter_idx)

                p_recon = tdot(ZPhi, a)
                self.ax[1,0].imshow(np.reshape(p_recon, (self.sz, self.sz)),cmap = cm.binary, interpolation='nearest')
                self.ax[1,0].set_title('Iter=%d\nReconstruct (t+1)' % self.iter_idx)

                # Plot SNR
                var = I.var().mean()
                R = I - recon
                mse = (R ** 2).mean()
                snr = 10 * log(var/mse, 10)
                color = 'r' if t == 0 else 'g'
                self.ax[0,1].scatter(self.iter_idx, snr, s=8, c=color)

                p_var = nI.var().mean()
                ZR = I - p_recon
                p_mse = (ZR ** 2).mean()
                p_snr = 10 * log(p_var/p_mse, 10)
                color = 'r' if t == 0 else 'g'
                self.ax[0,0].scatter(self.iter_idx, p_snr, s=8, c=color)

                if showme:
                    plt.draw()
                    plt.show()
                if self.save_cgraphs:
                    plt.savefig('animation/%d.jpeg' % self.iter_idx)
                self.iter_idx += 1

        return u, a


