import numpy as np
##import matplotlib.pyplot as plt

import Neurapse
from Neurapse.Synapses import *
from Neurapse.utils.SpikeTrains import *
from Neurapse.Networks import *
from Neurapse.utils.CURRENTS import *

def t_SNN():
    '''

    #usage : DRN_Const problem 2 and 3
    def DRN_Const_driver(N, exci_frac, connect_frac):
        N_exci = int(N*exci_frac)
        N_inhi = N - N_exci

        # Constructing the network
        exci_neuron_id = range(N_exci)
        inhi_neuron_id = range(N_exci, N)

        Fanout = []
        W = []
        Tau = []
        w0 = 3000
        gamma = 0.77 #for problem 3
        gamma = 1# for problem 2
        for i in exci_neuron_id:
            a = []
            for z in range(N):
                if z!=i:
                    a.append(z)
            tempFL = []
            tempWL = []
            tempTL = []
            for j in range(int(connect_frac*N)):
                t = np.random.choice(a)
                tempFL.append(t)
                tempWL.append(gamma*w0)
                tempTL.append(np.random.uniform(1e-3, 20e-3))
            W.append(tempWL)
            Fanout.append(tempFL)
            Tau.append(tempTL)

        for i in inhi_neuron_id:
            tempFL = []
            tempWL = []
            tempTL = []
            for j in range(int(connect_frac*N)):
                t = np.random.choice(exci_neuron_id)
                tempFL.append(t)
                tempWL.append(-w0)
                tempTL.append(1e-3)
            W.append(tempWL)
            Fanout.append(tempFL)
            Tau.append(tempTL)
        print('Fanout: ', Fanout)
        print('W: ', W)
        print('Tau: ', Tau)

        plt.plot(W)
        plt.title("W")
        plt.show()

        plt.plot(Tau)
        plt.title("Tau")
        plt.show()

        A = DRN_Const(Fanout, W, Tau_d=Tau, Tau=15e-3, I0=1e-12, N=N, T=20000, delta_t=1e-4)
        # Creating poisson spike trains as inputs for first 25 neurons
        # Note: Originally this was meant for poisson potential spikes,
        # here using as current
        T = 20000
        Tau = 15e-3
        delta_t = 1e-4
        n_out = 5
        I0 = 1e-12
        ST = POISSON_SPIKE_TRAIN(T=int(T*delta_t), delta_t=delta_t, lamb=100, n_out=n_out)
        V_poi_spikes = ST.V_train[:,:-1]
        # print(V_poi_spikes.shape)

        reference_alpha = np.zeros(T)
        for t in range(3*int(Tau//delta_t)):
            reference_alpha[t] = np.exp(-t*delta_t/Tau) - np.exp(-4*t*delta_t/Tau)
        plt.plot(reference_alpha)
        plt.show()

        I_poi = np.zeros(shape=(n_out, T))
        for idx in range(n_out):
            V_sp = V_poi_spikes[idx,:].reshape(1,-1)
            print(V_sp.shape)
            for t,v in enumerate(V_sp[0,:]):
                if v>0:
                    t1 = t
                    t2 = min(t+int(3*Tau//delta_t), T)
                    I_poi[idx, t1:t2] += w0*I0*reference_alpha[0:t2-t1]

        # for idx in range(n_out):
        #     plt.plot(I_poi[idx,:])
        #     plt.show()

        I_app = np.zeros(shape=(N, T))
        I_app[0:n_out, :] = I_poi
        El = -70e-3
        V_thresh = 20e-3
        V0 = El*np.ones(shape=(N,1))
        A.compute(I_app=I_app, V0=V0, delta_t=delta_t)

        V_response = A.V_collector
        I_app_feed = A.I_app_collector #same as I_app
        I_synapse_feed = A.I_synapse_feed

        exci_spike_instants = get_spike_instants_from_neuron(
            V_response[exci_neuron_id,:],
            V_thresh
        )

        inhi_spike_instants = get_spike_instants_from_neuron(
            V_response[inhi_neuron_id,:],
            V_thresh
        )

        # print(exci_spike_instants)
        # print(inhi_spike_instants)

        colorCodes = np.array(
            [[0,0,0]]*N_exci
            +
            [[0,0,1]]*N_inhi
        )
        plt.eventplot(exci_spike_instants + inhi_spike_instants, color=colorCodes, lineoffsets=2)
        plt.show()

        # for i in range(N):
        #     plt.plot(I_synapse_feed[i, :])
        #     plt.show()

        # Ret and Rit
        Ret = []
        Rit = []
        for l in exci_spike_instants:
            Ret = Ret+list(l)
        for l in inhi_spike_instants:
            Rit = Rit+list(l)
        Ret_sorted = sorted(Ret)
        Rit_sorted = sorted(Rit)
        t0 = 100
        plt.figure(figsize=(25, 25))
        plt.subplot(2,1,1)
        plt.hist(Ret_sorted, bins=int(T/t0))
        plt.xlabel('time')
        plt.ylabel('freq Ret')

        plt.subplot(2,1,2)
        plt.hist(Rit_sorted, bins=int(T/t0))
        plt.xlabel('time')
        plt.ylabel('freq Rit')
        plt.show()

    #usage : DRN_Plastic problem 4 and 5
    def DRN_Plastic_driver(N, exci_frac, connect_frac):
        N_exci = int(N*exci_frac)
        N_inhi = N - N_exci

        # Constructing the network
        exci_neuron_id = range(N_exci)
        inhi_neuron_id = range(N_exci, N)

        Fanout = []
        W = []
        Tau = []

        w0 = 3000
        # gamma = 1 #for problem 4
        gamma = 0.4 # for problem 5
        for i in exci_neuron_id:
            a = []
            for z in range(N):
                if z!=i:
                    a.append(z)
            tempFL = []
            tempWL = []
            tempTL = []
            for j in range(int(connect_frac*N)):
                t = np.random.choice(a)
                tempFL.append(t)
                tempWL.append(gamma*w0)
                tempTL.append(np.random.uniform(1e-3, 20e-3))
            W.append(tempWL)
            Fanout.append(tempFL)
            Tau.append(tempTL)

        for i in inhi_neuron_id:
            tempFL = []
            tempWL = []
            tempTL = []
            for j in range(int(connect_frac*N)):
                t = np.random.choice(exci_neuron_id)
                tempFL.append(t)
                tempWL.append(-w0)
                tempTL.append(1e-3)
            W.append(tempWL)
            Fanout.append(tempFL)
            Tau.append(tempTL)
        print('Fanout: ', Fanout)
        print('W: ', W)
        print('Tau: ', Tau)

        A = DRN_Plastic(Fanout, W, Tau_d=Tau, Tau=15e-3, Tau_l=20e-3, I0=1e-12, A_up=0.01, A_dn=-0.07, N=N, N_exci=N_exci, T=20000, delta_t=1e-4, gamma=gamma)
        # Creating poisson spike trains as inputs for first 25 neurons
        # Note: Originally this was meant for poisson potential spikes,
        # here using as current
        T = 20000
        Tau = 15e-3
        delta_t = 1e-4
        n_out = 5
        I0 = 1e-12
        ST = POISSON_SPIKE_TRAIN(T=int(T*delta_t), delta_t=delta_t, lamb=100, n_out=n_out)
        V_poi_spikes = ST.V_train[:,:-1]
        # print(V_poi_spikes.shape)

        reference_alpha = np.zeros(T)
        for t in range(3*int(Tau//delta_t)):
            reference_alpha[t] = np.exp(-t*delta_t/Tau) - np.exp(-4*t*delta_t/Tau)
        # plt.plot(reference_alpha)
        # plt.show()

        I_poi = np.zeros(shape=(n_out, T))
        for idx in range(n_out):
            V_sp = V_poi_spikes[idx,:].reshape(1,-1)
            print(V_sp.shape)
            for t,v in enumerate(V_sp[0,:]):
                if v>0:
                    t1 = t
                    t2 = min(t+int(3*Tau//delta_t), T)
                    I_poi[idx, t1:t2] += w0*I0*reference_alpha[0:t2-t1]

        # for idx in range(n_out):
        #     plt.plot(I_poi[idx,:])
        #     plt.show()

        I_app = np.zeros(shape=(N, T))
        I_app[0:n_out, :] = I_poi
        El = -70e-3
        V_thresh = 20e-3
        V0 = El*np.ones(shape=(N,1))
        A.compute(I_app=I_app, V0=V0, delta_t=delta_t)

        V_response = A.V_collector
        I_app_feed = A.I_app_collector #same as I_app
        I_synapse_feed = A.I_synapse_feed

        avg_weights = A.avg_weights
        plt.plot(avg_weights)
        plt.xlabel('time steps')
        plt.ylabel('mean excitatory synapse weights')
        plt.show()

        exci_spike_instants = get_spike_instants_from_neuron(
            V_response[exci_neuron_id,:],
            V_thresh
        )

        inhi_spike_instants = get_spike_instants_from_neuron(
            V_response[inhi_neuron_id,:],
            V_thresh
        )

        colorCodes = np.array(
            [[0,0,0]]*N_exci
            +
            [[0,0,1]]*N_inhi
        )
        plt.eventplot(exci_spike_instants + inhi_spike_instants, color=colorCodes, lineoffsets=2)
        plt.show()

        # Ret and Rit
        Ret = []
        Rit = []
        for l in exci_spike_instants:
            Ret = Ret+list(l)
        for l in inhi_spike_instants:
            Rit = Rit+list(l)
        Ret_sorted = sorted(Ret)
        Rit_sorted = sorted(Rit)
        t0 = 100
        plt.figure(figsize=(25, 25))
        plt.subplot(2,1,1)
        plt.hist(Ret_sorted, bins=int(T/t0))
        plt.xlabel('time')
        plt.ylabel('freq Ret')

        plt.subplot(2,1,2)
        plt.hist(Rit_sorted, bins=int(T/t0))
        plt.xlabel('time')
        plt.ylabel('freq Rit')
        plt.show()

    #DRN_Const_driver(N=5, exci_frac=0.5, connect_frac=0.5)

    #DRN_Plastic_driver(N=5, exci_frac=0.8, connect_frac=0.5)

    '''

    import numpy as np
    N=1
    C = 300e-12
    gL = 30e-9
    V_thresh = 20e-3
    El = -70e-3
    Rp = 2e-3
    all_neurons = LIF(C, gL, V_thresh, El, Rp, num_neurons=N)
    '''
    n_t= 15000
    start, stop = 0,1000
    I= np.zeros(n_t)
    I[start:stop] = 1
    
    
    I = I * 15e-5 #I0*I
    I=np.array([I])
    V0 = -0.06515672*np.ones((1,1))
    #T = 30e-3
    #delta_t = 1e-5
    '''

    I0 = 15e-8
    T = 30e-3
    delta_t = 1e-5
    n_t = int(5*T//delta_t)+1 #otherwise one-timestep is gone

    #Sq1 = Cur.SQUARE_PULSE(t_start=6000, t_end=9000, T=n_t)
    #I = Sq1.generate()

    start, stop = 0,1200
    I= np.zeros(n_t)
    I[start:stop] = 1


    I = I * 15e-8 #I0*I
    I=np.array([I])
    V0 = -0.06515672*np.ones((1,1))


    V,spike= all_neurons.compute (V0,I,delta_t=delta_t) # (I,V0)
    plt.plot(V[0],'-o')


    plt.plot(spike)
    plt.show()
    print("salut")


class POISSON_SPIKE_TRAIN():
    def __init__(self, T, delta_t, lamb, n_out=1):
        self.T = T
        self.delta_t = delta_t
        self.lamb = lamb
        self.n_out = n_out
        self.n_t = int(T / delta_t)
        self.V_train = np.zeros(shape=(self.n_out, self.n_t + 1))  # t=0 included
        self.spike_instants = [None] * self.n_out  # to hold the spike time instants of each neuron
        self.generate()

    def generate(self):
        '''
        spike_instants = list of length n_out each value being arr(time_instants_of_spikes)
        '''
        self.V_train = np.random.rand(self.n_out, self.n_t + 1)
        self.V_train = self.V_train < self.lamb * self.delta_t
        for i in range(self.n_out):
            self.spike_instants[i] = np.where(self.V_train[i, :] == 1.0)[0]
        # print(self.spike_instants)


import numpy as np


class CONST_SYNAPSE():
    '''
    This synapse can be represented
    by a single non changing weight
    '''

    def __init__(self, w, I0, tau, tau_s, tau_d):
        self.w = w
        self.I0 = I0
        self.tau = tau
        self.tau_s = tau_s
        self.tau_d = tau_d

    def getI(self, V_train, spike_instants, delta_t):
        '''
        V_train : 1 X n_t
        spike_instants : list(arr(num_of_spikes))

        returns It = 1 X n_t
        '''
        n_t = V_train.shape[1]
        self.It = np.zeros(shape=(1, n_t))
        spike_instants_delayed = [si + int(self.tau_d // delta_t) for si in spike_instants]
        # print(spike_instants_delayed)
        # return
        for t in range(n_t):
            contribution = np.array(spike_instants_delayed[0]) < t
            contribution_i = np.where(contribution == 1)[0]
            t_calc = np.array(spike_instants_delayed[0][contribution_i])
            if t_calc.size != 0:
                s = self.f(t * delta_t, t_calc * delta_t)
                self.It[0, t] = self.I0 * self.w * s
            else:
                self.It[0, t] = 0
        return self.It

    def f(self, t, t_calc):
        s1 = np.exp(-(t - t_calc) / self.tau)
        s2 = np.exp(-(t - t_calc) / self.tau_s)
        s = s1 - s2
        s = np.sum(s)
        return s


class PLASTIC_SYNAPSE_A():
    '''
    This synapse can be represented
    by a single weight, update rule as given in update function
    '''

    def __init__(self, w, I0, tau, tau_s, tau_d):
        self.w = w
        self.I0 = I0
        self.tau = tau
        self.tau_s = tau_s
        self.tau_d = tau_d

    def getI(self, V_train, spike_instants, delta_t):
        '''
        V_train : 1 X n_t
        spike_instants : list(arr(num_of_spikes))

        returns It = 1 X n_t
        '''
        n_t = V_train.shape[1]
        self.It = np.zeros(shape=(1, n_t))
        spike_instants_delayed = [si + int(self.tau_d // delta_t) for si in spike_instants]
        # print(spike_instants_delayed)
        # return
        for t in range(n_t):
            contribution = np.array(spike_instants_delayed[0]) < t
            contribution_i = np.where(contribution == 1)[0]
            t_calc = np.array(spike_instants_delayed[0][contribution_i])
            if t_calc.size != 0:
                s = self.f(t * delta_t, t_calc * delta_t)
                self.It[0, t] = self.I0 * self.w * s
            else:
                self.It[0, t] = 0
        return self.It

    def f(self, t, t_calc):
        s1 = np.exp(-(t - t_calc) / self.tau)
        s2 = np.exp(-(t - t_calc) / self.tau_s)
        s = s1 - s2
        s = np.sum(s)
        return s

    # upd_coeff is {-1,1} according to increment/decrement rule
    def weight_update(self, gamma, delta_tk, upd_coeff):
        '''
        update the weight and will return the delta by which it updated
        '''
        s1 = np.exp(- delta_tk / self.tau)
        s2 = np.exp(- delta_tk / self.tau_s)
        if upd_coeff == -1:
            if self.w <= 1:
                self.w = 1
                return 1 - self.w
            else:
                self.w = self.w + upd_coeff * self.w * gamma * (s1 - s2)
                return upd_coeff * self.w * gamma * (s1 - s2)
                # print('weights fixed to 10')
        elif upd_coeff == 1:
            if self.w >= 500:
                self.w = 500
                return 500 - self.w
                # print('weights fixed to 500')
            else:
                self.w = self.w + upd_coeff * self.w * gamma * (s1 - s2)
                return upd_coeff * self.w * gamma * (s1 - s2)


class PLASTIC_SYNAPSE_B():
    '''
    This synapse will use updates
    considering the delayed time effect
    '''

    def __init__(self, w, I0, tau, tau_s, tau_d, tau_l, A_up, A_dn):
        self.w = w
        self.I0 = I0
        self.tau = tau
        self.tau_s = tau_s
        self.tau_d = tau_d
        # for weight updates
        self.tau_l = tau_l
        self.A_up = A_up
        self.A_dn = A_dn

    def getI(self, V_train, spike_instants, delta_t):
        '''
        V_train : 1 X n_t
        spike_instants : list(arr(num_of_spikes))

        returns It = 1 X n_t
        '''
        n_t = V_train.shape[1]
        self.It = np.zeros(shape=(1, n_t))
        spike_instants_delayed = [si + int(self.tau_d // delta_t) for si in spike_instants]
        # print(spike_instants_delayed)
        # return
        for t in range(n_t):
            contribution = np.array(spike_instants_delayed[0]) < t
            contribution_i = np.where(contribution == 1)[0]
            t_calc = np.array(spike_instants_delayed[0][contribution_i])
            if t_calc.size != 0:
                s = self.f(t * delta_t, t_calc * delta_t)
                self.It[0, t] = self.I0 * self.w * s
            else:
                self.It[0, t] = 0
        return self.It

    def f(self, t, t_calc):
        s1 = np.exp(-(t - t_calc) / self.tau)
        s2 = np.exp(-(t - t_calc) / self.tau_s)
        s = s1 - s2
        s = np.sum(s)
        return s

    # upd_coeff represents upstream or downstream
    def weight_update(self, delta_tk, upd_coeff):
        '''
        update the weight and will return the delta by which it updated
        '''
        # print('old w: ', self.w)
        s1 = np.exp(- delta_tk / self.tau_l)
        # print('s1: ', s1)
        if upd_coeff == 1:
            # upstream
            self.w = self.w + self.w * (self.A_up * s1)
        elif upd_coeff == -1:
            self.w = self.w + self.w * (self.A_dn * s1)
        # print('new w: ', self.w)
        return self.w


# using SYNAPSE and SPIKETRAINS:

import matplotlib.pyplot as plt

# from utils.SpikeTrains import POISSON_SPIKE_TRAIN, RANDOM_SPIKE_TRAIN

n_out = 2
T = 500 * (10 ** -3)
delta_t = 0.1 * (10 ** -3)
n_t = int(T / delta_t)

ST = POISSON_SPIKE_TRAIN(T=T, delta_t=delta_t, lamb=10, n_out=n_out)
V, SI = ST.V_train, ST.spike_instants

w = 500
I0 = 1 * (10 ** -12)
tau = 15 * (10 ** -3)
tau_s = tau / 4
tau_d = 0.1

# w, I0, tau, tau_s, tau_d):
#                  w, I0, tau, tau_s, tau_d, tau_l, A_up, A_dn):
Sy = CONST_SYNAPSE(w, I0, tau, tau_s, tau_d)
I = Sy.getI(V, SI, delta_t)

plt.figure()
plt.suptitle('spike train and synaptic current')

plt.subplot(2, 1, 1)
plt.plot(list(range(n_t + 1)), V[0, :],'-o')
plt.plot(list(range(n_t + 1)), V[1, :],'-or')

plt.xlabel('time')
plt.ylabel('V')

plt.subplot(2, 1, 2)
plt.plot(list(range(n_t + 1)), I[0, :])
plt.xlabel('time')
plt.ylabel('I')
plt.show()


