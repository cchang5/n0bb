import numpy as np
import lsqfit
import gvar as gv
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

pidat = './l3248_pion.dat'
medat = './l3248_3pt.dat'

f = open(pidat,'r')
pi1 = []
pi2 = []
for l in f: # configs
    s0 = l.split("\",\"")
    pit1 = []
    pit2 = []
    for m in s0: # t
        s1 = m.split(',')
        pis = []
        for n in s1: # srcs
            dat = float(n.replace("\"",'').replace("{",'').replace("}",''))
            pis.append(dat)
        pit1.append(pis[0])
        pit2.append(pis[1])
    pi1.append(pit1)
    pi2.append(pit2)
pi1 = np.array(pi1)
pi2 = np.array(pi2)
f.close()

f = open(medat,'r')
me1 = []
me2 = []
for l in f: # configs
    s0 = l.split("}}\",\"{{")
    met1 = []
    meu1 = []
    for m in s0: # t1
        s1 = m.split('}}, {{')
        met2 = []
        meu2 = []
        for n in s1: # t1
            s2 = n.split(',')
            mes = []
            for o in s2: # srcs
                dat = float(o.replace("\"",'').replace("{",'').replace("}",'').replace('*^','E'))
                mes.append(dat)
            met2.append(mes[0])
            meu2.append(mes[1])
        met1.append(met2)
        meu1.append(meu2)
    me1.append(met1)
    me2.append(meu1)
print np.shape(me1)
me1 = np.array(me1)
me2 = np.array(me2)
f.close()

# make correlated data
def make_dat(pi,me):
    meflat = []
    T = len(me[0]) # this is actually T/2
    for c in me:
        meflat.append(c.reshape(T**2))
    print np.shape(meflat)
    print np.shape(pi)
    dat = {'pi':pi, 'me':meflat}
    gvdat = gv.dataset.avg_data(dat)
    foldpi = 0.5*(gvdat['pi']+np.roll(gvdat['pi'][::-1],-1))[:T]
    gvdat = {'pi':foldpi,'me':gvdat['me'].reshape((T,T))}
    return gvdat

dat1 = make_dat(pi1, me1)
dat2 = make_dat(pi2, me2)

# plot data
def plot_data(dat):
    # plot meff
    pidat = dat['pi']
    meff = np.arccosh(((np.roll(pidat,-1)+np.roll(pidat,1))/(2.*pidat))[1:len(pidat)-1])
    x = np.arange(1,len(pidat)-1)
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    ax.errorbar(x=x,y=[i.mean for i in meff], yerr=[i.sdev for i in meff])
    plt.draw()
    # plot amplitude
    zeff = pidat[1:len(pidat)-1]*np.exp(meff*x)
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    ax.errorbar(x=x,y=[i.mean for i in zeff], yerr=[i.sdev for i in zeff])
    plt.draw()
    # plot matrix elements
    medat = dat['me'][1:len(pidat)-1,1:len(pidat)-1]
    x = np.arange(1,len(pidat)-1)
    #y = np.arange(1,len(pidat)-1)
    #X, Y = np.meshgrid(x,y)
    mescale = np.diagonal(medat)*np.exp(2.*meff*x)*2.*meff/zeff #*np.exp(meff*Y)
    #Z = []
    #for i in range(len(mescale)):
    #    t = []
    #    for j in range(len(mescale)):
    #        t.append(mescale[i,j].mean)
    #    Z.append(t)
    #fig = plt.figure(figsize=(7,7))
    #ax = plt.axes([0.15,0.15,0.8,0.8])
    #ax.contourf(X,Y,Z,100)
    fig = plt.figure(figsize=(7,4.326237))
    ax = plt.axes([0.15,0.15,0.8,0.8])
    ax.errorbar(x=x,y=[i.mean for i in mescale], yerr=[i.sdev for i in mescale])
    ax.set_ylim([-0.0004,0.0001])
    ax.set_xlim([2.50,20.5])
    ax.set_xlabel('$t$', fontsize=20)
    ax.set_ylabel("$\mathcal{O}_3$", fontsize=20)
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    plt.draw()
    plt.savefig('ME_data.pdf',transparent=True)
    plt.show()
plot_data(dat1)
