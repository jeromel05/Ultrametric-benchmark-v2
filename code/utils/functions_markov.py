import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm.notebook import trange, tqdm

def fillmarkov(m,i1,j1,l,md):
    l2=int(l/2)
    for i in range(i1, (i1+l2)):
        for j in range(j1, (j1+l2)):
            m[i,j+l2]=md
            m[j+l2,i]=md

    if(md>1):
        m=fillmarkov(m,0,0,l2,md-1)
        for jjn in range(0, l2):
            for kkk in range(0, l2):
                m[kkk+l2,jjn+l2]=m[kkk,jjn]

    return m

def compute_P_0(maxh, tree_l, chain, verbose=0):
    if verbose>0: print("Computing P_0...")
    P0=np.zeros(maxh)

    for i in tqdm(range(0, tree_l), leave=False):
        P0stat = compute_P0_inner_loop(i, P0, chain, tree_l, maxh)
    
    P0stat[0]=1
    P0stat[1]=1
    if verbose>0: print("P_0 computed")
    return P0, P0stat

def compute_P0_inner_loop(i, P0, chain, tree_l, maxh):
    P0stat=np.zeros(maxh)
    locs=np.where(chain==i)[0] #find all occurrences of i
    if locs.shape[0] == 0: 
        return P0stat
    else:
        for j in range(0, locs.shape[0]):
            for k in range(j+1, locs.shape[0]):
                deltat=locs[k]-locs[j]
                if(deltat>=maxh): break            
                P0[deltat]=P0[deltat]+1

        P0=P0/locs.shape[0]
        P0stat=P0stat+P0/tree_l
    return P0stat

def plot_P0(P0stat):
    xv=np.concatenate(([1], np.arange(2, P0stat.shape[0], 2))) 
    # plot only initial value and even values to remove oscillations
    h=plt.plot(xv, P0stat[xv],'r-')
    ax = plt.gca()
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel('∆t')
    ax.set_ylabel('P0(∆t)')
    plt.show()

def normalize_matrix(dia, tree_l, markovme):
    if(dia==0):
        for i in range(0, tree_l):
            tot=np.sum(markovme[i])-1 # don't count the diagonal terms (which are ones)
            markovme[i]=markovme[i]/(tot+1e-6)
            markovme[i,i]=0
    else:
        # diagonal is set to zero to avoid too long sequences with the same
        # presentation
        for i in range(0, tree_l):
            tot=np.sum(markovme[i]) # count also the diagonal terms
            markovme[i]=markovme[i]/(tot+1e-6)   
            
    return markovme

def generate_chain(markovme, chain_length, verbose=0):
    chain=np.zeros((chain_length), dtype=np.int32)
    chain[0]=1
    if verbose>0: print("Generating Markov chain...")
    for i in range(1, chain_length):
        this_step_distribution = markovme[int(chain[i-1])]
        cumulative_distribution = np.cumsum(this_step_distribution)
        r = np.random.rand(1)[0]
        if np.any(cumulative_distribution>r):
            chain[i] = np.argmax(cumulative_distribution>r)
        
    return chain

def generate_markov_chain(chain_length=500000, T=0.2, tree_levels=3, dia=0):
    tree_l=2**tree_levels # number of leaves
    markovm=np.zeros((tree_l,tree_l))
    
    markovm=fillmarkov(markovm,0,0,tree_l,tree_levels)
    markovme=np.exp(-markovm/T)
    markovme = normalize_matrix(dia, tree_l, markovme)
    # generate the markov chain
    chain = generate_chain(markovme, chain_length, verbose=1)
    return chain 

def shuffle_blocks_v2(chain, b_len):
    shuff_chain = chain.copy() #bc we modifiy in-place afterward
    blocks = [shuff_chain[i:i+b_len] for i in range(0,len(shuff_chain),b_len)]
    np.random.shuffle(blocks)
    shuff_chain[:] = [b for bs in blocks for b in bs]
    return shuff_chain
