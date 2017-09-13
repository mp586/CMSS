import numpy as np 
import matplotlib.pyplot as plt 

def initialBell(x): 
    """Intitial Funtion"""
    print(initialBell.__doc__)
    return np.where(x%1.<0.5, np.power(np.sin(2*x*np.pi), 2), 0) 
    # x%1 = x mod 1. 

def periodic_BC(phi,c,nx):
    phi_start = phi[0] - 0.5*c[0]*(phi[1] - phi[nx-1])
    print('phi[0] is ',phi_start)
    return phi_start
    
def first_timestep(phi_old,c,nx):
    """ FTCS for the first timestep """
    phi = phi_old.copy()
    for j in xrange(1,nx):
        phi[j] = phi_old[j] - (c[j]/2.)*(phi_old[j+1] - phi_old[j-1])
        # use c in each spatial position j
    return phi
    
    
def Burger(phi_old,phi,c,nx):
    """ Non-linear advection, CTCS scheme """
    phi_new=phi_old.copy()
    for j in xrange(1,nx):
        phi_new[j] = phi_old[j] - c[j]*(phi[j+1] - phi[j-1])       
    return phi_new
    
def main():
    nx = 100
    nt = 30
    dt = 0.005
    dx = 1./nx
    
    x=np.linspace(0.0, 1.0, nx+1)
    phi = initialBell(x)
    phi_old = phi.copy()
    
    c = phi*dt/dx    # for boundary condition calculation    
    phi[0] = periodic_BC(phi,c,nx)
    phi[nx] = phi[0]
    
    phi = first_timestep(phi_old,c,nx)
    
    c=np.zeros((nx+1,nt+1))
    for i in xrange(1,nt):
        c[:,i] = phi*dt/dx # c is different for each time step
        print(np.max(c[:,i]))
        phi_new = Burger(phi_old,phi,c[:,i],nx)
        phi_old = phi.copy()
        phi = phi_new.copy()
    
    fig = plt.figure()
    plt.plot(x,initialBell(x),'b',label='initial')
    plt.plot(x,phi,'r',label='final_Burger')
    plt.annotate('c_max_Burger=%.2f'%(np.max(c)),xy=(0.15,0.8),xycoords='figure fraction')
    
# linear advection    
    u=0.5
    c_lin=np.ones((nx+1))*u*(dt/dx)
    for i in xrange(1,nt):
        phi_new = Burger(phi_old,phi,c_lin,nx)
        phi_old = phi.copy()
        phi = phi_new.copy()
    plt.plot(x,phi,'g',label='final_lin_adv')
    plt.annotate('c_lin_adv=%.2f'%(np.max(c_lin)),xy=(0.15,0.85),xycoords='figure fraction')
    plt.legend()
    plt.show()     
    
if __name__ == '__main__':
    main()
    
    
