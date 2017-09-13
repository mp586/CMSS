import numpy as np 
import matplotlib.pyplot as plt 

def initialStep(x): 
    """ Intitial Step for height """
    print(initialStep.__doc__)
    return np.where(x<0.1,1.,0) 
    # x%1 = x mod 1. 
    
def unstaggered_1dSW(u_old,h_old,nx,dx,dt,H,g):
    """ """
    u = u_old.copy()
    h = h_old.copy()
    
    for j in xrange(1,nx):      
        u[j] = u_old[j] - g*(dt/2*dx)*(h_old[j+1] - h_old[j-1])
        h[j] = h_old[j] - H*(dt/2*dx)*(u[j+1] - u[j-1])
    u[0] = u_old[0] - g*(dt/2*dx)*(h_old[1] - h_old[nx-1])
    h[0] = h_old[0] - H*(dt/2*dx)*(u[1] - u[nx-1])
    h[nx] = h[0]
    u[nx] = u[0]
    return u,h
    
# def staggered_1dSW(u_old,h_old,nx,dx,dt,H,g,x_half): 
 # u_half = np.interp(x_half,x,u_old)   
    
def main():
    nx = 100
    nt = 1000
    H = 10 # height of water column
    g = 9.81 
    dt = 0.1
    dx = 1./nx
    
    x = np.linspace(0.0, 1.0, nx+1)
    h = initialStep(x)
    u = np.ones((nx+1))*0.
    fig = plt.figure()
    plt.plot(x,h,'g',label='initial_h')
    plt.plot(x,u,'k',label='initial_u')
    
    u_old = u.copy()
    h_old = h.copy()
    
    for i in xrange(1,nt):
        u, h = unstaggered_1dSW(u_old,h_old,nx,dx,dt,H,g)
        u_old = u.copy()
        h_old = h.copy()
          
    plt.plot(x,u,'r',label='final_u')
    plt.plot(x,h,'b',label='final_h')
    plt.ylim((0,20))
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
    
