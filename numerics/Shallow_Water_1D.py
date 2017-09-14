import numpy as np 
import matplotlib.pyplot as plt 

# results are a bit weird for unstable (unstaggered) should be symmetric but instability is only on the side --> possibly a problem with the boundary conditions

def initialStep(x): 
    """ Intitial Step for height """
    print(initialStep.__doc__)
    return np.where(x<=0.1,1.,0) 
    # x%1 = x mod 1. 
    
def initialBell(x): 
    """Intitial Bell Funtion"""
    print(initialBell.__doc__)
    return np.where(x%1.<0.5, np.power(np.sin(2*x*np.pi), 2), 0) 
    # x%1 = x mod 1. 
    
def unstaggered_1dSW(u_old,h_old,nx,dx,dt,H,g):
    """ FTCS for non-rotating shallow water equations (1D) """
    u = u_old.copy()
    h = h_old.copy()
    for j in xrange(-1,nx-1): # -1 = last element phi[nx] --> avoid periodic BC    
        u[j] = u_old[j] - g*(dt/(2.*dx))*(h_old[j+1] - h_old[j-1])
        h[j] = h_old[j] - H*(dt/(2.*dx))*(u[j+1] - u[j-1])
#    u[0] = u_old[0] - g*(dt/2*dx)*(h_old[1] - h_old[nx-1])
#    h[0] = h_old[0] - H*(dt/2*dx)*(u[1] - u[nx-1])
#    h[nx] = h[0]
#    u[nx] = u[0]
    return u,h
    
def staggered_1dSW(u_stgg_old,h_stgg_old,nx,dx,dt,H,g): 
    u_stgg = u_stgg_old.copy()
    h_stgg = h_stgg_old.copy()   
    for j in xrange(-1,nx-1):
        u_stgg[j] = u_stgg_old[j] - g*(dt/dx)*(h_stgg_old[j+1] - h_stgg_old[j])
        h_stgg[j] = h_stgg_old[j] - H*(dt/dx)*(u_stgg[j] - u_stgg[j-1])        
    return u_stgg, h_stgg
    
def main():
    nx = 100
    nt = 500
    H = 5 # height of water column
    g = 9.81 
    dt = 0.0001
    dx = 1./nx
    
    x = np.linspace(0.0, 1.0, nx+1)
    h = initialBell(x)
    u = np.ones((nx+1))*0.5
    
    h_init = h.copy()
    u_init = u.copy()
    
    fig = plt.figure()
    plt.plot(x,h_init,'g',label='initial_h')
    plt.plot(x,u_init,'orange',label='initial_u')
    
    u_old = u.copy()
    h_old = h.copy()
    
    h_stgg_old = h.copy()    
    x_half = x + dx/2.
    u_stgg_old = np.interp(x_half,x,u) # interpolated to x_half
    h_stgg_old = h.copy()
    
    for i in xrange(1,nt):
        u, h = unstaggered_1dSW(u_old,h_old,nx,dx,dt,H,g)
        u_old = u.copy()
        h_old = h.copy()
        u_stgg, h_stgg = staggered_1dSW(u_stgg_old,h_stgg_old,nx,dx,dt,H,g)
        u_stgg_old = u_stgg.copy()
        h_stgg_old = h_stgg.copy()
         
    plt.plot(x,u,'r',label='final_u')
    plt.plot(x,h,'b',label='final_h')
    plt.legend()
    plt.annotate('nt = '+str(nt),xy=(0.7,0.15),xycoords='figure fraction')
    plt.annotate('dt = '+str(dt),xy=(0.7,0.13),xycoords='figure fraction')
    plt.annotate('dx = '+str(dx),xy=(0.7,0.11),xycoords='figure fraction')
    plt.annotate('H = '+str(H),xy=(0.7,0.20),xycoords='figure fraction')    
    plt.annotate('c = %.2f'%(np.sqrt(g*H)*(dt/dx)), xy=(0.7,0.17),xycoords='figure fraction')

    plt.show()
    plt.close()
    
    fig = plt.figure()
    plt.plot(x,u_stgg,'r',label='final_stgg_u')
    plt.plot(x,h_stgg,'b',label='final_stgg_h')
    plt.plot(x,h_init,'g',label='initial_h')
    plt.plot(x,u_init,'orange',label='initial_u')
    plt.annotate('nt = '+str(nt),xy=(0.7,0.15),xycoords='figure fraction')
    plt.annotate('dt = '+str(dt),xy=(0.7,0.13),xycoords='figure fraction')
    plt.annotate('dx = '+str(dx),xy=(0.7,0.11),xycoords='figure fraction')
    plt.annotate('H = '+str(H),xy=(0.7,0.20),xycoords='figure fraction')    
    plt.annotate('c = %.2f'%(np.sqrt(g*H)*(dt/dx)), xy=(0.7,0.17),xycoords='figure fraction')
    plt.legend()
    plt.show()
    
if __name__ == '__main__':
    main()
    
