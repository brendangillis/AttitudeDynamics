import numpy as np
from numpy import pi as pi
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import axes3d

import sys
sys.path.append('C:/Users/brend/Documents/GitHub/AttitudeDynamics') # Windows
#sys.path.append('/Users/brendangillis/Documents/AttitudeDynamics')  # Mac
from AAE440_Funcs import *

def sattelite_orientation_MRP_dual_spin(t, state, I1, I2, I3, Omega, R_O, J, Ww):
        """
        Return the derivative of omega and
        MRP for sattelite orientation
        Account for gravity gradient torque
        **note state is an array of [omega, MRP]
        R_O is the orbit radius in Orbit frame.  
        J and Ww coorspond to a reaction conttrol wheel
        """
        state_dot   = np.zeros_like(state)
        omega       = state[0:3]
        MRP         = state[3:6]
        omega_ON    = np.array([0, 0, Omega])

        BO          = MRPtoDCM(MRP)
        R_B         = np.dot(BO, R_O)
        omega_BO    = omega - np.dot(BO, omega_ON)

        omega_dot   = dwdt_gravTorque(omega, [I1, I2, I3], Omega, R_B)
        MRP_dot     = KDE_MRP(MRP, omega_BO)
        omega_dot[0]  = omega_dot[0] - J/I1*Ww*omega[1]
        omega_dot[1]  = omega_dot[1] + J/I2*Ww*omega[0]
        
        state_dot[0:3]   = omega_dot
        state_dot[3:6]   = MRP_dot
        return state_dot

def main():
    
    np.set_printoptions(precision=3)
    I   = np.array([300, 600, 500])
    J   = 0.05
    K1  = (I[1]-I[2])/I[0]
    K2  = (I[2]-I[0])/I[1]
    K3  = (I[0]-I[1])/I[2]
    
    Omega   = .1
    u       = np.linspace(-3*I[0]/J, 3*I[0]/J, 5000)
    
    b2  = J**2/(I[0]*I[1])*Omega**2
    b1  = (K2/I[0]-K1/I[1])*J*Omega**2
    b0  = (1+3*K2-K1*K2)*Omega**2
    
    c2  = J**2/(I[0]*I[1])*Omega**4
    c1  = (4*K2/I[0]-K1/I[1])*J*Omega**4
    c0  = -4*K1*K2*Omega**4
    
    b   = b2*u**2+b1*u+b0
    c   = c2*u**2+c1*u+c0
    
    fig, ax = plt.subplots(1, 1)
    ax.plot(u, b/Omega**2, 'b-', label = "b")
    ax.plot(u, c/Omega**4, 'g-', label = "c")
    ax.plot(u, (b**2-4*c)/Omega**4, 'm-', label = r"$b^2-4c$")
    ax.set_xlim([u[0], u[-1]])
    ax.set_ylim([-5, 15])
    
    region = np.logical_or(c<0, b<0)
    region = np.logical_or(region, (b**2-4*c)<0)
    ax.fill_between(u, -5, 0, facecolor='gray', alpha=0.3)
    ax.fill_between(u, -5, 0, where = region, facecolor='red', alpha=0.4, label = "unstable")   

    fig.suptitle(r"$I_1$ = {}, $I_2$ = {}, $I_3$ = {} [kg/$m^2$]".format(I[0], I[1], I[2]))   
    ax.set(xlabel=r'$\omega_w/\Omega$')
    ax.legend(shadow=True, fancybox=True)
    fig.tight_layout()  
    plt.grid()
    plt.show()
    plt.close(fig)
    
    roots1 = np.zeros([len(u), 2])
    roots2 = np.zeros([len(u), 4])
    for i, u1 in enumerate(u):
        roots1[i, :] = np.roots([1, 0, -K3*Omega**2])
        b = b2*u1**2+b1*u1+b0
        c = c2*u1**2+c1*u1+c0
        roots2[i, :] = np.roots([1, 0, b, 0, c])
        
    fig, ax = plt.subplots(1, 1)
    ax.plot(u, roots1, 'b-', label = ["Group 1", ""])
    ax.plot(u, roots2, 'g-', label = ["Group 2", "", "", ""])
    ax.set_xlim([u[0], u[-1]])
    ax.set_ylim([-0.1, 0.1])
    
    region = np.zeros_like(u)
    for i, _ in enumerate(u):
        region[i] = (any(roots1.real[i, :]>0.001) or any(roots2.real[i, :]>.001))
    
    ax.fill_between(u, 0, 0.8, facecolor='gray', alpha=0.3)
    ax.fill_between(u, 0, 0.8, where = region, facecolor='red', alpha=0.4, label = "unstable")

    fig.suptitle(r"$I_1$ = {}, $I_2$ = {}, $I_3$ = {} [kg/$m^2$]".format(I[0], I[1], I[2])) 
    ax.set(xlabel=r'$\omega_w/\Omega$')
    ax.set(ylabel=r'$Re(\lambda_i)$')
    ax.legend(shadow=True, fancybox=True)
    fig.tight_layout()  
    plt.grid()
    plt.show()
    plt.close(fig)
    
    mu      = 3.986e14  # [m^3/s^2]
    R       = 6500e3    # [m]
    R_O     = [-R, 0, 0]
    Omega   = (mu/R**3)**0.5
    J       = 0.05
    
    # theta0s = np.linspace(1, 50, 30)*pi/180
    # Wws     = Omega*np.linspace(-3*I[0]/J, 3*I[0]/J, 60)
    
    # theta_max = np.zeros([len(Wws), len(theta0s)]) 
    # for i, theta0 in enumerate(theta0s):    
    #     print("{:.1f}%".format(i/len(theta0s)*100))
    #     for j, Ww in enumerate(Wws):
    #         MRP0    = 1/np.sqrt(3)*np.tan(theta0/4)*np.array([1, 1, 1])
    #         omega0  = np.array([0, 0, Omega])
    
    #         state0  = np.zeros(6)
    #         state0[0:3] = omega0
    #         state0[3:6] = MRP0
            
    #         tf      = 3600*24
    #         t       = np.linspace(0, tf, 1000)
    #         sol = solve_ivp(sattelite_orientation_MRP_dual_spin , y0=state0, t_span=(0,tf), 
    #                         t_eval=t, args=(I[0], I[1], I[2], Omega, R_O, J, Ww), atol=1e-10, rtol=1e-10)
    
    #         t           = sol.t
    #         omega_hist  = sol.y[0:3, :]
    #         MRP_hist    = sol.y[3:6, :]
            
    #         theta   = np.zeros_like(t)
    #         for k, MRP in enumerate(MRP_hist.transpose()):
    #             [theta[k], _] = DCMtoPRP(MRPtoDCM(MRP))
                
    #         theta_max[j, i] = np.max(theta)
    
    # np.save('data/theta0s.npy', theta0s)
    # np.save('data/Wws.npy', Wws)
    # np.save('data/theta_max.npy', theta_max)
    # np.save('data/Omega.npy', Omega)
    theta0s     = np.load('data/theta0s.npy')
    Wws         = np.load('data/Wws.npy')
    theta_max   = np.load('data/theta_max.npy')
    Omega       = np.load('data/Omega.npy')
        
        
    fig, ax = plt.subplots(1, 1)        
    cf = ax.contourf(theta0s*180/pi, Wws*60/(2*pi), theta_max*180/pi, 60, cmap='RdYlGn_r')
        
    fig.suptitle(r"Maximum $\theta$ Perturbation [deg]") 
    ax.set(ylabel=r'$\omega_w$ [rpm]')
    ax.set(xlabel=r'$\theta_0$')
    plt.colorbar(cf, ax=ax);
    
    # Part e
    Ww          = 28/60*2*pi
    theta0s     =   pi/180*np.array([3, 15, 30])
    for theta0 in theta0s:
        MRP0    = 1/np.sqrt(3)*np.tan(theta0/4)*np.array([1, 1, 1])
        omega0  = np.array([0, 0, Omega])

        state0  = np.zeros(6)
        state0[0:3] = omega0
        state0[3:6] = MRP0
        
        tf      = 3600*24
        t       = np.linspace(0, tf, 1000)
        sol = solve_ivp(sattelite_orientation_MRP_dual_spin , y0=state0, t_span=(0,tf), 
                        t_eval=t, args=(I[0], I[1], I[2], Omega, R_O, J, Ww), atol=1e-10, rtol=1e-10)

        t           = sol.t
        omega_hist  = sol.y[0:3, :]
        MRP_hist    = sol.y[3:6, :]
        
        fig, axs = plt.subplots(2, 1)
        l1 = axs[0].plot(t/3600, omega_hist[0, :]*180/pi, label = r"$\omega_1$")
        l2 = axs[0].plot(t/3600, omega_hist[1, :]*180/pi, label = r"$\omega_2$")
        l3 = axs[0].plot(t/3600, omega_hist[2, :]*180/pi, label = r"$\omega_3$")
        axs[0].set_title(r'$\omega_i$')
        axs[0].legend(shadow=True, fancybox=True)
        axs[0].set(ylabel=r'$\omega$, [deg/s]')

        l1 = axs[1].plot(t/3600, MRP_hist[0, :], label = r"$\sigma_1$") 
        l2 = axs[1].plot(t/3600, MRP_hist[1, :], label = r"$\sigma_2$")
        l3 = axs[1].plot(t/3600, MRP_hist[2, :], label = r"$\sigma_3$")
        axs[1].set_title(r'$\sigma_i$')
        axs[1].legend(shadow=True, fancybox=True)
        axs[1].set_ylim([-1, 1])
        axs[1].set(ylabel=r'$\sigma$')

        fig.tight_layout(pad=2.5)
        for ax in axs.flat:
            ax.set(xlabel='Time [hr]')
            ax.set_xlim([0, t[-1]/3600])
            
        fig.suptitle(r"$I_1$={}, $I_2$={}, $I_3$={} [kg/$m^2$], $\theta_0:$ {:.0f} [deg], $\omega_w$={:.0f} [rpm]".format(I[0], I[1], I[2], theta0*180/pi, Ww*60/(2*pi)))   
        plt.show()
        plt.close(fig)

        theta   = np.zeros_like(t)
        for i, MRP in enumerate(MRP_hist.transpose()):
            [theta[i], _] = DCMtoPRP(MRPtoDCM(MRP))

        fig, ax = plt.subplots(1, 1)
        ax.plot(t/3600, theta*180/pi, label = r"$\theta(t)$")
        ax.axhline(y=theta0*180/pi, color='r', linestyle='-', label=r"$\theta_0$")
        
        ax.set(ylabel=r'$\theta$, [deg]')
        ax.set(xlabel='Time [hr]')
        ax.set_xlim([0, t[-1]/3600])

        fig.tight_layout(pad=2.5)  
        ax.legend(shadow=True, fancybox=True)
        fig.suptitle(r"$I_1$={}, $I_2$={}, $I_3$={} [kg/$m^2$], $\theta_0:$ {:.0f} [deg], $\omega_w$={:.0f} [rpm]".format(I[0], I[1], I[2], theta0*180/pi, Ww*60/(2*pi)))    
        plt.show()
        plt.close(fig)
        
        
        
        
        
        
        
        
        
        
        
    
    return


if __name__ == "__main__":
    main()