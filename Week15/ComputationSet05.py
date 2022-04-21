import numpy as np
from numpy import pi as pi
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt

import sys
#sys.path.append('C:/Users/brend/Documents/GitHub/AttitudeDynamics') # Windows
sys.path.append('/Users/brendangillis/Documents/AttitudeDynamics')  # Mac
from AAE440_Funcs import *


def sattelite_orientation_MRP(t, state, I1, I2, I3, Omega, R_O):
        """
        Return the derivative of omega and
        MRP for sattelite orientation
        Account for gravity gradient torque
        **note state is an array of [omega, MRP]
        R_O is the orbit radius in Orbit frame
        """
        state_dot   = np.zeros_like(state)
        omega       = state[0:3]
        MRP         = state[3:6]
        omega_ON    = np.array([0, 0, Omega])

        BO          = MRPtoDCM(MRP)
        R_B         = np.dot(BO, R_O)
        omega_BO    = omega - np.dot(BO, omega_ON)

        omega_dot   = dwdt_gravTorque(omega, [I1, I2, I3], Omega, R_B)
        MRP_dot          = KDE_MRP(MRP, omega_BO)
        state_dot[0:3]   = omega_dot
        state_dot[3:6]   = MRP_dot
        return state_dot
  
    
def main():
    
    np.set_printoptions(precision=3)

    # Problem 01
    # Part C
    mu      = 3.989e14 # m/s^2
    m       = 100   # [kg]
    l       = 0.7   # [m] 
    I       = m*l**2 * np.array([[5/12, 0, 0],
                                 [0, 5/6, 0],
                                 [0, 0, 13/12]])
    
    Rc      = 6578e3  # [m] 200km altitude
    Rc_O    = np.array([Rc, 0, 0]) 
    
    MRP_BO  = np.array([1/3, 1/4, 1/5])
    DCM_BO  = MRPtoDCM(MRP_BO)
    Rc_B    = np.dot(DCM_BO, Rc_O)    
    
    Fg      = -mu/Rc**3*(m*Rc_B+3/Rc**2*np.dot(I, Rc_B)+3/(2*Rc**2)*np.trace(I)*Rc_B 
                         - 15/(2*Rc**4)*np.dot(Rc_B, np.dot(I, Rc_B))*Rc_B)
    Lg      = 3*mu/(Rc**5)*np.cross(Rc_B, np.dot(I, Rc_B))
    
    #print("\n\nFg: {}".format(Fg))
    #print("Lg: {}".format(Lg))
    
    Fg_1st = -mu/Rc**3*(m*Rc_B)

    #print("Fg2-Fg1: {}".format(Fg-Fg_1st))
    
    Rcg_norm    = (mu*m/np.linalg.norm(Fg))**0.5
    Rcg         = -Fg/np.linalg.norm(Fg)*Rcg_norm
    #print("||Rcg-Rc||: {}".format(np.linalg.norm(Rcg-Rc_B)))
    
    # Part d
    m       = 4.5e5   # [kg]
    l       = 33   # [m] 
    I       = m*l**2 * np.array([[5/12, 0, 0],
                                 [0, 5/6, 0],
                                 [0, 0, 13/12]])
    
    Rc      = 6578e3  # [m] 200km altitude
    Rc_O    = np.array([Rc, 0, 0]) 
    
    MRP_BO  = np.array([1/3, 1/4, 1/5])
    DCM_BO  = MRPtoDCM(MRP_BO)
    Rc_B    = np.dot(DCM_BO, Rc_O)    
    
    Fg      = -mu/Rc**3*(m*Rc_B+3/Rc**2*np.dot(I, Rc_B)+3/(2*Rc**2)*np.trace(I)*Rc_B 
                         - 15/(2*Rc**4)*np.dot(Rc_B, np.dot(I, Rc_B))*Rc_B)
    Lg      = 3*mu/(Rc**5)*np.cross(Rc_B, np.dot(I, Rc_B))
    
    #print("\n\nFg: {}".format(Fg))
    #print("Lg: {}".format(Lg))
    
    Fg_1st = -mu/Rc**3*(m*Rc_B)

    #print("Fg2-Fg1: {}".format(Fg-Fg_1st))
    
    Rcg_norm    = (mu*m/np.linalg.norm(Fg))**0.5
    Rcg         = -Fg/np.linalg.norm(Fg)*Rcg_norm
    #print("||Rcg-Rc||: {}".format(np.linalg.norm(Rcg-Rc_B)))
    
    # Part e
    Rc      = 358e5  # [m] 200km altitude
    Rc_O    = np.array([Rc, 0, 0]) 
    
    MRP_BO  = np.array([1/3, 1/4, 1/5])
    DCM_BO  = MRPtoDCM(MRP_BO)
    Rc_B    = np.dot(DCM_BO, Rc_O)    
    
    Fg      = -mu/Rc**3*(m*Rc_B+3/Rc**2*np.dot(I, Rc_B)+3/(2*Rc**2)*np.trace(I)*Rc_B 
                         - 15/(2*Rc**4)*np.dot(Rc_B, np.dot(I, Rc_B))*Rc_B)
    Lg      = 3*mu/(Rc**5)*np.cross(Rc_B, np.dot(I, Rc_B))
    
    #print("\n\nFg: {}".format(Fg))
    #print("Lg: {}".format(Lg))
    
    Fg_1st = -mu/Rc**3*(m*Rc_B)

    #print("Fg2-Fg1: {}".format(Fg-Fg_1st))
    
    Rcg_norm    = (mu*m/np.linalg.norm(Fg))**0.5
    Rcg         = -Fg/np.linalg.norm(Fg)*Rcg_norm
    #print("||Rcg-Rc||: {}".format(np.linalg.norm(Rcg-Rc_B)))


    ######################################
    # Problem 02
    Is       = [np.array([400, 600, 800]),np.array([600, 400, 800]),np.array([400, 800, 600])]

    for I in Is:
        I1      = I[0]
        I2      = I[1]
        I3      = I[2]

        mu      = 3.986e14  # [m^3/s^2]
        R       = 6800e3    # [m]
        R_O     = [-R, 0, 0]

        Omega   = (mu/R**3)**0.5
        MRP0    = 1/np.sqrt(3)*np.tan(pi/(4*60))*np.array([1, 1, 1])
        omega0  = np.array([0, 0, Omega])

        state0  = np.zeros(6)
        state0[0:3] = omega0
        state0[3:6] = MRP0
        
        tf      = 3600*24
        t       = np.linspace(0, tf, 1000)
        sol = solve_ivp(sattelite_orientation_MRP , y0=state0, t_span=(0,tf), 
                        t_eval=t, args=(I1, I2, I3, Omega, R_O), atol=1e-10, rtol=1e-10)

        t           = sol.t
        omega_hist  = sol.y[0:3, :]
        MRP_hist    = sol.y[3:6, :]
        
        fig, axs = plt.subplots(2, 1)
        l1 = axs[0].plot(t/3600, omega_hist[0, :]*180/pi, label = r"$\omega_1$")
        l2 = axs[0].plot(t/3600, omega_hist[1, :]*180/pi, label = r"$\omega_1$")
        l3 = axs[0].plot(t/3600, omega_hist[2, :]*180/pi, label = r"$\omega_1$")
        axs[0].set_title(r'$\omega_i$')
        axs[0].legend(shadow=True, fancybox=True)
        axs[0].set(ylabel=r'$\omega$, [deg/s]')

        l1 = axs[1].plot(t/3600, MRP_hist[0, :], label = r"$\sigma_1$") 
        l2 = axs[1].plot(t/3600, MRP_hist[1, :], label = r"$\sigma_2$")
        l3 = axs[1].plot(t/3600, MRP_hist[2, :], label = r"$\sigma_3$")
        axs[1].set_title(r'$\sigma_i$')
        axs[1].legend(shadow=True, fancybox=True)
        #axs[1].set_ylim([-0.2, 0.2])
        axs[1].set(ylabel=r'$\sigma$')

        fig.tight_layout(pad=2.5)
        for ax in axs.flat:
            ax.set(xlabel='Time [hr]')
            ax.set_xlim([0, t[-1]/3600])
            
        fig.suptitle(r"$I_1$ = {}, $I_2$ = {}, $I_3$ = {} [kg/$m^2$] $\theta:$ 3 [deg]".format(I1, I2, I3))    
        plt.show()
        plt.close(fig)

        theta   = np.zeros_like(t)
        for i, MRP in enumerate(MRP_hist.transpose()):
            [theta[i], _] = DCMtoPRP(MRPtoDCM(MRP))

        fig, ax = plt.subplots(1, 1)
        ax.plot(t/3600, theta*180/pi)
        ax.set_title(r'$\theta$')
        ax.set(ylabel=r'$\theta$, [deg]')
        ax.set(xlabel='Time [hr]')
        ax.set_xlim([0, t[-1]/3600])

        fig.tight_layout(pad=2.5)     
        fig.suptitle(r"$I_1$ = {}, $I_2$ = {}, $I_3$ = {} [kg/$m^2$] $\theta:$ 3 [deg]".format(I1, I2, I3))    
        plt.show()
        plt.close(fig)
        print(max((theta*180/pi)/3))

    return


if __name__ == "__main__":
    main()