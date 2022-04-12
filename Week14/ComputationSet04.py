import numpy as np
from numpy import pi as pi
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt

import sys
sys.path.append('C:/Users/brend/Documents/GitHub/AttitudeDynamics')
#sys.path.append('/Users/brendangillis/Documents/AttitudeDynamics')
from AAE440_Funcs import *



def sattelite_orientation_omega(t, omega, I1, I2, I3):
    """
    Return the derivative of omega for 
    sattelite orientation.
    """
    omega_dot  = dwdt_torqueFree(omega, [I1, I2, I3])
    return omega_dot

def sattelite_orientation_MRP(t, state, I1, I2, I3):
        """
        Return the derivative of omega and
        MRP for sattelite orientation.
        **note state is an array of [omega, MRP]
        """
        state_dot   = np.zeros_like(state)
        omega       = state[0:3]
        MRP         = state[3:6]

        omega_dot   = dwdt_torqueFree(omega, [I1, I2, I3])
        MRP_dot      = KDE_MRP(MRP, omega)
        state_dot[0:3]   = omega_dot
        state_dot[3:6]   = MRP_dot
        return state_dot
    

def main():
    
    np.set_printoptions(precision=3)
    
    # Problem 1
    # Part a
    I1  = 3/4
    I2  = 1
    I3  = 3/2

    omega0  = np.array([-0.1, 0.05, 0.1])
    tf  = 200
    t   = np.linspace(0, tf, 1000)
    sol = solve_ivp(sattelite_orientation_omega , y0=omega0, t_span=(0,tf), 
                    t_eval=t, args=(I1, I2, I3), atol=1e-10, rtol=1e-10)
    t           = sol.t
    omega_hist  = sol.y
    
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, omega_hist[0, :])
    axs[0].set_title(r'$\omega_1$')
    axs[1].plot(t, omega_hist[1, :])    
    axs[1].set_title(r'$\omega_2$')
    axs[2].plot(t, omega_hist[2, :])
    axs[2].set_title(r'$\omega_3$')

    fig.tight_layout(pad=2.5)
    for ax in axs.flat:
        ax.set(xlabel='Time [s]')
        ax.set_xlim([0, t[-1]])
    fig.suptitle('Sattelite Angular Velocity History')    
    plt.show()
    plt.close(fig)
    
    # Part b
    I1o  = 1
    I2o  = 1
    I3o  = 3/2

    omega0  = np.array([-0.1, 0.05, 0.1])
    tf  = 200
    t   = np.linspace(0, tf, 1000)
    sol = solve_ivp(sattelite_orientation_omega , y0=omega0, t_span=(0,tf), 
                    t_eval=t, args=(I1o, I2o, I3o), atol=1e-10, rtol=1e-10)
    t                   = sol.t
    omega_hist_CPS3     = sol.y
    
    fig, axs = plt.subplots(3, 1)
    l1 = axs[0].plot(t, omega_hist_CPS3[0, :])
    l2 = axs[0].plot(t, omega_hist[0, :], '--')
    axs[0].set_title(r'$\omega_1$')
    axs[1].plot(t, omega_hist_CPS3[1, :]) 
    axs[1].plot(t, omega_hist[1, :], '--')
    axs[1].set_title(r'$\omega_2$')
    axs[2].plot(t, omega_hist_CPS3[2, :])
    axs[2].plot(t, omega_hist[2, :], '--')
    axs[2].set_title(r'$\omega_3$')

    fig.tight_layout(pad=2.5)
    for ax in axs.flat:
        ax.set(xlabel='Time [s]')
        ax.set_xlim([0, t[-1]])
        
    fig.suptitle('Axisymetric vs Non-Axisymetric                 ')    
    fig.legend([l1, l2], labels=["Axisymmetric  Body", "Non-axisymmetric Body"],
           loc="upper right")
    plt.show()
    plt.close(fig)
    # print(np.amax(omega_hist_CPS3, axis=1))
    # print(np.amax(omega_hist, axis=1))
    
    
    # Part d
    I       = np.array([[3/4, 0, 0],
                        [0, 1, 0],
                        [0, 0, 3/2]])
    T_rot   = np.zeros_like(t)
    H       = np.zeros_like(omega_hist)
    H_l2    = np.zeros_like(t)
    
    for i, omega in enumerate(omega_hist.transpose()):
        H[:, i]     = np.dot(I, omega)
        H_l2[i]     = np.linalg.norm(H[:, i])
        T_rot[i]    = 1/2 * np.dot(omega, H[:, i])
    
    fig, axs = plt.subplots(2, 1)
    axs[0].plot(t, H_l2)
    axs[0].set_title(r'$||H||_2$')
    axs[1].plot(t, T_rot)    
    axs[1].set_title(r'$T_{rot}$')
    
    fig.tight_layout(pad=2.5)
    for ax in axs.flat:
        ax.set(xlabel='Time [s]')
        ax.set_xlim([0, t[-1]])
        ax.set_ylim([0, 0.2])
    fig.suptitle('||H|| and Rotational Energy Over Time')    
    plt.show()
    plt.close(fig)

    # Part e
    A       = np.zeros(3)
    B       = np.zeros(3)
    C       = np.zeros_like(omega_hist)
    Trot    = T_rot[5]
    H       = H_l2[5]
    A[0]      = ((I1-I2)*(2*I3*Trot-H**2)+(I3-I1)*(H**2-2*I2*Trot))/(I1*I2*I3)
    A[1]      = ((I2-I3)*(2*I1*Trot-H**2)+(I1-I2)*(H**2-2*I3*Trot))/(I1*I2*I3)
    A[2]      = ((I3-I1)*(2*I2*Trot-H**2)+(I2-I3)*(H**2-2*I1*Trot))/(I1*I2*I3)
    B[0]      = 2*(I1-I2)*(I1-I3)/(I2*I3)
    B[1]      = -2*(I1-I2)*(I2-I3)/(I1*I3)
    B[2]      = 2*(I1-I3)*(I2-I3)/(I1*I2)
    
    omega_dot = np.zeros_like(omega_hist)
    for i, omega in enumerate(omega_hist.transpose()):
        omega_dot[:, i]     = dwdt_torqueFree(omega, [I1, I2, I3])
    
    for i, _ in enumerate(A):
        C[i, :] = omega_dot[i, :]**2 + A[i]*omega_hist[i, :]**2 + B[i]/2*omega_hist[i, :]**4
    
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, C[0, :])
    axs[0].set_title(r'$C_1$')
    axs[1].plot(t, C[1, :])    
    axs[1].set_title(r'$C_2$')
    axs[2].plot(t, C[2, :])
    axs[2].set_title(r'$C_3$')

    fig.tight_layout(pad=2.5)
    for ax in axs.flat:
        ax.set(xlabel='Time [s]')
        ax.set_xlim([0, t[-1]])
        #ax.set_ylim([0, 0.0])
    fig.suptitle('3 Additional Integrals of Motion')    
    plt.show()
    plt.close(fig)
    
    # Part f
    state0  = np.zeros(6)
    MRP0    = np.array([1/3, 1/3, 1/3])
    omega0  = np.array([-0.1, 0.05, 0.1])
    state0[0:3] = omega0
    state0[3:6] = MRP0
    
    tf      = 200
    t       = np.linspace(0, tf, 1000)
    sol = solve_ivp(sattelite_orientation_MRP , y0=state0, t_span=(0,tf), 
                    t_eval=t, args=(I1, I2, I3), atol=1e-10, rtol=1e-10)
    
    t           = sol.t
    omega_hist  = sol.y[0:3, :]
    MRP_hist    = sol.y[3:6, :]
    
    H_n         = np.zeros_like(omega_hist)
    omega_n     = np.zeros_like(omega_hist)
    for i, MRP in enumerate(MRP_hist.transpose()):
        H_b             = np.dot(I, omega_hist[:, i])
        BN              = MRPtoDCM(MRP)
        H_n[:, i]       = np.dot(BN.T, H_b)
        omega_n[:, i]   = np.dot(BN.T, omega_hist[:, i])

    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, MRP_hist[0, :])
    axs[0].set_title(r'$\sigma_1$')
    axs[1].plot(t, MRP_hist[1, :])    
    axs[1].set_title(r'$\sigma_2$')
    axs[2].plot(t, MRP_hist[2, :])
    axs[2].set_title(r'$\sigma_3$')

    fig.tight_layout(pad=2.5)
    for ax in axs.flat:
        ax.set(xlabel='Time [s]')
        ax.set_xlim([0, t[-1]])
        #ax.set_ylim([0, 0.0])
    fig.suptitle('CRP Time Evolution')    
    plt.show()
    plt.close(fig)
    
    fig, axs = plt.subplots(3, 1)
    axs[0].plot(t, H_n[0, :])
    axs[0].set_title(r'$H_1$')
    axs[1].plot(t, H_n[1, :])    
    axs[1].set_title(r'$H_2$')
    axs[2].plot(t, H_n[2, :])
    axs[2].set_title(r'$H_3$')

    fig.tight_layout(pad=2.5)
    for ax in axs.flat:
        ax.set(xlabel='Time [s]')
        ax.set_xlim([0, t[-1]])
        #ax.set_ylim([0, 0.0])
    fig.suptitle('Angular Momentum N-frame')    
    plt.show()
    plt.close(fig)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(omega_n[0,:], omega_n[1,:], omega_n[2,:])
    ax.plot([0, H_n[0, 1]], [0, H_n[1, 1]], [0, H_n[2, 1]])

    ax.set_xlabel(r'$n_1$')
    ax.set_ylabel(r'$n_2$')
    ax.set_zlabel(r'$n_3$')
    ax.set_xlim([.08, .16])
    ax.set_ylim([-0.1, -0.02])
    ax.set_zlim([0, 0.08])
    plt.show()
    
    # Part g
    sol = solve_ivp(sattelite_orientation_MRP , y0=state0, t_span=(0,tf), 
                    t_eval=t, args=(I1o, I2o, I3o), atol=1e-10, rtol=1e-10)
    
    t                   = sol.t
    omega_hist_CPS3     = sol.y[0:3, :]
    MRP_hist_CPS3       = sol.y[3:6, :]
    omega_n_CPS3        = np.zeros_like(omega_hist)
    
    for i, MRP in enumerate(MRP_hist_CPS3.transpose()):
        H_b             = np.dot(I, omega_hist[:, i])
        BN              = MRPtoDCM(MRP)
        omega_n_CPS3[:, i]   = np.dot(BN.T, omega_hist_CPS3[:, i])
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    l1 = ax.plot(omega_n[0,:], omega_n[1,:], omega_n[2,:])
    l2 = ax.plot(omega_n_CPS3[0,:], omega_n_CPS3[1,:], omega_n_CPS3[2,:])

    ax.set_xlabel(r'$n_1$')
    ax.set_ylabel(r'$n_2$')
    ax.set_zlabel(r'$n_3$')
    ax.set_xlim([.08, .16])
    ax.set_ylim([-0.1, -0.02])
    ax.set_zlim([0, 0.08])
    fig.legend([l2, l1], labels=["Axisymmetric  Body", "Non-axisymmetric Body"],
           loc="upper right")
    
    plt.show()

    
    return


if __name__ == "__main__":
    main()