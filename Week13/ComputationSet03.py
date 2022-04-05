import numpy as np
from numpy import pi as pi
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt

import sys
#sys.path.append('C:/Users/brend/Documents/GitHub/AttitudeDynamics')
sys.path.append('/Users/brendangillis/Documents/AttitudeDynamics')
from AAE440_Funcs import *


def sattelite_orientation_CRP(t, CRP):
        """
        Return the derivative of CRP for 
        sattelite orientation.
        """
        omega1  = 0.1*np.cos(0.2*t) + 0.15*np.sin(0.2*t)
        omega2  = 0.15*np.cos(0.2*t) - 0.1*np.sin(0.2*t)
        omega3  = 0.3
        omega   = np.array([omega1, omega2, omega3])
        CRP_dot  = KDE_CRP(CRP, omega)
        return CRP_dot

def sattelite_orientation_MRP(t, MRP):
        """
        Return the derivative of MRP for 
        sattelite orientation.
        """
        omega1  = 0.1*np.cos(0.2*t) + 0.15*np.sin(0.2*t)
        omega2  = 0.15*np.cos(0.2*t) - 0.1*np.sin(0.2*t)
        omega3  = 0.3
        omega   = np.array([omega1, omega2, omega3])
        MRP_dot  = KDE_MRP(MRP, omega)
        return MRP_dot

def sattelite_orientation_omega(t, omega):
        """
        Return the derivative of omega for 
        sattelite orientation.
        """
        I1 = 1
        I2 = 1
        I3 = 3/2
        omega_dot  = dwdt_torqueFree(omega, [I1, I2, I3])
        return omega_dot
    
def main():
    
    np.set_printoptions(precision=3)
    
    # # Problem 1
    # # Part a.1
    # BN  = EAtoDCM(EA=[-pi/4, pi/8, pi/5], axis=[3,2,1])
    # CRP = DCMtoCRP(BN) 
    # print("CRP_BN: {}".format(CRP))
    
    # # Part a.3
    # t   = np.linspace(0, 9.5, 1000)
    # CRP0 = CRP
    # sol = solve_ivp(sattelite_orientation_CRP , y0=CRP0, t_span=(0,9.5), 
    #                 t_eval=t, atol=1e-10, rtol=1e-10)
    
    # t           = sol.t
    # CRP_hist   = sol.y
    # CRP_hist_mag = np.linalg.norm(CRP_hist, axis=0)
    
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(t, CRP_hist[0, :])
    # axs[0, 0].set_title(r'$\rho_1$')
    # axs[1, 0].plot(t, CRP_hist[1, :])    
    # axs[1, 0].set_title(r'$\rho_2$')
    # axs[0, 1].plot(t, CRP_hist[2, :])
    # axs[0, 1].set_title(r'$\rho_3$')
    # axs[1, 1].plot(t, CRP_hist_mag)
    # axs[1, 1].set_title(r'||$\rho$||')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_xlim([0, 9.5])
    # fig.suptitle('Sattelite CRP History')    
    # plt.show()
    # plt.close(fig) 
    
    # # Part a.4
    # prp_axis = np.zeros_like(CRP_hist)
    # prp_angle  = np.zeros_like(t)
    # for i, CRP in enumerate(CRP_hist.transpose()):
    #     [prp_angle[i], axis] = DCMtoPRP(CRPtoDCM(CRP))
    #     prp_axis[:, i] = np.squeeze(axis)
    
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(t, prp_axis[0, :])
    # axs[0, 0].set_title(r'$\lambda_1$')
    # axs[1, 0].plot(t, prp_axis[1, :])    
    # axs[1, 0].set_title(r'$\lambda_2$')
    # axs[0, 1].plot(t, prp_axis[2, :])
    # axs[0, 1].set_title(r'$\lambda_3$')
    # axs[1, 1].plot(t, prp_angle*180/pi)
    # axs[1, 1].set_title(r'$\theta$')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_xlim([0, 9.5])
    # fig.suptitle('Sattelite PRP History')    
    # plt.show()
    # plt.close(fig) 
    
    # # Part a.4
    # prp_axis = np.zeros_like(CRP_hist)
    # prp_angle  = np.zeros_like(t)
    # for i, CRP in enumerate(CRP_hist.transpose()):
    #     [prp_angle[i], axis] = DCMtoPRP(CRPtoDCM(CRP))
    #     prp_axis[:, i] = np.squeeze(axis)
    
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(t, prp_axis[0, :])
    # axs[0, 0].set_title(r'$\lambda_1$')
    # axs[1, 0].plot(t, prp_axis[1, :])    
    # axs[1, 0].set_title(r'$\lambda_2$')
    # axs[0, 1].plot(t, prp_axis[2, :])
    # axs[0, 1].set_title(r'$\lambda_3$')
    # axs[1, 1].plot(t, prp_angle)
    # axs[1, 1].set_title(r'$\theta$')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_xlim([0, 9.5])
    # fig.suptitle('Sattelite EP History from CRP')    
    # plt.show()
    # plt.close(fig) 
    
    # # Part a.5
    # EP = np.zeros((4, np.prod(t.shape)))
    # for i, CRP in enumerate(CRP_hist.transpose()):
    #     EP[:, i] = DCMtoEP(CRPtoDCM(CRP))
        
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(t, EP[0, :])
    # axs[0, 0].set_title(r'$\epsilon_1$')
    # axs[1, 0].plot(t, EP[1, :])    
    # axs[1, 0].set_title(r'$\epsilon_2$')
    # axs[0, 1].plot(t, EP[2, :])
    # axs[0, 1].set_title(r'$\epsilon_3$')
    # axs[1, 1].plot(t, EP[3, :])
    # axs[1, 1].set_title(r'$\epsilon_4$')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_xlim([0, 9.5])
    # fig.suptitle('Sattelite EP History from CRP')    
    # plt.show()
    # plt.close(fig) 
    
    # # Part b.1
    # BN  = EAtoDCM(EA=[-pi/4, pi/8, pi/5], axis=[3,2,1])
    # MRP = DCMtoMRP(BN) 
    # print("MRP_BN: {}".format(MRP))
    
    # # Part a.3
    # t   = np.linspace(0, 100, 10000)
    # MRP0 = MRP
    # sol = solve_ivp(sattelite_orientation_MRP , y0=MRP0, t_span=(0,100), 
    #                 t_eval=t, atol=1e-10, rtol=1e-10)
    
    # t           = sol.t
    # MRP_hist   = sol.y
    # MRP_hist_mag = np.linalg.norm(MRP_hist, axis=0)
    
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].plot(t, MRP_hist[0, :])
    # axs[0, 0].set_title(r'$\sigma_1$')
    # axs[1, 0].plot(t, MRP_hist[1, :])    
    # axs[1, 0].set_title(r'$\sigma_2$')
    # axs[0, 1].plot(t, MRP_hist[2, :])
    # axs[0, 1].set_title(r'$\sigma_3$')
    # axs[1, 1].plot(t, MRP_hist_mag)
    # axs[1, 1].set_title(r'||$\sigma$||')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_xlim([0, 100])
    # fig.suptitle('Sattelite MRP History')    
    # plt.show()
    # plt.close(fig) 
    
    # # Part b.4
    # prp_axis = np.zeros_like(MRP_hist)
    # prp_angle  = np.zeros_like(t)
    # for i, MRP in enumerate(MRP_hist.transpose()):
    #     [prp_angle[i], axis] = DCMtoPRP(MRPtoDCM(MRP))
    #     prp_axis[:, i] = np.squeeze(axis)
    
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].scatter(t, prp_axis[0, :], s=1, marker = ",")
    # axs[0, 0].set_title(r'$\lambda_1$')
    # axs[1, 0].scatter(t, prp_axis[1, :], s=1, marker = ",")    
    # axs[1, 0].set_title(r'$\lambda_2$')
    # axs[0, 1].scatter(t, prp_axis[2, :], s=1, marker = ",")
    # axs[0, 1].set_title(r'$\lambda_3$')
    # axs[1, 1].scatter(t, prp_angle, s=1, marker = ",")
    # axs[1, 1].set_title(r'$\theta$')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_xlim([0, 100])
    # fig.suptitle('Sattelite PRP History from MRP')    
    # plt.show()
    # plt.close(fig) 
    
    # # Part b.5
    # EP = np.zeros((4, np.prod(t.shape)))
    # for i, MRP in enumerate(MRP_hist.transpose()):
    #     EP[:, i] = DCMtoEP(MRPtoDCM(MRP))
        
    # fig, axs = plt.subplots(2, 2)
    # axs[0, 0].scatter(t, EP[0, :], s=1, marker = ",")
    # axs[0, 0].set_title(r'$\epsilon_1$')
    # axs[1, 0].scatter(t, EP[1, :], s=1, marker = ",")    
    # axs[1, 0].set_title(r'$\epsilon_2$')
    # axs[0, 1].scatter(t, EP[2, :], s=1, marker = ",")
    # axs[0, 1].set_title(r'$\epsilon_3$')
    # axs[1, 1].scatter(t, EP[3, :], s=1, marker = ",")
    # axs[1, 1].set_title(r'$\epsilon_4$')
    
    # fig.tight_layout(pad=2.5)
    # for ax in axs.flat:
    #     ax.set(xlabel='Time [s]')
    #     ax.set_ylim([-1, 1])
    #     ax.set_xlim([0, 100])
    # fig.suptitle('Sattelite EP History from MRP')    
    # plt.show()
    # plt.close(fig) 
    

    # Problem 2
    # Part a
    I       = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 3/2]])

    EP0     = np.array([np.sqrt(3)/2, 0, 0, 1/2])
    omega0  = np.array([-0.1, 0.005, 0.1])
    tf  = 100
    t   = np.linspace(0, tf, 1000)
    sol = solve_ivp(sattelite_orientation_omega , y0=omega0, t_span=(0,tf), 
                    t_eval=t, atol=1e-10, rtol=1e-10)
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
        ax.set_ylim([-.15, .15])
    fig.suptitle('Sattelite Angular Velocity History')    
    plt.show()
    plt.close(fig) 

    return


if __name__ == "__main__":
    main()