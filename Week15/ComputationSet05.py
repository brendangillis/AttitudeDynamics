import numpy as np
from numpy import pi as pi
from scipy.integrate import solve_ivp
import matplotlib.pylab as plt

import sys
sys.path.append('C:/Users/brend/Documents/GitHub/AttitudeDynamics')
#sys.path.append('/Users/brendangillis/Documents/AttitudeDynamics')
from AAE440_Funcs import *

  
    
def main():
    
    np.set_printoptions(precision=3)

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
    
    print("\n\nFg: {}".format(Fg))
    print("Lg: {}".format(Lg))
    
    Fg_1st = -mu/Rc**3*(m*Rc_B)

    print("Fg2-Fg1: {}".format(Fg-Fg_1st))
    
    Rcg_norm    = (mu*m/np.linalg.norm(Fg))**0.5
    Rcg         = -Fg/np.linalg.norm(Fg)*Rcg_norm
    print("||Rcg-Rc||: {}".format(np.linalg.norm(Rcg-Rc_B)))
    
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
    
    print("\n\nFg: {}".format(Fg))
    print("Lg: {}".format(Lg))
    
    Fg_1st = -mu/Rc**3*(m*Rc_B)

    print("Fg2-Fg1: {}".format(Fg-Fg_1st))
    
    Rcg_norm    = (mu*m/np.linalg.norm(Fg))**0.5
    Rcg         = -Fg/np.linalg.norm(Fg)*Rcg_norm
    print("||Rcg-Rc||: {}".format(np.linalg.norm(Rcg-Rc_B)))
    
    # Part e
    Rc      = 358e5  # [m] 200km altitude
    Rc_O    = np.array([Rc, 0, 0]) 
    
    MRP_BO  = np.array([1/3, 1/4, 1/5])
    DCM_BO  = MRPtoDCM(MRP_BO)
    Rc_B    = np.dot(DCM_BO, Rc_O)    
    
    Fg      = -mu/Rc**3*(m*Rc_B+3/Rc**2*np.dot(I, Rc_B)+3/(2*Rc**2)*np.trace(I)*Rc_B 
                         - 15/(2*Rc**4)*np.dot(Rc_B, np.dot(I, Rc_B))*Rc_B)
    Lg      = 3*mu/(Rc**5)*np.cross(Rc_B, np.dot(I, Rc_B))
    
    print("\n\nFg: {}".format(Fg))
    print("Lg: {}".format(Lg))
    
    Fg_1st = -mu/Rc**3*(m*Rc_B)

    print("Fg2-Fg1: {}".format(Fg-Fg_1st))
    
    Rcg_norm    = (mu*m/np.linalg.norm(Fg))**0.5
    Rcg         = -Fg/np.linalg.norm(Fg)*Rcg_norm
    print("||Rcg-Rc||: {}".format(np.linalg.norm(Rcg-Rc_B)))
    
    return


if __name__ == "__main__":
    main()