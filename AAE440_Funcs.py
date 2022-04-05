
import numpy as np
from numpy import linalg as LA
from numpy import pi as pi
from numpy import sqrt as sqrt
from numpy import matmul as matmul

def Rx(theta):
    """
    Returns a rotation matrix about the x-axis
    """
    return np.array([[1, 0, 0],
                     [0, np.cos(theta), np.sin(theta)],
                     [0, -np.sin(theta), np.cos(theta)]])

def Ry(theta):
    """
    Returns a rotation matrix about the y-axis
    """
    return np.array([[np.cos(theta), 0, -np.sin(theta)],
                     [0, 1, 0],
                     [np.sin(theta), 0, np.cos(theta)]])

def Rz(theta):
    """
    Returns a rotation matrix about the z-axis
    """
    return np.array([[np.cos(theta), np.sin(theta), 0],
                     [-np.sin(theta), np.cos(theta), 0],
                     [0, 0, 1]])
    
def CrossMatrix(v):
    """
    Returns the cross product matrix of a vector
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

def EAtoDCM(EA, axis):
    """
    Converts an Euler angle to a direction cosine matrix
    using the 3 provided angles and 3 axis order:
    Ex. EA=[pi, pi/2, pi/4] about axis=[3-1-3]
    """
    DCM = np.eye(3)
    for i, angle in enumerate(EA):
        if axis[i] == 1:
            DCM = matmul(Rx(angle), DCM)
        if axis[i] == 2:
            DCM = matmul(Ry(angle), DCM)
        if axis[i] == 3:
            DCM = matmul(Rz(angle), DCM)
    return DCM

def DCMtoEA313(DCM):
    """
    Converts a direction cosine matrix to an Euler angle
    using the 3-1-3 convention
    """
    EA      = np.zeros(3)
    EA[0]   = np.arctan2(DCM[2,0], -DCM[2,1])
    EA[1]   = np.arccos(DCM[2,2])
    EA[2]   = np.arctan2(DCM[0,2], DCM[1,2])
    return EA

def DCMtoEA323(DCM):
    """
    Converts a direction cosine matrix to an Euler angle
    using the 3-2-3 convention
    """
    EA      = np.zeros(3)
    EA[0]   = np.arctan2(DCM[2,1], DCM[2,0])
    EA[1]   = np.arccos(DCM[2,2])
    EA[2]   = np.arctan2(DCM[1,2], -DCM[0,2])
    return EA

def DCMtoPRP(DCM):
    """
    Converts a direction cosine matrix to a principal rotation
    parameters theta and axis
    """
    theta = np.arccos((DCM[0,0]+DCM[1,1]+DCM[2,2]-1)/2)
    axis  = 1/(2*np.sin(theta))*np.array([[DCM[1,2]-DCM[2,1]], 
                                          [DCM[2,0]-DCM[0,2]], 
                                          [DCM[0,1]-DCM[1,0]]])
    return [theta, axis]

def DCMtoEP(DCM):
    """
    Converts a direction cosine matrix to an Euler parameter
    using Sheppard's method
    """
    # Compute largest parameter
    EP      = np.zeros(4)
    EP[0]   = sqrt(0.25*(1+2*DCM[0,0] - np.trace(DCM)))
    EP[1]   = sqrt(0.25*(1+2*DCM[1,1] - np.trace(DCM)))
    EP[2]   = sqrt(0.25*(1+2*DCM[2,2] - np.trace(DCM)))
    EP[3]   = sqrt(0.25*(1 + np.trace(DCM)))
    largest = np.argmax(EP)
    
    # Use largest parameter to computer other 3 parameters
    if largest == 0:
        EP[1] = (DCM[0,1] + DCM[1,0]) / (4*EP[largest])
        EP[2] = (DCM[2,0] + DCM[0,2]) / (4*EP[largest])
        EP[3] = (DCM[1,2] - DCM[2,1]) / (4*EP[largest])
        
    elif largest == 1:
        EP[0] = (DCM[0,1] + DCM[1,0]) / (4*EP[largest])
        EP[2] = (DCM[1,2] + DCM[2,1]) / (4*EP[largest])
        EP[3] = (DCM[2,0] - DCM[0,2]) / (4*EP[largest])
    
    elif largest == 2:
        EP[0] = (DCM[2,0] + DCM[0,2]) / (4*EP[largest])
        EP[1] = (DCM[1,2] + DCM[2,1]) / (4*EP[largest])
        EP[3] = (DCM[0,1] - DCM[1,0]) / (4*EP[largest])
    
    elif largest == 3:
        EP[0] = (DCM[1,2] - DCM[2,1]) / (4*EP[largest])
        EP[1] = (DCM[2,0] - DCM[0,2]) / (4*EP[largest])
        EP[2] = (DCM[0,1] - DCM[1,0]) / (4*EP[largest])
    
    EP = -EP if EP[3] < 0 else EP
    return EP

def DCMtoCRP(DCM):
    """
    Converts a direction cosine matrix to the classical
    Rodrigues parameter
    """
    CRP = 1/(np.trace(DCM) + 1) * np.array([[DCM[1,2]-DCM[2,1]],
                                            [DCM[2,0]-DCM[0,2]],
                                            [DCM[0,1]-DCM[1,0]]])
    CRP = np.squeeze(CRP)
    return CRP

def DCMtoMRP(DCM):
    """"
    Converts a directred cosine matrix to modified Rodrigues
    parameters
    """
    d = sqrt(np.trace(DCM) + 1)
    MRP = 1/(d*(d+2)) * np.array([[DCM[1,2]-DCM[2,1]],
                                  [DCM[2,0]-DCM[0,2]],
                                  [DCM[0,1]-DCM[1,0]]])
    MRP = np.squeeze(MRP)
    return MRP

def PRPtoDCM(theta, axis):
    """
    Converts a principal rotation parameter to a direction cosine
    matrix
    """
    DCM = np.zeros([3, 3])
    DCM[0,0] = axis[0]**2 * (1-np.cos(theta)) + np.cos(theta)
    DCM[0,1] = axis[0]*axis[1] * (1-np.cos(theta)) + axis[2]*np.sin(theta)
    DCM[0,2] = axis[0]*axis[2] * (1-np.cos(theta)) - axis[1]*np.sin(theta)
    DCM[1,0] = axis[0]*axis[1] * (1-np.cos(theta)) - axis[2]*np.sin(theta)
    DCM[1,1] = axis[1]**2 * (1-np.cos(theta)) + np.cos(theta)
    DCM[1,2] = axis[1]*axis[2] * (1-np.cos(theta)) + axis[0]*np.sin(theta)
    DCM[2,0] = axis[0]*axis[2] * (1-np.cos(theta)) + axis[1]*np.sin(theta)
    DCM[2,1] = axis[1]*axis[2] * (1-np.cos(theta)) - axis[0]*np.sin(theta)
    DCM[2,2] = axis[2]**2 * (1-np.cos(theta)) + np.cos(theta)
    return DCM

def EPtoDCM(EP):
    """
    Converts Euler parameters to directed cosine matrix
    """
    DCM = np.zeros([3, 3])
    DCM[0,0] = 1 - 2*EP[1]**2 - 2*EP[2]**2
    DCM[0,1] = 2*EP[0]*EP[1] + 2*EP[2]*EP[3]
    DCM[0,2] = 2*EP[0]*EP[2] - 2*EP[1]*EP[3]
    DCM[1,0] = 2*EP[0]*EP[1] - 2*EP[2]*EP[3]
    DCM[1,1] = 1 - 2*EP[0]**2 - 2*EP[2]**2
    DCM[1,2] = 2*EP[1]*EP[2] + 2*EP[0]*EP[3]
    DCM[2,0] = 2*EP[0]*EP[2] + 2*EP[1]*EP[3]
    DCM[2,1] = 2*EP[1]*EP[2] - 2*EP[0]*EP[3]
    DCM[2,2] = 1 - 2*EP[0]**2 - 2*EP[1]**2
    return DCM

def CRPtoDCM(CRP):
    """
    Converts classical Rodrigues parameters to a 
    direction cosine matrix
    """
    DCM = np.zeros([3, 3])
    DCM = (1-np.dot(CRP, CRP)) * np.eye(3) + 2*np.outer(CRP, CRP) - 2*CrossMatrix(CRP)
    DCM = DCM/(1+np.dot(CRP, CRP))
    return DCM

def MRPtoDCM(MRP):
    """
    Converts modified Rodrigues parameters to a 
    direction cosine matrix
    """
    DCM = (8*matmul(CrossMatrix(MRP),CrossMatrix(MRP))
                - 4*(1-np.dot(MRP,MRP)) * CrossMatrix(MRP))
    DCM = DCM / (1 + np.dot(MRP, MRP))**2 + np.eye(3)
    return DCM

def addEP(EP1, EP2):
    """
    Adds two Euler parameters
    """
    EP_mat = np.array([[EP2[3], EP2[2], -EP2[1], EP2[0]],
                       [-EP2[2], EP2[3], EP2[0], EP2[1]],
                       [EP2[1], -EP2[0], EP2[3], EP2[2]],
                       [-EP2[0], -EP2[1], -EP2[2], EP2[3]]])
    EP = np.dot(EP_mat, EP1)
    return EP

def KDE_EP(EP, omega):
    """
    Compute and return the derivative of the 
    Euler Parameter given the euler parameter
    and angular velocity 
    """
    EP_mat  = np.array([[EP[3], -EP[2], EP[1], EP[0]],
                       [EP[2], EP[3], -EP[0], EP[1]],
                       [-EP[1], EP[0], EP[3], EP[2]],
                       [-EP[0], -EP[1], -EP[2], EP[3]]])
    
    omega4  = [omega[0], omega[1], omega[2], 0]
    EP_dot  = 0.5*np.dot(EP_mat, omega4)
    return EP_dot

def KDE_EA313(EA, omega):
    """
    Compute and return the derivative of the 3-1-3 
    Euler Angle given the euler angle and angular
    velocity 
    """
    EA_mat          = np.zeros((3,3))
    EA_mat[0, 0]    = np.sin(EA[2])
    EA_mat[0, 1]    = np.cos(EA[2])
    EA_mat[0, 2]    = 0
    EA_mat[1, 0]    = np.sin(EA[1])*np.cos(EA[2])
    EA_mat[1, 1]    = -np.sin(EA[1])*np.sin(EA[2])
    EA_mat[1, 2]    = 0
    EA_mat[2, 0]    = -np.sin(EA[2])*np.cos(EA[1])
    EA_mat[2, 1]    = -np.cos(EA[1])*np.cos(EA[2])
    EA_mat[2, 2]    = np.sin(EA[1])

    EA_dot = 1/np.sin(EA[1]) * np.dot(EA_mat, omega)
    return EA_dot

def KDE_CRP(CRP, omega):
    """
    Compute the CRP derivative given CRP
    and angular velocity
    """
    CRP_mat          = np.zeros((3,3))
    CRP_mat[0, 0]    = 1+CRP[0]**2
    CRP_mat[0, 1]    = CRP[0]*CRP[1]-CRP[2]
    CRP_mat[0, 2]    = CRP[0]*CRP[2]+CRP[1]
    CRP_mat[1, 0]    = CRP[1]*CRP[0]+CRP[2]
    CRP_mat[1, 1]    = 1+CRP[1]**2
    CRP_mat[1, 2]    = CRP[1]*CRP[2]-CRP[0]
    CRP_mat[2, 0]    = CRP[2]*CRP[0]-CRP[1]
    CRP_mat[2, 1]    = CRP[2]*CRP[1]+CRP[0]
    CRP_mat[2, 2]    = 1+CRP[2]**2
    
    CRP_dot = 1/2*np.dot(CRP_mat, omega)
    return CRP_dot

def KDE_MRP(MRP, omega):
    """
    Compute the MRP derivative given CRP
    and angular velocity
    """
    mrpL2 = np.linalg.norm(MRP)
    MRP_mat          = np.zeros((3,3))
    MRP_mat[0, 0]    = 1-mrpL2**2+2*MRP[0]**2
    MRP_mat[0, 1]    = 2*(MRP[0]*MRP[1]-MRP[2])
    MRP_mat[0, 2]    = 2*(MRP[0]*MRP[2]+MRP[1])
    MRP_mat[1, 0]    = 2*(MRP[1]*MRP[0]+MRP[2])
    MRP_mat[1, 1]    = 1-mrpL2**2+2*MRP[1]**2
    MRP_mat[1, 2]    = 2*(MRP[1]*MRP[2]-MRP[0])
    MRP_mat[2, 0]    = 2*(MRP[2]*MRP[0]-MRP[1])
    MRP_mat[2, 1]    = 2*(MRP[2]*MRP[1]+MRP[0])
    MRP_mat[2, 2]    = 1-mrpL2**2+2*MRP[2]**2
    
    MRP_dot = 1/4*np.dot(MRP_mat, omega)
    return MRP_dot


