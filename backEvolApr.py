"""                                                                                            
/********************************************************/                                         
/* plotEvol.py                                          */                                         
/* Author: Luca Iliesiu                                 */                                          
/* Adviser: David Marsh                                 */                                          
/* Last date modified: 08/01/2013                       */                                          
/* Compiler: Python                                     */                                          
/* Dependencies: os, copy, matplotlib, scipy, sys       */                                          
/*               math                                   */                                          
/*                                                      */                                          
/* Description: Implements simulation for generalized   */                                          
/* reheating model. We particularize it for the         */                                          
/* curvaton model. Approximate the Klein-Gordon eq.     */
/* for the scalar fluids during oscillation as a matter */
/* fluid.                                               */                                          
/********************************************************/                                          
"""


from numpy import *
seterr(all='ignore') 
import sys
sys.path.append("/u/liliesiu/pp")

import os
import copy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as p
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
import matplotlib.mlab as mlab


class Background:
    """Class representing fields, fields velocities and densities."""

    """ phi[i]: value of field i
        phid[i]: value of speed of field i
        dens[j]: value of density of species j
        w[j]: equation of state for species j """
    
    def __init__(self, phi, phid, dens, w):
        self._phi, self._phid, self._dens, self._w = phi, phid, dens, w
    def __repr__(self):
        return 'phi:{phi} phid:{phid} dens:{dens} w:{w}'.format(
            phi=self._phi, phid=self._phid, dens=self._dens, w=self._w)


class Perturbation:
    """ Class representing perturbations in fields, 
        field velocities, densities and density speed gradients."""

    """ dphi[i]: perturbation of field i
        dphid[i]: velocity of perturbation of field i
        ddens[j]: density perturbation of species j
        sdens[j]: gradient of speed od density perturbation of species j
        sigma[i, j]: two-point correlation function between 
                     perturbations in species i and j
        pert[i]: array of all species that numerically varifies the 
                 accuracy of our calculations
        np: Newtonian potential
        k: mode of perturbation in Fourier space """
    
    def __init__(self, dphi, dphid, ddens, sdens, np, k):
        self._dphi, self._dphid, self._ddens, self._sdens, self._np, self._k = dphi, dphid, ddens, sdens, np, k
    def __repr__(self):
        return 'dphi:{dphi} dphid:{dphid} ddens:{ddens} sdens:{sdens} np:{np} k:{k}'.format(
            dphi=self._dphi, dphid=self._dphid, ddens=self._ddens, 
            sdens=self._sdens, np = self._np, k = self._k)


class State:
    """ Class representing backround parameters and perturbations."""

    """ b: background class
        p[i]: perturbation class corresponding to mode p[i]._k """
    
    def __init__(self, b, p): 
        self._b, self._p = b, p
    def __repr__(self):
        return 'b:{b} p:{p}'.format(b = self._b, p = self._p)



"""Calculate potential."""
def V(st):
    pot = 0
    for i in arange(len(st._phi)):
        pot += (mass[i] ** 2) * (st._phi[i] ** 2)/2
    return pot   


"""Calculate the derivative of the potential."""
def derivV(st):
    global mass
    return [(mass[i] ** 2) * st._phi[i] for i in range(len(st._phi))]


"""Calculate the density of the scalar fields."""
def fielddens(st):
    fieldd = 0
    for i in arange(len(st._phi)):
        fieldd += (st._phid[i] ** 2)/2.
    fieldd += V(st)
    return fieldd



"""Calculate the Hubble parameter."""
def Hubble(st):
    global H, MPL
    H = sqrt((1/(3.0 * MPL ** 2)) * 
             (sum([st._dens[i] for i in range(1, len(st._dens))]) + fielddens(st)))
             


"""Evolve the background."""    
def fb(st):
    global H, coupl, gamma, startDecay
    """ Background dynamics equations."""
    if (startDecay == 0):
        dSpeed =   [(-3 * H * (1 + st._w[i]) * st._dens[i]) + (gamma * coupl[i] * st._dens[0])
                           for i in range(len(st._dens))]
        dSpeed[0] = 0.0
        return Background(st._phid,
                          [-(3 * H + gamma) * st._phid[i] - derivV(st)[i] for i in range(len(st._phi))],
                          dSpeed, st._w)
    else:
        return Background([0.0 for i in range(len(st._phid))],
                          [0.0 for i in range(len(st._phi))],
                          [(-3 * H * (1 + st._w[i]) * st._dens[i]) + (gamma * coupl[i] * st._dens[0])
                           for i in range(len(st._dens))], st._w)


"""Evolve the perturbation."""
def fp(st, stb):
    global startDecay, H, coupl, gamma, mass, a, MPL
    k = st._k
    """ Auxilliarry variables."""
    sumphid = sum([stb._phid[i]**2 for i in range(len(stb._phid))])
    phidphi = sum([stb._phid[i] * st._dphi[i] for i in range(len(stb._phi))])
    if (startDecay == 0):
        sumphid = sum([stb._phid[i]**2 for i in range(len(stb._phid))])
        phidphi = sum([stb._phid[i] * st._dphi[i] for i in range(len(stb._phi))])

        fdens = fielddens(stb)
        dfdens = sum([(stb._phid[i] * st._dphid[i]) for i in range(len(st._dphi))]) + sum([derivV(stb)[i] * st._dphi[i] for i in range(len(st._dphi))]) - (sumphid * st._np)
        sigmafield = - (k**2) * phidphi/sumphid
        st._ddens[0] = dfdens/fdens
        st._sdens[0] = sigmafield

    else:
        phiSpeed = [0. for i in range(len(st._dphi))]
        phiAcc = [0. for i in range(len(st._dphi))]

    
    fdens = stb._dens[0]
    dfdens = st._ddens[0] * fdens
    
    sigmafield = st._sdens[0]
    """ Derivatives of each perturbation."""
    nps = -(st._np * (((k ** 2)/(3 * (a ** 2) * H)) +  H)) - ((1./(6. * (MPL ** 2) * H))
          * (sum([stb._dens[i] * st._ddens[i] for i in range(1, len(st._ddens))]) + dfdens))
    densSpeed = [-((1 + stb._w[i]) * st._sdens[i]/a) + (3 * (1 + stb._w[i]) * nps) + 
                  ((coupl[i] * gamma) * (fdens/stb._dens[i]) * ((dfdens/fdens) + st._np - st._ddens[i]))
                  for i in range(len(st._ddens))]
    ddensSpeed = [(((k ** 2)/a) * (st._np + (3 * stb._w[i] * st._ddens[i]/4.)))
                  - ((1 - (3 * stb._w[i])) * H * st._sdens[i]) 
                  + ((coupl[i] * gamma) * (fdens/stb._dens[i]) 
                  * ((sigmafield/(1+stb._w[i])) - st._sdens[i])) 
                  for i in range(len(st._sdens))]
    phiSpeed =  st._dphid
    phiAcc = [ -((3 * H + gamma) * st._dphid[i]) - (((mass[i] ** 2) + ((k/a)**2)) * st._dphi[i])
                - ((2 * (mass[i] ** 2) * stb._phi[i] * st._np) - ((4. * stb._phid[i]) * nps))
                for i in range(len(st._dphi))]

    """
    # Tests for numerical accuracy:
    if not (pertSpeed[0] == phiSpeed):
        print "Different speeds..."
    if not (pertSpeed[1] == phiAcc):
        print "Different acc.."
    if not (pertSpeed[2] == densSpeed[0]):
        print "Different dens speed1.."    
    if not (pertSpeed[3] == densSpeed[1]):
        print "Different dens speed2.."    
    if not (pertSpeed[4] == ddensSpeed[0]):
        print "Different dens speed3.."
    if not (pertSpeed[5] == ddensSpeed[1]):
        print "Different dens speed4.."
    if not (pertSpeed[6] == nps):
        print "Different nps" 
    print "Speed 1...", pertSpeed
    print "Speed 2...", phiSpeed, phiAcc, densSpeed, ddensSpeed, nps
    """
    """Perturbation dynamics ODE."""
#    print phiSpeed, phiAcc, densSpeed, ddensSpeed, nps, k
#    print gamma
    return Perturbation(phiSpeed, phiAcc, densSpeed, ddensSpeed, nps, k)




"""Evolve the function."""
def f(st):
    return State(fb(st._b), [fp(st._p[i], st._b) for i in range(len(st._p))])



"""Sum between two backgrounds."""
def l_sum(st1, st2):
    return Background([st1._phi[i] + st2._phi[i] for i in range(len(st1._phi))],
                      [st1._phid[i] + st2._phid[i] for i in range(len(st1._phid))],
                      [st1._dens[i] + st2._dens[i] for i in range(len(st1._dens))],
                      st1._w)



"""Product between background class and a scalar."""
def l_product(st1, r):
    return Background([st1._phi[i] * r for i in range(len(st1._phi))],
                      [st1._phid[i] * r for i in range(len(st1._phid))],
                      [st1._dens[i] * r for i in range(len(st1._dens))],
                      st1._w)



"""Sum between two perturbations."""
def p_sum(st1, st2):
    return Perturbation([st1._dphi[i] + st2._dphi[i] for i in range(len(st1._dphi))],
                        [st1._dphid[i] + st2._dphid[i] for i in range(len(st1._dphid))],
                        [st1._ddens[i] + st2._ddens[i] for i in range(len(st1._ddens))],
                        [st1._sdens[i] + st2._sdens[i] for i in range(len(st1._sdens))],
                        st1._np + st2._np, st1._k)



"""Product between the perturbation and a scalar."""
def p_product(st1, r):
    return Perturbation([st1._dphi[i] * r for i in range(len(st1._dphi))],
                        [st1._dphid[i] * r for i in range(len(st1._dphid))],
                        [st1._ddens[i] * r for i in range(len(st1._ddens))],
                        [st1._sdens[i] * r for i in range(len(st1._sdens))],
                        st1._np * r, st1._k)




"""Sum between two general classes."""
def s_sum(st1, st2):
    return State(l_sum(st1._b, st2._b), [p_sum(st1._p[i], st2._p[i]) for i in range(len(st1._p))])




"""Product between two general classes."""
def s_product(st1, r):
    return State(l_product(st1._b, r), [p_product(st1._p[i], r) for i in range(len(st1._p))])




"""Fourth order Runge Kutta."""
def rk4(st):
    global startDecay
    a1 = f(st)
    a2 = f(s_sum(st, s_product(a1, dt/2)))
    a3 = f(s_sum(st, s_product(a2, dt/2)))
    a4 = f(s_sum(st, s_product(a3, dt)))
    """    if (startDecay == 1):
        print "a1,", a1
        print "a4", a4"""
    return s_sum(st, s_product(s_sum(a1, s_sum(a4, s_sum(s_product(a2, 2.), s_product(a3, 2.)))), dt/6))




"""Calculate curvature perturbations."""
def curvature(st, stb):
    global H, gamma, coupl
    k = st._k
    zetaphi = -st._np + (st._ddens[0]/(3*(1+stb._w[0]) + (coupl[0] * gamma/H)))
    zetadens = [-st._np + (st._ddens[i]/(3*(1+stb._w[i]) - (coupl[i] * gamma * stb._dens[0]/(H * stb._dens[i])))) for i in range(1, len(stb._dens))]
    return (zetaphi, zetadens)



""" Calculate curvature correlation matrix for all the fields and species. """
def curvatureMatrix(curv):
    cMatrix = zeros(shape = (1 + len(curv[1]), 1 + len(curv[1])))
    cMatrix[0, 0] = curv[0] ** 2
    for i in range(len(curv[1])):
        cMatrix[0, 1 + i] = curv[0] * curv[1][i]
        cMatrix[1 + i, 0] = cMatrix[0, 1 + i]
    for i in range(len(curv[1])):
        for j in range(len(curv[1])):
            cMatrix[1 + i, 1 + j] = curv[1][i] * curv[1][j]
    return cMatrix



""" Calculate the power spectra of total curvature perturbation. """
def curvatureSum(curv, stb):
    total = sum([stb._dens[i] * (1 + stb._w[i])  for i in range(len(stb._dens))])
    cSumTotal = stb._dens[0] * (1 + stb._w[0]) * curv[0]/total
    cSumTotal += sum([stb._dens[i + 1] * (1 + stb._w[i + 1]) * curv[1][i] / total for i in range(len(curv[1]))])
    return cSumTotal


""" Calculate total curvature perturbation. """
def curvTotal(curvMatrix, stb):
    curv = 0.
    total = sum([stb._dens[i] * (1 + stb._w[i])  for i in range(len(stb._dens))]) ** 2
    for i in range(len(curvMatrix)):
        for j in range(len(curvMatrix[0])):
            curv += stb._dens[i] * (1 + stb._w[i]) * stb._dens[j] * (1 + stb._w[j]) * curvMatrix[i, j] / total
    return curv
                    

"""Evolve the system using the ODEs."""
def evolve():
    global H, a, state, dt, dtInit, dN, gamma, FINAL_DENS, GAMMA_FINAL, tau, epsilon, sigmaInit, lambd, dtInit, fielddens, startDecay, mass
    time = 0
    N = 0
    startDecay = 0
    endDecay = 0
    gamma = 0
    evolution = []
    steps = 0
    while ((state._b._dens[2]/100.00 < fielddens(state._b)) |(gamma < 50.0 * H)):
        if (startDecay == 0):
            dt = dtInit
            state._b._dens[0] = fielddens(state._b)
            for stp in state._p:
                fdens  = state._b._dens[0]
                sumphid = sum([state._b._phid[i]**2 for i in range(len(state._b._phid))])
                phidphi = sum([state._b._phid[i] * stp._dphi[i] for i in range(len(state._b._phi))])
                dfdens = sum([(state._b._phid[i] * stp._dphid[i]) for i in range(len(stp._dphi))]) + sum([derivV(state._b)[i] * stp._dphi[i] for i in range(len(stp._dphi))]) -(sumphid * stp._np)
                sigmafield = - (stp._k**2) * phidphi/sumphid
                stp._ddens[0] = dfdens/fdens
                stp._sdens[0] = sigmafield
            w = (V(state._b) - ((state._b._phid[0] ** 2)/2.0))/(V(state._b) + ((state._b._phid[0] ** 2)/2.0))
            if (state._b._phi[0] < 0):
                print "***************************************"
                print "Decay has started...."
                print state._b._phi
                gamma = GAMMA_FINAL
                print fielddens(state._b)
                print state._b._dens
                print H
                ratio = fielddens(state._b)/(state._b._dens[0] +  fielddens(state._b))
                r = 3.* ratio/(4 - ratio)
                print r
                lambd[0] =  (8./9.) * (r ** 2) * 0.1 * (1/sigmaInit) ** 2

                dt = dtInit * 5e-2

                gamma = GAMMA_FINAL
                if (w > -0.95):
                    print "**********************************##################*********************"
                    startDecay = 1
            
        else:
            dt = 0.0002/H
            state._b._phi[0] = sqrt(state._b._dens[0])/mass[0]
            state._b._phid[0] = sqrt(state._b._dens[0])
            for stp in state._p:
                fieldd = (state._b._dens[0] * stp._ddens[0]) + ((state._b._phid[0] ** 2) * stp._np)
                #print fieldd
                stp._dphi[0] = (fieldd/2.)/(mass[0] ** 2 * state._b._phi[0])
                stp._dphid[0] = (fieldd/2.)/(state._b._phid[0])
        if (steps % 1000 == 0):
            print "Number of steps..." + str(steps)
            print gamma
            print a
            print H
            print state._p[0]._k/(a * H)
            print state._b._dens[0]
            print "a...", state._b._phi, state._b._dens
            print "b...", state._p[0]._dphi, state._p[0]._dphid, state._p[0]._ddens, state._p[0]._sdens, state._p[0]._np
            print "b2...", state._p[1]._dphi, state._p[1]._dphid, state._p[1]._ddens, state._p[1]._sdens, state._p[1]._np
            curv = curvature(state._p[0], state._b)
            print "curvature", curv
            curvMatrix = curvatureMatrix(curv)
            sumphid = sum([(state._b._phid[i]**2/2 + V(state._b)) 
                           for i in range(len(state._b._phid))])
            print curvMatrix
            print "Total curvature...", curvatureSum(curv, state._b) ** 2
            curv = curvature(state._p[1], state._b)
            print "curvature", curv
            curvMatrix = curvatureMatrix(curv)
            print curvMatrix
            print "Total curvature...", curvatureSum(curv, state._b) ** 2
            if ((gamma > H) & (endDecay == 0)):
                ratio =fielddens(state._b)/(state._b._dens[0] + fielddens(state._b))
                r = 3.* ratio/(4 + ratio)
                lambd[1] =  (8./9.) * (r ** 2) * epsilon * (1/(sigmaInit))  ** 2
                print r
                print "Attention! lambda is",lambd
                endDecay = 1
            if (gamma > H):
                print "Attention! lambda is",lambd
        N += H * dt
        a = exp(N)
        time += dt
        tau += dt/a
        steps += 1
       # if (startDecay == 1):
       #     print state, "hhaaa"
        Hubble(state._b)
        state = rk4(state)
       # if (startDecay == 1):
       #     print state, "heheh"
        if (((steps % 6 == 0) & (N < 2.0)) | ((steps % 50 == 0) & (N > 2.0))):
            cur = [curvature(state._p[i], state._b) for i in range(len(state._p))]
            curvMatrix = [curvatureMatrix(cur[i]) for i in range(len(state._p))]
            curvatureTotal = [curvatureSum(cur[i], state._b) ** 2 for i  in range(len(state._p))] 
            evolution.append((N, time, state, 
                              [curvature(state._p[i], state._b) for i in range(len(state._p))], 
                              curvMatrix, curvatureTotal))
    return evolution


"""Set initial conditions for perturbations."""
def setInitPert(np, ks):
    global tau, MPL, epsilon
    p = []
    for i in range(len(ks)):
        
        """ The perturvations in the adiabatic universe. """
        p.append(Perturbation([0.000], [0.000], [0.00, -2 * np[i], -(3./2.) * np[i], -2 * np[i]], [0.00, (ks[i] ** 2) * tau * np[i]/2.,  (ks[i] ** 2) * tau * np[i]/2.,  (ks[i] ** 2) * tau * np[i]/2.], np[i], ks[i]))
        
        """ The perturvations in the adiabatic universe. """
        p.append(Perturbation([-3. * MPL * np[i] * sqrt(epsilon)], [0.000], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], 0.0, ks[i]))
    return p


""" Calculate parameter \lambda. """
def calculateLambda(evolution):
    last = evolution[-1][5][0] + evolution[-1][5][1]
    begin = evolution[0][5][0] + evolution[0][5][1]
    return last/begin - 1.


""" Find directory number to output data. """
def dirNumber():
    global dir_number, datapath, plotpath, statpath
    dir_number = 0 
    while(os.path.isdir('data/'+ str(dir_number))):
        dir_number += 1
    os.mkdir('data/' + str(dir_number))
    os.mkdir('stats/' + str(dir_number))
    os.mkdir('plot_data/'+str(dir_number))
    datapath = 'data/' + str(dir_number) + '/'
    statpath = 'stats/' + str(dir_number) + '/'
    plotpath = 'plot_data/' + str(dir_number) + '/'
    


""" Find file number to output data."""
def fileNumber():
    global file_number, datapath
    file_number = 0 
    while(os.path.isfile(datapath + 'evolution'+ str(file_number) +'.out')):
        file_number += 1
    return file_number



""" Output data with evolution of the curvaton. """
def writeData(evolution):
    global file_number, mass, gamma, sigmaInit, lambd, datapath, statpath, epsilon
    datafile = open(datapath + 'evolution' + str(file_number) + '.out', 'w')
    print "Printing data to evolution" + str(file_number) + ".out"
    writeOut = "[" + str(mass) + ", " + str(sigmaInit) + ", " + str(gamma) + ", " + str(epsilon) + "]"
    print >>datafile, writeOut
    writeOut = "["
    for ev in evolution:
        densities = [ev[2]._b._dens[i] for i in range(1, len(ev[2]._b._dens))]
        ws =  [ev[2]._b._w[i] for i in range(1, len(ev[2]._b._w))]
        writeOut += "["
        writeOut += str(ev[0]) + ", "
        writeOut += str(ev[1]) + ", "
        writeOut += "[[" + str(ev[2]._b._phi) + ", " + str(ev[2]._b._phid) +  ", " + str(densities) + ", " + str(ws) +  "],["
        for i in range(len(ev[2]._p)):
            ddensities = [ev[2]._p[i]._ddens[j] for j in range(1, len(ev[2]._p[i]._ddens))]
            sdensities =  [ev[2]._p[i]._sdens[j] for j in range(1, len(ev[2]._p[i]._sdens))]
            writeOut += "[" + str(ev[2]._p[i]._dphi) +  ", " + str(ev[2]._p[i]._dphid) + ", " + str(ddensities) +  ", " + str(sdensities) +  ", " + str(ev[2]._p[i]._np) + ", " + str(ev[2]._p[i]._k) +  "]"
            if (i != len(ev[2]._p) - 1): writeOut += ","
        writeOut += "]],"
        writeOut += str(ev[3]) + ", "
        curvMatrices = [el.tolist() for el in ev[4]]
        writeOut += str(curvMatrices) + ", "
        writeOut += str(ev[5]) + "]"
        if (ev != evolution[len(evolution) - 1]):
            writeOut += ", "
    writeOut += "]"
    print >>datafile, writeOut    
    datafile.close()

    print "Printing statistics to stats" + str(file_number) + ".out"
    print statpath
    datafile = open(statpath + 'stats' + str(file_number) + '.out', 'w')
    print >> datafile, "Number of fields: ", len(evolution[0][2]._b._phi)
    print >> datafile, "Number of species: ", len(evolution[0][2]._b._dens) - 1
    print >> datafile, "Mass of the fields: ", mass
    print >> datafile, "Initial value of the fields: ", sigmaInit
    print >> datafile, "Initial density of species: ", [evolution[0][2]._b._dens[i] for i in range(1, len(evolution[0][2]._b._dens))]
    print >> datafile, "Value of the slow-roll inflationary parameter: ", epsilon
    print >>datafile, "Bounds for approximated lambda: ", lambd[0], lambd[1]
    print >>datafile, "Actual value of lambda: ", calculateLambda(evolution)
    print >>datafile, "Value of Gamma: ", gamma
    datafile.close()
    


""" Print values for the testes masses, sigmas, gammas, epsilons, and coupl."""
def printInitVal():
    global datapath, masses, sigmas, gammas, epsilons, coupl
    datafile = open(datapath + 'initValues.out', 'w')
    
    print >>datafile, masses
    print >>datafile, sigmas
    print >>datafile, gammas
    print >>datafile, epsilons
    print >>datafile, coupl

    datafile.close()

    

""" Program main."""
if __name__ == "__main__":
    datapath = ''
    statpath = ''
    plotpath = ''
    dirNumber()
    MPL = 1.0
    masses = [1.00e-13]
    sigmas = [1e-6 * MPL]
    #masses = [1.25e-14, 2.5e-14, 5.0e-14, 1.0e-13, 2.0e-13, 4.0e-13, 8.0e-13, 1.6e-12, 3.2e-12]
    #sigmas = [1e-6 * MPL, 2e-6 * MPL, 4e-6, 8e-6, 1.6e-5, 3.2e-5, 6.4e-5, 1.28e-4] 
    #gammas = [5e-06, 1e-05, 2e-05, 4e-05, 8e-05, 0.00016, 0.00032, 0.00064, 0.00128, 0.00256, 0.00512]
    gammas = [1.25e-29, 2.5e-29, 5.0e-29, 1.0e-28, 2.0e-28, 4.0e-28, 8.0e-28, 1.6e-27, 3.2e-27, 6.4e-27, 1.28e-26, 2.56e-26, 5.12e-26, 1e-25, 2e-25, 4e-25, 8e-25, 1.6e-24, 3.2e-24, 6.4e-24, 1.28e-23, 2.56e-23, 5.12e-23, 1e-22] 
    #gammas = [1e-30, 1e-29, 1e-28, 1e-27, 1e-26, 1e-25, 1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4]
    epsilons = [0.1]
    Neff = 1.0
    ratioDRDM = (7./8.) * ((4./11.)**(4./3.)) * (3 + Neff)
    frac = 1.0e-15
    coupl = [-1.0, -(frac * 2) + (1/(1 + ratioDRDM)) , 2. * frac, (ratioDRDM/(1 + ratioDRDM))]
    coupl = [0, 0.499999999999999, 2e-15, 0.499999999999999]
    printInitVal()

    evolutions = []
    lam = []
    for m in masses:
        for sigmaInit in sigmas:
            for gammafinal in gammas:
                for epsilon in epsilons:  
                    dtInit = (1.e+10)/MPL
                    dt = dtInit
                    mass = [m] 
                    initNumb = 2
                    lambd = [0.00, 0.00]
                    initBack = Background([sigmaInit], [(1e-50) * MPL ** 2], [0.0, 100.00 * (m ** 2) * (MPL ** 2), (1.0e-58) * MPL**4,  (21./8.) * ((4./11.) ** (4./3.)) * 100.00 * (m ** 2) * (MPL ** 2)], [0.0, 1./3., 0, 1./4.])
                    nps = [-1.00]
                    H = 0.
                    a = 1.00
                    Hubble(initBack)
                    GAMMA_FINAL = gammafinal * H 
                    # Theoretically predicted \Gamma.
                    # GAMMA_FINAL = 100 * (m ** 3)/(MPL ** 2)
                    print GAMMA_FINAL
                    gamma = GAMMA_FINAL
                    tau = -1/(a * H)
                    ks = [-1e-20 * (1/tau)]
                    print "k/a...",ks[0]/a
                    print "a...", a
                    initPert = setInitPert(nps, ks)
                    state = State(initBack, initPert)
                    FINAL_DENS = fielddens(initBack) * (1e-4)
                    """ Initialize the transformation matrix u                 
                        Evolve ODE system. """
                    ev = evolve()   
                
                    print "Saving data..."
                    file_number = 0 
                    file_number = fileNumber()
                    writeData(ev)
                
                    lam.append(calculateLambda(ev))
                    evolutions.append(ev)
    




