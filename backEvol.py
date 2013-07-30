"""
/********************************************************/
/* plot_data.py                                         */
/* Author: Luca Iliesiu                                 */
/* Last date modified: 06/05/2013                       */
/* Compiler: Python                                     */
/* Dependencies: os, copy, matplotlib, scipy, sys       */
/1;3201;0c*                                                      */
/* Description: Implements plotting methods for the     */
/* evolution of a single inflationary model.            */
/********************************************************/
"""

from numpy import *
seterr(all='ignore') # Sometimes it has weird value potentials
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
    H = sqrt((1/(3.0 * MPL ** 2)) * (sum(st._dens) + fielddens(st)))



"""Evolve the background."""    
def fb(st):
    global H, coupl, gamma

    """ Background dynamics equations."""
    return Background(st._phid, 
                 [-(3 * H + gamma) * st._phid[i] - derivV(st)[i] for i in range(len(st._phi))],
                 [(-3 * H * (1 + st._w[i]) * st._dens[i]) + (gamma * coupl[i] * fielddens(st)) 
                  for i in range(len(st._dens))], st._w)


"""Evolve the perturbation."""
def fp(st, stb):
    global H, coupl, gamma, mass, a, MPL
    k = st._k
    """ Auxilliarry variables."""
    sumphid = sum([stb._phid[i]**2 for i in range(len(stb._phid))])
    phidphi = sum([stb._phid[i] * st._dphi[i] for i in range(len(stb._phi))])
    
    fdens = fielddens(stb)
    dfdens = sum([(stb._phid[i] * st._dphid[i])
             for i in range(len(st._dphi))]) + sum([derivV(stb)[i] * st._dphi[i]
             for i in range(len(st._dphi))])- (sumphid * st._np)
    sigmafield = - (k**2) * phidphi/sumphid
    """ Derivatives of each perturbation."""
    nps = -(st._np * (((k ** 2)/(3 * (a ** 2) * H)) +  H)) - ((1./(6. * (MPL ** 2) * H))
          * (sum([stb._dens[i] * st._ddens[i] for i in range(len(st._ddens))]) + dfdens))
    
    phiSpeed = st._dphid
    phiAcc = [ -((3 * H + gamma) * st._dphid[i]) - (((mass[i] ** 2) + ((k/a)**2)) * st._dphi[i])
                - ((2 * (mass[i] ** 2) * stb._phi[i] * st._np) - ((4. * stb._phid[i]) * nps))
               for i in range(len(st._dphi))]
    
    densSpeed = [-((1 + stb._w[i]) * st._sdens[i]/a) + (3 * (1 + stb._w[i]) * nps) + 
                  ((coupl[i] * gamma) * (fdens/stb._dens[i]) * ((dfdens/fdens) + st._np - st._ddens[i])) 
                  for i in range(len(st._ddens))]
    ddensSpeed = [(((k ** 2)/a) * (st._np + (3 * stb._w[i] * st._ddens[i]/4.)))
                  - ((1 - (3 * stb._w[i])) * H * st._sdens[i]) 
                  + ((coupl[i] * gamma) * (fdens/stb._dens[i]) 
                  * ((sigmafield/(1+stb._w[i])) - st._sdens[i])) 
                  for i in range(len(st._sdens))]
        
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
    a1 = f(st)
    a2 = f(s_sum(st, s_product(a1, dt/2)))
    a3 = f(s_sum(st, s_product(a2, dt/2)))
    a4 = f(s_sum(st, s_product(a3, dt)))
    return s_sum(st, s_product(s_sum(a1, s_sum(a4, s_sum(s_product(a2, 2.), s_product(a3, 2.)))), dt/6))



"""Calculate curvature perturbations."""
def curvature(st, stb):
    global H, gamma, coupl
    k = st._k
    sumphid = sum([(stb._phid[i]**2/2 + V(stb)) 
                   for i in range(len(stb._phid))])
    phidphi =  sum([(derivV(stb)[i] * st._dphi[i])  
                    for i in range(len(st._dphi))])
    phidphi = sum([(stb._phid[i] * st._dphid[i]) + (derivV(stb)[i] * st._dphi[i])  
                   for i in range(len(st._dphi))]) - (sumphid * st._np)
    #phidphi =  sum([(derivV(stb)[i] * st._dphi[i])  
    #               for i in range(len(st._dphi))])
    zetaphi = - st._np + ((phidphi)/(3 * (1 + (gamma/H)) * sumphid))
    zetadens = [-st._np + (st._ddens[i]/(3*(1+stb._w[i]) - (coupl[i] * gamma * fielddens(stb)/(H * stb._dens[i]))))
                 for i in range(len(stb._dens))]
    return (zetaphi, zetadens)


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


def curvatureSum(curv, stb):
    sumphid = sum([(stb._phid[i]**2/2 + V(stb)) for i in range(len(stb._phid))])
    total = (sum([(stb._phid[i]**2/2 + V(stb)) for i in range(len(stb._phid))]) + sum([stb._dens[i] * (1 + stb._w[i])  for i in range(len(stb._dens))]))
    cSumTotal = sumphid * curv[0]/total
    cSumTotal += sum([stb._dens[i] * (1 + stb._w[i]) * curv[1][i] / total for i in range(len(curv[1]))])
    return cSumTotal


""" Calculate total curvature perturbation. """
def curvTotal(curvMatrix, stb):
    curv = 0.
    sumphid = sum([(stb._phid[i]**2/2 + V(stb)) for i in range(len(stb._phid))])
    total = (sumphid + sum([stb._dens[i] * (1 + stb._w[i])  for i in range(len(stb._dens))])) ** 2
    curv += (sumphid ** 2) * curvMatrix[0,0] / total               
    for i in range(1, len(curvMatrix)):
        curv += 2. * sumphid * stb._dens[i - 1] * (1 + stb._w[i - 1]) * curvMatrix[0, i]/ total
    for i in range(1, len(curvMatrix)):
        for j in range(1, len(curvMatrix[0])):
            curv += stb._dens[i - 1] * (1 + stb._w[i - 1]) * stb._dens[j - 1] * (1 + stb._w[j - 1]) * curvMatrix[i, j] / total
    return curv
                    

"""Evolve the system using the ODEs."""
def evolve():
    global H, a, state, dt, dtInit, dN, gamma, FINAL_DENS, GAMMA_FINAL, tau, epsilon, sigmaInit, lambd, dtInit
    time = 0
    N = 0
    startDecay = 0
    endDecay = 0
    gamma = 0.0
    evolution = []
    steps = 0
    while ((fielddens(state._b) > FINAL_DENS) | (gamma < 20.0 * H)):
        dt = dtInit
        Hubble(state._b)
        if (startDecay == 0):
            if (state._b._phi[0] < 0): 
                print "***************************************"
                print "Decay has started...."
                gamma = GAMMA_FINAL
                startDecay = 1
                print fielddens(state._b)
                print state._b._dens
                print H
                ratio = fielddens(state._b)/(state._b._dens[0] +  fielddens(state._b))
                r = 3.* ratio/(4 - ratio)
                print r
                lambd[0] =  (8./9.) * (r ** 2) * 0.1 * (1/sigmaInit) ** 2
                print "Lower bound for lambda: ", lambd[0] 
                print "***************************************"
        if (steps % 1000 == 0):
            print "Number of steps..." + str(steps)
            print a
            print H
            print state._p[0]._k/(a * H)
            print fielddens(state._b)
            print "a...", state._b._phi, state._b._dens[0]
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
        state = rk4(state)
        if (((steps % 5 == 0) & (N < 2.0)) | ((steps % 10 == 0) & (N > 2.0))):
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
        p.append(Perturbation([0.000], [0.000], [-2 * np[i], -(3./2.) * np[i], -2 * np[i]], [(ks[i] ** 2) * tau * np[i]/2.,  (ks[i] ** 2) * tau * np[i]/2.,  (ks[i] ** 2) * tau * np[i]/2.], np[i], ks[i]))
        p.append(Perturbation([-3. * MPL * np[i] * sqrt(epsilon)], [0.000], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.0, ks[i]))
    return p



"""Density evolution plotter."""
def plotEvol(evolution):
    global file_number, plotpath
    fig = p.figure()
    p.title(r'Evolution of $\phi$ over a number of e-folds')
    p.plot([N[0] for N in evolution], [st[2]._b._phi for st in evolution])
    p.xlabel(r'$N$')
    p.ylabel(r'$\phi(M_{PL})$')
    p.savefig(plotpath + str(file_number) + "phi.png", format = 'png')
    
    fig = p.figure()
    p.title(r'Evolution of densities over a number of e-folds')
    p.ylabel(r'$\log (\rho_i)$')
    p.xlabel(r'$N$')
    p1 = p.plot([N[0] for N in evolution],
                [math.log(st[2]._b._dens[0], 100) for st in evolution], 
                label = r'$\log(\rho_r)$')
    p2 = p.plot([N[0] for N in evolution],
                [math.log(st[2]._b._dens[1], 100) for st in evolution], 
                label = r'$\log(\rho_{DM})$')
    p3 = p.plot([N[0] for N in evolution],
                [math.log(fielddens(st[2]._b), 100) for st in evolution], 
                label = r'$\log(\rho_{\sigma})$')
    p.legend()
    p.savefig(plotpath + str(file_number) + "dens_evolution.png", format = 'png')
  


"""Curvature perturbation plotter."""      
def plotPert(evolution):
    global file_number, initNumb
    for i in range(len(evolution[0][2]._p)/initNumb):
        fig = p.figure()
        p.title(r'Evolution of the curvature perturbations over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\zeta_i$')
        p.plot([N[0] for N in evolution],
               [st[3][2 * i][0] + st[3][2 * i + 1][0] for st in evolution], 
               label = r'$\zeta_{\phi}$')
        p.plot([N[0] for N in evolution],
               [st[3][2 * i][1][0] + st[3][2 * i + 1][1][0] for st in evolution]
               , label = r'$\zeta_{r}$')
        p.plot([N[0] for N in evolution],
               [st[3][2 * i][1][1] + st[3][2 * i + 1][1][1] for st in evolution]
               , label = r'$\zeta_{DM}$')
        p.plot([N[0] for N in evolution], 
               [curvatureSum(st[3][2 * i], st[2]._b) 
                + curvatureSum(st[3][2 * i + 1], st[2]._b)
                for st in evolution], label = '$\zeta_{total}$')
        p.ylim(-10,10)
        p.legend()
        p.text(0.1, -5.0, r'$k = $' + str(evolution[0][2]._p[0]._k))
        p.savefig(plotpath + str(file_number) + "curv_pert_" + str(i) + ".png", format = 'png')


        fig = p.figure()
        p.title(r'Evolution of the curvature perturbations over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$S_i$')
        p.plot([N[0] for N in evolution],
               [3 * (st[3][2 * i][0] + st[3][2 * i + 1][0] 
                     - curvatureSum(st[3][2 * i], st[2]._b) 
                     - curvatureSum(st[3][2 * i + 1], st[2]._b))
                for st in evolution], label = r'$S_{\phi}$')
        p.plot([N[0] for N in evolution],
                 [3 * (st[3][2 * i][1][1] + st[3][2 * i + 1][1][1] 
                     - curvatureSum(st[3][2 * i], st[2]._b) 
                     - curvatureSum(st[3][2 * i + 1], st[2]._b))
                for st in evolution], label = r'$S_{DM}$')
        p.ylim(-10,10)
        p.legend()
        p.text(0.1, -5.0, r'$k = $' + str(evolution[0][2]._p[0]._k))
        p.savefig(plotpath + str(file_number) + "iso_curv_pert_" + str(i) + ".png", format = 'png')




"""Density perturbation plotting."""
def plotDensPert(evolution):
    global plotpath
    for i in range(len(evolution[0][2]._p)/2):
        fig = p.figure()
        p.title(r'Evolution of the density perturbations over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\delta_i$')
        p.plot([N[0] for N in evolution],
               [st[2]._p[2 * i]._ddens[0] + st[2]._p[2 * i + 1]._ddens[0] 
                for st in evolution], label = r'$\delta_r$')
        p.plot([N[0] for N in evolution],
               [st[2]._p[2 * i]._ddens[1] + st[2]._p[2 * i + 1]._ddens[1] 
                for st in evolution], label = r'$\delta_{DM}$')
        p.plot([N[0] for N in evolution],
               [st[2]._p[2 * i]._np + st[2]._p[2 * i + 1]._np 
                for st in evolution], label = r'$\Phi$')
        p.ylim(-10,10)
        p.legend()
        p.text(0.1, -5.0, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "dens_pert_" + str(i) + ".png", format = 'png')
    

        fig = p.figure()
        p.title(r'Evolution of the $\delta\phi$ over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\delta \phi$')
        p.plot([N[0] for N in evolution], 
               [st[2]._p[2 * i]._dphi[0] + st[2]._p[2 * i + 1]._dphi[0] for st in evolution])
        p.text(0.1, -0.1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "dphi_" + str(i) + ".png", format = 'png')



def plotW(evolution):
    global file_number, plotpath
    fig = p.figure()
    p.title(r'Evolution of the equation of state for the scalar field')
    p.xlabel(r'$N$')
    p.ylabel(r'$w_i$')
    p.plot([N[0] for N in evolution], [(fielddens(st[2]._b) - (2 * V(st[2]._b)))/fielddens(st[2]._b) for st in evolution])
    p.savefig(plotpath + str(file_number) + "eq_of_state.png", format = 'png')



def plotCurvCorrelation(evolution):
    global lambd, plotpath
    for i in range(len(evolution[0][2]._p)/2):
        fig = p.figure()
        p.title(r'Evolution of the curvature power spectrum over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$P_{\zeta_i}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][0, 0] + st[4][2 * i + 1][0, 0]
                for st in evolution], label = r'$P_{\zeta_{\phi_1 ... \phi_M}}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][1, 1] + st[4][2 * i + 1][1, 1] 
                for st in evolution], label = r'$P_{\zeta_{r}}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][2, 2] + st[4][2 * i + 1][2, 2] 
                for st in evolution], label = r'$P_{\zeta_{DM}}$')
        p.plot([N[0] for N in evolution],
               [st[5][2 * i] + st[5][2 * i + 1]
                for st in evolution], label = r'$P_{\zeta_{total}}$')
        p.legend()
        p.text(0.1, -1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        last = evolution[len(evolution) - 1][5][2 * i] + evolution[len(evolution) - 1][5][2 * i + 1]
        begin =  evolution[0][5][2 * i] + evolution[0][5][2 * i + 1]
        actual_lambd = last/begin - 1
        p.axhline(y = last, linestyle = '--')
        p.ylim(0.0, 10.0)
        p.text(0.1, 1, r'$\lambda = $' + str(actual_lambd))
        p.savefig(plotpath + str(file_number) + "curvCorrel_"
 + str(i) + ".png", format = 'png')


        fig = p.figure()
        p.title(r'Evolution of cross-species spectrum over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$P_{\zeta_i \zeta_j}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][0, 1] + st[4][2 * i + 1][0, 1] for st in evolution], label = r'$P_{\zeta_{\phi_1 ... \phi_M} \zeta_{r}}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][0, 2] + st[4][2 * i + 1][0, 2] for st in evolution], label = r'$P_{\zeta_{\phi_1 ... \phi_M} \zeta_{DM}}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][1, 2] + st[4][2 * i + 1][1, 2] for st in evolution], label = r'$P_{\zeta_{r} \zeta_{DM}}$')
        p.legend()
        p.ylim(-10.0, 10.0)       
        p.text(0.1, -0.1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "crossCorrel_" + str(i) + ".png", format = 'png')

        fig = p.figure()
        p.title(r'Evolution of isocurvature spectrum over a number of e-folds'\
)
        p.xlabel(r'$N$')
        p.ylabel(r'$P_{S_i \zeta}$')
        coreldm = [3 * (curvatureSum((st[4][2 * i][2, 0] + st[4][2*i + 1][2, 0]
                                     , [st[4][2*i][2,1] + st[4][2*i + 1][2,1], st[4][2*i][2,2] + st[4][2*i + 1][2,2]]), st[2]._b) 
                        - (st[5][2 * i] + st[5][2 * i + 1])) for st in evolution]
        corelsigma = [3 * (curvatureSum((st[4][2 * i][0, 0] + st[4][2*i + 1][0, 0]
                                         , [st[4][2*i][0,1] + st[4][2*i + 1][0,1], st[4][2*i][0,2] + st[4][2*i + 1][0,2]]), st[2]._b) 
                        - (st[5][2 * i] + st[5][2 * i + 1])) for st in evolution]
        p.plot([N[0] for N in evolution], coreldm, label = r'$P_{S_{DM} \zeta_{total}}$')
        p.plot([N[0] for N in evolution], corelsigma, label = r'$P_{S_{\sigma} \zeta_{DM}}$')
        p.legend()
        p.ylim(-50.0, 50.0)
        p.text(0.1, -0.1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "isoCorrel_" + str(i) + ".p\
ng", format = 'png')

        

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
    


def fileNumber():
    global file_number, datapath
    file_number = 0 
    while(os.path.isfile(datapath + 'evolution'+ str(file_number) +'.out')):
        file_number += 1
    return file_number



def writeData(evolution):
    global file_number, mass, gamma, sigmaInit, lambd, datapath, statpath, epsilon
    datafile = open(datapath + 'evolution' + str(file_number) + '.out', 'w')
    print "Printing data to evolution" + str(file_number) + ".out"
    writeOut = "[" + str(mass) + ", " + str(sigmaInit) + ", " + str(gamma) + ", " + str(epsilon) + "]"
    print >>datafile, writeOut
    writeOut = "["
    for ev in evolution:
        writeOut += "["
        writeOut += str(ev[0]) + ", "
        writeOut += str(ev[1]) + ", "
        writeOut += "[[" + str(ev[2]._b._phi) + ", " + str(ev[2]._b._phid) +  ", " + str(ev[2]._b._dens) + ", " + str(ev[2]._b._w) +  "],["
        for i in range(len(ev[2]._p)):
            writeOut += "[" + str(ev[2]._p[i]._dphi) +  ", " + str(ev[2]._p[i]._dphid) + ", " + str(ev[2]._p[i]._ddens) +  ", " + str(ev[2]._p[i]._sdens) +  ", " + str(ev[2]._p[i]._np) + ", " + str(ev[2]._p[i]._k) +  "]"
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
    print >> datafile, "Number of species: ", len(evolution[0][2]._b._dens)
    print >> datafile, "Mass of the fields: ", mass
    print >> datafile, "Initial value of the fields: ", sigmaInit
    print >> datafile, "Initial density of species: ", evolution[0][2]._b._dens
    print >> datafile, "Value of the slow-roll inflationary parameter: ", epsilon
    print >>datafile, "Bounds for approximated lambda: ", lambd[0], lambd[1]
    print >>datafile, "Actual value of lambda: ", calculateLambda(evolution)
    print >>datafile, "Value of Gamma: ", gamma
    datafile.close()
    


def plotCurvatures():
    global plotpath
    fig = p.figure()
    p.title(r'Evolution of the curvature power spectrum over a number of e-folds')
    p.xlabel(r'$N$')
    p.ylabel(r'$\frac{P_{\zeta_{total}}}{P_{\zeta_{total}}(0)}$')
    nr = 0
    separation = max([(st[-1][5][0] + st[-1][5][1])
                      /(st[0][5][0] + st[0][5][1]) - 1.
                      for st in evolutions])/(0.95 * len(evolutions))
    print "separation", separation
    for evolution in evolutions:
        nr += 1
        last = evolution[len(evolution) - 1][5][0] + evolution[len(evolution) - 1][5][1]
        begin =  evolution[0][5][0] + evolution[0][5][1]
        p.plot([N[0] for N in evolution],
               [(st[5][0] + st[5][1])/begin
                for st in evolution], label = r'$P_{\zeta_{total}}$')
        actual_lambd = last/begin - 1
        p.axhline(y = 1 + actual_lambd, linestyle = '--')
        p.text(evolution[len(evolution) - 1][0], 1.00 + actual_lambd, str(nr))
        p.text(1.2 * evolution[-1][0], 1. + (separation * (nr - 1)),
               r'' + str(nr)+'.$\lambda = $' + str(actual_lambd))
    p.savefig(plotpath + "curvCorrel_all.png", format = 'png', bbox_inches='tight')

    fig = p.figure()
    p.title(r'Evolution of the isocurvature over a number of e-folds')
    p.xlabel(r'$N$')
    p.ylabel(r'$\frac{S_{DM}}{S_{DM}(0)}$')
    nr = 0
    for evolution in evolutions:
        nr += 1
        fa = 0
        iso =  [isoCorrelCurv(st, evolution) for st in evolution]
        p.plot([N[0] for N in evolution], iso
               , label = r'$S_{DM}$')
        final = 3 * (st[3][0][1][1] + st[3][1][1][1] 
                     - curvatureSum(st[3][0], st[2]._b) 
                     - curvatureSum(st[3][1], st[2]._b))/begin
        p.text(evolution[len(evolution) - 1][0], 0.1 + iso[-1], str(nr))
    p.ylim(-50, 50)
    p.savefig(plotpath + 'iso_all.png', format = 'png')


def initCurvSpectrum():
    global plotpath, epsilons
    fig = p.figure()
    p.title(r'Evolution of the curvature power spectrum over a number of e-folds')
    p.xlabel(r'$N$')
    p.ylabel(r'$P_{\zeta_{total}}$')
    nr = 0
    separation = 0.1
    fact = 2.4 * (1.e-9)
    for evolution in evolutions:
        nr += 1
        last = evolution[-1][5][0] + evolution[-1][5][1]
        begin =  evolution[0][5][0] + evolution[0][5][1]
        p.plot([N[0] for N in evolution],
               [fact * (st[5][0] + st[5][1])/last
                for st in evolution], label = r'$P_{\zeta_{total}}$')
        actual_lambd = last/begin - 1
        p.axhline(y = fact * begin/last, linestyle = '--')
        p.text(evolution[-1][0], 0.9 * fact, str(nr))
        p.text(1.2 * evolution[len(evolution) - 1][0], fact * (1. - (separation * (nr - 1))), r'' + str(nr)+'.$\lambda = $' + str(actual_lambd))
    p.savefig(plotpath + 'curvCorrelSpectrum.png', format = 'png', bbox_inches='tight')
    


def isoCorrelCurv(st, evolution):
    begin = 3 * (evolution[0][3][0][1][1] + evolution[0][3][0][1][1] 
                 - curvatureSum(evolution[0][3][0], evolution[0][2]._b) 
                 -  curvatureSum(evolution[0][3][0], evolution[0][2]._b)) 
    iso =  3 * (st[3][0][1][1] + st[3][1][1][1] 
                  - curvatureSum(st[3][0], st[2]._b) 
                  - curvatureSum(st[3][1], st[2]._b))/begin
    return iso
    


def isoCorrelCurvature(curvMatrix, j):
    return 3. * (curvMatrix[1, j] - curvMatrix[1, 1])



def isoSelfCorrel(curvMatrix, j):
    if (len(curvMatrix) < j):
        print "Error: Index for isocurvature correlation out-of-bounds"
        return 
    return 9. * (curvMatrix[1, 1] + curvMatrix[j, j] - (2 * curvMatrix[1, j]))

    

def calculateLambda(evolution):
    last = evolution[len(ev) - 1][5][0] + evolution[len(ev) - 1][5][1]
    begin = evolution[0][5][0] + evolution[0][5][1]
    return last/begin - 1.



def calculateAlpha(evolution):
    st = evolution[len(evolution) - 1]
    ratio = (isoSelfCorrel(st[4][0], 2) + isoSelfCorrel(st[4][1], 2))/(st[5][0] + st[5][1])
    return ratio/(1. + ratio)



def calculateR(evolution):
    st = evolution[len(evolution) - 1]
    ratio = isoCorrelCurvature(st[4][0], 2)/sqrt((isoSelfCorrel(st[4][0], 2) + isoSelfCorrel(st[4][1], 2)) * (st[5][0] + st[5][1]))

    

def plotLambdasGamma():
    global evolutions, lam, gammas, epsilons
    if (len(gammas) <= len(evolutions)):
        fig = p.figure()
        p.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)

        ax1 = p.subplot2grid((3, 3), (0,0), colspan = 3)
        ax2 = p.subplot2grid((3, 3), (1,0), colspan = 3)
        ax3 = p.subplot2grid((3, 3), (2,0), colspan = 3)
                
        lambds = [lam[i * len(epsilons)] for i in range(len(gammas))]
        ax1.set_ylabel(r'$\lambda$')
        ax1.set_xlabel(r'$\Gamma$')
        ax1.plot(lambds, gammas) 
            
        alphas = [calculateAlpha(evolutions[i * len(epsilons)]) for i in range(len(gammas))]
        ax2.set_ylabel(r'$\alpha$')
        ax2.set_xlabel(r'$\Gamma$')
        ax2.plot(alphas, gammas)

        rs = [calculateR(evolutions[i * len(epsilons)]) for i in range(len(gammas))]
        ax3.set_ylabel(r'$r$')
        ax3.set_xlabel(r'$\Gamma$')
        ax3.plot(rs, gammas)
        fig.savefig(plotpath +  "lamalpgam.png", format = 'png') 
        
 
        fig = p.figure()
        axes = []
        for i in range(3):
            lineAxes = []
            for j in range(3):
                lineAxes.append(p.subplot2grid((3, 3), (i, j)))
            axes.append(lineAxes)
        for i in range(3):
            for j in range(3):
                if(j + 3 * i < len(gammas)):
                    axes[i][j].set_xlabel(r'$\rho$')
                    axes[i][j].set_ylabel(r'$N$')
                    axes[i][j].plot([N[0] for N in evolutions[(j + 3 * i) * len(epsilons)]],
                                    [math.log(st[2]._b._dens[0], 100) for st in evolutions[(j + 3 * i) * len(epsilons)]], 
                                    label = r'$\log(\rho_r)$')
                    axes[i][j].plot([N[0] for N in evolutions[(j + 3 * i) * len(epsilons)]],
                                    [math.log(st[2]._b._dens[1], 100) for st in evolutions[(j + 3 *i) * len(epsilons)]], 
                                    label = r'$\log(\rho_{DM})$')
                    axes[i][j].plot([N[0] for N in evolutions[(j + 3 * i) * len(epsilons)]],
                                    [math.log(fielddens(st[2]._b), 100) for st in evolutions[(j + 3 * i) * len(epsilons)]],
                                    label = r'$\log(\rho_{\sigma})$')
                     #axes[i][j].legend()
                     #axes[i][j].text()
        print "hahaha"
        fig.savefig(plotpath +  "rholamalpgam.png", format = 'png') 



def contMassSigma():
    global masses, sigmas, evolutions, epsilons
    if (len(evolutions) >= len(masses) * len(sigmas)):
        logmasses = [log(mass) for mass in masses]
        logsigmas = [log(sig) for sig in sigmas]
        M, S = meshgrid(logmasses, logsigmas)
        lambdaMatrix = []
        for i in range(len(logmasses)):
            lambdaLines = []
            for j in range(len(logsigmas)):
                lambdaLines.append(lam[(len(logmasses) * i + j) * len(gammas) * len(sigmas)])
            lambdaMatrix.append(lambdaLines)
        fig = p.figure()
        im = p.imshow(lambdaMatrix, interpolation='bilinear', origin='lower',
                      cmap=cm.gray, extent=(logmasses[0], logmasses[-1], 
                                            logsigmas[0], logsigmas[-1]))
        levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
        CS = p.contour(lambdaMatrix, levels, origin = 'lower', 
                       linewidths = 2, 
                       extent=(logmasses[0], logmasses[-1], 
                               logsigmas[0], logsigmas[-1]))
        p.clabel(CS, levels[1::2],  # label every second level
           inline=1,
           fmt='%1.1f',
           fontsize=14)
        CB = p.colorbar(CS, shrink=0.8, extend='both')
        CBI = p.colorbar(im, orientation='horizontal', shrink=0.8)
        p.title(r'Contour of $\lambda$ dependence in the $m_{\sigma} - \sigma (0)$ space')
        p.xlabel(r'$\log(m_{\sigma})$')
        p.ylabel(r'$\log(\sigma(0))$')
        p.savefig(plotpath + str(file_number) + "contMassSigma.png", format = 'png') 

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
    masses = [1.0e-13]
    sigmas = [0.16 * MPL]
    gammas = [1.e-3]
#gammas = [0.125e-7, 0.25e-7, 0.5e-7, 1.0e-6, 2.0e-6, 4.0e-6, 8.0e-6, 1.6e-5, 3.2e-5, 6.4e-5, 1.28e-4, 2.56e-5, 5.12e-4]
    # gammas = [0.15e-4, 0.3125e-4, 0.625e-4, 1.25e-4, 2.5e-4, 5e-4, 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2, 6.4e-2, 1.28e-1]
    epsilons = [1.0]
    frac = 1.0e-15
    coupl = [-frac + 0.4, 2 * frac, -frac + 0.6]

    printInitVal()
    
    evolutions = []
    lam = []
    for m in masses:
        for sigmaInit in sigmas:
            for gammafinal in gammas:
                for epsilon in epsilons:  
                    dtInit = (1.e+11)/MPL
                    dt = dtInit
                    mass = [m] 
                    initNumb = 2
                    lambd = [0.00, 0.00]
                    initBack = Background([sigmaInit], [(1e-50) * MPL ** 2], [50.00 * (m ** 2) * (MPL ** 2), (1.0e-58) * MPL**4, (21./8.) * ((4./11.) ** (4./3.)) * 50.00 * (m ** 2) * (MPL ** 2)], [1./3., 0., 1./3.])
                    nps = [-1.00]
                    H = 0.
                    a = 1.00
                    Hubble(initBack)
                    gamma = 0
                    GAMMA_FINAL = H * gammafinal
                    tau = -1/(a * H)
                    ks = [-1.e-10 * (1/tau)]
                    print "k/a...",ks[0]/a
                    print "a...", a
                    initPert = setInitPert(nps, ks)
                    state = State(initBack, initPert)
                    FINAL_DENS = fielddens(initBack) * (5e-1)
                    print FINAL_DENS
                    # Initialize the transformation matrix u
                                       
                    #Evolve ODE system
                    ev = evolve()   
                
                    print "Saving data..."
                    file_number = 0 
                    file_number = fileNumber()
                    writeData(ev)
                
                
                    print "Plotting data..."
                    plotCurvCorrelation(ev)
        
                    plotEvol(ev)
                    plotDensPert(ev)
                    plotPert(ev)
                    init = ev[0][5][0] + ev[0][5][1]
                    lam.append(calculateLambda(ev))
                    evolutions.append(ev)
    print "Plotting all..."
    plotCurvatures()
    initCurvSpectrum()
    plotLambdasGamma()
    contMassSigma()




