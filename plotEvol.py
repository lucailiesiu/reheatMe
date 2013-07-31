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
/* Description: Implements plotting methods for the     */
/* evolution of a curvaton field. Normalizes the        */   
/* evolution by finding the value of \epsilon such that */
/* the energy scale of inflation is the fixed.          */
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
    while ((fielddens(state._b) > FINAL_DENS)): #| (gamma < 10.0 * H)):
        dt = dtInit * (1 + N)
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
        p.append(Perturbation([0.000], [0.000], [-2 * np[i], -(3./2.) * np[i]], [(ks[i] ** 2) * tau * np[i]/2.,  (ks[i] ** 2) * tau * np[i]/2.], np[i], ks[i]))
        p.append(Perturbation([-3. * MPL * np[i] * sqrt(epsilon)], [0.000], [0.0, 0.0], [0.0, 0.0], 0.0, ks[i]))
    return p



"""Density evolution plotter."""
def plotEvol(evolution):
    global file_number, plotpath, speciesName, MPLTEV
    fig = p.figure()
    p.title(r'$\mathrm{Evolution\ of\ }\phi\mathrm{\ over\ a\ number\ of\ e-folds}$')
    p.plot([N[0] for N in evolution], [st[2]._b._phi for st in evolution], color = 'purple')
    p.xlabel(r'$N$')
    p.ylabel(r'$\phi[M_{PL}]$')
    p.savefig(plotpath + str(file_number) + "phi.png", format = 'png')
    
    fig = p.figure()
    ax = p.subplot(111)
    ax.set_ylabel(r'$\log_{100}(\rho_i[M_{PL}])$')
    ax.set_xlabel(r'$N$')
    p.text(0.1, math.log(evolution[0][2]._b._dens[0], 100) + 0.2, 
           r'$T_{RH} = $' + str((evolution[-1][2]._b._dens[0]) ** (0.25) 
                                * MPLTEV) + '$TeV$')
    for i in range(len(evolution[0][2]._b._dens)):
        p.plot([N[0] for N in evolution],
               [math.log(st[2]._b._dens[i], 100) for st in evolution],
               label = r'$\log_{100}(\rho_{' + speciesName[i] + '})$', color = speciesColor[i])
    p.plot([N[0] for N in evolution],
           [math.log(fielddens(st[2]._b), 100) for st in evolution], 
           label = r'$\log_{100}(\rho_{\sigma})$', color = 'purple')
    axT = ax.twinx()
    axT.set_xlabel(r'$log_{100}(T[MeV])$')
    axT.semilogy([N[0] for N in evolution],
             [(st[2]._b._dens[0] ** (0.25)) * MPLTEV * (1.e6) for st in evolution],
             visible = False, basey = 100)
    lg = ax.legend()
    lg.draw_frame(False)
    p.savefig(plotpath + str(file_number) + "dens_evolution.png", format = 'png',  bbox_inches='tight')
  


"""Curvature perturbation plotter."""      
def plotPert(evolution):
    global file_number, initNumb, speciesName
    for i in range(len(evolution[0][2]._p)/initNumb):
        fig = p.figure()
        p.title(r'Evolution of the curvature perturbations over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\zeta_i$')
        p.plot([N[0] for N in evolution],
               [st[3][2 * i][0] + st[3][2 * i + 1][0] for st in evolution], 
               label = r'$\zeta_{\sigma}$', color = 'purple')
        for j in range(len(evolution[0][2]._b._dens) - 1): 
            p.plot([N[0] for N in evolution],
                   [st[3][2 * i][1][j] + st[3][2 * i + 1][1][j] for st in evolution]
                   , label = r'$\zeta_{' + speciesName[j] + '}$', color = speciesColor[j])
        p.plot([N[0] for N in evolution], 
               [curvatureSum(st[3][2 * i], st[2]._b) 
                + curvatureSum(st[3][2 * i + 1], st[2]._b)
                for st in evolution], label = '$\zeta_{total}$', color = 'black')
        p.ylim(-10,10)
        p.legend()
        p.text(0.1, -5.0, r'$k = $' + str(evolution[0][2]._p[0]._k))
        p.savefig(plotpath + str(file_number) + "curv_pert_" + str(i) + ".png", format = 'png')


        fig = p.figure()
        p.title(r'Evolution of isocurvature over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$S_i$')
        p.plot([N[0] for N in evolution],
               [3 * (st[3][2 * i][0] + st[3][2 * i + 1][0] 
                     - curvatureSum(st[3][2 * i], st[2]._b) 
                     - curvatureSum(st[3][2 * i + 1], st[2]._b))
                for st in evolution], label = r'$S_{\sigma}$', color = 'purple')

        for j in range(1, len(evolution[0][2]._b._dens)):
            p.plot([N[0] for N in evolution],
                   [3 * (st[3][2 * i][1][j] + st[3][2 * i + 1][1][j] 
                         - curvatureSum(st[3][2 * i], st[2]._b) 
                         - curvatureSum(st[3][2 * i + 1], st[2]._b))
                    for st in evolution], label = r'$S_{' + speciesName[j] + '}$', color = speciesColor[j])
        p.ylim(-10,10)
        p.legend()
        p.text(0.1, -5.0, r'$k = $' + str(evolution[0][2]._p[0]._k))
        p.savefig(plotpath + str(file_number) + "iso_curv_pert_" + str(i) + ".png", format = 'png')




"""Density perturbation plotting."""
def plotDensPert(evolution):
    global plotpath, speciesName
    for i in range(len(evolution[0][2]._p)/2):
        fig = p.figure()
        p.title(r'Evolution of the density perturbations over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\delta_i$')
        for j in range(len(evolution[0][2]._b._dens)):
            p.plot([N[0] for N in evolution],
                   [st[2]._p[2 * i]._ddens[j] + st[2]._p[2 * i + 1]._ddens[j] 
                    for st in evolution], label = r'$\delta_{' + speciesName[j] + ' }$', color = speciesColor[j])
        p.plot([N[0] for N in evolution],
               [st[2]._p[2 * i]._np + st[2]._p[2 * i + 1]._np 
                for st in evolution], label = r'$\sigma$', color = 'purple')
        p.ylim(-10,10)
        p.legend()
        p.text(0.1, -5.0, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "dens_pert_" + str(i) + ".png", format = 'png')
    

        fig = p.figure()
        p.title(r'Evolution of the $\delta\phi$ over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\delta \sigma[M_PL]$')
        p.plot([N[0] for N in evolution], 
               [st[2]._p[2 * i]._dphi[0] + st[2]._p[2 * i + 1]._dphi[0] for st in evolution], color = 'purple')
        p.text(0.1, -0.1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "dphi_" + str(i) + ".png", format = 'png')



def plotW(evolution):
    global file_number, plotpath
    fig = p.figure()
    p.title(r'Evolution of the equation of state for the scalar field')
    p.xlabel(r'$N$')
    p.ylabel(r'$w_i$')
    p.plot([N[0] for N in evolution], [(fielddens(st[2]._b) - (2 * V(st[2]._b)))/fielddens(st[2]._b) for st in evolution],
           color = 'black')
    p.savefig(plotpath + str(file_number) + "eq_of_state.png", format = 'png')



def plotCurvCorrelation(evolution):
    global lambd, plotpath, speciesName
    for i in range(len(evolution[0][2]._p)/2):
        fig = p.figure()
        p.title(r'Evolution of the curvature power spectrum over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$P_{\zeta_i}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][0, 0] + st[4][2 * i + 1][0, 0]
                for st in evolution], label = r'$P_{\zeta_{\sigma}}$', color = 'purple')
        for j in range(1, 1 + len(evolution[0][2]._b._dens)):
            p.plot([N[0] for N in evolution],
                   [st[4][2 * i][j, j] + st[4][2 * i + 1][j, j] 
                    for st in evolution], label = r'$P_{\zeta_{' + speciesName[j - 1]  + '}}$', 
                   color = speciesColor[j - 1])
        p.plot([N[0] for N in evolution],
               [st[5][2 * i] + st[5][2 * i + 1]
                for st in evolution], label = r'$P_{\zeta_{total}}$', color = 'black')
        p.legend()
        # p.text(0.1, -1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        last = evolution[len(evolution) - 1][5][2 * i] + evolution[len(evolution) - 1][5][2 * i + 1]
        begin =  evolution[0][5][2 * i] + evolution[0][5][2 * i + 1]
        actual_lambd = last/begin - 1
        p.axhline(y = last, linestyle = 'k--')
        p.ylim(0.0, 10.0)
        p.text(0.1, 1, r'$\lambda = \mathrm{' + str(actual_lambd) + '}$')
        p.savefig(plotpath + str(file_number) + "curvCorrel_" + str(i) + ".png", format = 'png')


        fig = p.figure()
        p.title(r'Evolution of cross-species spectrum over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$P_{\zeta_i \zeta_j}$')
        p.plot([N[0] for N in evolution],
               [st[4][2 * i][0, 1] + st[4][2 * i + 1][0, 1] for st in evolution], label = r'$P_{\zeta \zeta_{\sigma}}$', 
               color = 'purple')
        for j in range(2, 1 + len(evolution[0][2]._b._dens)):
            p.plot([N[0] for N in evolution],
                   [st[4][2 * i][1, j] + st[4][2 * i + 1][1, j] for st in evolution], 
                   label = r'$P_{\zeta_{r} \zeta_{' + speciesName[j - 1]  + '}}$', color = speciesColor[j - 1])
        p.legend()
        p.ylim(-10.0, 10.0)       
        # p.text(0.1, -0.1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "crossCorrel_" + str(i) + ".png", format = 'png')

        fig = p.figure()
        p.title(r'Evolution of isocurvature spectrum over a number of e-folds'\
)
        p.xlabel(r'$N$')
        p.ylabel(r'$P_{S_i \zeta}$')
        corel = []
        for j in range(2, 1 + len(evolution[0][2]._b._dens)):
            corel.append([3 * (curvatureSum((st[4][2 * i][j, 0] + st[4][2*i + 1][j, 0]
                                               , [st[4][2*i][j, q] + st[4][2*i + 1][j, q] for q in range(len(evolution[0][2]._b._dens))]), st[2]._b) 
                            - (st[5][2 * i] + st[5][2 * i + 1])) for st in evolution])
        corelsigma = [3 * (curvatureSum((st[4][2 * i][0, 0] + st[4][2*i + 1][0,0]
                                         , [st[4][2*i][0,1] + st[4][2*i + 1][0,1], st[4][2*i][0,2] + st[4][2*i + 1][0,2]]), st[2]._b) 
                           - (st[5][2 * i] + st[5][2 * i + 1])) for st in evolution]
        for j in range(2, 1 + len(evolution[0][2]._b._dens)):                    
            p.plot([N[0] for N in evolution], corel[j - 2], label = r'$P_{S_{' + speciesName[j - 1] + '} \zeta_{total}}$',
                   color = speciesColor[j-1])
        p.plot([N[0] for N in evolution], corelsigma, label = r'$P_{S_{\sigma} \zeta_{total}}$', color = 'purple')
        p.legend() 
        p.ylim(-50.0, 50.0)
        # p.text(0.1, -0.1, r'$k = $' + str(evolution[0][2]._p[2 * i]._k))
        p.savefig(plotpath + str(file_number) + "isoCorrel_" + str(i) + ".p\
ng", format = 'png')

        

def dirNumber(dir_number):
    global datapath, statpath, plotpath
    if (not os.path.isdir('data/' + str(dir_number))):
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
        writeOut += str(ev[2]._p[0]._dphi) +  ", " + str(ev[2]._p[0]._dphid) + ", " + str(ev[2]._p[0]._ddens) +  ", " + str(ev[2]._p[0]._sdens) +  ", " + str(ev[2]._p[0]._np) + ", " + str(ev[2]._p[0]._k) +  "]],"
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
    print >>datafile, "ActuSal value of lambda: ", calculateLambda(evolution)
    print >>datafile, "Value of Gamma: ", gamma
    datafile.close()
    


def plotCurvatures():
    global plotpath, proc, speciesName
    fig = p.figure()
    p.title(r'Evolution of the curvature power spectrum over a number of e-folds')
    p.xlabel(r'$N$')
    p.ylabel(r'$\frac{P_{\zeta_{total}}}{P_{\zeta_{total}}(0)}$')
    nr = 0
    yTextLam = 1.1 * max([N[-1][0] for N in evolutions])
    separation = max([(evolutions[i * proc][-1][5][0] + 
                       evolutions[i * proc][-1][5][1])/
                      (evolutions[i * proc][0][5][0] + 
                       evolutions[i * proc][0][5][1]) - 1.
                      for i in range(len(evolutions)/proc)])/(0.95 * len(evolutions)/(1.0 * proc))
    print "separation", separation
    for i in range(len(evolutions)/proc):
        evolution = evolutions[i * proc]
        nr += 1
        last = evolution[len(evolution) - 1][5][0] + evolution[len(evolution) - 1][5][1]
        begin =  evolution[0][5][0] + evolution[0][5][1]
        p.plot([N[0] for N in evolution],
               [(st[5][0] + st[5][1])/begin
                for st in evolution], label = r'$P_{\zeta_{total}}$')
        actual_lambd = last/begin - 1
        p.axhline(y = 1 + actual_lambd, linestyle = '--')
        p.text(evolution[len(evolution) - 1][0], 1.00 + actual_lambd, r'$\mathrm{' + str(nr) + '}$')
        p.text(yTextLam, 1. + (separation * (nr - 1)),
               r'$\mathrm{' + str(nr)+'}. \lambda = \mathrm{' + str(actual_lambd)+  '}$')
    p.savefig(plotpath + "curvCorrel_all.png", format = 'png', bbox_inches='tight')

    fig = p.figure()
    p.title(r'Evolution of DM isocurvature over a number of e-folds')
    p.xlabel(r'$N$')
    p.ylabel(r'$\frac{S_{DM}}{S_{DM}(0)}$')
    nr = 0         
    for i in range(len(evolutions)/proc):
        evolution = evolutions[i * proc]
        nr += 1
        fa = 0
        iso =  [isoCorrelCurv(st, evolution) for st in evolution]
        p.plot([N[0] for N in evolution], iso
               , label = r'$S_{DM}$')
        p.text(evolution[len(evolution) - 1][0], 0.1 + iso[-1], r'$\mathrm{' + str(nr) + '}$')
        
    p.ylim(-50, 50)
    p.savefig(plotpath + 'iso_DM.png', format = 'png')

    if (len(evolutions[0][0][2]._b._dens) > 2):  
        p.title(r'Evolution of $' + speciesName[2] + ' $ isocurvature over a number of e-folds')
        p.xlabel(r'$N$')
        p.ylabel(r'$\frac{S_{' + speciesName[2] + ' }}{S_{' + speciesName[2] +  ' }(0)}$')
        nr = 0
        for i in range(len(evolutions)/proc):
            evolution = evolutions[i * proc]
            nr += 1
            fa = 0
            begin =  3 * (evolution[0][3][0][1][2] + evolution[0][3][1][1][2]
                          - curvatureSum(evolution[0][3][0], st[2]._b)
                          - curvatureSum(evolution[0][3][1], st[2]._b))
            iso =  [3 * (st[3][0][1][2] + st[3][1][1][2]
                         - curvatureSum(st[3][0], st[2]._b)
                         - curvatureSum(st[3][1], st[2]._b))/begin
                    for st in evolution]
            
            p.plot([N[0] for N in evolution], iso,
                   label = r'$S_{' + speciesName[2] + '}$')
            p.text(evolution[len(evolution) - 1][0], 0.1 + iso[-1], r'$\mathrm{' + str(nr) + '}$')
        p.ylim(-50, 50)
        p.savefig(plotpath + 'iso_DR.png', format = 'png')


def initCurvSpectrum():
    global evolutions, plotpath, epsilons, proc, speciesName
    fig = p.figure()
    p.title(r'Evolution of the curvature power spectrum over a number of e-folds')
    p.xlabel(r'$N$')
    p.ylabel(r'$P_{\zeta_{total}}$')
    yTextLam = 1.2 * max([N[-1][0] for N in evolutions])
    nr = 0
    separation = 0.1
    fact = 2.4 * (1.e-9)
    for i in range(len(evolutions)/proc):
        evolution = evolutions[i * proc]
        nr += 1
        last = evolution[-1][5][0] + evolution[-1][5][1]
        begin =  evolution[0][5][0] + evolution[0][5][1]
        actual_lambd = last/begin - 1
        if (actual_lambd > 0):
            p.plot([N[0] for N in evolution],
                   [fact * (st[5][0] + st[5][1])/last
                    for st in evolution], label = r'$P_{\zeta_{total}}$')
            p.axhline(y = fact * begin/last, linestyle = '--')
            p.text(evolution[0][0] + 0.1, 1.05 * fact * begin/last, r'$\mathrm{' + str(nr) + '}$')
        if (len(eps) != 0):
            print "blablab"
            p.text(yTextLam, fact * (1. - (separation * (nr - 1))),
                   r'$\mathrm{' + str(nr)+'}.\epsilon = \mathrm{' + str(eps[i * proc]) + '}$')
    if (len(eps) != 0):
        p.text(yTextLam, fact * (1. + (separation)),
               r'$H_* = \mathrm{' + str(round(sqrt(8 * (3.14 ** 2) * (MPL ** 2) * eps[0]
                                       * fact * begin / last) * MPLTEV, 3)) + '} TeV$')
    p.ylim(0, fact * 1.2)
    p.savefig(plotpath + 'curvCorrelSpectrum.png',
              format = 'png', bbox_inches='tight')
    


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


def isoCorrelij(curvMatrix, i, j):
    return 3. * (curvMatrix[i, j] - curvMatrix[1, j] - curvMatrix[1, i] + curvMatrix[1, 1])


def isoSelfCorrel(curvMatrix, j):
    if (len(curvMatrix) < j):
        print "Error: Index for isocurvature correlation out-of-bounds"
        return 
    return 9. * (curvMatrix[1, 1] + curvMatrix[j, j] - (2 * curvMatrix[1, j]))

    

def calculateLambda(evolution):
    last = evolution[-1][5][0] + evolution[-1][5][1]
    begin = evolution[0][5][0] + evolution[0][5][1]
    return last/begin - 1.



def calculateAlpha(evolution, j):
    st = evolution[-1]
    ratio = (isoSelfCorrel(st[4][0], j + 1) + isoSelfCorrel(st[4][1], j + 1))/(st[5][0] + st[5][1])
    return ratio/(1. + ratio)



def calculateR(evolution, j):
    st = evolution[len(evolution) - 1]
    print "calculate R ", j
    print (isoSelfCorrel(st[4][0], j + 1) + isoSelfCorrel(st[4][1], j + 1))
    print isoCorrelCurvature(st[4][0], j + 1) + isoCorrelCurvature(st[4][1], j + 1)
    print st[5][0] + st[5][1]
    ratio = (isoCorrelCurvature(st[4][0], j + 1) + isoCorrelCurvature(st[4][1], j + 1))/sqrt((isoSelfCorrel(st[4][0], j + 1) + isoSelfCorrel(st[4][1], j + 1)) * (st[5][0] + st[5][1]))
    print ratio
    return ratio


def calculateRij(evolution, i, j):
    st = evolution[- 1]
    print "Calculate Rij", i, " ", j
    print isoCorrelij(st[4][0], i +1 , j + 1) + isoCorrelij(st[4][1], i + 1, j + 1)
    print isoSelfCorrel(st[4][0], j + 1) + isoSelfCorrel(st[4][1], j + 1)
    print isoSelfCorrel(st[4][0], i + 1) + isoSelfCorrel(st[4][1], i + 1)
    ratio = (isoCorrelij(st[4][0], i +1 , j + 1) + isoCorrelij(st[4][1], i + 1, j + 1))/sqrt((isoSelfCorrel(st[4][0], j + 1) + isoSelfCorrel(st[4][1], j + 1)) * (isoSelfCorrel(st[4][0], i + 1) + isoSelfCorrel(st[4][1], i + 1)))
    print ratio
    return ratio
    

def plotLambdasGamma():
    global evolutions, lam, gammas, epsilons, speciesName, MPLTEV
    if (len(gammas) <= len(evolutions)):
        fig = p.figure()
        p.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=.5, hspace=.5)

        ax1 = p.subplot2grid((3, 3), (0,0), colspan = 3)
        ax2 = p.subplot2grid((3, 3), (1,0), colspan = 3)
        ax3 = p.subplot2grid((3, 3), (2,0), colspan = 3)
        lambds = [lam[i * len(epsilons)] for i in range(len(gammas))]
        print lambds
        ax1.set_ylabel(r'$\lambda$')
        ax1.set_xlabel(r'$\log_{10}(\Gamma[M_{PL}])$')
        ax1.plot([math.log(gam, 10) for gam in gammas], [lam for lam in lambds], color = 'black') 
        ax1.text(math.log(gammas[-1], 10) - 2, lambds[0]/2., r'$\sigma(0) = \mathrm{' + str(sigmas[0]) +'}M_{PL}$', fontsize = 12)
        ax1.text(math.log(gammas[-1], 10) - 2, lambds[0] * 4./5., r'$m_{\sigma} = \mathrm{' + str(masses[0] * MPLTEV) + '} TeV$', fontsize = 12)
        ax4 = ax1.twiny()
        ax4.set_xlabel(r'$log_{10}(T_{RH}[MeV])$')
        #ax4.plot([math.log((evolution[-1][2]._b._dens[0] ** (0.25)) * MPLTEV * (1.e6), 10)
        #          for evolution in evolutions], [0 for evolution in evolutions]
        #         , visible = False)
        
        alphasDM = [calculateAlpha(evolutions[i * len(epsilons)], 1) for i in range(len(gammas))]
        ax2.set_ylabel(r'$\alpha$')
        ax2.set_xlabel(r'$\log_{10}(\Gamma[M_{PL}])$')
        print "alphaDM", alphasDM
        ax2.plot([math.log(gam, 10) for gam in gammas], alphasDM, color = speciesColor[1])
        if (len(evolutions[0][0][2]._b._dens) > 2):
            alphasDR = [calculateAlpha(evolutions[i * len(epsilons)], 2) for i in range(len(gammas))]
            ax2.plot([math.log(gam, 10) for gam in gammas], alphasDR, color = speciesColor[2])
            
        rsDM = [calculateR(evolutions[i * len(epsilons)], 1) for i in range(len(gammas))]
        print "rDM", rsDM
        ax3.set_ylabel(r'$r$')
        ax3.set_xlabel(r'$\log_{10}(\Gamma)$')
        ax3.plot([math.log(gam, 10) for gam in gammas], rsDM, color = speciesColor[1], label = r'$\mathrm{dark\ matter}$')

        
        if (len(evolutions[0][0][2]._b._dens) > 2):
            alphasDR = [calculateAlpha(evolutions[i * len(epsilons)], 2) for i in range(len(gammas))]
            ax2.plot([math.log(gam, 10) for gam in gammas], alphasDR, color = speciesColor[2])
            rsDR = [calculateR(evolutions[i * len(epsilons)], 2) for i in range(len(gammas))]
            ax3.plot([math.log(gam, 10) for gam in gammas], rsDR, color = speciesColor[2], label = r'$\mathrm{dark\ radiation}$')
            print "rsDR", rsDR
            rsDRDM = [calculateRij(evolutions[i * len(epsilons)], 1, 2) for i in range(len(gammas))]
            print "rsDRM", rsDRDM
            ax3.plot([math.log(gam, 10) for gam in gammas], rsDRDM, linestyle = '--', color = speciesColor[1])
            ax3.plot([math.log(gam, 10) for gam in gammas], rsDRDM, linestyle = ':', color = speciesColor[2])
        lg = ax3.legend(loc = 0, prop = {'size': 12})
        lg.draw_frame(False)
        
        upGamma = math.log(100 * (masses[0] ** 3)/(MPL ** 2), 10)
        downGamma = math.log(0.1 *  (masses[0] ** 3)/(MPL ** 2), 10)

        ax1.axvline(downGamma,  color = 'black', linestyle = '--')
        ax1.axvline(upGamma,  color = 'black', linestyle = '--')
        ax1.fill_between([downGamma, upGamma], 0, 10000, color= 'DarkGreen', alpha = 0.4)
        ax1.set_ylim(0, max(lambds))

        ax2.axvline(downGamma,  color = 'black', linestyle = '--')
        ax2.axvline(upGamma,  color = 'black', linestyle = '--')
        ax2.fill_between([downGamma, upGamma], 0, 10000, color='DarkGreen', alpha = 0.4)
        ax2.set_ylim(0, 1.1)

        ax3.axvline(downGamma,  color = 'black', linestyle = '--')
        ax3.axvline(upGamma,  color = 'black', linestyle = '--')
        ax3.text(upGamma + 0.1, 0.05, r'$100 \times m_{\sigma}^3/M_{PL}^2$', fontsize = 12)
        ax3.text(downGamma + 0.1, 0.05, r'$0.1 \times  m_{\sigma}^3/M_{PL}^2$', fontsize = 12)
        ax3.fill_between([downGamma, upGamma], 0, 10000, color='DarkGreen', alpha = 0.4)
        ax3.set_ylim(0, 1.1)

        ax2.axhline(0.067, color = 'black')

        fig.savefig(plotpath +  "lamalpgam.png", format = 'png') 
 
        fig = p.figure()
        axes = []
        for i in range(3):
            lineAxes = []
            for j in range(1):
                lineAxes.append(p.subplot2grid((3, 3), (i, 0), colspan = 3))
            axes.append(lineAxes)
        axes[1][0].set_ylabel(r'$\log_{10}(\rho_i)$')
        axes[2][0].set_xlabel(r'$N$')
        for i in range(3):
            for j in range(1):
                if(j + 3 * i < len(gammas)):
                    axes[i][j].plot([N[0] for N in evolutions[(i * (len(gammas)/3)) * len(epsilons)]],
                                    [math.log(st[2]._b._dens[0], 10) for st in evolutions[(i * (len(gammas)/3)) * len(epsilons)]], 
                                    label = r'$\log_{10}(\rho_r)$', color = speciesColor[0])
                    axes[i][j].plot([N[0] for N in evolutions[(i * (len(gammas)/3)) * len(epsilons)]],
                                    [math.log(st[2]._b._dens[0], 10) for st in evolutions[(i * (len(gammas)/3)) * len(epsilons)]],
                                    label = r'$\log_{10}(\rho_{DR})$', color = speciesColor[2])
                    axes[i][j].plot([N[0] for N in evolutions[(i * (len(gammas)/3)) * len(epsilons)]],
                                    [math.log(fielddens(st[2]._b), 10) for st in evolutions[(i * (len(gammas)/3))  * len(epsilons)]],
                                    label = r'$\log_{10}(\rho_{\sigma})$', color = 'purple')
                    bott = math.log(fielddens(evolutions[0][-1][2]._b), 10)
                    axes[i][j].text(1, bott + 2.00, r'$\Gamma = \mathrm{' + str(gammas[i * (len(gammas)/3)]) + '} M_{PL}$')
                    axes[i][j].set_xlim(0, evolutions[0][-1][0])
                    axes[i][j].set_ylim(bottom = bott)
                     #axes[i][j].legend()
                     #axes[i][j].text()
        lg = axes[0][0].legend(loc = 0, prop = {'size': 10})
        lg.draw_frame(False)
        print "hahaha"
        fig.savefig(plotpath +  "rholamalpgam.png", format = 'png') 



def contMassSigma():
    global masses, sigmas, evolutions, epsilons, lam, file_number, speciesName
    if (len(evolutions) >= len(masses) * len(sigmas)):
        logmasses = [math.log(mass * MPLTEV, 10) for mass in masses]
        logsigmas = [math.log(sig, 10) for sig in sigmas]
        M, S = meshgrid(logmasses, logsigmas)
        print len(lam) 
        for q in range(1, len(evolutions[0][0][2]._b._dens)):
            lambdaMatrix = []
            for i in range(len(logmasses)):
                lambdaLines = []
                for j in range(len(logsigmas)):
                    lambdaLines.append(lam[((len(logsigmas) * i) + j) * len(gammas) * len(epsilons) + file_number])
                lambdaMatrix.append(lambdaLines)
        
            alphaMatrix = []
            for i in range(len(logmasses)):
                alphaLines = []
                for j in range(len(logsigmas)):
                    alphaLines.append(calculateAlpha(evolutions[((len(logsigmas) * i) + j) * len(gammas) * len(epsilons) + file_number], q))
                alphaMatrix.append(alphaLines)
                
            rMatrix = []
            for i in range(len(logmasses)):
                rLines = []
                for j in range(len(logsigmas)):
                    rLines.append(calculateR(evolutions[((len(logsigmas) * i) + j) * len(gammas) * len(epsilons) + file_number], q))
                rMatrix.append(alphaLines)


            fig = p.figure()
            im = p.imshow(lambdaMatrix, interpolation='bilinear', origin='lower',
                          cmap=cm.gray, extent=(logmasses[0], logmasses[-1], 
                                                logsigmas[0], logsigmas[-1]))
            levels = [0.5, 1.0, 5.0, 25.0]
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
            p.title(r'Contour of $\lambda$ dependence for $\Gamma/H = $' +
                    str(round(gammas[file_number], 3)))
            p.xlabel(r'$\log_{10}(m_{\sigma}[TeV])$')
            p.ylabel(r'$\log_{10}(\sigma(0)[M_{PL}])$')
            print file_number
            p.savefig(plotpath + speciesName[q] + 'contLambdaMassSigma'    
                      + str(file_number) + ".png", format = 'png') 


            fig = p.figure()
            im = p.imshow(rMatrix, interpolation='bilinear', origin='lower',
                          cmap=cm.gray, extent=(logmasses[0], logmasses[-1],
                                                logsigmas[0], logsigmas[-1]))
            levels = [0.0, 0.2, 0.4, 0.6, 0.8]
            CS = p.contour(rMatrix, levels, origin = 'lower',
                           linewidths = 2,
                           extent=(logmasses[0], logmasses[-1],
                                   logsigmas[0], logsigmas[-1]))
            p.clabel(CS, levels[1::2],  # label every second level            
                     inline=1,
                     fmt='%1.1f',
                     fontsize=14)
            CB = p.colorbar(CS, shrink=0.8, extend='both')
            CBI = p.colorbar(im, orientation='horizontal', shrink=0.8)
            p.title(r'Contour of $r_{' + speciesName[q] +  '}$ dependence for $\Gamma/H = $' + 
                    str(round(gammas[file_number], 3)))
            p.xlabel(r'$\log_{10}(m_{\sigma})[TeV]$')
            p.ylabel(r'$\log_{10}(\sigma(0))[M_{PL}]$')
            p.savefig(plotpath + speciesName[q] +'contrMassSigma' 
                      + str(file_number) + ".png" , format = 'png')


            fig = p.figure()
            im = p.imshow(alphaMatrix, interpolation='bilinear', origin='lower',
                          cmap=cm.gray, extent=(logmasses[0], logmasses[-1],
                                                logsigmas[0], logsigmas[-1]))
            levels = [0.0, 0.067, 0.1]
            CS = p.contour(alphaMatrix, levels, origin = 'lower',
                           linewidths = 2,
                           extent=(logmasses[0], logmasses[-1],
                                   logsigmas[0], logsigmas[-1]))
            p.clabel(CS, levels[1::2],  # label every second level                
                     inline=1,
                     fmt='%1.1f',
                     fontsize=14)
            CBI = p.colorbar(im, orientation='horizontal', shrink=0.8)
            p.title(r'Contour of $\alpha_{' + speciesName[q] +  '}$ dependence for $\Gamma/H = $' +
                    str(round(gammas[file_number], 3)))
            p.xlabel(r'$\log_{10}(m_{\sigma}[TeV])$')
            p.ylabel(r'$\log_{10}(\sigma(0)[M_{PL}])$')
            p.savefig(plotpath + speciesName[q] + 'contAlphaMassSigma' 
                      + str(file_number) + ".png", format = 'png')
        for q in range(1, len(evolutions[0][0][2]._b._dens)):
            for t in range(q + 1, len(evolutions[0][0][2]._b._dens)):
                rijMatrix = []
                for i in range(len(logmasses)):
                    rijLines = []
                    for j in range(len(logsigmas)):
                       rijLines.append(calculateRij(evolutions[((len(logsigmas) * i) + j) * len(gammas) 
                                                                   * len(epsilons) + file_number], q, t))
                    rijMatrix.append(rijLines)
                print '********rijMatrix***************'
                print rijMatrix
                fig = p.figure()
                im = p.imshow(rijMatrix, interpolation='bilinear', origin='lower',
                              cmap=cm.gray, extent=(logmasses[0], logmasses[-1],
                                                logsigmas[0], logsigmas[-1]))
                levels = [0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.34, 0.35, 0.36]
                CS = p.contour(rijMatrix, levels, origin = 'lower',
                               linewidths = 2,
                               extent=(logmasses[0], logmasses[-1],
                                       logsigmas[0], logsigmas[-1]))
                p.clabel(CS, levels[1::2],  # label every second level  
                         inline=1,
                         fmt='%1.1f',
                         fontsize=14)
                CB = p.colorbar(CS, shrink=0.8, extend='both')
                CBI = p.colorbar(im, orientation='horizontal', shrink=0.8)
                p.title(r'Contour of $r_{' + speciesName[q] + speciesName[t] + '}$ dependence for $\Gamma/H = $' +
                    str(round(gammas[file_number], 3)))
                p.xlabel(r'$\log_{10}(m_{\sigma}[TeV])$')
                p.ylabel(r'$\log_{10}(\sigma(0)[M_{PL}])$')
                p.savefig(plotpath + speciesName[q] + speciesName[t] + 'contrMassSigma'
                          + str(file_number) + ".png" , format = 'png')



def readInitVal(dir_number):
    global masses, sigmas, gammas, epsilons
    datafile = open('data/' + str(dir_number) + '/' + 'initValues.out', 'r')
    masses = eval(datafile.readline())
    sigmas = eval(datafile.readline())
    gammas = eval(datafile.readline())
    epsilons = eval(datafile.readline())
    coupl = eval(datafile.readline())
    datafile.close()



def readFile(dir_number, file_number):
    global evol, mass, sigmaInit, gammafinal, epsilon
    datafile = open('data/' + str(dir_number) + '/' + 'evolution' + str(file_number) + '.out', 'r')
    initValuesSim = eval(datafile.readline())
    if (mass != initValuesSim[0]):
        print "Problem with the mass..."
    if (sigmaInit != initValuesSim[1]):
        print "Problem with the sigma..."
    if (gammafinal != initValuesSim[2]):
        print "Problem with the gamma..."
    if (epsilon != initValuesSim[3]):
        print "Problem with epsilon..."
    ev = eval(datafile.readline())
    evol = []
    count = 0 
    for st in ev:
        background = Background(st[2][0][0], st[2][0][1], st[2][0][2], st[2][0][3])
        perturbation = []
        for i in range(len(st[2][1])):
            perturbation.append(Perturbation(st[2][1][i][0], st[2][1][i][1], st[2][1][i][2], st[2][1][i][3], st[2][1][i][4], st[2][1][i][5]))
        currentState = State(background, perturbation)
        curvMatrices = [asarray(curvMatrix) for 
                        curvMatrix in st[4]]
        if ((count < 100) | (count % 5 == 0)):
            evol.append((st[0], st[1], currentState, st[3], 
                         curvMatrices, st[5]))
        count += 1 


def normalizeEvolution():
    global epsilonAct, epsilon
    print "Modifying epsilon..."
    for st in evol:
        st[2]._p[1] =  p_product(st[2]._p[1], 1/sqrt(epsilon))
        st[3][1] = (st[3][1][0] / sqrt(epsilon), [st[3][1][1][i] / sqrt(epsilon) for i in range(len(st[3][1][1]))])
        for curv in st[3][1][1]: curv = curv/sqrt(epsilon)
        for i in range(1, len(st[4])): st[4][i] *= 1/epsilon
        for i in range(1, len(st[5])): st[5][i] *= 1/epsilon
    return evol


def modEvolution():
    global epsilonAct, epsilon
    first = evol[1]
    last = evol[-1]
    epsilon = last[5][0]/((first[5][0]/epsilonAct) - last[5][1])
    print epsilon 
    #if ((epsilon < 0)):
    #    print "Problem with epsilon..."
    #    return
    if (epsilon < 0): epsilon = 0
    eps.append(epsilon)
    for st in evol:
        st[2]._p[1] =  p_product(st[2]._p[1], sqrt(epsilon))
        st[3][1] = (st[3][1][0] * sqrt(epsilon), [st[3][1][1][i] * sqrt(epsilon) for i in range(len(st[3][1][1]))])
        for curv in st[3][1][1]: curv = curv * sqrt(epsilon)
        for i in range(1, len(st[4])): st[4][i] *= epsilon
        for i in range(1, len(st[5])): st[5][i] *= epsilon
    return evol


def plotHubbleGamma():
    global gammas
    fact = 2.4 * (1.e-9)
    for evol in evolutions:
        evol = normalizeEvolution()
    hubbleGamma = []
    allEpsilons = []
    hubbleGammaSL = []
    allEpsilonsSL = []
    for i in range(len(gammas) - 1):
         first = evolutions[i * len(masses)  * len(sigmas)][0]
         last = evolutions[i * len(masses) * len(sigmas)][-1]
         actEpsilon = findEpsilonMax(100.00, i)
         
         epsilon = last[5][0]/((first[5][0]/actEpsilon) - last[5][1])
                  
         hubbleGamma.append(math.log(sqrt(8 * (3.14 ** 2) * (MPL ** 2) * epsilon
                                 * fact * (1/101.0)) * MPLTEV, 10))
         allEpsilons.append(math.log(epsilon, 10))

         actEpsilon = findEpsilonMax(5.00, i)

         epsilon = last[5][0]/((first[5][0]/actEpsilon) - last[5][1])

         hubbleGammaSL.append(math.log(sqrt(8 * (3.14 ** 2) * (MPL ** 2) * epsilon
                                 * fact * (1/101.0)) * MPLTEV, 10))
         allEpsilonsSL.append(math.log(epsilon, 10))

    print hubbleGamma
    print hubbleGammaSL
    fig = p.figure()
    ax = fig.add_subplot(111)
    print len(gammas), len(hubbleGamma)
    p.plot([math.log(gammas[i], 10) for i in range(len(gammas) - 1)], hubbleGamma)
    p.text(math.log(gammas[0], 10), hubbleGamma[0], r'$\lambda = 100$')
    p.plot([math.log(gammas[i], 10) for i in range(len(gammas) - 1)], hubbleGammaSL)
    p.text(math.log(gammas[0], 10), hubbleGamma[0], r'$\lambda = 5$')
    A = array([[math.log(gammas[i], 10) for i in range(len(gammas) - 1)], ones(len(gammas) - 1)])

    w = linalg.lstsq(A.T, hubbleGamma)[0]
    wSL = linalg.lstsq(A.T, hubbleGammaSL)[0]
    
    gamm = [math.log(gammas[0], 10), math.log(gammas[-1], 10)]
    line = [(w[0] * g) + w[1] for g in gamm]
    lineSL = [(wSL[0] * g) + wSL[1] for g in gamm]
    print w, wSL
    print line, lineSL, gamm
    p.plot(gamm, line, 'k--')
    p.plot(gamm, lineSL, 'k--')
    ax.set_xlabel(r'$\log_{10}(\Gamma)$')
    ax.set_ylabel(r'$\log_{10}(H_*(\Gamma))$')
    p.title(r'Hubble parameter for constant $\lambda(\Gamma) = 5$ and $\lambda(\Gamma) = 100$')
    p.savefig(plotpath + str(file_number) + "hubbleGamma.png", format = 'png') 
    
    fig = p.figure()
    p.plot([math.log(gammas[i], 10) for i in range(len(gammas) - 1)], allEpsilons)
    p.text(math.log(gammas[0], 10), allEpsilons[0], r'$\lambda = 100$')
    p.plot([math.log(gammas[i], 10) for i in range(len(gammas) - 1)], allEpsilonsSL)
    p.text(math.log(gammas[0], 10), allEpsilonsSL[0], r'$\lambda = 5$')

    w = linalg.lstsq(A.T, allEpsilons)[0]
    wSL = linalg.lstsq(A.T, allEpsilonsSL)[0]

    p.axhline(y = log(0.1/16), linestyle = 'k--')

    p.xlabel(r'$\log_{10}(\Gamma)$')
    p.ylabel(r'$\log_{10}(\epsilon(\Gamma))$')
    p.title(r'Slow-roll parameter for constant $\lambda(\Gamma) = 100$')
    p.savefig(plotpath + str(file_number) + "epsilonGamma.png", format = 'png')
    

def findEpsilonMax(giveLambda, j):
    first = evolutions[j][0]                                                

    last = evolutions[j][-1]    
    return ((first[5][0] * (1 + giveLambda)) - last[5][0])/((1 + giveLambda) * last[5][1])


""" Program main."""
if __name__ == "__main__":
    dir_number = 374
    proc = 20          # Fraction of points plotted (integer)
    speciesName = ['r', 'DM', 'DR']
    speciesColor = ['DarkRed', 'blue', 'DarkOrange']
    datapath = ''
    statpath = ''
    plotpath = ''
    dirNumber(dir_number)
    MPL = 1.0
    MPLTEV = 2.43 * (10 ** 15)
    masses = []
    sigmas = []
    gammas = []
    epsilons = []
    eps = []
    coupl = []
    readInitVal(dir_number)
    evol = []
    lam = []
    file_number = 0
    H = 0
    evolutions = []
    for i in range(len(masses)):
        for j in range(len(sigmas)):
            for l in range(len(gammas)):
                for q in range(len(epsilons)):
                    m = masses[i]
                    sigmaInit = sigmas[j]
                    gammafinal = gammas[l]
                    epsilon  = epsilons[q]
                    print "Reading"
                    mass = [m] 
                    frac = 1.0e-17
                    coupl = [-frac + 1.0, frac]
                    initNumb = 2
                    lambd = [0.00, 0.00]
                    
                    readFile(dir_number, file_number)
                    evol = normalizeEvolution()
                    Hubble(evol[0][2]._b)
                    print H
                    gammas[l] = gammafinal * H

                    evolutions.append(evol)
                    file_number += 1
    print gammas
    file_number = 0
    element = argmax([ev[-1][5][1]/ev[0][5][0] for ev in evolutions])
    epsilonAct = findEpsilonMax(100.0, element)
    print "Actual epsilon: ", epsilonAct
    for m in masses:
        for sigmaInit in sigmas:
            for gammafinal in gammas:
                for epsilon in epsilons:
                    evol = evolutions[file_number]
                    evol = modEvolution()
                    if (evol !=None):
                        print "Plotting data..."
                        """
                        plotCurvCorrelation(evol)
                        
                        plotEvol(evol)
                        plotDensPert(evol)
                        plotPert(evol)
                        """
                        init = evol[0][5][0] + evol[0][5][1]
                        lam.append(calculateLambda(evol))
                    file_number += 1
                    
    print "Plotting all..."
    plotCurvatures()
    initCurvSpectrum()
    if ((len(masses) != 1) | (len(sigmas) != 1)):
        file_number = 0
        for i in range(len(gammas)):
            contMassSigma()
            file_number += 1
    plotLambdasGamma()        
    plotHubbleGamma()


