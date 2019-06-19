#! /usr/bin/env python3

import os
import sys
import glob
import time
import json
import colorsys
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from korali.plotter.helpers import plt_pause_light, plt_multicolored_lines

# Get a list of evenly spaced colors in HLS huse space.
# Credits: seaborn package
def hls_colors(num, h = 0.01, l=0.6, s=0.65):
    hues = np.linspace(0, 1, num + 1)[:-1]
    hues += h
    hues %= 1
    hues -= hues.astype(int)
    palette = [ list(colorsys.hls_to_rgb(h_i, l, s)) for h_i in hues ]
    return palette


# Get a list of strings for json keys of current results or best ever results
def objstrings(obj='current'):
    if obj == 'current':
        return ['CurrentBestFunctionValue', 'CurrentBestVector']
    elif obj == 'ever':
        return ['BestEverFunctionValue', 'BestEverVector']
    else:
        raise ValueError("obj must be 'current' or 'ever'")


# Plot DEA results (read from .json files)
def plot_dea(src, live = False, obj='current'):

    idx    = 0 # generation
    numdim = 0 # problem dimension
    names    = [] # description params
    colors   = [] # rgb colors
    numeval  = [] # number obj function evaluations
    dfval    = [] # abs diff currentBest - bestEver
    fval     = [] # best fval current generation
    fvalXvec = [] # location fval
    meanXvec = [] # location mean population
    width    = [] # spread population

    plt.style.use('seaborn-dark')

    fig, ax = plt.subplots(2,2,num='DEA live diagnostics: {0}'.format(src),figsize=(8,8))
    if live == True:
        fig.show()

    while( (live == False) or (plt.fignum_exists(fig.number)) ):

        path = '{0}/s{1}.json'.format(src, str(idx).zfill(5))
       
        if ( not os.path.isfile(path) ):
            if ( (live == True) and (idx > 0) ):
              plt_pause_light(0.5)
              continue
            else:
                break

        if live == True:
            plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                          fontweight='bold',\
                          fontsize=12 )

        with open(path) as f:
            data  = json.load(f)
            state = data['DE']['State']

            if idx == 0:
                numdim = len(data['Variables'])
                names  = [ data['Variables'][i]['Name'] for i in range(numdim) ]
                colors = hls_colors(numdim)
                for i in range(numdim):
                    fvalXvec.append([])
                    meanXvec.append([])
                    width.append([])

                idx = idx + 1
                continue

            numeval.append(state['EvaluationCount'])
            dfval.append(abs(state["CurrentBestFunctionValue"] - state["BestEverFunctionValue"]))
            
            fval.append(state[objstrings(obj)[0]])

            for i in range(numdim):
                fvalXvec[i].append(state[objstrings(obj)[1]][i])
                meanXvec[i].append(state['CurrentMeanVector'][i])
                width[i].append(state['MaxWidth'][i])

        if (live == False or idx < 2):
            idx = idx + 1
            continue

        draw_figure(fig, ax, src, idx, numeval, numdim, fval, dfval, fvalXvec, meanXvec, width, colors, names, live)
        idx = idx+1

    if live == False:
        draw_figure(fig, ax, src, idx, numeval, numdim, fval, dfval, fvalXvec, meanXvec, width, colors, names, live)
            
    fig.show()


# Create Plot from Data
def draw_figure(fig, ax, src, idx, numeval, numdim, fval, dfval, fvalXvec, meanXvec, width, colors, names, live):
    #fig, ax = plt.subplots(2,2,num='DEA live diagnostics: {0}'.format(src),figsize=(8,8))

    plt.suptitle( 'Generation {0}'.format(str(idx).zfill(5)),\
                      fontweight='bold',\
                      fontsize=12 )

    # Upper Left Plot
    ax[0,0].grid(True)
    ax[0,0].set_yscale('log')
    plt_multicolored_lines(ax[0,0], numeval, fval, 0.0, 'r', 'b', '$| F |$')
    ax[0,0].plot(numeval, dfval, 'x', color = '#34495e', label = '$| F - F_{best} |$')
    if ( (idx == 2) or (live == False) ):
        ax[0,0].legend(bbox_to_anchor=(0,1.00,1,0.2), loc="lower left", mode="expand", ncol = 3, handlelength=1, fontsize = 8)

    # Upper Right Plot
    ax[0,1].set_title('Objective Variables')
    ax[0,1].grid(True)
    for i in range(numdim):
        ax[0,1].plot(numeval, fvalXvec[i], color = colors[i], label=names[i])
    if ( (idx == 2) or (live == False) ):
        ax[0,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)

    # Lower Right Plot
    ax[1,0].set_title('Width Population')
    ax[1,0].grid(True)
    for i in range(numdim):
        ax[1,0].plot(numeval, width[i], color = colors[i])

    # Lower Left Plot
    ax[1,1].set_title('Mean Population')
    ax[1,1].grid(True)
    for i in range(numdim):
        ax[1,1].plot(numeval, meanXvec[i], color = colors[i], label=names[i])
    if ( (idx == 2) or (live == False) ):
        ax[1,1].legend(bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, handlelength=1)
    
    if (live == True):
        plt_pause_light(0.05)
    else:
        plt.pause(3600) #fix this (DW)


