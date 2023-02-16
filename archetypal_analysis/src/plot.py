"""
Created on Jan 4th 2023
@author: May Levin
"""

import plotly.graph_objects as go
import plotly.express as px
from itertools import groupby
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from math import *

def run_plotting(outputPath, plotType, supplement=None, sorted = False, dataTitle = "", dpi_num = 200):
    data, suppDf = read_in(outputPath, supplement, sorted)
    print("got data with", len(data), "rows")
    if plotType == "bar_simple":
        bar_simple(data, suppDf, dataTitle, dpi_num)
    elif plotType == "bar_html":
        bar_html(data, suppDf, dataTitle)
    elif plotType == "plot_simplex":
        plot_simplex(data,suppDf, dataTitle = dataTitle)
    elif plotType == "plot_simplex_html":
        plot_simplex_html(data,suppDf, dataTitle = dataTitle)
    elif plotType == "bar_labeled":
        if suppDf is None:
            print("need supplement for labeled plot")
        else:
            bar_labeled(data, suppDf, dataTitle, dpi_num)
    else:
         print(f"{plotType} does not exist")
   


"""
This function lets you read in the archetypal analysis result produced, and any supplementary data.

Parameters:
-----------
outputPath:
    string path where the archetypal analysis result was saved
supplement = none:
    string path for any supplementary description data. There should no column name or index and values are separated by spaces
sorted = false:
    boolean, whether you want the data to be sorted by maximum archetype component

Output:
    A panda dataframe of the archetypal analysis results, a supplementary panda dataframe
-----------
"""


def read_in(outputPath, supplement=None, sorted=False):

    data = pd.read_csv(f'{outputPath}', header=None, sep= " ")

    if sorted and supplement is not None:
        supplementDF = pd.read_csv(
            f'{supplement}', header=None, index_col=None, sep = " ")
        supplementDF = supplementDF.add_prefix("col_")
        df = pd.concat((data, supplementDF), axis=1)
        df['Max'] = df.drop(supplementDF.columns, axis=1).idxmax(axis=1)
        df = df.sort_values(by='Max')
        df.drop('Max', axis=1, inplace=True)
        return df.iloc[:, 0:data.shape[1]], df.iloc[:, data.shape[1]:]

    elif sorted:
        data['Max'] = data.idxmax(axis=1)
        data = data.sort_values(by='Max')
        data.drop('Max', axis=1, inplace=True)

    elif supplement is not None:
        supplementDF = pd.read_csv(
            f'{supplement}', header=None,  index_col=None, sep = " ")
        supplementDF = supplementDF.add_prefix("col_")
        return data, supplementDF

    return data, None


"""
Creating a simple matplotlib bar plot and saving it.

Parameters:
-----------
data:
    pandas dataframe, read in archetypal analysis results
supplement = []:
    pandas dataframe, read in supplemental information. The first column becomes the axis columns
dataTitle = "":
    str, title for plot
dpi_num = 200:
    int, quality of saved plot graphic
"""


def bar_simple(data, supplement= None, dataTitle="", dpi_num=200):
 
    fig, ax = plt.subplots()
    k = data.shape[1]

    if supplement is not None:
        data = data.copy()
        data.index = supplement.iloc[:, 0]    

    
    if len(data) > 500:
        data.plot(kind='bar', stacked=True, width=1, ax = ax)
        ax.set_xticks([])
    else: 
        data.plot(kind='bar', stacked=True, width=.8, ax = ax)
        plt.xticks(fontsize=10, rotation=90)
    
    plt.title(f'{dataTitle} k={k}')
    plt.savefig(f'aa.{dataTitle}_k{k}.jpg', dpi=dpi_num, bbox_inches='tight')
    #plt.show()


"""
Creating an html bar plot and saving it.

Parameters:
-----------
data:
    pandas dataframe, read in archetypal analysis results
supplement = []:
    pandas dataframe, read in supplemental information. The whole dataframe will be used as hover information
dataTitle = "":
    str, title for plot
"""


def bar_html(data, supplement=None, dataTitle=""):
    k = data.shape[1]
    df = data

    if supplement is not None:
        df = pd.concat((data, supplement), axis=1)

        fig = px.bar(data_frame=df, x=data.index, y=list(range(k)), hover_data=supplement.columns,
                 color_discrete_sequence=px.colors.qualitative.Alphabet,
                 title=f"{dataTitle} k = {k}")

    else:
        fig = px.bar(data_frame=df, x=data.index, y=list(range(k)),
                 color_discrete_sequence=px.colors.qualitative.Alphabet,
                 title=f"{dataTitle} k = {k}")
    
    fig.update_layout(
        xaxis_type='category'
    )

    # fig.update_layout(
    #    xaxis_tickfont_size=5,
    #    bargap=0.1,  # gap between bars of adjacent location coordinates.)
    fig.update_traces(dict(marker_line_width=0))

    fig.write_html(f'aa.{dataTitle}_k{k}_interactive.html')
    fig.show()


"""
Creating a simplex html graph and saving it. Supplemental dataframe can provide coloring based on the first column.
This was taken from Benyamin Motevalli code in the original archetypal analysis package code.

Parameters:
-----------
alfa:
    pandas dataframe, read in archetypal analysis results
supplement = []:
    pandas dataframe, read in supplemental information. 
plot_args

grid_on = true

dataTitle = "":
    str, title for plot
"""


def plot_simplex_html(alfa, supplement=None, plot_args={}, grid_on=True, dataTitle=""):
    """
    # group_color = None, color = None, marker = None, size = None
    group_color:    

        Dimension:      n_data x 1

        Description:    Contains the category of data point.
    """
    alfa = alfa.T
    archetypeNums = alfa.shape[0]

    labels = ('A'+str(i + 1) for i in range(archetypeNums))
    rotate_labels = True
    label_offset = 0.10
    data = alfa.T
    scaling = False
    sides = archetypeNums

    basis = np.array(
        [
            [
                np.cos(2*_*pi/sides + 90*pi/180),
                np.sin(2*_*pi/sides + 90*pi/180)
            ]
            for _ in range(sides)
        ]
    )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T, basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data, basis)

    #fig = plt.figure(figsize=(10,10))
    #ax = fig.add_subplot(111)

    for i, l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i, 0]
        y = basis[i, 1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = (angle + 180) % 360  # mod(angle + 180,360)
        else:
            angle = 0

    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False
#
            if not (ignore):
                lst_ax_0.append(basis[i, 0] + [0, ])
                lst_ax_1.append(basis[i, 1] + [0, ])
                lst_ax_0.append(basis[j, 0] + [0, ])
                lst_ax_1.append(basis[j, 1] + [0, ])

    # ax.plot(lst_ax_0,lst_ax_1, color='#FFFFFF',linewidth=1, alpha = 0.5, zorder=1)

    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    for _ in range(sides):
        lst_ax_0.append(basis[_, 0] + [0, ])
        lst_ax_1.append(basis[_, 1] + [0, ])

    lst_ax_0.append(basis[0, 0] + [0, ])
    lst_ax_1.append(basis[0, 1] + [0, ])

    dfX = pd.DataFrame(lst_ax_0, columns=['x'])
    dfY = pd.DataFrame(lst_ax_1, columns=['y'])
    df = pd.concat(objs=[dfX, dfY], axis=1)

    # ax.plot(lst_ax_0,lst_ax_1,linewidth=1, zorder=2) #, **edge_args )

    if supplement is not None:
        supArray = []
        for col in supplement.columns:
            supArray.append(supplement[col])

        fig1 = px.scatter(x=newdata[:, 0], y=newdata[:, 1], color=supplement.iloc[:, 0],
                          hover_data=supArray
                          )

    else:
        fig1 = px.scatter(x=newdata[:, 0], y=newdata[:, 1])

    fig2 = px.line(df, x="x", y="y")
    fig3 = go.Figure(data=fig1.data + fig2.data)
    fig3.update_layout(title=f'{dataTitle} K = {archetypeNums}')
    fig3.update_yaxes(scaleanchor="x", scaleratio=1)

    fig3.write_html(
        f"aa.simplex_k_{archetypeNums}_{dataTitle}_interactive_admix.html")

    fig3.show()


"""
Creating a simplex graph and saving it. Supplemental dataframe can provide coloring based on the first column.
This was taken from Benyamin Motevalli code in the original archetypal analysis package code.

Parameters:
-----------
alfa:
    pandas dataframe, read in archetypal analysis results
supplement = []:
    pandas dataframe, read in supplemental information. 
plot_args

grid_on = true

dataTitle = "":
    str, title for plot
"""


def plot_simplex(alfa,  supplement=None, plot_args={}, grid_on=True, dataTitle=""):
    """
    # group_color = None, color = None, marker = None, size = None
    group_color:    

        Dimension:      n_data x 1

        Description:    Contains the category of data point.
    """
    alfa = alfa.T
    archetypeNum = alfa.shape[0]

    labels = ('A'+str(i + 1) for i in range(archetypeNum))
    rotate_labels = True
    label_offset = 0.10
    data = alfa.T
    scaling = False
    sides = archetypeNum

    basis = np.array(
        [
            [
                np.cos(2*_*pi/sides + 90*pi/180),
                np.sin(2*_*pi/sides + 90*pi/180)
            ]
            for _ in range(sides)
        ]
    )

    # If data is Nxsides, newdata is Nx2.
    if scaling:
        # Scales data for you.
        newdata = np.dot((data.T / data.sum(-1)).T, basis)
    else:
        # Assumes data already sums to 1.
        newdata = np.dot(data, basis)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)

    for i, l in enumerate(labels):
        if i >= sides:
            break
        x = basis[i, 0]
        y = basis[i, 1]
        if rotate_labels:
            angle = 180*np.arctan(y/x)/pi + 90
            if angle > 90 and angle <= 270:
                angle = (angle + 180) % 360  # mod(angle + 180,360)
        else:
            angle = 0
        ax.text(
            x*(1 + label_offset),
            y*(1 + label_offset),
            l,
            horizontalalignment='center',
            verticalalignment='center',
            rotation=angle
        )

    # Clear normal matplotlib axes graphics.
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_frame_on(False)

    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    ignore = False
    for i in range(sides):
        for j in range(i + 2, sides):
            if (i == 0 & j == sides):
                ignore = True
            else:
                ignore = False
            if not (ignore):
                lst_ax_0.append(basis[i, 0] + [0, ])
                lst_ax_1.append(basis[i, 1] + [0, ])
                lst_ax_0.append(basis[j, 0] + [0, ])
                lst_ax_1.append(basis[j, 1] + [0, ])

    ax.plot(lst_ax_0, lst_ax_1, color='#FFFFFF',
            linewidth=1, alpha=0.5, zorder=1)

    # Plot border
    lst_ax_0 = []
    lst_ax_1 = []
    for _ in range(sides):
        lst_ax_0.append(basis[_, 0] + [0, ])
        lst_ax_1.append(basis[_, 1] + [0, ])

    lst_ax_0.append(basis[0, 0] + [0, ])
    lst_ax_1.append(basis[0, 1] + [0, ])

    ax.plot(lst_ax_0, lst_ax_1, linewidth=1, zorder=2)  # , **edge_args )

    if supplement is not None:
        sc = ax.scatter(newdata[:, 0], newdata[:, 1],  zorder=3, alpha=0.5,
                        c=supplement.iloc[:, 0].factorize()[0], cmap='Spectral', label=supplement.iloc[:, 0].unique
                        )

        plt.legend(sc.legend_elements(num=len(supplement.iloc[:, 0].unique()))[0], list(supplement.iloc[:, 0].unique()), loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   fancybox=True, ncol=10)

    elif supplement is None:
        ax.scatter(newdata[:, 0], newdata[:, 1],
                   color='red', zorder=3, alpha=0.5)

    else:
        if ('marker' in plot_args):
            marker_vals = plot_args['marker'].values
            marker_unq = np.unique(marker_vals)

            for marker in marker_unq:
                row_idx = np.where(marker_vals == marker)
                tmp_arg = {}
                for keys in plot_args:
                    if (keys != 'marker'):
                        tmp_arg[keys] = plot_args[keys].values[row_idx]

                ax.scatter(newdata[row_idx, 0], newdata[row_idx, 1],
                           **tmp_arg, marker=marker, alpha=0.5, zorder=3)
        else:
            ax.scatter(newdata[:, 0], newdata[:, 1], **plot_args,
                       color='pink', marker='s', zorder=3, alpha=0.5)

    plt.title(f'{dataTitle}, K = {archetypeNum}', fontsize=15)

    plt.savefig(f"aa.simplex_k_{archetypeNum}_{dataTitle}.jpg",
                dpi=500, bbox_inches='tight')


def add_line(ax, xpos, ypos):
    line = plt.Line2D([xpos, xpos], [ypos + .1, ypos],
                      transform=ax.transAxes, color='black', linewidth=3)
    line.set_clip_on(False)
    ax.add_line(line)


def label_len(my_index, level):
    labels = my_index.get_level_values(level)
    return [(k, sum(1 for i in g)) for k, g in groupby(labels)]


def label_group_bar_table(ax, df):
    ypos = -.4
    scale = 1. / df.index.size
    for level in range(df.index.nlevels - 1)[::-1]:
        pos = 0
        for label, rpos in label_len(df.index, level):
            lxpos = (pos + .5 * rpos) * scale

            if (rpos < 5):
                ax.text(lxpos, ypos + .05, label,  transform=ax.transAxes, rotation="vertical",
                        ha='center', va="top")

            else:
                ax.text(lxpos, ypos + .05, label,
                        transform=ax.transAxes, ha='center', va="top")
            add_line(ax, pos * scale, ypos)
            pos += rpos
        add_line(ax, pos * scale, ypos)
        ypos -= .2


"""
Creating a labeled bar plot and saving it. Supplemental dataframe will be used with each column providing a label for the plot.

Parameters:
-----------
data:
    pandas dataframe, read in archetypal analysis results
supplement = []:
    pandas dataframe, read in supplemental information. 
dataTitle = "":
    str, title for plot
dpi_num = 200:
    int, quality of saved plot graphic
"""


def bar_labeled(data, supplement, dataTitle="", dpi_num=200):
    k = data.shape[1]
    levels = supplement.shape[1] - 1
    fig = plt.figure()
    ax = fig.add_subplot(111)
    supplement = supplement.add_prefix("col_")

    data.plot(ax=fig.gca(), figsize=(20, 10), kind='bar',
              stacked=True, width=.9, title=f"{dataTitle} k = {k}")

    df = pd.concat((data, supplement), axis=1)

    df = df.groupby(list(supplement.columns)).sum()

    ax.set_xticklabels(df.index.get_level_values(levels), rotation=90)
    labels = ['' for item in ax.get_xticklabels()]

    ax.set_xlabel('')

    label_group_bar_table(ax, df)
    fig.subplots_adjust(bottom=.1 * df.index.nlevels)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.savefig(f'aa.{dataTitle}_k{k}_sorted.jpg',
                dpi=dpi_num, bbox_inches='tight')
    # plt.show()
