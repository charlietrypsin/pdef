########################
# Personal Definitions #
########################

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import plotly
import plotly.graph_objs as go
import plotly.offline as offline
import seaborn as sns 
from scipy.signal import argrelmin, argrelmax


def percentage(input_list):
    # Percentage:
    # converts list into a percentage.
    # input_list = data to be converted to percentage
    float_list = []
    output_list = []
    for item in input_list:
        float_list.append(float(item))
    for item in float_list:
        output_list.append((100/max(float_list)*item))
    return output_list


def ccs_conv(data, pusher, wave_speed, z, mi, gas, A, N):
    # CCS Calculator:
    # Calculates CCS from either a list of scan bin or drift time.
    # data = pandas dataframe, with columns named 'Time' and 'Int'
    # pusher = pusher frequency
    # wave_speed = TWIMS wave velocity
    # z = charge of ion
    # mi = mass of ion
    # gas = drift-cell gas choice (currently N2 supported only)
    # A, N = coefficients from y = Ax^(N) calibration curve
    if np.isnan(data['Time'][0]) == True:
        print('Data appears to be CCS converted.')
        return

    else:
        if gas == 'N2':
            mn = (14.0067*2)
            print('Gas mass = \t' + str(mn))
        else:
            print('N2 not selected')

        mz = mi/z
        print('m/z = \t' + str(mz))

        t_offset = ((0.01*300)/wave_speed)*61 + ((0.01*300)/wave_speed)*31
        print('Offset time = \t' + str(t_offset))

        mu =((1/mn)-(1/mi))**(0.5)
        print('Reduced mass = \t' + str(mu))

        if data['Time'][0]==1:
            td = data['Time']*pusher
            print('Data format = \t' + '(scan)')
        else:
            td = data['Time']
            print('Data format = \t' +  '(td)')

        td_corrected = ((td - t_offset) - np.sqrt((mz/1000)*0.085))

        CCS = (td_corrected**N)*A*z*(mu)

        data['Time'] = CCS
        print('Data converted to CCS. \n')

        return data


def dataframe_wipe(df, column, x_min, x_max):
    # DataFrame Wiper:
    # removes rows from pandas.df above and below minimum and maximum values 
    # df = dataframe
    # column = column name
    # x_min = minimum value bound
    # x_max = maximum value bound
    value_list = []
    index_list = []
    for value in df[column]:
        if (x_min < value < x_max) == False:
            value_list.append(value)
    for value in value_list:
        index = df[df[column] == value].index[0]
        index_list.append(index)
    df = df.drop(labels = index_list, axis=0)
    return df


def offset(input_list,value):
    # Offset:
    # Offsets list by given value.
    # input_list = data to be altered
    # value = offset value
    output_list = []
    for item in input_list:
        output_list.append(float(item) + float(value))
    return output_list


def triplicate_plot(list1, list2, list3, name, state):
    # Triplicate Plotter:
    # Plots 3 repeats on the same graph, assuming data input is ['Time']['Int']
    # list1 = first data set
    # list2 = second data set
    # list3 = third data set
    # name = Experiment name
    # state = differentiator eg. charge state etc.
    sns.set_style('whitegrid')
    fig = plt.gca()
    plt.title(str(name) + ' ' + str(state), fontsize = 20)
    plt.ylim(0,110)
    plt.ylabel('Relative Intensity (%)', fontsize = 15)
    plt.xlabel('Drift Time (ms)', fontsize  = 15)
    plt.plot(list1['Time'], percentage(list1['Int']), label = '1')
    plt.plot(list2['Time'], percentage(list2['Int']), label = '2')
    plt.plot(list3['Time'], percentage(list3['Int']), label = '3')
    plt.legend()
    plt.savefig('triplicate_figures\\' + name + '_' + state + '_triplicate.png')
    plt.show()


def ciu_maker(int_list, volt_list, time_list, x_min, x_max, colour, name):
    # CIU-Maker:
    # Creates collision induced unfolding plot, using plotly.
    # int_list = list of intensity values
    # volt_list = list of strings of voltage values
    # time_list = drift-time list 
    # colour = colour palette for input
    # name = name of experiment for file output
    plotly.offline.init_notebook_mode()
    trace = go.Heatmap(z = int_list,
                       x = volt_list,
                       y = time_list,
                       zsmooth = 'best',
                       colorscale = colour,
                       colorbar = dict(title = 'Int',
                                       titleside = 'top',
                                       tickmode = 'array',
                                       tickvals = [0,50,100],
                                       ticks = 'outside'
                                    ),
                       transpose = True)

    layout = go.Layout(font = dict(size = 20),
                       autosize = False,
                       width = 750,
                       height = 750,

                       xaxis = dict(title = 'Voltage (V)',
                                    type = 'category',
                                    side = 'bottom',
                                    titlefont = dict(size = 25),
                                    tickfont = dict(size = 25)
                                   ),

                       yaxis = dict(title = 'Arrival Time (ms)',
                                    side = 'left',
                                    type = 'linear',
                                    tickvals = np.arange(x_min, x_max+10,10),
                                    titlefont = dict(size = 25),
                                    tickfont = dict(size = 25)
                                   )

                      )

    data=[trace]

    fig = go.Figure(data = data, layout = layout)
    offline.iplot(fig, filename = name, image = 'svg', image_height = 750, image_width = 750)
    # fig.write_image(name +)


def movingaverage(interval, window_size):
    # Moving Average:
    # Creates an average value from a window of values
    # interval = 
    # window_size = length of mean window
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')


def mean_smooth(data, window, smooth_no):
    # Mean Window Smoothing:
    # Performs iterative mean window smoothing on a single dataset
    # data = input list
    # window = window size for moving average
    # smooth_no = number of iterative smooths to be performed
    mean_smooth_counter = smooth_no
    data_output = []
    while (mean_smooth_counter >= 1):
        window_smooth = int(window)
        if len(data_output) == 0:
            data_array = np.array(data)
            data_smooth = movingaverage(data_array, window_smooth)
            mean_smooth_counter = mean_smooth_counter -1
            data_output = data_smooth
        else:
            data_smooth = movingaverage(data_output, window_smooth)
            mean_smooth_counter = mean_smooth_counter - 1
    return data_smooth


def peak_top(data, name, rel_max = False):
    # Peak Top Printing:
    # Finds the peak top values from ATD and prints the values, the plots the data 
    # with the maxima annotated.
    # data = input list as Data['Time']['Int']
    fig = plt.gca()
    plt.title(name)
    data = data.reset_index(drop = True) # to reset index if undergone dataframe_wipe
    max_int = max(data['Int'])
    index = data[data['Int']== max_int].index.values.astype(int)[0]
    maximum = data['Time'][index]
    plt.plot(data['Time'],data['Int'])
    if rel_max == False:
        print('Peak maximum = ' + str(maximum))
        fig.annotate(s = maximum, xy = (data['Time'][index],data['Int'][index]))
    if rel_max == True:
        print('Peak maximum = ' + str(maximum))
        indices = (argrelmax(np.array(data['Int'])))
        maxima = []
        print_str = 'Other maxima include: '
        for item in indices[0]:
            maxima.append(float(data['Time'][item]))
            print_str = print_str + str(data['Time'][item]) + ', '
        print(print_str)
        for txt,i in enumerate(indices[0]):
            fig.annotate(s = maxima[txt], xy = (data['Time'][i],data['Int'][i]))
    plt.savefig( name + '_peak_top.png')
    plt.savefig(name + '_peak_top.svg')
    plt.show()
    plt.clf()


def iwm(data):
    # Calculate intensity weighted mean
    # data = input list, format DataFrame['Time']['Int']
    it = []
    i = []
    data =data[np.isfinite(data['Time'])] # Remove NaN values.
    it.append(data['Time']*data['Int'])
    i.append(data['Int'])     
    sum_it = np.sum(it)
    sum_i = np.sum(i)
    IWM = sum_it/sum_i
    return IWM
    

def iwsd(data):
    # Calculate intensity weighted standard deviation
    # data = input list, format DataFrame['Time']['Int']
    data =data[np.isfinite(data['Time'])] # Remove NaN values.
    IWM = iwm(data)
    sd_i = []
    i = []
    sd_i.append((data['Int']*((data['Time'] - IWM)**2)))
    i.append(data['Int'])
    sum_sd_i = np.sum(sd_i)
    sum_i = np.sum(i)
    IWSD = np.sqrt(sum_sd_i/sum_i)
    return IWSD


def gradient_palette(colour,length):
    # Create a light to dark colour palette based on a given colour
    # colour = initial colour in rgb format
    # length = required length of gradient
    pre_colours = sns.dark_palette(colour, length+3, input ='rgb')
    colours = pre_colours[2:-1] # I find this stops the final colour being too light
    return colours

def stacked_im(data, x_min, x_max, colour, name, labels):
    #Creates a stacked IM plot with labelled axis
    # data = input list, format DataFrame['Time']['Int']
    # x_min = numerical value
    # x_max = numerical value
    # colour = list of rgb values
    # name = string of output filename
    # labels = list of strings for label values
    sns.set_style('white')
    fig = plt.gca()
    length = len(data)
    
    colours = itertools.cycle(colour)
    
    counter_list = []
    step = float(100./length)
    for n in range(0,6):
        counter_list.append(step*n)

    counter = itertools.cycle(counter_list)
    
    for item in data:
        plt.plot(item['Time'], item['Int']+counter.next(), c=colours.next(), lw=2)
    
    label_list = labels
    y_max = counter_list[-1] + 100 + step/2


    fig.set_ylim(0,y_max)
    for item in label_list:
        fig.text(x_max - 4, counter.next()+5,  item, fontsize=15)
            
    fig.set_xlabel('Drift Time (ms)', fontsize = 20)
    fig.set_xlim(x_min, x_max)
    fig.set_ylabel('Relative Intensity', fontsize = 20)
    fig.axes.get_yaxis().set_ticks([])
    fig.axes.tick_params(labelsize=15)
    
    plt.savefig(name + '.png')
    plt.savefig(name + '.svg')
    plt.show()