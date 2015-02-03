# -*- coding: utf-8 -*-

""" Calculate some times of flight
of some noise and dispersive signals from UPV
Tomas Gomez, CSIC, t.gomez@csic.es, 2015 JAN 30"""

import os
import numpy as np
import matplotlib.pyplot as plt
from os import listdir

#Define the path where the folder with the signals is located
#path = 'c:\\users\\tomas\downloads\\valencia\\t3\\t3_1parte\\'
path = 'c:\\users\\tomas\downloads\\t3\\t3\\t3_1parte\\'
#path = 'i:\\valencia\\t3\\t3_1parte\\'

def crossCorr(D1a,D2b,shift = 500):
    """Function to calculate the delay between two signals using a cross-correlation algorithm.
    it takes two signals 1D np.array and and optional argument that the determines the amplitud shift
    in the cross correlation: integer (number of points).
    Returns an integer that is the delay (in number of points) between the two signals"""
    
    #Limito las señales al rango 10000-25000 que es donde aparece la señal de interes
    
    D1a = D1a[10000:25000]
    D2b = D2b[10000:25000]
    
    #Ventana del tramo de señal para la correlacion
    n0 = 6500
    nF = 10500
    
    ddv = np.where(D1a[n0:nF] == max(D1a[n0:nF]))
    dd = np.mean(ddv) + n0
       
    ddv2 = np.where(D2b[n0:nF] == max(D2b[n0:nF]))
    dd2 = np.mean(ddv2) + n0

    #Desplazamiento en numero de puntos para la correlacion
    n0p = dd - 1200
    nFp = dd + 1200
    
    n0p2 = dd2 - 1200
    nFp2 = dd2 + 1200
   
    cc = []
    
    for n in range(shift):
        cc.append(sum(D1a[n0p:nFp] * D2b[n0p2-shift+n:nFp2-shift+n]))
    
    for n in range(shift):
        cc.append(sum(D1a[n0p:nFp] * D2b[n0p2+n:nFp2+n]))

    return int(cc.index(max(cc)) - shift - (n0p - n0p2))

def maxDelay(DM1,DM2):
    """This function takes two signals (np.array 1D) and returns one integer that is the separation
    in number of points between the two maxima
    If there are more than one maxima (equal maximum value), the function returns the separation between the first two, 
    the last two, the crossed terms and the average the maximum and the minimum values"""
    
    #Ventana para la busqueda del maximo
    n0 = 16800
    nF = 20300

    ddv1 = np.where(DM1[n0:nF] == max(DM1[n0:nF])) 
    dd1 = np.mean(ddv1) + n0
    ddv2 = np.where(DM2[n0:nF] == max(DM2[n0:nF]))
    dd2 = np.mean(ddv2) + n0 
    
    dd1b = (ddv1[0][0]) + n0
    dd2b = (ddv2[0][0]) + n0
    dd1c = (ddv1[0][-1]) + n0
    dd2c = (ddv2[0][-1]) + n0   
    dd1d = (ddv1[0][0]) + n0
    dd2d = (ddv2[0][-1]) + n0
    dd1e = (ddv1[0][-1]) + n0
    dd2e = (ddv2[0][0]) + n0 
    tt = [dd2 - dd1, dd2b - dd1b, dd2c - dd1c, dd2d - dd1d, dd2e - dd1e]
           
    return np.mean(tt), np.max(tt), np.min(tt)
        
def analyzePeak(N):
    """This function takes the files in a given folder (and path) and calculates the delay
    from the shift in the maxium peak, see previous functions.
    the files in the folder must contain a column of floats with the signal amplitude
    Takes one argument N
    N = 1 Mean displacement between maxima
    N = 2 Maximum displacement between maxima
    N = 3 Minimum displacement between maxima"""

  
    lista = os.listdir(path)   # genero una lista con todos los nombres de fichero en la carpeta
    
    tempM = []
    data1 = np.loadtxt(path + lista[0])
    
    #Calculo delay por cross-corr y por maximo pico entre señales consecutivas
    for acqNo in range(len(lista)-1):

        data2 = np.loadtxt(path + lista[acqNo+1])
        tempM.append(maxDelay(data1,data2)[N-1]) 
        data1 = data2
    tempMAc = []
    
    #Calculo los tiempos acumulados
    for n in range(len(tempM)):
        tempMAc.append(sum(tempM[0:n]))

    return tempM, tempMAc # Devuelvo una lista de dos items: tiempo entre señales y tiempo acumulado
                                                
def analyze(shiftA):
    """This function takes the files in a given folder (and path) and calculates the cross correlation between them, see previous function.
    The files in the folder must contain a column of floats with the signal amplitude"""
    
    lista = os.listdir(path)   # genero una lista con todos los nombres de fichero en la carpeta
    temp = []
    data1 = np.loadtxt(path + lista[0])
    
    #Calculo delay por cross-corr y por maximo pico entre señales consecutivas
    
    for acqNo in range(len(lista)-1):

        data2 = np.loadtxt(path + lista[acqNo+1])
        temp.append(crossCorr(data1,data2,shiftA))
        data1 = data2
   
    tempAc = []
    
    #Calculo los tiempos acumulados

    for n in range(len(temp)):
        tempAc.append(sum(temp[0:n]))

    return temp, tempAc # Devuelvo una lista de dos items: tiempo entre señales (cross-corr) y tiempo acumulado
    
def analyzePhase(SampFreq = 1):
    """Function to calculate the phase shift between the signals in a given folder (and path),
    takes one argument, which is the Sampling frequency, (default value =1), in this casem the function returns 
    the number of points of delay instead of the time.
    Returns the phase shift and the time delay"""
    
    lista = os.listdir(path)   # genero una lista con todos los nombres de fichero en la carpeta
    tempF = []

    data1 = np.loadtxt(path + lista[0])
    
    #Limito el rango de señal para tomar la FFT a: 13000 22000 ptos
    
    data1_r = data1[13000:22000]
    ydata1_r = np.fft.fft(data1_r)
    # Tranducers bandwidth (frequency window) in number of points of the FFT
    F0 = 75
    FF = 95

    for acqNo in range(len(lista)-1):
 
        data2 = np.loadtxt(path + lista[acqNo+1])
        data2_r = data2[13000:22000]
        ydata2_r = np.fft.fft(data2_r)

        Fase = np.unwrap(np.angle(ydata1_r)-np.angle(ydata2_r))   # Calculo la diferencia de fase
        Fase = Fase[F0:FF]       # limito los resultados a la banda de frecuencia de los transductores
        Fase[0] = 0       # Introduzco una fase inicial de cero para activar la acción del unwrap
        Fase = np.unwrap(Fase)    # unwrap de fase
        FaseAVG = np.mean(Fase)   # tomo como diferencia de fase el valor medio de las diferencias de fase dentro de la banda de frecuencia de los trans

        tempF.append(FaseAVG)
        ydata1_r = ydata2_r
        
    # Calculo la fase acumulada
    tempFAc = []
    for n in range(len(tempF)):
        tempFAc.append(sum(tempF[0:n]))

    #Calculo la frecuencia para obtener los tiempos a partir de las fases
    frecStep = 1.0 / len(data1_r) * SampFreq
    frecCent = (FF + F0) /2.0 * frecStep
    time = np.array(tempFAc)/frecCent/2/np.pi
    return tempF, tempFAc, time  # devolvemos la diferencia de fase entre pares, la diferencia de fase acumulada y el tiempo (numero de puntos) acumulados

def analyze0(shiftB):
    """This function returns the time delay obtained from cross-correlation, between all signals respecto to the first one"""
    
    #Define the path where the folder with the signals is allocated
    #path = 'c:\\users\\tomas\downloads\\valencia\\t3\\t3_1parte\\'
    path = 'c:\\users\\tomas\downloads\\t3\\t3\\t3_1parte\\'
    #path = 'i:\\valencia\\t3\\t3_1parte\\'
    lista = os.listdir(path)
    data1 = np.loadtxt(path + lista[0])
    temp = []
    tempM = []
    
    for acqNo in range(len(lista)-1):
        
        if acqNo%2 != 0:
            shiftB += 1
        
        data2 = np.loadtxt(path + lista[acqNo+1])
        
        temp.append(crossCorr(data1,data2,shiftB))
        tempM.append(maxDelay(data1,data2))

    return temp, tempM
        
        
        
    
#path = 'c:\\users\\tomas\downloads\\valencia\\t3\\t3_1parte\\'
path = 'c:\\users\\tomas\downloads\\t3\\t3\\t3_1parte\\'
#path = 'i:\\valencia\\t3\\t3_1parte\\'
lista = os.listdir(path)
data = np.loadtxt(path+lista[0])
y = []
for i in range(len(data)): y.append(i)
data2 = data[14000:240000]
datafft = abs(np.fft.fft(data2))
y2 = []
for i in range(len(datafft)): y2.append(i)

#plt.plot(y2,datafft)
#plt.show()

data1 = np.loadtxt(path + lista[0])
data2 = np.loadtxt(path + lista[1])

#plt.plot(y,data1,y,data2)
#plt.show()

delay = crossCorr(data1,data2)
print delay

        
    
