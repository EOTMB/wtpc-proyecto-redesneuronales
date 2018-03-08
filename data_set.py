# -*- coding: utf-8 -*-
"""
Created on Tue Mar 06 14:44:36 2018

@author: Matias Garavaglia
"""
import numpy as np

def GenerarSRO(x,y):
    """Creación del array de ceros de dimensión "y"."""
    S_signal = np.zeros((y,), dtype=np.int)
    R_signal = np.zeros((y,), dtype=np.int)
    O_signal = np.zeros((y,), dtype=np.int)
    """Creación de un array que vaya de 10 en 10 para luego cambiar el array
    principal"""
    ventana = np.arange(0, y-x+1, x)
    """Loop central donde se toman los valores del array principal S_total y 
    R_total y según como sale el flip coin se cambian simultaneamente sus 
    valores, al mismo tiempo se crea el dataset Ouput, teniendo en cuenta 
    los valores de S y R"""
    for i in ventana:
        moneda = np.random.random_integers(1, high=3, size=1)
        if moneda == 1:
            S_signal[i:i+x] = 1
            R_signal[i:i+x] = 0
            O_signal[i:i+x] = 1
        elif moneda == 2:
            S_signal[i:i+x] = 0
            R_signal[i:i+x] = 1
            O_signal[i:i+x] = 0
        elif np.size(O_signal[i-x:i])> 0:
            O_signal[i:i+x] = O_signal[i-x:i]
        else:
            S_signal[i:i+x] = 0
            R_signal[i:i+x] = 0
            O_signal[i:i+x] = 0
    Resultado = np.array([S_signal,R_signal,O_signal])
    return Resultado
