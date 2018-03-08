# -*- coding: utf-8 -*-
"""
Created on Thu Mar 08 13:55:08 2018

@author: Yo
"""
import numpy as np

def trainingSRO(x,y,a,b,c,d):
    """Creación del array de ceros de dimensión "y"."""
    S_signal = np.zeros((y,), dtype=np.int)
    R_signal = np.zeros((y,), dtype=np.int)
    O_signal = np.zeros((y,), dtype=np.int)
    entrada_array = np.zeros((a, b, c))
    salida_array = np.zeros((a, b, d))
    """Creación de un array que vaya de 10 en 10 para luego cambiar el array
    principal"""
    ventana = np.arange(0, y-x+1, x)
    """Loop central donde se toman los valores del array principal S_Signal y 
    R_Signal y según como sale el flip coin se cambian simultaneamente sus 
    valores, al mismo tiempo se crea el dataset O_Signal, teniendo en cuenta 
    los valores de S y R"""
    for i in ventana:
        moneda = np.random.random_integers(1, high=4, size=1)
        if moneda == 1:
            S_signal[i:i+x] = 1
            R_signal[i:i+x] = 0
            O_signal[i:i+x] = 1
        elif moneda == 2:
            S_signal[i:i+x] = 0
            R_signal[i:i+x] = 1
            O_signal[i:i+x] = 1
        elif moneda == 3:
            S_signal[i:i+x] = 0
            R_signal[i:i+x] = 0
            O_signal[i:i+x] = 0
        else:
            S_signal[i:i+x] = 1
            R_signal[i:i+x] = 1
            O_signal[i:i+x] = 0
    S = S_signal
    R = R_signal
    O = O_signal
    for i in range(b):
        set_seed = S[i * a : i * a + a]
        res_seed = R[i : i + a]
        entrada_array[:,i,0] = set_seed
        entrada_array[:,i,1] = res_seed
        out_seed = O[i : i + a]
        salida_array[:,i,0] = out_seed
        x_train = entrada_array
        y_train = salida_array
    return x_train, y_train