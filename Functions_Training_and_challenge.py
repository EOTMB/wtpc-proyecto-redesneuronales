# -*- coding: utf-8 -*-
"""
Created on Wed Mar 07 11:45:12 2018

@author: Yo
"""
import numpy as np
"""
Creación de las funciones para generar tanto el training set como el challenge
set. Las variables necesarias para cada función son las siguientes:
x = longitud_de_la_ventana = 10
y = longitud_del_array_S_R_O = 100000 
a = largo_seed = 10000
b = cantidad_de_timesteps = 100
c = dim_imp = 2
d = dim_out = 1
"""
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

def ChallengeSR(x,y,a,b,c):
    """Creación del array de ceros de dimensión "y"."""
    S_signal = np.zeros((y,), dtype=np.int)
    R_signal = np.zeros((y,), dtype=np.int)
    entrada_array = np.zeros((a, b, c))
    """Creación de un array que vaya de "x" en "x" para luego cambiar el array
    principal"""
    ventana = np.arange(0, y-x+1, x)
    """Loop central donde se toman los valores del array principal S_Signal y 
    R_Signal y según como sale el flip coin se cambian simultaneamente sus 
    valores"""
    for i in ventana:
        moneda = np.random.random_integers(1, high=3, size=1)
        if moneda == 1:
            S_signal[i:i+x] = 1
            R_signal[i:i+x] = 0
        elif moneda == 2:
            S_signal[i:i+x] = 0
            R_signal[i:i+x] = 1
        else:
            S_signal[i:i+x] = 0
            R_signal[i:i+x] = 0
    S = S_signal
    R = R_signal
    for i in range(b):
        set_seed = S[i * a : i * a + a]
        res_seed = R[i : i + a]
        entrada_array[:,i,0] = set_seed
        entrada_array[:,i,1] = res_seed
        x_challenge = entrada_array
    return x_challenge