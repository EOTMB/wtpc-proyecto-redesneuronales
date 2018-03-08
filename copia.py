import numpy as np

def atanasov(largo_seed, cantidad_de_timesteps):
    nturns           = cantidad_de_timesteps
    input_wait       = 3
    quiet_gap        = 4
    stim_dur         = 3
    var_delay_length = 0
    var_delay        = 0
    stim_noise       = 0
    sample_size      = largo_seed

    input_times  = np.zeros([sample_size, nturns],dtype=np.int)
    output_times = np.zeros([sample_size, nturns],dtype=np.int)
    
    turn_time = np.zeros(sample_size, dtype=np.int)

    for sample in np.arange(sample_size):
        turn_time[sample] =  stim_dur + quiet_gap# + var_delay[sample]
        for i in np.arange(nturns): 
            input_times[sample, i]  = input_wait + i * turn_time[sample]
            output_times[sample, i] = input_wait + i * turn_time[sample] + stim_dur

    seq_dur = int(max([output_times[sample, nturns-1] + quiet_gap, sample in np.arange(sample_size)]))
    
    x_train = np.zeros([sample_size, seq_dur, 2])
    y_train = 0.5 * np.ones([sample_size, seq_dur, 1])
    for sample in np.arange(sample_size):
        for turn in np.arange(nturns):
            firing_neuron = np.random.randint(2)                # 0 or 1
            x_train[sample, 
                    input_times[sample, turn]:(input_times[sample, turn] + stim_dur),
                    firing_neuron] = 1
            y_train[sample, 
                    output_times[sample, turn]:(input_times[sample, turn] + turn_time[sample]),
                    0] = firing_neuron
       
    mask = np.zeros((sample_size, seq_dur))
    for sample in np.arange(sample_size):
        mask[sample,:] = [0 if x == .5 else 1 for x in y_train[sample,:,:]]

    x_train = x_train + stim_noise * np.random.randn(sample_size, seq_dur, 2)
    #params['input_times']   = input_times
    #params['output_times']  = output_times
    return (x_train, y_train, mask)





