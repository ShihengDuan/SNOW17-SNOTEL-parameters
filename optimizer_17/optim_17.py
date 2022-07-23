import numpy as np
import spotpy
from optimizer import spot_setup as Setup
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import pickle
import os

all_params = {}
for n_station in range(219, 581):
    spot_setup = Setup(n_station=n_station, obj_func=spotpy.objectivefunctions.rmse)
    sampler = spotpy.algorithms.sceua(
        spot_setup, dbname='SCEUA_17_'+str(n_station), dbformat='csv')
    rep = 4000
    sampler.sample(rep, ngs=7, kstop=3, peps=0.1, pcento=0.1)
    results = spotpy.analyser.load_csv_results('SCEUA_17_'+str(n_station))
    '''fig = plt.figure(1, figsize=(9, 5))
    plt.plot(results['like1'])
    plt.show()
    plt.ylabel('RMSE')
    plt.xlabel('Iteration')
    fig.savefig('SCEUA_objectivefunctiontrace.png', dpi=300)'''

    bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results) # best run. 
    best_model_run = results[bestindex]
    params = spotpy.analyser.get_best_parameterset(results, maximize=False)
    # print(type(params), ' ', params.shape, ' ', type(params[0]), params[0])
    all_params[str(n_station)] = params[0]
    fields = [
        word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = list(best_model_run[fields])
    '''fig = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_simulation, color='black', linestyle='solid',
            label='Best objf.='+str(bestobjf))
    ax.plot(spot_setup.evaluation(), color='red', label='Observation data')
    plt.xlabel('Number of Observation Points')
    plt.ylabel('Discharge [l s-1]')
    plt.legend(loc='upper right')
    fig.savefig('SCEUA_best_modelrun.png', dpi=300)'''
    
    os.remove('single.txt')
    pred = np.array(best_simulation).flatten()
    real = spot_setup.evaluation()
    print(r2_score(spot_setup.evaluation(), best_simulation))
    with open('snow17_params_'+str(n_station)+'.pickle', 'wb') as handle:
        pickle.dump(params[0], handle, protocol=pickle.HIGHEST_PROTOCOL)
# Save best parameters
'''with open('snow17_params.pickle', 'wb') as handle:
    pickle.dump(all_params, handle, protocol=pickle.HIGHEST_PROTOCOL)'''
