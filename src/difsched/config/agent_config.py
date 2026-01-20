def getSubAgentConfig(configIdx):
    if configIdx == 0:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'dataflow': 'thumb_fr',
            'N_aggregation': 4,
            'sigma_list': [0.9], # channel quality
            'N_kappa': 10,
            'kappa_range': (0.0, 1.0),
            'M_list': [4, 5, 6],
            'r_bar': 4,
            'B': 50,
            'randomSeed': 999,
        }
    elif configIdx == 1:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'dataflow': 'thumb_fr',
            'N_aggregation': 4,
            'sigma_list': [0.8], # channel quality
            'N_kappa': 10,
            'kappa_range': (0.0, 1.0),
            'M_list': [4, 5, 6],
            'r_bar': 4,
            'B': 50,
            'randomSeed': 999,
        }
    elif configIdx == 2:
        return {
            'EnvType': 'HYBRID',
            'N_user': 4,
            'LEN_window': 20,
            'dataflow': 'thumb_fr',
            'N_aggregation': 4,
            'sigma_list': [0.7], # channel quality
            'N_kappa': 10,
            'kappa_range': (0.0, 1.0),
            'M_list': [4, 5, 6],
            'r_bar': 4,
            'B': 50,
            'randomSeed': 999,
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")
    
def visualizeSubAgentConfig(simParams):
    print(f"{'='*50}")
    print(f"Environment Configuration")
    print(f"{'='*50}")
    print(f"Environment Type:       {simParams['EnvType']}")
    print(f"Number of Users:        {simParams['N_user']}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"N_aggregation:          {simParams['N_aggregation']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"Random Seed:            {simParams['randomSeed']}")
    print(f"Resource Bar:           {simParams['r_bar']}")
    
    if simParams['EnvType'] == 'SPS':
        print(f"N_r:                    {simParams['N_r']}")
        print(f"Resource Bar:           {simParams['r_bar']}")
        print(f"Sigmoid K List:         {simParams['sigmoid_k_list']}")
        print(f"Sigmoid S List:         {simParams['sigmoid_s_list']}")
    elif simParams['EnvType'] == 'HYBRID':
        print(f"Sigma List:             {simParams['sigma_list']}")
        print(f"N_kappa:                {simParams['N_kappa']}")
        print(f"Kappa Range:            {simParams['kappa_range']}")
        print(f"M List:                 {simParams['M_list']}")
    
    print(f"{'='*50}")