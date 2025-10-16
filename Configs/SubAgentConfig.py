def getSubAgentConfig(configIdx):
    if configIdx == 0:
        return {
            'N_user': 4,
            'LEN_window': 200,
            'dataflow': 'thumb_fr',
            'r_bar': 5,
            'B': 60,
            'sigmoid_k_list': [0.3],
            'sigmoid_s_list': [10.0],
            'randomSeed': 999,
            'N_aggregation': 4,
            'N_r': 5,  
        }
    elif configIdx == 1:
        return {
            'N_user': 4,
            'LEN_window': 200,
            'dataflow': 'thumb_bk',
            'r_bar': 5,
            'B': 60,
            'sigmoid_k_list': [0.3],
            'sigmoid_s_list': [10.0],
            'randomSeed': 999,
            'N_aggregation': 4,
            'N_r': 5,  
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")
    

def visualizeSubAgentConfig(simParams):
    print(f"{'='*50}")
    print(f"Environment Configuration")
    print(f"{'='*50}")
    print(f"Number of Users:        {simParams['N_user']}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"N_aggregation:          {simParams['N_aggregation']}")
    print(f"N_r:                    {simParams['N_r']}")
    print(f"Resource Bar:           {simParams['r_bar']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"Sigmoid K List:         {simParams['sigmoid_k_list']}")
    print(f"Sigmoid S List:         {simParams['sigmoid_s_list']}")
    print(f"Random Seed:            {simParams['randomSeed']}")
    print(f"{'='*50}")