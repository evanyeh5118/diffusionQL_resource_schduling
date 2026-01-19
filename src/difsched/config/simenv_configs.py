def getEnvConfig(configIdx):
    if configIdx == 0:
        return {
            'N_user': 8,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 100,
            'sigma_list': [0.1, 0.5, 0.9],
            'sub_agents_idx': [[0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizeEnvConfig(simParams):
    print(f"{'='*50}")
    print(f"Environment Configuration")
    print(f"{'='*50}")
    print(f"Number of Users:        {simParams['N_user']}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"Sigma List:             {simParams['sigma_list']}")
    print(f"Resource Bar:           {simParams['r_bar']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"Sub Agents:             {simParams['sub_agents_idx']}")
    print(f"User Map:               {simParams['user_map']}")
    print(f"{'='*50}")