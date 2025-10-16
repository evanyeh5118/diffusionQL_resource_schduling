def getDatasetConfig(configIdx):
    if configIdx == 0:
        return {
            'N_user': 8,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 100,
            'sigmoid_k_list': [0.3],
            'sigmoid_s_list': [10.0],
            'sub_agents_idx': [[0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 1:
        return {
            'N_user': 20,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 200,
            'sigmoid_k_list': [0.3],
            'sigmoid_s_list': [10.0],
            'sub_agents_idx': [[0,0,0,0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7],[8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 2:
        return {
            'N_user': 8,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 100,
            'sigmoid_k_list': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sigmoid_s_list': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'sub_agents_idx': [[0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 3:
       return {
            'N_user': 20,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_fr',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 200,
            'sigmoid_k_list': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sigmoid_s_list': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'sub_agents_idx': [[0,0,0,0,0]],
            'user_map': [[0,1,2,3], [4,5,6,7],[8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 4:
        return {
                'N_user': 8,
                'LEN_window': 200,
                'N_aggregation': 4,
                'dataflow': 'thumb_bk',
                'randomSeed': 999,
                'r_bar': 5,
                'B': 100,
                'sigmoid_k_list': [0.3],
                'sigmoid_s_list': [10.0],
                'sub_agents_idx': [[1,1]],
                'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 5:
        return {
            'N_user': 20,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_bk',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 200,
            'sigmoid_k_list': [0.3],
            'sigmoid_s_list': [10.0],
            'sub_agents_idx': [[1,1,1,1,1]],
            'user_map': [[0,1,2,3], [4,5,6,7],[8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    elif configIdx == 6:
        return {
            'N_user': 8,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_bk',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 100,
            'sigmoid_k_list': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sigmoid_s_list': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'sub_agents_idx': [[1,1]],
            'user_map': [[0,1,2,3], [4,5,6,7]],
        }
    elif configIdx == 7:
       return {
            'N_user': 20,
            'LEN_window': 200,
            'N_aggregation': 4,
            'dataflow': 'thumb_bk',
            'randomSeed': 999,
            'r_bar': 5,
            'B': 200,
            'sigmoid_k_list': [0.1, 0.2, 0.3, 0.4, 0.5],
            'sigmoid_s_list': [10.0, 10.0, 10.0, 10.0, 10.0, 10.0],
            'sub_agents_idx': [[1,1,1,1,1]],
            'user_map': [[0,1,2,3], [4,5,6,7],[8,9,10,11], [12,13,14,15], [16,17,18,19]],
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizeDatasetConfig(simParams):
    print(f"{'='*50}")
    print(f"Dataset Configuration")
    print(f"{'='*50}")
    print(f"Number of Users:        {simParams['N_user']}")
    print(f"Window Length:          {simParams['LEN_window']}")
    print(f"N_aggregation:          {simParams['N_aggregation']}")
    print(f"Dataflow:               {simParams['dataflow']}")
    print(f"Random Seed:            {simParams['randomSeed']}")
    print(f"Resource Bar:           {simParams['r_bar']}")
    print(f"Bandwidth:              {simParams['B']}")
    print(f"Sigmoid K List:         {simParams['sigmoid_k_list']}")
    print(f"Sigmoid S List:         {simParams['sigmoid_s_list']}")
    print(f"Sub Agents:             {simParams['sub_agents_idx']}")
    print(f"User Map:               {simParams['user_map']}")
    print(f"{'='*50}")