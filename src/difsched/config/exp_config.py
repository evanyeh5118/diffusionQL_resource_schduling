def getExpConfig(configIdx):
    if configIdx == 0:
        return {
            'dataset_config_idx': 0,
            'env_config_idx': 0,
        }
    elif configIdx == 1:
        return {
            'dataset_config_idx': 1,
            'env_config_idx': 1,
        }
    elif configIdx == 2:
        return {
            'dataset_config_idx': 4,
            'env_config_idx': 2,
        }
    elif configIdx == 3:
        return {
            'dataset_config_idx': 5,
            'env_config_idx': 3,
        }
    else:
        raise ValueError(f"Invalid configIdx: {configIdx}")


def visualizeExpConfig(expParams): 
    print(f"dataset_config_idx: {expParams['dataset_config_idx']}")
    print(f"env_config_idx: {expParams['env_config_idx']}")