import os
import numpy as np

position_dir = "./positions"

# To load the best position later:
def load_best_pos(save_path=position_dir):
    best_pos_path = os.path.join(save_path, 'best_pos.npy')
    if os.path.exists(best_pos_path):
        saved_dict = np.load(best_pos_path, allow_pickle=True).item()
        return saved_dict['best_pos'], saved_dict['best_error']
    return None, None

if __name__ == '__main__':
    print(load_best_pos())
