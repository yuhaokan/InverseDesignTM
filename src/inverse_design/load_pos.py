import os
import numpy as np

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
position_dir = os.path.join(current_dir, "positions")


# To load the best position later:
def load_best_pos(best_pos_file_name, save_path=position_dir):
    best_pos_path = os.path.join(save_path, best_pos_file_name)
    if os.path.exists(best_pos_path):
        saved_dict = np.load(best_pos_path, allow_pickle=True).item()
        return saved_dict['best_pos'], saved_dict['best_error']
    return None, None

if __name__ == '__main__':
    print(load_best_pos(''))
