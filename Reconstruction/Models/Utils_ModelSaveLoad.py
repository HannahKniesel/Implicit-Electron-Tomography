import torch

# Utils to save and load model parameters

def load_param(default_val, name, ckpt):
    try:
        variable = ckpt[name]
    except: 
        print("Did not load variable "+str(name)+" from checkpoint.")
        variable = default_val
    return variable

def save_dict(dict_val, path):
    try:
        torch.save(dict_val, path)
        return True
    except: 
        return False

def set_param(val, name, dict_save):
    try:
        dict_save[name] = val
    except: 
        print("WARNING:: Did not save parameter: "+str(name))
    return

