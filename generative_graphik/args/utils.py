import argparse

def str2inttuple(v):
    return tuple([int(item) for item in v.split(',')] if v else [])

def str2floattuple(v):
    return tuple([float(item) for item in v.split(',')] if v else [])

def str2tuple(v):
    return tuple([item for item in v.split(',')] if v else [])

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
