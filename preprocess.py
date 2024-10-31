import numpy as np

def preprocess_data(in_data, feature_range=(0,1), copy=True, reshape_dim=None):
    """
    Perform feature scaling on input dataset. 
    The input dataset is transformed to lie in the `feature_range` requested.
    
    Inputs:
    in_data: (ndarray) Input dataset array.
    feature_range: (tuple(min,max), default=(0,1)) Desired range of transformed data.
    copy: (bool, default=True) set to False to perform inplace row normalization and avoid a copy (if the input is already a numpy array).
    reshape_dim: (tuple, default=None) Reshape the feature scaled dataset.
    
    Output:
    out_data: (ndarray) Scaled dataset.
    """
    
    if reshape_dim is not None:
        reshape_dim = reshape_dim
    else:
        reshape_dim = in_data.shape
        
    from sklearn.preprocessing import MinMaxScaler
    
    scaler = MinMaxScaler(feature_range=feature_range, copy=copy)
    out_data = scaler.fit_transform(in_data)
    out_data = np.reshape(out_data, reshape_dim)
    
    return out_data


def preprocess_images(in_images, scale_type="minmax", reshape_dim=None):
    """
    Perform feature scaling on input images
    
    Input:
    in_images: (ndarray) Input images
    scale_type: (str, default=minmax) Type of feature scaling to perform. (Options: minmax, mean)
    reshape_dim: (tuple, default=None) Reshape the feature scaled dataset.
    
    Output:
    out_images: (ndarray) Preprocessed images
    """
    
    if reshape_dim is not None:
        reshape_dim = reshape_dim
    else:
        reshape_dim = in_images.shape
    
    if len(in_images.shape) > 3:
        squeezed_images = np.squeeze(in_images, axis=-1)
   
    shift_value = 0.0
    range_value = 1.0
    
    if scale_type == "minmax":
        shift_value = squeezed_images.min(axis=0) # minimum across all images
        range_value = squeezed_images.max(axis=0) - squeezed_images.min(axis=0)
    
    elif scale_type == "standardize":
        shift_value = squeezed_images.mean(axis=0)
        range_value = squeezed_images.std(axis=0)
    elif scale_type == "mean":
        shift_value = squeezed_images.mean(axis=0)
        range_value = squeezed_images.max(axis=0) - squeezed_images.min(axis=0)
    else:
        print("Unknown scale_type option")
    
    scaled_images = (squeezed_images - shift_value) / (range_value)
    
    out_images = np.reshape(scaled_images, reshape_dim)
    
    return out_images