from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

def visualizeModel(model):
    """ Visualizes the structure of a model.
    
    Arguments:
        model: A Keras model
    
    Returns:
        Inline SVG
    """
    return SVG(model_to_dot(model, show_shapes=True).create(prog='dot', format='svg'))