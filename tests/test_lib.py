# tests for functions in lib.py

from toolbox import lib

def test_number_of_layers():
    model = lib.init_binary_model()
    assert(len(model.layers) == 3)
