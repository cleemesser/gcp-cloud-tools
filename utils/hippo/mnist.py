from ..config import *

def mnist_test():

    return prod(
        [
            flag("dataset", ["mnist"]),
            flag("model", ["s4"]),
            flag("trainer.limit_train_batches", [0.1]),
        ]
    )

def mnist_test_128():

    return prod(
        [
            flag("dataset", ["mnist"]),
            flag("model", ["s4"]),
            flag("model.layer.d_state", [128]),
            flag("trainer.limit_train_batches", [0.1]),
        ]
    )

def mnist_transformer():
    return prod(
        [
            flag("train.seed", [0]),
            flag("pipeline", ["mnist"]),
            flag("model", ["transformer"]),
            flag("model.layer.0.n_heads", [8]),
            flag("model.d_model", [128]),
            flag("model.layer.0.causal", [False]),
            flag("model.n_layers", [2]),
            flag("decoder.mode", ['pool']),
            flag("model.prenorm", [True]),
            flag("optimizer.lr", [0.001, 0.0005]),
            flag("loader.batch_size", [50]),
        ]
    )

def mnist_performer():

    return prod(
        [
            flag("train.seed", [0]),
            flag("pipeline", ["mnist"]),
            flag("model", ["transformer"]),
            flag("+model/layer", ["performer"]),
            flag("model.d_model", [128]),
            flag("model.layer.0.causal", [False]),
            flag("model.n_layers", [2]),
            flag("decoder.mode", ['pool']),
            flag("model.prenorm", [True]),
            flag("optimizer.lr", [0.001, 0.0005]),
            flag("loader.batch_size", [50]),
        ]
    )

def hdmb_s4_first_sweep():

    return prod(
        [
            flag("experiment", ['s4-hmdb-something']),
            flag("trainer.max_epochs", [200]),
        ]
    )
