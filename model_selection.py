import numpy as np


def normalize(values):
    return (values - np.min(values))/(np.max(values)-np.min(values))


def non_zeros(model_weights):
    return sum(np.count_nonzero(layer_weights) for layer_weights in model_weights)


def get_rmse_ranks(all_models_mse):
    return normalize(np.sqrt(all_models_mse))


def get_sparsity_ranks(all_models_weights):
    all_models_non_zeros = [non_zeros(model_weights) for model_weights in all_models_weights]
    return normalize(all_models_non_zeros)


def get_ranks(all_models_weights, all_models_mse, all_models_extrapolation_mse=None, alpha=0.5, beta=0.5):

    # do nothing if we only have one model
    if len(all_models_weights) == 1:
        return [0, ]

    if all_models_extrapolation_mse is None:
        # without extrapolation points
        mse_ranks = get_rmse_ranks(all_models_mse)
        sparsity_ranks = get_sparsity_ranks(all_models_weights)
        ranks = alpha * np.square(mse_ranks) + beta * np.square(sparsity_ranks)
        return ranks
    else:
        # with extrapolation points
        mse_ranks = get_rmse_ranks(all_models_mse)
        extrapol_ranks = get_rmse_ranks(all_models_extrapolation_mse)
        ranks = alpha * np.square(mse_ranks) + beta * np.square(extrapol_ranks)
        return ranks
