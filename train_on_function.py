import tensorflow as tf

from tqdm import tqdm
import os

import numpy as np
import matplotlib.pyplot as plt

from EQL_div.EQL_div import EQL_div_network
from EQL_div.inputs_needed import inputs_needed
from EQL_div.formula_writer import get_formulas


def train(input_length, funcs, target_fn, T, t_1, t_2, batchsize, train_low, train_high, eval_low, eval_high,
          l1_reg, l0_threshold, penalty_strength, eval_bound, expected_param_range=3., formula_info=False,
          plot_prelim=False):
    # trains for all three phases, using the given parameters
    # returns the trained model and a list of losses after each training step

    costs = np.zeros(T)

    def train_phase(phase, t_begin, t_end):

        if formula_info:
            pbar = range(t_begin, t_end)
        else:
            pbar = tqdm(range(t_begin, t_end))
            pbar.set_description("Phase {} training".format(phase))
        for i in pbar:

            lo = train_low
            hi = train_high

            # inject penalty epochs every 50 steps
            if i % 50 == 0:
                model.set_penalty_epoch(True)
                if phase == 2:
                    model.set_l1_reg(0.0)
                lo = eval_low
                hi = eval_high

            xs_batch = np.random.uniform(low=lo, high=hi, size=(batchsize, input_length))
            ys_batch = target_fn(xs_batch)
            model.train_on_batch(xs_batch, ys_batch)
            # costs should always show the mse, without regularization or penalty terms
            costs[i] = tf.reduce_mean(tf.math.squared_difference(ys_batch, model.predict_on_batch(xs_batch)))

            # disable penalty epoch again after it occurred
            if i % 50 == 0:
                model.set_penalty_epoch(False)
                if phase == 2:
                    model.set_l1_reg(l1_reg)

            if i % 50 == 0:
                if formula_info:
                    # delete old formula:
                    os.system('clear')
                    # write some info about how far we are with training
                    print("Currently training in phase {}, and we're at step {} of {}".format(phase,
                                                                                              i - t_begin,
                                                                                              t_end - t_begin))
                    # write new formula
                    formulas = get_formulas(model.trainable_weights, funcs, simplify=True)
                    for j in range(output_length):
                        formulas[j] = "f_{} = ".format(j) + formulas[j] + "\n"
                        print(formulas[j])

                if plot_prelim:
                    # plot graphs
                    plot_prelims(model=model,
                                 output_length=output_length,
                                 eval_low=eval_low,
                                 eval_high=eval_high,
                                 axs=axs,
                                 pred_artists=pred_artists)

    # for plotting the function as we go
    output_length = len(funcs[-1])
    fig = plt.figure(1)
    axs = list()
    pred_artists = list()
    if plot_prelim:
        plt.ion()
        xx = list()
        for i in range(input_length):
            xx.append(np.linspace(eval_low[i], eval_high[i], 1000))
        xx = np.array(xx).transpose()
        y_true = target_fn(xx)

        cols = int(np.ceil(np.sqrt(output_length)))
        rows = int(np.ceil(output_length / cols))

        y_true = np.reshape(y_true, [-1, output_length])

        for i in range(output_length):
            axs.append(fig.add_subplot(rows, cols, i + 1))
            axs[i].set_title("Output node {}".format(i))
            lo = np.min(y_true[:, i]) - 0.1 * (np.abs(np.min(y_true[:, i])) + 1.)
            hi = np.max(y_true[:, i]) + 0.1 * (np.abs(np.max(y_true[:, i])) + 1.)
            axs[i].plot([train_low[i], train_low[i]], [lo, hi], 'g', linestyle='dashed')
            axs[i].plot([train_high[i], train_high[i]], [lo, hi], 'g', linestyle='dashed')
            axs[i].set_ylim([lo, hi])
            axs[i].plot(xx[:, i], y_true[:, i], 'r-', label='$f$ true')
            pred_artists.extend(
                axs[i].plot(xx[:, i], (lo - 2) * np.ones(1000), 'b', linestyle='dashed', label='$f$ pred'))
            axs[i].legend(loc='upper right')
        fig.suptitle("Phase 1")
        plt.show()
        plt.pause(.001)

    # phase 1 training
    model = EQL_div_network(funcs, 0., 0., penalty_strength, eval_bound, expected_param_range)
    model.compile(optimizer='adam', loss='mse', metrics=['mse'])
    # make sure the weights and biases are already created
    model.predict(np.zeros((1, input_length)))

    # set denominators to constants at the start
    model_weights = model.get_weights()
    ws = model_weights[:len(funcs)]
    bs = model_weights[len(funcs):2 * len(funcs)]
    for i in range(len(funcs)):
        at_func = 0
        for j in range(len(funcs[i])):
            if funcs[i][j] == 'div':
                for k in range(ws[i].shape[0]):
                    ws[i][k][at_func + 1] = 0.
                bs[i][at_func + 1] = 1.
            at_func += inputs_needed(funcs[i][j])

    model_weights[:len(funcs)] = ws
    model_weights[len(funcs):2 * len(funcs)] = bs
    model.set_weights(model_weights)

    train_phase(phase=1, t_begin=0, t_end=t_1)

    # phase 2
    if plot_prelim:
        fig.suptitle("Phase 2")

    # turn on L1-regularizer
    model.set_l1_reg(l1_reg)

    # turn on L0-threshold
    model.set_l0_thresh(l0_threshold)

    train_phase(phase=2, t_begin=t_1, t_end=t_2)

    # phase 3
    if plot_prelim:
        fig.suptitle("Phase 3")

    # turn off L1-regularizer
    model.set_l1_reg(0.)

    train_phase(phase=3, t_begin=t_2, t_end=T)

    if plot_prelim:
        plt.ioff()
        plt.close()

    return model, costs


def plot_prelims(model,
                 output_length,
                 eval_low,
                 eval_high,
                 axs,
                 pred_artists):
    weights = model.get_weights()
    input_length = weights[0].shape[0]

    xx = list()
    for i in range(input_length):
        xx.append(np.linspace(eval_low[i], eval_high[i], 1000))
    xx = np.array(xx).transpose()

    y_pred = model.predict_on_batch(xx)
    for i in range(output_length):
        pred_artists[i].set_ydata(y_pred[:, i])
        axs[i].draw_artist(pred_artists[i])
    plt.pause(.001)
    return


def plot_results(model,
                 f,
                 output_length,
                 train_low,
                 train_high,
                 eval_low,
                 eval_high,
                 block=True
                 ):
    weights = model.get_weights()
    input_length = weights[0].shape[0]

    plt.close()

    xx = list()
    for i in range(input_length):
        xx.append(np.linspace(eval_low[i], eval_high[i], 1000))
    xx = np.array(xx).transpose()

    y_true = f(xx)
    y_pred = model.predict_on_batch(xx)

    cols = int(np.ceil(np.sqrt(output_length)))
    rows = int(np.ceil(output_length / cols))

    fig, axs = plt.subplots(rows, cols)
    if cols != 1:
        axs = axs.flatten()

    if output_length == 1:
        axs.plot(xx[:, 0], y_true, 'r-', label='$f$ true')
        axs.plot(xx[:, 0], y_pred, 'b', linestyle='dashed', label='$f$ pred')
        lo = np.min(y_true) - 0.1 * (np.abs(np.min(y_true)) + 1.)
        hi = np.max(y_true) + 0.1 * (np.abs(np.max(y_true)) + 1.)
        axs.plot([train_low, train_low], [lo, hi], 'g', linestyle='dashed')
        axs.plot([train_high, train_high], [lo, hi], 'g', linestyle='dashed')
        axs.set_ylim([lo, hi])
        axs.legend(loc='upper right')
    else:
        for i in range(output_length):
            axs[i].plot(xx[:, i], y_true[:, i], 'r-', label='$f$ true')
            axs[i].plot(xx[:, i], y_pred[:, i], 'b', linestyle='dashed', label='$f$ pred')
            lo = np.min(y_true[:, i]) - 0.1 * (np.abs(np.min(y_true[:, i])) + 1.)
            hi = np.max(y_true[:, i]) + 0.1 * (np.abs(np.max(y_true[:, i])) + 1.)
            axs[i].plot([train_low[i], train_low[i]], [lo, hi], 'g', linestyle='dashed')
            axs[i].plot([train_high[i], train_high[i]], [lo, hi], 'g', linestyle='dashed')
            axs[i].set_ylim([lo, hi])
            axs[i].legend(loc='upper right')

    if block:
        plt.show(block=True)
    else:
        plt.show(block=False)
        plt.pause(.001)