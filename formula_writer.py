import numpy as np
import sympy as sp

from EQL_div.inputs_needed import inputs_needed


def get_formulas(weights, funcs, symbols=None, simplify=False):
    # number of layers (including input layer)
    nr_layers = len(funcs)

    w = weights[:nr_layers]
    b = weights[nr_layers:]

    # get number of dense nodes and number of nodes after arithmetic operations in each layer
    dense_node_nr = [sum(inputs_needed(func) for func in funcs[i]) for i in range(nr_layers)]
    final_node_nr = [len(func) for func in funcs]

    # prev_formulas always stores the formulas of the previous layer
    prev_formulas = ["x_{}".format(i) for i in range(w[0].shape[0])]

    for curr_layer in range(nr_layers):

        dense_formulas = ["" for _ in range(dense_node_nr[curr_layer])]

        # get the formulas after the dense nodes
        for i in range(dense_node_nr[curr_layer]):
            # flag that indicates if we need brackets
            brackets_needed = False

            # weight*prev_formula pairs
            for j in range(len(prev_formulas)):
                # get the current weight:
                curr_weight = w[curr_layer][j][i]
                # skip if the weight is zero
                if curr_weight == 0.:
                    continue
                # also skip if prev_formula is zero
                if prev_formulas[j] in {'0.', '-0.'}:
                    continue
                if curr_weight < 0.:
                    brackets_needed = True
                elif dense_formulas[i] != '':
                    # if curr_weight is positive, but there is already another factor in front, insert a "+"
                    dense_formulas[i] += '+'
                    brackets_needed = True
                dense_formulas[i] += "{:.2f}*{}".format(curr_weight, prev_formulas[j])

            # bias
            curr_bias = b[curr_layer][i]
            if curr_bias != 0.:
                if curr_bias < 0.:
                    brackets_needed = True
                elif dense_formulas[i] != "":
                    # if curr_bias is positive, but there is already another factor in front, insert a "+"
                    dense_formulas[i] += "+"
                    brackets_needed = True
                dense_formulas[i] += "{:.2f}".format(curr_bias)

            # put brackets if we need them
            if brackets_needed:
                dense_formulas[i] = "({})".format(dense_formulas[i])

            # put a zero if we don't have anything in the formula now
            if dense_formulas[i] == "":
                dense_formulas[i] = '0.'

        final_formulas = ["" for _ in range(final_node_nr[curr_layer])]

        # apply the function nodes
        at_node = 0
        for i in range(final_node_nr[curr_layer]):
            curr_func = funcs[curr_layer][i]
            if curr_func == 'id':
                final_formulas[i] = dense_formulas[at_node]
            elif curr_func == 'sin':
                if dense_formulas[at_node] in {'0.', '-0.'}:
                    final_formulas[i] = '0.'
                else:
                    final_formulas[i] = "sin({})".format(dense_formulas[at_node])
            elif curr_func == 'cos':
                if dense_formulas[at_node] in {'0.', '-0.'}:
                    final_formulas[i] = '1.'
                else:
                    final_formulas[i] = "cos({})".format(dense_formulas[at_node])
            elif curr_func == 'square':
                if dense_formulas[at_node] in {'0.', '-0.'}:
                    final_formulas[i] = '0.'
                else:
                    final_formulas[i] = "({})**2".format(dense_formulas[at_node])
            elif curr_func == 'sqrt':
                if dense_formulas[at_node] in {'0.', '-0.'}:
                    final_formulas[i] = '0.'
                else:
                    final_formulas[i] = "sqrt({})".format(dense_formulas[at_node])
            elif curr_func == 'prod':
                if dense_formulas[at_node] in {'0.', '-0.'} or dense_formulas[at_node + 1] in {'0.', '-0.'}:
                    final_formulas[i] = '0.'
                else:
                    final_formulas[i] = "{}*{}".format(dense_formulas[at_node], dense_formulas[at_node + 1])
                at_node += 1
            elif curr_func == 'div':
                if dense_formulas[at_node] in {'0.', '-0.'}:
                    final_formulas[i] = '0.'
                elif dense_formulas[at_node + 1] in {'0.', '-0.'}:
                    final_formulas[i] = 'oo'
                else:
                    final_formulas[i] = "{}/({})".format(dense_formulas[at_node], dense_formulas[at_node + 1])
                at_node += 1
            at_node += 1

        # store the obtained formulas for the next layer
        prev_formulas = final_formulas

    formulas = prev_formulas

    if simplify:
        for i in range(len(formulas)):
            formulas[i] = str(sp.sympify(formulas[i]).evalf(2))

    if symbols is not None and w[0].shape[0] == len(symbols):
        for s in range(len(symbols)):
            for i in range(len(formulas)):
                formulas[i] = formulas[i].replace("x_{}".format(s), symbols[s])

    return formulas
