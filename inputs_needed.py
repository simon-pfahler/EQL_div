def inputs_needed(node):
    # returns the number of input nodes needed for a specific function node (unary or binary)
    if node in {'id', 'sin', 'cos', 'square', 'sqrt'}:
        return 1
    if node in {'prod', 'div'}:
        return 2
    else:
        return
