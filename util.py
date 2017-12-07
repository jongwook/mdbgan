
def is_notebook():
    try:
        return type(get_ipython().config['IPKernelApp']['connection_file']) is str
    except:
        return False
