'''
    A debug py file to teste integration with Google colab

'''

def debug_msg():
    return "Debug message!"

def sum(a,b):
    try:
        return a+b
    except:
        debug_msg()