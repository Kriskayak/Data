
# Define each function below by copying and pasting the code from each
# function into an ipython console, one at a time.

# If the function gives you an error as you are trying to define it,
# you will need to fix the error so that the function can be defined.
# Once it is defined, run the function to make sure it works.

# If the function is defined without errors, run the function to see
# if an error is thrown. If so, you must debug it!

# These functions are simple enough such that you may not need to
# utilize the python debugger enviroment. However, I encourage you
# to play around and get used to the tool. It is very powerful!

# To use python debugger, run this line in your console:
#import ipdb

# Then uncomment the below line and place it somewhere in the code
# immediately before where you think an error is happening. This will
# initiate the "debugging" environment when the function is called.

#ipdb.set_trace()


def syntax_error(a=1):
    b = a*34
    return b

def runtime_error():
    print("Run time error demo")
    
    first = 1
    second = 2
    third = 3
    extra = input("What is the extra value?")
    
    total = first + second + third + extra
    print(total)
    return total

def symantic_error(a=4):
    sqrta = 4**(1/2)
    p = 3.14 * sqrta
    return p

def embedded_error(a=1):
    b = a + 45
    def func(x):
        y = x- 5 +3
        return y
    c = func(b)
    return c+2


# js comments
#------------
# - symantic_error does not function properly. The sqrta variable is not the square root of a

# 9/10

