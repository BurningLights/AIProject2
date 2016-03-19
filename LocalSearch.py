def hill_climb_iteration(function_to_optimize, step_size, start_x, start_y, xmin, xmax, ymin, ymax, plot=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pyplot
    import random
    random.seed()
    
    # Start at given position and go from there
    x = start_x
    y = start_y
    num_repeats = 0
    # Find current min for starting spot
    current_min = function_to_optimize(x, y)
    
    # Plot starting spot
    if plot != None:
        plot.scatter(x, y, current_min, c='r')
    
    while num_repeats < 100:
        # Randomly pick which direction to go
        newx = x
        newy = y
        direction = random.randint(0, 3)
        if direction == 0:
            # Go left
            newx -= step_size
        elif direction == 1:
            # Go right
            newx += step_size
        elif direction == 2:
            # Go up
            newy += step_size
        elif direction == 3:
            # Go down
            newy -= step_size
            
        if function_to_optimize(newx, newy) < current_min:
            # Good move
            x = newx
            y = newy
            current_min = function_to_optimize(x, y)
            num_repeats = 0
            
            # Plot new position
            if plot != None:
                plot.scatter(x, y, current_min, c='r')
        else:
        # Movement did not yield an increase in the function
            num_repeats += 1
            
    # Found minimum
    return x, y

def hill_climb(function_to_optimize, step_size, xmin, xmax, ymin, ymax, plot=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pyplot
    import random
    random.seed()
    
    # Do a single hill climb iteration starting at random position
    startx = random.uniform(xmin, xmax)
    starty = random.uniform(ymin, ymax)
    x, y = hill_climb_iteration(function_to_optimize, step_size, startx, starty, xmin, xmax, ymin, ymax, plot)
    if plot != None:
        # Offset z a little so appears on top
        plot.scatter(x, y, function_to_optimize(x, y) + 0.01, c='y', marker='^', s=80)
    
    return x, y

def hill_climb_random_restart(function_to_optimize, step_size, num_restarts, xmin, xmax, ymin, ymax, plot=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pyplot
    # Seed random numbers
    import random
    random.seed()
    
    # Do first iteration of hill climbing at random position to find initial min
    x = random.uniform(xmin, xmax)
    y = random.uniform(ymin, ymax)
    min_x, min_y = hill_climb_iteration(function_to_optimize, step_size, x, y, xmin, xmax, ymin, ymax, plot)
    current_min = function_to_optimize(min_x, min_y)
    # Already did first iteration
    restarts_left = num_restarts - 1
    
    # Do the requested number of random restarts
    while restarts_left > 0:
        # Randomly pick starting x and y, and randomly pick sign
        x = random.uniform(xmin, xmax)
        y = random.uniform(ymin, ymax)
        
        # Do hill climbing for those starting values
        x, y = hill_climb_iteration(function_to_optimize, step_size, x, y, xmin, xmax, ymin, ymax, plot)
        result = function_to_optimize(x, y)
        
        # Check if result is new min
        if result < current_min:
            current_min = result
            min_x = x
            min_y = y
            
        restarts_left -= 1
    
    if plot != None:
        # Offset z a little so appears on top
        plot.scatter(x, y, function_to_optimize(x, y) + 0.01, c='y', marker='^', s=80)
        
    # Return minimum found amongst all hill climbing attempts
    return min_x, min_y

# This function determines if a move is acceptable in simulated annealing
def annealing_probability(new_value, old_value, current_temp):
    import random
    import math
    
    # Calculate probability
    if new_value < old_value:
        # Moving downhill, so just do it
        return True
    elif current_temp == 0 and new_value > old_value:
        # Temperature is 0 and moving uphill, so won't happen
        return False
    else:
        # General case
        # Seed random number generator
        random.seed()
        prob = math.e**(-(new_value - old_value) / current_temp)
        # Pick random number, and if probability is greater, allow
        return prob > random.uniform(0, 1.0)
    
def simulated_annealing(function_to_optimize, step_size, max_temp, xmin, xmax, ymin, ymax, plot=None):
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pyplot
    import random
    random.seed()
    
    # Current value of function (starts at minimum position)
    current_value = function_to_optimize(xmin, ymin)
            
    # Current x and y start at minimum
    x = xmin
    y = ymin
    finished = False
    # Start temp at max temp
    current_temp = max_temp
    
    # Plot new position
    if plot != None:
        plot.scatter(x, y, function_to_optimize(x, y), c='r')
    
    while not finished:
        # pick new spot to move to
        if current_temp > 0:
            newx = random.uniform(xmin, xmax)
            newy = random.uniform(ymin, ymax)
            if annealing_probability(function_to_optimize(newx, newy), current_value, current_temp):
                # Move to new location accepted
                x = newx
                y = newy
                current_value = function_to_optimize(x, y)
            
            # Determine new current value
            current_value = function_to_optimize(x, y)
        
            # Plot new position
            if plot != None:
                plot.scatter(x, y, current_value, c='r')
            
            # Reduce temperature and keep >= 0
            current_temp -= step_size
            current_temp = max(0, current_temp)
        else:
            # Hill climb rest of the way
            min_x, min_y = hill_climb_iteration(function_to_optimize, step_size, x, y, xmin, xmax, ymin, ymax, plot)
            finished = True
            
    if plot != None:
        # Offset z a little so appears on top
        plot.scatter(x, y, function_to_optimize(x, y) + 0.01, c='y', marker='^', s=80)
        
    # The last value is the minimum
    return min_x, min_y

def plot_graph(graph_function, x_start, x_end, y_start, y_end, samples, wire_frame = True):
    import numpy
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as pyplot
    
    # Convert Python function to numpy function
    func = numpy.frompyfunc(graph_function, 2, 1)
    # Get X, Y, and Z values to plot
    X = numpy.linspace(x_start, x_end, num=samples);
    Y = numpy.linspace(y_start, y_end, num=samples);
    X, Y = numpy.meshgrid(X, Y)
    Z = func(X, Y);

    # Create the pyplot and display it
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    if (wire_frame):
        ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1, color='b', linewidth=1)
    else:
        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, color='b', linewidth=0.1)
    
    return ax

def test_func(x, y):
    import math
    r = math.sqrt(x**2 + y**2)
    return (math.sin(x**2 + 3 * y**2) / (0.1 + r**2)) + (x**2 + 5 * y**2) * (math.e**(1 - r**2)) / 2

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pyplot

# Plot solid graph
#plot = plot_graph(test_func, -2.5, 2.5, -2.5, 2.5, 40, False)
# Plot the wireframe graph for path plotting
#plot = plot_graph(test_func, -2.5, 2.5, -2.5, 2.5, 40, True)
# Do straight up hillclimbing
#print(hill_climb(test_func, 0.01, -2.5, 2.5, -2.5, 2.5, plot))
# Do hillclimbing with random restarts
#print(hill_climb_random_restart(test_func, 0.01, 10, -2.5, 2.5, -2.5, 2.5, plot))
# Do simulated annealing
#print(simulated_annealing(test_func, 0.001, 2, -2.5, 2.5, -2.5, 2.5, plot))

#pyplot.show()
