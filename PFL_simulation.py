
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from scipy.stats import iqr
from IPython.display import display, clear_output
from scipy.stats import iqr
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import cdist
import pandas as pd
from scipy.integrate import quad

y = np.linspace(0, 10, 1000)
plt.plot(y,np.log(1 + np.exp(y-5)))
# Define the x range
x = np.linspace(0, 2*np.pi, 1000)

# Calculate the y values for each sine wave with the necessary phase shifts
y1 = np.sin(x - (np.pi/6))   # Peaks at pi/3
y2 = 1.5 + np.sin(x)               # Peaks at pi/2
y3 = np.sin(x + (7*np.pi/6)) # Peaks at 5*pi/3 (we add because the phase shift is negative)
# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Compute softplus of the sums
softplus_y1_y2 = softplus(y1 + y2)
softplus_y2_y3 = softplus(y2 + y3)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='left - Peak at $\pi/3$',color = 'darkgoldenrod')
plt.plot(x, y2, label='goal - Peak at $\pi/2$', color = 'tab:red')
plt.plot(x, y3, label='right - Peak at $5\pi/3$',color = 'tab:blue')
#plt.plot(x, softplus_y1_y2, label='Softplus of $left + goal$',color = 'goldenrod')
#plt.plot(x, softplus_y2_y3, label='Softplus of $goal + right$',color = 'blue')
#plt.title('Sine Waves with Different Peaks and transformed Output')
plt.ylim([-1.5,4])
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Define the x range
x = np.linspace(0, 2*np.pi, 1000)

# Calculate the y values for each sine wave with the necessary phase shifts
y1 = np.sin(x - (np.pi/6))   # Peaks at pi/3
y2 = 1.5 + np.sin(x)               # Peaks at pi/2
y3 = np.sin(x + (7*np.pi/6)) # Peaks at 5*pi/3 (we add because the phase shift is negative)
# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Compute softplus of the sums
softplus_y1_y2 = softplus(y1 + y2)
softplus_y2_y3 = softplus(y2 + y3)
# Plotting
plt.figure(figsize=(10, 6))
#plt.plot(x, y1, label='left - Peak at $\pi/3$',color = 'darkgoldenrod')
#plt.plot(x, y2, label='goal - Peak at $\pi/2$', color = 'tab:red')
#plt.plot(x, y3, label='right - Peak at $5\pi/3$',color = 'tab:blue')
plt.plot(x, softplus_y1_y2, label='Softplus of $left + goal$',color = 'darkgoldenrod')
plt.plot(x, softplus_y2_y3, label='Softplus of $goal + right$',color = 'tab:blue')
#plt.title('Sine Waves with Different Peaks and transformed Output')
plt.ylim([0,1])
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Define the x range
x = np.linspace(0, 2*np.pi, 1000)

# Calculate the y values for each sine wave with the necessary phase shifts
y1 = np.sin(x - (np.pi/6))   # Peaks at pi/3
y2 = 0.5 + 2*np.sin(x)               # Peaks at pi/2
y3 = np.sin(x + (7*np.pi/6)) # Peaks at 5*pi/3 (we add because the phase shift is negative)
# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Compute softplus of the sums
softplus_y1_y2 = softplus(y1 + y2)
softplus_y2_y3 = softplus(y2 + y3)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='left - Peak at $\pi/3$',color = 'darkgoldenrod')
plt.plot(x, y2, label='goal - Peak at $\pi/2$', color = 'tab:red')
plt.plot(x, y3, label='right - Peak at $5\pi/3$',color = 'tab:blue')
plt.plot(x, softplus_y1_y2, label='Softplus of $left + goal$',color = 'goldenrod')
plt.plot(x, softplus_y2_y3, label='Softplus of $goal + right$',color = 'blue')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Define the x range
x = np.linspace(0, 2*np.pi, 1000)

# Calculate the y values for each sine wave with the necessary phase shifts
y1 = np.sin(x - (np.pi/6))   # Peaks at pi/3
y2 = 0.5 + np.sin(x - np.pi/2)     # Peaks at pi/2
y3 = np.sin(x + (7*np.pi/6)) # Peaks at 5*pi/3 (we add because the phase shift is negative)
# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Compute softplus of the sums
softplus_y1_y2 = softplus(y1 + y2)
softplus_y2_y3 = softplus(y2 + y3)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='left - Peak at $\pi/3$',color = 'darkgoldenrod')
plt.plot(x, y2, label='goal - Peak at $\pi$', color = 'tab:red')
plt.plot(x, y3, label='right - Peak at $5\pi/3$',color = 'tab:blue')
plt.plot(x, softplus_y1_y2, label='Softplus of $left + goal$',color = 'goldenrod')
plt.plot(x, softplus_y2_y3, label='Softplus of $goal + right$',color = 'blue')
#plt.title('Sine Waves with Different Peaks and transformed Output')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Define the x range
x = np.linspace(0, 2*np.pi, 1000)

# Calculate the y values for each sine wave with the necessary phase shifts
y1 = np.sin(x - (np.pi/6))   # Peaks at pi/3
y2 = 3+np.sin(x)               # Peaks at pi/2
y3 = np.sin(x + (7*np.pi/6)) # Peaks at 5*pi/3 (we add because the phase shift is negative)
# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Compute softplus of the sums
softplus_y1_y2 = softplus(y1 + y2)
softplus_y2_y3 = softplus(y2 + y3)
# Plotting
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='left - Peak at $\pi/3$',color = 'darkgoldenrod')
plt.plot(x, y2, label='goal - Peak at $\pi$', color = 'tab:red')
plt.plot(x, y3, label='right - Peak at $5\pi/3$',color = 'tab:blue')
#plt.plot(x, softplus_y1_y2, label='Softplus of $left + goal$',color = 'goldenrod')
#plt.plot(x, softplus_y2_y3, label='Softplus of $goal + right$',color = 'blue')
plt.ylim([-1.5,4])
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# Define the x range
x = np.linspace(0, 2*np.pi, 1000)

# Calculate the y values for each sine wave with the necessary phase shifts
y1 = np.sin(x - (np.pi/6))   # Peaks at pi/3
y2 = 3+np.sin(x)               # Peaks at pi/2
y3 = np.sin(x + (7*np.pi/6)) # Peaks at 5*pi/3 (we add because the phase shift is negative)
# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Compute softplus of the sums
softplus_y1_y2 = softplus(y1 + y2)
softplus_y2_y3 = softplus(y2 + y3)
# Plotting
plt.figure(figsize=(10, 6))
#plt.plot(x, y1, label='left - Peak at $\pi/3$',color = 'darkgoldenrod')
#plt.plot(x, y2, label='goal - Peak at $\pi$', color = 'tab:red')
#plt.plot(x, y3, label='right - Peak at $5\pi/3$',color = 'tab:blue')
plt.plot(x, softplus_y1_y2, label='Softplus of $left + goal$',color = 'darkgoldenrod')
plt.plot(x, softplus_y2_y3, label='Softplus of $goal + right$',color = 'tab:blue')
plt.ylim([0,1])
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.grid(True)
plt.show()

# integral

# Define the sinusoidal functions
def y1(x):
    return np.sin(x - np.pi / 6)  # Phase shift for peak at pi/3

def y2(x):
    return 1.5+np.sin(x)  # No phase shift, peak at pi/2

def y3(x):
    return np.sin(x + 7 * np.pi / 6)  # Phase shift for peak at 5*pi/3

# Define the softplus function
def softplus(x):
    return np.log(1 + np.exp(x-5))

# Define the combined functions to integrate
def softplus_y1_y2(x):
    return softplus(y1(x) + y2(x))

def softplus_y2_y3(x):
    return softplus(y2(x) + y3(x))

# Compute the integrals
integral_y1_y2, _ = quad(softplus_y1_y2, 0, 2 * np.pi)
integral_y2_y3, _ = quad(softplus_y2_y3, 0, 2 * np.pi)

diff = integral_y1_y2 - integral_y2_y3
diff

# integral with upshifted baseline

# Define the sinusoidal functions
def y1(x):
    return np.sin(x - np.pi / 6)  # Phase shift for peak at pi/3

def y2(x):
    return 3+np.sin(x)  # No phase shift, peak at pi/2

def y3(x):
    return np.sin(x + 7 * np.pi / 6)  # Phase shift for peak at 5*pi/3


# Define the combined functions to integrate
def softplus_y1_y2(x):
    return softplus(y1(x) + y2(x))

def softplus_y2_y3(x):
    return softplus(y2(x) + y3(x))

# Compute the integrals
integral_y1_y2, _ = quad(softplus_y1_y2, 0, 2 * np.pi)
integral_y2_y3, _ = quad(softplus_y2_y3, 0, 2 * np.pi)

diff = integral_y1_y2 - integral_y2_y3
diff
