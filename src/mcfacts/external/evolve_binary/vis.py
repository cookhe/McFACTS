import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
radius = 10          # Radius of the orbit
speed = 0.05         # Speed of the orbit (angular velocity)
center = (0, 0)      # Orbit center (the star or planet's position)
frames = 360         # Number of frames for the animation (1 full orbit)

# Create a figure and axis for the plot
fig, ax = plt.subplots()
ax.set_xlim(-radius-1, radius+1)
ax.set_ylim(-radius-1, radius+1)
ax.set_aspect('equal', 'box')

# Set up the orbit (a circle)
theta = np.linspace(0, 2*np.pi, 100)   # Circle parameter
x_orbit = radius * np.cos(theta)
y_orbit = radius * np.sin(theta)
ax.plot(x_orbit, y_orbit, color='blue', linestyle='--')  # Orbit path

# Create a point that will move along the orbit
orbiting_object, = ax.plot([], [], 'ro', markersize=10)

# Function to initialize the plot
def init():
    orbiting_object.set_data([], [])
    return orbiting_object,

# Function to update the position of the orbiting object
def update(frame):
    angle = frame * speed
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    orbiting_object.set_data(x, y)
    return orbiting_object,

# Create the animation
ani = FuncAnimation(fig, update, frames=frames, init_func=init, blit=True, interval=50)

# Show the plot
plt.title('2D Orbit Animation')
plt.show()



from manim import *

class PointMovingOnShapes(Scene):
    def construct(self):
        circle = Circle(radius=1, color=BLUE)
        dot = Dot()
        dot2 = dot.copy().shift(RIGHT)
        self.add(dot)

        line = Line([3, 0, 0], [5, 0, 0])
        self.add(line)

        self.play(GrowFromCenter(circle))
        self.play(Transform(dot, dot2))
        self.play(MoveAlongPath(dot, circle), run_time=2, rate_func=linear)
        self.play(Rotating(dot, about_point=[2, 0, 0]), run_time=1.5)
        self.wait()