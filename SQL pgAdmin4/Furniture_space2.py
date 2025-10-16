!pip install py3dbp
!pip install matplotlib

from py3dbp import Packer, Bin, Item
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# Define your storage unit as a 'Bin' (length, width, height, weight)
storage_unit = Bin('StorageUnit', 3000, 2500, 2500, 1)

item_data = [
    {"name": "Couch", "length": 2000, "width": 700, "height": 2100, "weight": 0},
    {"name": "CoffeeTable", "length": 500, "width": 800, "height": 1400, "weight": 0},
    {"name": "SideBoard", "length": 1500, "width": 600, "height": 800, "weight": 0},
    {"name": "Table", "length": 800, "width": 1000, "height": 2100, "weight": 0},
    {"name": "DoubleBed", "length": 1400, "width": 700, "height": 1900, "weight": 0},
    {"name": "SingleBed", "length": 900, "width": 300, "height": 1900, "weight": 0},
    # Add more items as needed
]

# Initialize the Packer
packer = Packer()

# Add the storage unit to the packer (ensure this is done)
packer.add_bin(storage_unit)

# Add items to the packer dynamically
for item in item_data:
    new_item = Item(item["name"], item["length"], item["width"], item["height"], item["weight"])
    packer.add_item(new_item)
    print(f"Added item: {item['name']} with dimensions {item['length']}x{item['width']}x{item['height']}")

# Run the packing algorithm
packer.pack()

# Debugging: Print the packing result
for bin in packer.bins:
    print("Items in bin:", bin.name)
    for item in bin.items:
        print(f" Item: {item.name} at position {item.position} with dimensions {item.get_dimension()}")
      
# Function to generate a random color
def get_random_color():
    return np.random.rand(3,)

color_mapping = {}

# Function to add a 3D box (representing an item)
def add_box(ax, item, color):
    # Extracting position and dimensions
    pos = np.array(item.position, dtype=float)
    dim = np.array(item.get_dimension(), dtype=float)

    # Create a rectangular prism
    xx, yy = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[1], pos[1]+dim[1]])
    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]), color=color, alpha=0.5)
    ax.plot_surface(xx, yy, np.full_like(xx, pos[2]+dim[2]), color=color, alpha=0.5)
    
    yy, zz = np.meshgrid([pos[1], pos[1]+dim[1]], [pos[2], pos[2]+dim[2]])
    ax.plot_surface(np.full_like(yy, pos[0]), yy, zz, color=color, alpha=0.5)
    ax.plot_surface(np.full_like(yy, pos[0]+dim[0]), yy, zz, color=color, alpha=0.5)
    
    xx, zz = np.meshgrid([pos[0], pos[0]+dim[0]], [pos[2], pos[2]+dim[2]])
    ax.plot_surface(xx, np.full_like(xx, pos[1]), zz, color=color, alpha=0.5)
    ax.plot_surface(xx, np.full_like(xx, pos[1]+dim[1]), zz, color=color, alpha=0.5)

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Adding each item in the storage unit to the plot dynamically
#for item in storage_unit.items:
#    color = get_random_color()  # Assign a random color to each item
#    add_box(ax, item, color)

# Adding each item in the storage unit to the plot
for item in storage_unit.items:
    color = get_random_color()  # Get a random color for each item
    add_box(ax, item, color)
    color_mapping[item.name] = color  # Store the color mapping

# Create a legend/key for the items
legend_labels = [plt.Line2D([0], [0], color=color, lw=4, label=name) for name, color in color_mapping.items()]
plt.legend(handles=legend_labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

# Setting the limits to match the storage unit size
ax.set_xlim([0, 3000])
ax.set_ylim([0, 2500])
ax.set_zlim([0, 2500])

# Labels and title
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
ax.set_title('3D Visualization of Furniture in Storage Unit')

plt.show()