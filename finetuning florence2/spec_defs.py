import os

import rasterio
# import shapely.geometry
import geojson

from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import set_matplotlib_formats

set_matplotlib_formats('retina')

import torch
import torchvision.transforms as transforms
if torch.cuda.is_available():
    device = torch.device("cuda:3")
else:
    device = torch.device("cpu")
    print("CUDA is not available, using CPU")

def load_image_and_label(image_path, label_path, metadata):
    """
    Loads a GeoTIFF image, its corresponding YOLO-OBB label, and relevant metadata.

    Args:
        image_path: Path to the GeoTIFF image file.
        label_path: Path to the corresponding YOLO-OBB label file.
        metadata: The parsed GeoJSON metadata (as a Python dictionary).

    Returns:
        image: The loaded image (e.g., as a NumPy array or PIL Image).
        label: The parsed YOLO-OBB label data (format to be determined).
        image_metadata:  The metadata associated with this specific image.
    """

    # 1. Extract x and y from the filename:
    tup = [int(i) for i in image_path.replace(".tif", "").replace(".png", "").split("/")[-1].split("_")]
    x = tup[0]
    y = tup[1]
    # 2. Find the corresponding metadata entry:
    if metadata == []:
        image_metadata=[]
    else:
        image_metadata = find_metadata_entry(metadata, x, y)

    # 3. Load the GeoTIFF image using rasterio:
    with rasterio.open(image_path) as src:
        image = src.read()  # Read all bands into a NumPy array
        image_tensor = torch.from_numpy(image).float().to(device)/255

    # 4. Load and parse the YOLO-OBB label:
    label_data = parse_yolo_obb_label(label_path)
    label_tensor = yolo_obb_to_pytorch_tensor(label_data, device)    

    # 5. Preprocess the image (if needed):
    #image_tensor = preprocess_image(image) # Implement this function

    return image_tensor, label_tensor, image_metadata

def find_metadata_entry(metadata, x, y):
    """
    Finds the metadata entry corresponding to the given x and y values.
    """
    for feature in metadata['features']:
        if feature['properties']['x'] == x and feature['properties']['y'] == y:
            return feature
    return None

def parse_yolo_obb_label(label_path):
    """
    Parses the YOLO-OBB label file.

    Args:
        label_path: Path to the label file.

    Returns:
        A list of dictionaries, where each dictionary represents an object
        and contains the class index and bounding box coordinates.
    """
    objects = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            class_index = int(parts[0])
            coordinates = [float(x) for x in parts[1:]]
            
            # Store the coordinates as a list of (x, y) pairs
            points = [(coordinates[i], coordinates[i+1]) for i in range(0, len(coordinates), 2)]

            objects.append({
                "class_index": class_index,
                "points": points,  # List of (x, y) tuples
            })

    return objects

def yolo_obb_to_pytorch_tensor(label_data, device):
    """
    Converts parsed YOLO-OBB label data to a PyTorch tensor.

    Args:
        label_data: The list of dictionaries from parse_yolo_obb_label.
        device: The torch.device  to use.

    Returns:
        A PyTorch tensor representing the label data, on the specified device.
        The structure of this tensor will depend on how you want to use it
        in your VLM (e.g., you might concatenate class and coordinates).
    """
    tensor_list = []
    for obj in label_data:
        # Convert points to a tensor
        points_tensor = torch.tensor(obj["points"], dtype=torch.float, device=device)

        # Create a tensor for the class index
        class_tensor = torch.tensor([obj["class_index"]], dtype=torch.long, device=device)

        # Combine class index and points into a single tensor for each object
        object_tensor = torch.cat((class_tensor, points_tensor.view(-1)))

        tensor_list.append(object_tensor)

    # Stack all object tensors into a single tensor
    if tensor_list:
        label_tensor = torch.stack(tensor_list)
    else:
        label_tensor = torch.empty((0, 9), dtype=torch.float, device=device) 
        
    return label_tensor

def preprocess_image(image_tensor):
    """
    Preprocesses the image tensor using torchvision transforms.

    Args:
        image_tensor: The image as a PyTorch tensor.

    Returns:
        The preprocessed image tensor.
    """
    # Example transformations (adapt these to your VLM's requirements):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize (adjust size as needed)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        # Add other transformations as needed
    ])
    
    # Ensure the tensor has 3 channels (e.g., by repeating a single channel 3 times)
    if image_tensor.shape[0] != 3:
        image_tensor = image_tensor.repeat(3, 1, 1)[:3, :, :]

    # Permute the dimensions to (C, H, W) if it's not already
    if image_tensor.shape[0] > 3:
      image_tensor = image_tensor.permute(2, 0, 1)

    image_tensor = preprocess(image_tensor)
    return image_tensor

def visualize_data(image_tensor, label_tensor):
    """
    Visualizes the image and bounding boxes, with different colors for each class.

    Args:
        image_tensor: The image as a PyTorch tensor (C, H, W).
        label_tensor: The label tensor containing bounding box info.
        class_colors: A dictionary mapping class indices to colors.
    """
    class_colors = {
        0: (1, 0, 0),  # Red for class 0
        1: (0, 1, 0),  # Green for class 1
        2: (0, 0, 1),  # Blue for class 2
    }

    # Move tensors to CPU and convert to numpy for plotting
    image_np = image_tensor.cpu().numpy()
    label_np = label_tensor.cpu().numpy()

    # Convert image to (H, W, C) format for plotting
    image_np = np.transpose(image_np, (1, 2, 0))

    # Denormalize the image (if it was normalized during preprocessing)
    #image_np = denormalize_image(image_np)  # Implement this function if needed

    # Create a figure and axes
    fig, ax = plt.subplots(1, figsize=(20, 20))

    # Display the image
    ax.imshow(image_np)
    ax.axis("off")

    # Iterate over bounding boxes
    for i in range(label_np.shape[0]):
        label = label_np[i]
        class_index = int(label[0])  # Assuming class index is the first element
        points = label[1:].reshape(-1, 2)  # Reshape to pairs of (x, y)

        # Convert normalized coordinates back to pixel coordinates
        height, width, _ = image_np.shape
        points[:, 0] *= width  # Scale x coordinates
        points[:, 1] *= height  # Scale y coordinates

        # Get the color for this class
        color = class_colors.get(class_index)  # Default to random color if not found

        # Create a polygon patch and add it to the plot
        polygon = patches.Polygon(points, closed=True, linewidth=4, edgecolor=color, facecolor='none')
        ax.add_patch(polygon)

    # Show the plot
    plt.show()