import mmap
import struct
import numpy as np
import cv2
import sys

# Structure to match C++ SharedData
SHARED_DATA_FORMAT = 'iiii'  # imageWidth, imageHeight, imageChannels, numBoxes
SHARED_DATA_SIZE = struct.calcsize(SHARED_DATA_FORMAT)

# Structure to match C++ BoundingBox
BOUNDING_BOX_FORMAT = 'ifiiii'  # classId, confidence, x, y, width, height
BOUNDING_BOX_SIZE = struct.calcsize(BOUNDING_BOX_FORMAT)

SHARED_MEMORY_NAME = "YOLODetectionSharedMemory"

def read_shared_memory():
    """Read image and bounding boxes from shared memory."""
    try:
        # Open shared memory
        print("Opening shared memory...")
        shm = mmap.mmap(-1, 0, SHARED_MEMORY_NAME, access=mmap.ACCESS_READ)
        
        # Read header
        header_data = shm.read(SHARED_DATA_SIZE)
        image_width, image_height, image_channels, num_boxes = struct.unpack(SHARED_DATA_FORMAT, header_data)
        
        print(f"Image dimensions: {image_width}x{image_height}x{image_channels}")
        print(f"Number of bounding boxes: {num_boxes}")
        
        # Read image data
        image_size = image_width * image_height * image_channels
        image_data = shm.read(image_size)
        
        # Convert to numpy array and reshape
        image_array = np.frombuffer(image_data, dtype=np.uint8)
        image = image_array.reshape((image_height, image_width, image_channels))
        
        # Read bounding boxes
        bounding_boxes = []
        for i in range(num_boxes):
            box_data = shm.read(BOUNDING_BOX_SIZE)
            class_id, confidence, x, y, width, height = struct.unpack(BOUNDING_BOX_FORMAT, box_data)
            bounding_boxes.append({
                'class_id': class_id,
                'confidence': confidence,
                'x': x,
                'y': y,
                'width': width,
                'height': height
            })
            print(f"Box {i+1}: Person at ({x}, {y}), size: {width}x{height}, confidence: {confidence:.2f}")
        
        shm.close()
        
        return image, bounding_boxes
    
    except FileNotFoundError:
        print(f"Error: Shared memory '{SHARED_MEMORY_NAME}' not found.")
        print("Make sure the C++ detector is running and has written data to shared memory.")
        return None, None
    except Exception as e:
        print(f"Error reading shared memory: {e}")
        return None, None

def draw_bounding_boxes(image, bounding_boxes):
    """Draw bounding boxes on the image."""
    output_image = image.copy()
    
    for box in bounding_boxes:
        x = box['x']
        y = box['y']
        width = box['width']
        height = box['height']
        confidence = box['confidence']
        
        # Draw rectangle
        cv2.rectangle(output_image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        
        # Draw label
        label = f"Person: {int(confidence * 100)}%"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        
        # Draw background for text
        cv2.rectangle(output_image, 
                     (x, y - label_size[1] - 10), 
                     (x + label_size[0], y), 
                     (0, 255, 0), 
                     -1)
        
        # Draw text
        cv2.putText(output_image, label, (x, y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    
    return output_image

def main():
    """Main function to read from shared memory and display results."""
    print("=" * 60)
    print("YOLO Human Detection - Python Reader")
    print("=" * 60)
    print("\nReading data from shared memory...")
    
    # Read from shared memory
    image, bounding_boxes = read_shared_memory()
    
    if image is None:
        print("\nFailed to read from shared memory. Exiting...")
        return
    
    print(f"\nSuccessfully read image and {len(bounding_boxes)} bounding box(es)")
    
    # Draw bounding boxes
    print("\nDrawing bounding boxes on image...")
    output_image = draw_bounding_boxes(image, bounding_boxes)
    
    # Save output image
    output_filename = "python_output_result.jpeg"
    cv2.imwrite(output_filename, output_image)
    print(f"Output image saved as: {output_filename}")
    
    # Display image
    print("\nDisplaying image (press any key to close)...")
    cv2.imshow("Human Detection Result", output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    print("\nDone!")

if __name__ == "__main__":
    main()
