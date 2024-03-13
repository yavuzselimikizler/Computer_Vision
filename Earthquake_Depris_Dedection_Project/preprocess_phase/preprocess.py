import cv2
import numpy as np
import os

def apply_windowing(image, window_center, window_width):
    # Calculate the window boundaries
    window_min = window_center - window_width / 2
    window_max = window_center + window_width / 2

    # Clip pixel values to the window boundaries
    windowed_img = np.clip(image, window_min, window_max)

    # Normalize to 8-bit for display (adjust if using a different target bit depth)
    windowed_img = cv2.normalize(windowed_img, None, 0, 255, cv2.NORM_MINMAX)

    return windowed_img.astype(np.uint8)
def apply_median_filter(image, kernel_size):
    # Apply median filter for noise reduction
    filtered_img = cv2.medianBlur(image, kernel_size)
    return filtered_img

def filter_tumors(original_img):
    # Load the original image
    
    blurred_img = cv2.GaussianBlur(original_img, (5, 5), 0)

    # Threshold the image to create a binary mask
    _, binary_img = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to clean up the mask
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out contours based on size (assuming tumors are larger than a certain threshold)
    min_tumor_area = 3000
    tumor_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_tumor_area]

    # Create a mask for tumors
    tumor_mask = np.zeros_like(original_img)
    cv2.drawContours(tumor_mask, tumor_contours, -1, 255, thickness=cv2.FILLED)

    # Invert the tumor mask to obtain a mask for non-tumor regions
    non_tumor_mask = cv2.bitwise_not(tumor_mask)

    # Apply the non-tumor mask to the original image
    cleared_img = cv2.bitwise_and(original_img, original_img, mask=non_tumor_mask)

    return cleared_img


def remove_contours(img):
    # Convert the image to binary using adaptive thresholding
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Perform morphological operations to remove small noise and close gaps in the binary image
    kernel = np.ones((5, 5), np.uint8)
    closing = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours in the binary image
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter out small contours (assuming the skull is a large region)
    min_contour_area = 5000
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    # Create a mask for the valid contours
    mask = np.zeros_like(img)
    cv2.drawContours(mask, valid_contours, -1, 255, thickness=cv2.FILLED)

    # Apply the mask to eliminate the skull part
    result_img = cv2.bitwise_and(img, img, mask=mask)

    return result_img

def adaptive_crop_image(image, padding=5):
    # Find non-white region for cropping
    non_white_pixels = np.column_stack(np.where(image < 255))

    if non_white_pixels.size > 0:
        # Calculate bounding box
        min_row = np.min(non_white_pixels[:, 0])
        min_col = np.min(non_white_pixels[:, 1])
        max_row = np.max(non_white_pixels[:, 0])
        max_col = np.max(non_white_pixels[:, 1])

        # Add padding to the bounding box
        min_row = max(0, min_row - padding)
        min_col = max(0, min_col - padding)
        max_row = min(image.shape[0] - 1, max_row + padding)
        max_col = min(image.shape[1] - 1, max_col + padding)

        # Crop the image
        cropped_image = image[min_row:max_row+1, min_col:max_col+1]
        return cropped_image
    else:
        return image

def remove_shadows(image):
    # Apply adaptive thresholding to remove shadows
    shadow_removed = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, blockSize=11, C=2)
    # Apply median filter to further remove noise
    shadow_removed = cv2.medianBlur(shadow_removed, 5)
    return shadow_removed

def gamma_correction(image, gamma=1.0):
    # Build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    # Apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def dynamic_range_compression(image, target_brightness):
    # Convert image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split LAB image into channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)

    # Compute adjustment factor based on target brightness
    mean_luminance = cv2.mean(l_channel)[0]
    adjustment_factor = target_brightness / mean_luminance

    # Apply adjustment factor to L channel
    adjusted_l_channel = np.clip(np.uint8(l_channel * adjustment_factor), 0, 255)

    # Merge adjusted L channel with original A and B channels
    adjusted_lab_image = cv2.merge((adjusted_l_channel, a_channel, b_channel))

    # Convert adjusted LAB image back to BGR color space
    adjusted_image = cv2.cvtColor(adjusted_lab_image, cv2.COLOR_LAB2BGR)

    return adjusted_image

def local_contrast_enhancement(image, clip_limit=10.0, tile_grid_size=(10, 10),l_scale=0.5):
    # Convert the image to LAB color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Split the LAB image into individual channels
    l_channel, a_channel, b_channel = cv2.split(lab_image)
    
    # Apply CLAHE to the L channel
    enhanced_l_channel = cv2.convertScaleAbs(l_channel, alpha=l_scale, beta=0)
    
    # Apply CLAHE to the modified L channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    enhanced_l_channel = clahe.apply(enhanced_l_channel)
    
    # Merge the enhanced L channel with the original A and B channels
    enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))
    
    # Convert the enhanced LAB image back to BGR color space
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    
    return enhanced_image


def histogram_matching(image):

    filename='1_4.png'
    input_folder='working_data4'

    input_path = os.path.join(input_folder, filename)
    reference_img = cv2.imread(input_path)
    reference_img = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((40,40), np.uint8)  # Adjust the kernel size as needed
    
    reference_img = cv2.medianBlur(reference_img, 15)

    reference_hist = cv2.calcHist([reference_img], [0], None, [256], [0,256])
    reference_cdf = np.cumsum(reference_hist) / np.sum(reference_hist)

    image_hist = cv2.calcHist([image], [0], None, [256], [0,256])
    image_cdf = np.cumsum(image_hist) / np.sum(image_hist)
    
    # Match histograms
    matched_image = np.interp(image.flatten(), np.arange(256), reference_cdf * 255).reshape(image.shape).astype(np.uint8)
    
    return matched_image

def preprocess_image(image_path, output_path,output_path4, target_pixels, target_width, target_height, target_bit_depth, darkness_threshold,reverse,counter):
    # Load the original image
    path1=0
    original_img = cv2.imread(image_path)
    resized_img = original_img.copy()
    cv2.imwrite(output_path + f"{path1}_normal_colored.jpg", resized_img)
    # Convert red-tone pixels to white-tone pixels
    """
    resized_img = dynamic_range_compression(resized_img,150)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_range_colored.jpg", resized_img)
    """

    resized_img = local_contrast_enhancement(resized_img)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_contrast_enhancement.jpg", resized_img)
    
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    kernel_size = (15, 15)  # Adjust the kernel size as needed
    blurred_img = cv2.GaussianBlur(resized_img, kernel_size, 0)
    
    
    """
    alpha = 100  # Increase alpha for more sharpening
    resized_img = cv2.addWeighted(resized_img, 1.0 + alpha, blurred_img, -alpha, 0) # sharpining


    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_sharping_colored.jpg", resized_img)
    """
    """
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_bluring_colored.jpg", resized_img)
    """
    # Subtract the blurred image from the original to obtain the sharpened image
    #resized_img = cv2.addWeighted(resized_img, 1.5, blurred_img, -0.5, 0)

   # blurred_img = cv2.GaussianBlur(resized_img_gray, kernel_size, 0)

# Unsharp masking
   
    # Apply median filter

    
    kernel = np.ones((40,40), np.uint8)  # Adjust the kernel size as needed
    
    resized_img = cv2.medianBlur(resized_img, 15)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_median_blur_colored.jpg", resized_img)
    
    
    resized_img = histogram_matching(resized_img)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_matched_colored.jpg", resized_img)

    resized_img = cv2.bilateralFilter(resized_img, d=9, sigmaColor=300, sigmaSpace=300)

    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_bilateral_blur_colored.jpg", resized_img)

    """
    resized_img = gamma_correction(resized_img, gamma=0.3)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_gama_blur_colored.jpg", resized_img)
    """


    

    """
    clip_limit = 50.0  # Clip limit to prevent amplification of noise
    tile_size = (2, 2)  # Size of each local neighborhood

# Create a CLAHE (Contrast Limited Adaptive Histogram Equalization) object
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_size)

# Apply adaptive histogram equalization
    resized_img = clahe.apply(resized_img)
    
    
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_histogram_colored.jpg", resized_img)
    """

   
    # Adjust the bit depth
    resized_img = cv2.convertScaleAbs(resized_img)
    resized_img = cv2.normalize(resized_img, None, 0, 2 ** target_bit_depth - 1, cv2.NORM_MINMAX)

    # Normalize the pixel values of the adjusted image
    resized_img = cv2.normalize(resized_img, None, 0, 255, cv2.NORM_MINMAX)
   
    # Threshold dark pixels
    # Create a CLAHE object (Contrast Limited Adaptive Histogram Equalization)
    
    
    window_center = 2000  # Example value, adjust accordingly
    window_width =  3620  # Example value, adjust accordingly
    resized_img = apply_windowing(resized_img, window_center, window_width)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_windowed_colored.jpg", resized_img)
    
    
   
    
    resized_img = cv2.dilate(resized_img,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)),iterations = 7)
    path1=path1+1
    cv2.imwrite(output_path + f"{path1}_dialate_colored.jpg", resized_img)


    if reverse:
        resized_img = 255 - resized_img
        dark_pixels = resized_img < darkness_threshold * 1.7

        resized_img[dark_pixels] = 0
        kernel = np.ones((10,10), np.uint8)  # Adjust the kernel size as needed

# Perform erosion
        resized_img = cv2.erode(resized_img, kernel)
    
    contours, _ = cv2.findContours(resized_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the maximum dimensions of contours
    max_width = max([cv2.boundingRect(contour)[2] for contour in contours])
    max_height = max([cv2.boundingRect(contour)[3] for contour in contours])

    # Create a blank canvas to draw contours
    canvas = np.zeros((max_height, max_width), dtype=np.uint8)
    
    min_width_threshold = 75  # Example value, adjust as needed
    max_width_threshold = 1000  # Example value, adjust as needed
    min_height_threshold = 75  # Example value, adjust as needed
    max_height_threshold = 1000  # Example value, adjust as needed
   
    #output_folder = "cropped_images3"
    #os.makedirs(output_folder, exist_ok=True)
    # Draw contours on the canvas
    c=25
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_width_threshold <= w <= max_width_threshold and min_height_threshold <= h <= max_height_threshold:
            cv2.rectangle(original_img, (x-c, y-c), (x + w+c, y + h+c), (0, 255, 0), 2)  # Draw green rectangles around objects within specified size range
            # Crop the contour rectangle from the original image
            if y-c>=0 and x-c>=0 and y+h+c<= original_img.shape[1] and x + w +c <= original_img.shape[0] :
                cropped_img = original_img[y - c:y + h + c, x - c:x + w + c]
        
        # Save the cropped image
            
                #cv2.imwrite(os.path.join(output_folder, f"cropped{counter}_{x}_{y}_{w}_{h}.jpg"), cropped_img)
    # Save the canvas with all contours as a single image
    cv2.imwrite(output_path4 + "contour_image.jpg", original_img)

    # Save the preprocessed images
    cv2.imwrite(output_path + "_colored.jpg", resized_img)

def red_to_white(output_path3,image_path):
    original_img = cv2.imread(image_path)

# Convert red-tone pixels to white-tone pixels
    red_pixels = (original_img[:, :, 2] > 70) & (original_img[:, :, 1] < 70) & (original_img[:, :, 0] < 70)
    original_img[red_pixels] = [255, 255, 255]  # Set red-tone pixels to white

    cv2.imwrite(output_path3 + '_colored.jpg', original_img)




def preprocess_images_in_folder(input_folder, output_folder,output_folder3,output_folder4,target_pixels, target_width, target_height, target_bit_depth, darkness_threshold):
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    reverse=0
    counter=0
    # Iterate over all files in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Full path to the input image
            input_path = os.path.join(input_folder, filename)

            # Output path for the preprocessed image
            output_path = os.path.join(output_folder, os.path.splitext(filename)[0])
            output_path3 =  os.path.join(output_folder3, os.path.splitext(filename)[0])
            output_path4 =  os.path.join(output_folder4, os.path.splitext(filename)[0])
            # Perform preprocessing
            counter=counter+1
            preprocess_image(input_path, output_path,output_path4,target_pixels, target_width, target_height, target_bit_depth, darkness_threshold,reverse,counter)
            red_to_white(output_path3,input_path)

# Example usage:
input_folder = 'working_data4'
output_folder = 'output_folder2'
output_folder3='output_folder3'
output_folder4='output_folder4'
# Set target dimensions, bit depth, and darkness threshold
target_width = 400
target_height = 400
target_bit_depth = 8
target_pixels = 30000
darkness_threshold = 130  # Adjust this threshold value as needed

# Process images in the input folder and save to the output folder
preprocess_images_in_folder(input_folder, output_folder,output_folder3,output_folder4,target_pixels, target_width, target_height, target_bit_depth, darkness_threshold)


