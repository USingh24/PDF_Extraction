import streamlit as st
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from PIL import Image,ImageDraw, ImageFont
from PIL import ImageOps
import torch
from detectron2.utils.visualizer import Visualizer
import cv2
import numpy as np
import fitz
import io
import tempfile
import os
import math
import time
import base64
import pytesseract
from pytesseract import Output
from pix2tex.cli import LatexOCR
from rapid_latex_ocr import LatexOCR as rapid_model
from IPython.display import display, Latex
import sympy

model = LatexOCR()
model1 = rapid_model()

st.title('Detectron 2 Streamlit App')
# Accept PDF file uploads
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "jpg", "jpeg", "png"])
bicolumn_page = st.radio("Bicolumn page", ["Yes", "No"], index=1)
# Function to calculate the center point of a box
def calculate_center(box):
    x1, y1, x2, y2 = box[1:5]
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    return center_x, center_y


def is_valid_latex(latex_str):
    try:
        sympy.latex(sympy.sympify(latex_str))
        return True
    except:
        return False
# Function to get the value for sorting
def get_sort_value(index,LatexFigure):
    # Retrieve the corresponding LatexFigure item using the index
    latex_item = LatexFigure[index]
    # Return the average x-coordinate as the value by which to sort
    return (latex_item[1] + latex_item[3]) / 2
# Function to get the value for sorting
def get_sort_lineboxes(box_list):
    # Calculate the average y-coordinate of the first box in the list
    return (box_list[0][2] + box_list[0][4]) / 2

def add_margin(img):
    width = img.shape[1]
    print(width)
    height = img.shape[0]
    print(height)
    left_pad = int(width*0.1)
    top_pad = int(height*0.1)
    # Add padding to the image array
    im_np_padded = np.pad(im_np, ((top_pad, top_pad), (left_pad, left_pad), (0, 0)), mode='constant',constant_values=255)

    return im_np_padded
def remove_lines(im_np):
    cropped_image = im_np

    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,
        25,
        15
    )
    # Create the images that will use to extract the horizontal and vertical lines
    horizontal = np.copy(thresh)
    vertical = np.copy(thresh)

    # Specify size on horizontal axis
    cols = horizontal.shape[1]
    horizontal_size = math.ceil(cols / 20)

    # Create structure element for extracting horizontal lines through morphology operations
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))

    # Apply morphology operations
    horizontal = cv2.erode(horizontal, horizontalStructure)
    horizontal = cv2.dilate(horizontal, horizontalStructure)


    # Specify size on vertical axis
    rows = vertical.shape[0]
    verticalsize = math.ceil(rows / 20)

    # Create structure element for extracting vertical lines through morphology operations
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))

    # Apply morphology operations
    vertical = cv2.erode(vertical, verticalStructure)
    vertical = cv2.dilate(vertical, verticalStructure)
    res=vertical+horizontal
    im_np[np.where(res != 0)] = 255
    return im_np
def get_LineBoxes(cnt,im_np):
    LineBoxes = []
    im_np=remove_lines(im_np)
    #im_np= remove_lines(im_np)

    cropped_image = im_np
    # Convert the image to grayscale
    gray = cv2.cvtColor(im_np, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Create a horizontal kernel for dilation
    kernel = np.ones((1, 320), np.uint8)

    # Apply horizontal dilation
    dilated = cv2.dilate(thresh, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # print(contours)
    # Filter out small contours (adjust the threshold as needed)
    min_contour_area = 50
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    bounding_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]
    bounding_boxes = [(x, y - 3, w, h + 6) for x, y, w, h in bounding_boxes if (h > 6)]
    # Display the result image with bounding boxes
    result_image = cropped_image.copy()

    for box in bounding_boxes:
        x, y, w, h = box
        cv2.rectangle(result_image, (max(5,x), y), (min(cropped_image.shape[1]-5,x + w), y + h), (0, 255, 0), 1)
        LineBoxes.append([4, max(5,x), y, min(cropped_image.shape[1]-5,x + w), y + h, cnt])
        cnt += 1
    #st.image(result_image)
    sortedBoxes = sorted(LineBoxes, key=lambda x: x[2])
    return sortedBoxes
def arranged_row_boxes(LineBoxes,LatexFigure):
    matches = {}
    UsedLatexFigure = {}

    # Iterate through each item in LineBoxes
    for i, line_item in enumerate(LineBoxes):
        class_id_line, x1_line, y1_line, x2_line, y2_line, i_line = line_item
        # Calculate the center point of the LineBoxes box
        center_line = calculate_center(line_item)
        # Check if the center point is inside or on the border of any item in LatexFigure
        contained_latex_items = []
        for j, latex_item in enumerate(LatexFigure):
            class_id_latex, x1_latex, y1_latex, x2_latex, y2_latex, i_latex = latex_item
            # Calculate the center point of the LatexFigure box
            center_latex = calculate_center(latex_item)

            if (
                    ((x1_line <= center_latex[0] <= x2_line and
                      y1_line - 3 <= center_latex[1] <= y2_line + 3) or
                     (y1_latex <= y1_line <= y2_latex and
                      y1_latex <= y2_line <= y2_latex)) and (j not in UsedLatexFigure)
            ):
                # Store the information that LineBoxes item contains LatexFigure item
                contained_latex_items.append(j)
                UsedLatexFigure[j] = True

        # Store the matches in the map
        matches[i] = contained_latex_items

    for key, value in matches.items():
        matches[key] = sorted(value, key=lambda idx: get_sort_value(idx, LatexFigure))
    # Print the sorted matches
    for key, value in matches.items():
        print(f"LineBoxes item {key} contains LatexFigure items {value}")

    l = len(LineBoxes)
    for i, latex_item in enumerate(LatexFigure):

        # Check if the index is not in UsedLatexBoxes
        if i not in UsedLatexFigure:
            # Append the latex_item to LineBoxes
            LineBoxes.append(latex_item)

    UpdatedLineBoxes = []
    # Iterate through each item in LineBoxes
    for i, line_item in enumerate(LineBoxes):
        class_id_line, x1_line, y1_line, x2_line, y2_line, i_line = line_item
        segments = []
        if (i < l):
            LF_List = matches[i]

            for item in LF_List:
                class_id, x1, y1, x2, y2, i_latex = LatexFigure[item]
                segments.append([class_id_line, x1_line, y1_line, min((x1 + x2) // 2, x2_line), y2_line])
                segments.append([class_id, x1, y1, x2, y2])
                x1_line = max(x1_line, (x1 + x2) // 2)
        segments.append([class_id_line, x1_line, y1_line, x2_line, y2_line])
        UpdatedLineBoxes.append(segments)
    # Sort UpdatedLineBoxes based on the average y-coordinate of the first box in each list
    sorted_line_boxes = sorted(UpdatedLineBoxes, key=get_sort_lineboxes)
    return sorted_line_boxes

def Handle_BiColumn(im_np, updated_predictions,flag):
    mask = np.zeros_like(im_np, dtype=np.uint8)
    if flag==True:
        left_boxes=[]
        right_boxes=[]
        LatexFigure_left=[]
        LatexFigure_right=[]
        half_width=int((im_np.shape[1])*0.5)
        for i, value in enumerate(updated_predictions):
            class_id = value[0]
            x1 = value[1]
            y1 = value[2]
            x2 = value[3]
            y2 = value[4]
            mid=int((x1+x2)*0.5)
            if(mid<= half_width):
                left_boxes.append(value)
                if class_id == 0 or class_id == 1:
                    LatexFigure_left.append([class_id, x1, y1, x2, y2, i])
                    mask[y1:y2, x1:x2, :] = 255  # Set pixels to white
            else:
                right_boxes.append(value)
                if class_id == 0 or class_id == 1:
                    LatexFigure_right.append([class_id, x1, y1, x2, y2, i])
                    mask[y1:y2, x1:x2, :] = 255  # Set pixels to white

        # # Apply the mask to the original image
        new_im_np = np.where(mask == 255, 255, im_np)
        cnt = len(updated_predictions)
        LineBoxes1 = get_LineBoxes(cnt, new_im_np[:,:int((new_im_np.shape[1])*0.49)])
        sorted_line_boxes1=arranged_row_boxes(LineBoxes1,LatexFigure_left)
        cnt += len(LineBoxes1)
        LineBoxes2 = get_LineBoxes(cnt, new_im_np[:,int((new_im_np.shape[1])*0.51):])
        diff=int((new_im_np.shape[1])*0.51)
        LineBoxes2 = [(c_id, a+diff, b , c+diff, d,new_id) for c_id, a, b , c, d,new_id in LineBoxes2]
        sorted_line_boxes2 = arranged_row_boxes(LineBoxes2, LatexFigure_right)
        sorted__boxes= sorted_line_boxes1+sorted_line_boxes2
        return sorted__boxes


    LatexFigure=[]
    for value in updated_predictions:
        class_id = value[0]
        x1 = value[1]
        y1 = value[2]
        x2 = value[3]
        y2 = value[4]
        i = value[5]
        if class_id == 0 or class_id == 1:
            LatexFigure.append([class_id, x1, y1, x2, y2, i])
            mask[y1:y2, x1:x2, :] = 255  # Set pixels to white

    # # Apply the mask to the original image
    im_np = np.where(mask == 255, 255, im_np)
    cnt = len(updated_predictions)
    LineBoxes = get_LineBoxes(cnt, im_np)
    sorted_line_boxes = arranged_row_boxes(LineBoxes, LatexFigure)
    return sorted_line_boxes





def generate_html(sorted_line_boxes, output_dir, img, im_np, output_file, selected_page):
    html_content = "<html>\n<head>\n<style>\n.container {\n  display: flex;\n  align-items: center;\n}\n</style>\n<script type = 'text/javascript' src = 'https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n<script>$(document).ready(function(){renderMathInElement(document.body, {delimiters: [{left:'$$', right:'$$', display: true},{left:'$', right:'$', display: false},{left: '\\[', right: '\\]', display: true}]});});</script>\n</head>\n<body>\n"
    #html_content = "<html>\n<head>\n<style>\n.container {\n  display: flex;\n  align-items: center;\n}\n</style>\n<script type='text/javascript' async src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'></script>\n<script type='text/x-mathjax-config'>MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']], processEscapes: true, ignoreClass: 'text'}});</script>\n</head>\n<body>\n"
    img_dir_path = os.path.join(output_dir, str(selected_page))
    os.makedirs(img_dir_path, exist_ok=True)
    for line_idx, line in enumerate(sorted_line_boxes):
        line_html = "<div class='container'>\n"

        for b_ind, box in enumerate(line):
            class_id, x1, y1, x2, y2 = box
            if 0 <= x1 < x2 < img.shape[1] and 0 <= y1 < y2 < img.shape[0]:
                if class_id == 1:
                    cropped_image = img[y1:y2, x1:x2]
                    res, elapse = model1(cropped_image)
                    if elapse < 5:
                        latex_expression = f"$\\displaystyle {res}$"
                        line_html += f"<span style='color: black; font-size:14px'>{latex_expression}</span>&nbsp;\n"
                    else:
                        cropped_image = img[y1:y2, x1:x2]
                        pil_image = Image.fromarray(cropped_image)
                        # Save image and get its path
                        image_name = f"image_{line_idx}_{b_ind}.png"

                        image_path = os.path.join(img_dir_path, image_name)
                        pil_image.save(image_path)
                        line_html += f"<img src='{image_path}'/>\n"
                    # Wrap LaTeX expression in MathJax delimiters

                elif class_id == 0:
                    cropped_image = img[y1:y2, x1:x2]
                    pil_image = Image.fromarray(cropped_image)
                    # Save image and get its path
                    image_name = f"image_{line_idx}_{b_ind}.png"

                    image_path = os.path.join(img_dir_path, image_name)
                    pil_image.save(image_path)
                    line_html += f"<img src='{image_path}'/>\n"
                else:
                    cropped_image = im_np[y1:y2, x1:x2]
                    pil_line = Image.fromarray(cropped_image)
                    # Add whitespace padding to the PIL Image
                    padding = 10  # You can adjust the padding value
                    padded_image = ImageOps.expand(pil_line, border=padding, fill='white')
                    # Pass the padded image to pytesseract
                    text = pytesseract.image_to_string(pil_line, config='--psm 6')
                    print(text)
                    text = text[:-1]
                    line_html += f"<span class='text' style='color: black; font-size:14px'>{text}</span>&nbsp;\n"



        line_html += "</div>\n"
        #st.markdown(line_html,
        #             unsafe_allow_html=True)
        html_content += line_html

    html_content += "</body>\n</html>"
    #st.markdown(html_content, unsafe_allow_html=True)
    # Save the HTML content to the output file
    with open(output_file, "w") as f:
        f.write(html_content)



def Create_page(image,im_np,flag,output_dir, output_file, selected_page):

    # im_np = cv2.convertScaleAbs(im_np, alpha=1.5, beta=0)
    # Run model inference
    config_path = r'C:\Users\Brigosha_Guest\Desktop\streamlit_projects\config.yaml'
    assert os.path.exists(config_path), f"Config file '{config_path}' does not exist!"
    cfg = get_cfg()
    cfg.merge_from_file(config_path)
    cfg.MODEL.WEIGHTS = r'C:\Users\Brigosha_Guest\Desktop\streamlit_projects\model_0002999 (1).pth'
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im_np)

    # Visualize results
    v = Visualizer(im_np[:, :, ::-1], scale=1.0)  # Convert BGR to RGB
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    # Get the predicted boxes and classes
    pred_boxes = outputs["instances"].to("cpu").pred_boxes.tensor
    pred_classes = outputs["instances"].to("cpu").pred_classes
    # Define the classes you want to visualize (e.g., classes 0 to 5) and their respective colors
    classes_to_visualize = {0: (255, 0, 0),  # Red
                            1: (0, 255, 0),  # Green
                            2: (0, 0, 255),  # Blue
                            3: (255, 255, 0),  # Yellow
                            4: (255, 0, 255),  # Magenta
                            5: (0, 255, 255)}  # Cyan

    predictions = []
    to_remove = {}
    cnt = 0
    show_image=im_np.copy()

    for i, (box, class_id) in enumerate(zip(pred_boxes, pred_classes)):
        class_id = class_id.item()
        x1, y1, x2, y2 = box.tolist()
        # You can adjust the coordinates here as needed
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        predictions.append([class_id, x1, y1, x2, y2, i])


    # Loop through the predicted boxes and classes
    for i, value in enumerate(predictions):
        class_id = value[0]
        if class_id in classes_to_visualize:
            x1 = value[1]
            y1 = value[2]
            x2 = value[3]
            y2 = value[4]
            # Iterate over other boxes to check if the current box is inside
            for j, other_value in enumerate(predictions):
                if j == i:
                    continue
                other_class_id = other_value[0]
                if other_class_id == class_id:  # Skip boxes of the same class
                    ox1 = other_value[1]
                    oy1 = other_value[2]
                    ox2 = other_value[3]
                    oy2 = other_value[4]
                    # Calculate the intersection area
                    intersection_area = max(0, min(x2, ox2) - max(x1, ox1)) * max(0, min(y2, oy2) - max(y1, oy1))
                    # Calculate the area of the smaller box
                    smaller_box_area = (x2 - x1) * (y2 - y1)
                    # Check if at least 80% of the smaller box's area is inside the other box
                    if intersection_area >= 0.5 * smaller_box_area:
                        to_remove[i] = True
                        cnt += 1
                        ox1 = min(ox1, x1)
                        oy1 = min(oy1, y1)
                        ox2 = max(ox2, x2)
                        oy2 = max(oy2, y2)
                        predictions[j] = [other_class_id, ox1, oy1, ox2, oy2, j]
                        break  # No need to check with other boxes once inside

    updated_predictions = [value for index, value in enumerate(predictions) if index not in to_remove]
    #check = Check_Two_Column_page(updated_predictions, im_np.shape[1])
    img = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
    # if flag == 1:
    #     img = cv2.resize(img, None, fx=1.9, fy=1.9)
    # if Bi_Cloumn==True:


    #LatexFigure = []
    mask = np.zeros_like(im_np, dtype=np.uint8)
    for value in updated_predictions:
        class_id = value[0]
        x1 = value[1]
        y1 = value[2]
        x2 = value[3]
        y2 = value[4]
        i = value[5]
        if class_id == 0 or class_id == 1:
            #LatexFigure.append([class_id, x1, y1, x2, y2, i])
            mask[y1:y2, x1:x2, :] = 255  # Set pixels to white
        cv2.rectangle(show_image, (x1, y1), (x2, y2), classes_to_visualize[class_id], 1)
    #st.image(show_image)

    # # Apply the mask to the original image
    im_np = np.where(mask == 255, 255, im_np)

    TwoColumn=0
    if bicolumn_page == "Yes":
        TwoColumn=1

    sorted_line_boxes= Handle_BiColumn(im_np, updated_predictions,TwoColumn)

    #html_content = "<html>\n<head>\n<style>\n.container {\n  display: flex;\n  align-items: center;\n}\n</style>\n<script type = 'text/javascript' src = 'https://www.maths.nottingham.ac.uk/plp/pmadw/LaTeXMathML.js'></script>\n<script>$(document).ready(function(){renderMathInElement(document.body, {delimiters: [{left:'$$', right:'$$', display: true},{left:'$', right:'$', display: false},{left: '\\[', right: '\\]', display: true}]});});</script>\n</head>\n<body>\n"
    html_content = "<html>\n<head>\n<style>\n.container {\n  display: flex;\n  align-items: center;\n}\n</style>\n<script type='text/javascript' async src='https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/MathJax.js?config=TeX-MML-AM_CHTML'></script>\n<script type='text/x-mathjax-config'>MathJax.Hub.Config({tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']], displayMath: [['$$','$$'], ['\\[','\\]']], processEscapes: true, ignoreClass: 'text'}});</script>\n</head>\n<body>\n"
    img_dir_path = os.path.join(output_dir, str(selected_page))
    os.makedirs(img_dir_path, exist_ok=True)

    for line_idx,lists in  enumerate(sorted_line_boxes):
        ncol = len(lists)
        ratio_arr = []
        total_len = 0
        for i, box in enumerate(lists):
            class_id, x1, y1, x2, y2 = box
            if 0 <= x1 < x2 < img.shape[1] and 0 <= y1 < y2 < img.shape[0]:
                if class_id == 0 or class_id == 1:
                    ratio_arr.append(x2 - x1)
                    total_len += (x2 - x1)
                else:
                    cropped_image = im_np[y1:y2, x1:x2]
                    # Convert the image to grayscale
                    gray = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
                    # Apply adaptive thresholding
                    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
                    kernel = np.ones((1, 25), np.uint8)
                    # Apply horizontal dilation
                    dilated = cv2.dilate(thresh, kernel, iterations=2)
                    # Find contours
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    min_contour_area = 50
                    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
                    bounding_boxes = [cv2.boundingRect(contour) for contour in filtered_contours]
                    curr_width = 1
                    for x, y, w, h in bounding_boxes:
                        if (h > 6):
                            curr_width += w
                    ratio_arr.append(curr_width / 1.8)
                    total_len += (curr_width / 1.8)

            else:
                ratio_arr.append(0.01)
                total_len += 0.01
        for i in range(ncol):
            ratio_arr[i] = ratio_arr[i] / total_len

        print(ratio_arr)
        cols = st.columns(ratio_arr)
        #var1 = []
        line_html = "<div class='container'>\n"
        for i, box in enumerate(lists):
            b_ind=i
            class_id, x1, y1, x2, y2 = box
            print("box in lists")
            print(x1, y1, x2, y2)
            # Get the color for the current class
            color = classes_to_visualize[class_id]
            col = cols[i % ncol]
            if 0 <= x1 < x2 < img.shape[1] and 0 <= y1 < y2 < img.shape[0]:
                if class_id == 1:
                    cropped_image = img[y1:y2, x1:x2]
                    pil_image = Image.fromarray(cropped_image)
                    #padded_image = add_margin(cropped_image)
                    res, elapse = model1(cropped_image)

                    # latex_result(padded_image)
                    # print(elapse)
                    print(res)

                    if elapse < 5:
                        col.markdown(
                            f" <b style='text-align: center; color: black; font-size:12px'>$\\displaystyle {res}$",
                            unsafe_allow_html=True)
                        latex_expression = f"$\\displaystyle {res}$"
                        line_html += f"<span style='color: black; font-size:14px'>{latex_expression}</span>&nbsp;\n"
                    else:

                        col.image(pil_image)
                        # Save image and get its path
                        image_name = f"image_{line_idx}_{b_ind}.png"

                        image_path = os.path.join(img_dir_path, image_name)
                        pil_image.save(image_path)
                        line_html += f"<img src='{image_path}'/>\n"
                    #var1.append(res)
                    # col.markdown(f" <b style='text-align: center; color: black; font-size:12px'>$\\displaystyle {res}$",unsafe_allow_html=True)
                    # st.latex(res)

                elif class_id == 0:
                    cropped_image = img[y1:y2, x1:x2]
                    pil_image = Image.fromarray(cropped_image)
                    # st.image(pil_image)
                    col.image(pil_image)
                    image_name = f"image_{line_idx}_{b_ind}.png"

                    image_path = os.path.join(img_dir_path, image_name)
                    pil_image.save(image_path)
                    line_html += f"<img src='{image_path}'/>\n"
                    # line_output += st.image(pil_image).to_html()

                else:
                    cropped_image = im_np[y1:y2, x1:x2]
                    pil_line = Image.fromarray(cropped_image)
                    # Add whitespace padding to the PIL Image
                    padding = 10  # You can adjust the padding value
                    padded_image = ImageOps.expand(pil_line, border=padding, fill='white')
                    # Pass the padded image to pytesseract
                    text = pytesseract.image_to_string(pil_line, config='--psm 6')

                    print(text)
                    #var1.append(text)
                    col.markdown(f"<b style='text-align: centre; color: black; font-size:13px'>{text}",
                                 unsafe_allow_html=True)
                    text = text[:-1]
                    line_html += f"<span class='text' style='color: black; font-size:14px'>{text}</span>&nbsp;\n"


            else:
                print("Invalid coordinates for cropping.")
        line_html += "</div>\n"
        html_content += line_html
    html_content += "</body>\n</html>"
    with open(output_file, "w") as f:
        f.write(html_content)
        # st.write('-------------------------------------------------------------------------')
    return html_content,sorted_line_boxes, img,im_np


if uploaded_file is not None:
    file_details = {
        "FileName": uploaded_file.name,
        "FileType": uploaded_file.type,
        "NumPages": None  # Placeholder for now
    }
    temp_pdf_path = None  # Initialize temp PDF path variable

    if uploaded_file.type == "application/pdf":
        # Read the contents of the PDF file
        pdf_content = uploaded_file.read()
        # Use BytesIO to create a file-like object from the PDF content
        pdf_file = io.BytesIO(pdf_content)

        # Create a temporary file and write the PDF content to it
        temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
        temp_pdf.write(pdf_content)
        temp_pdf_path = temp_pdf.name
        temp_pdf.close()

        # Open the temporary file with fitz
        with fitz.open(temp_pdf_path) as doc:
            num_pages = len(doc)
            file_details["NumPages"] = num_pages
            st.write(file_details)
            # Dropdown for selecting page number
            selected_page = st.selectbox("Select Page Number", range(1, num_pages + 1), key="page_selector")

            # Process the selected page
            page_num = selected_page - 1  # Convert to 0-based index
            page = doc.load_page(page_num)  # Load the page
            # Get the cropped pixmap
            pix = page.get_pixmap(matrix=fitz.Matrix(1.9, 1.9), alpha=False)
                                 # clip=(left_offset, top_offset, new_width, new_height))
            img_bytes = pix.tobytes()  # Convert to bytes
            # Convert image bytes to NumPy array
            image = np.array(bytearray(img_bytes), dtype="uint8")
            # Decode and prepare image for Detectron2
            im_np = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
            height = im_np.shape[0]
            width = im_np.shape[1]
            x1 = int(width * 0.03)
            x2= int(width * 0.97)
            y1 = int(height * 0.02)
            y2 = int(height * 0.98)
            mask = np.zeros_like(im_np, dtype=np.uint8)
            mask[:y1,:, :] = 255  # Set pixels to white
            mask[y2:, :, :] = 255
            mask[:, :x1, :] = 255
            mask[:, x2:, :] = 255

            # # Apply the mask to the original image
            im_np = np.where(mask == 255, 255, im_np)
            file_name_parts = uploaded_file.name.split('.')
            directory_name = '.'.join(file_name_parts[:-1])

            # Construct the directory path
            output_dir = rf"C:\Users\Brigosha_Guest\Desktop\streamlit_projects\Output_html\{directory_name}"

            # Ensure the directory exists
            os.makedirs(output_dir, exist_ok=True)

            output_file = os.path.join(output_dir, f"page{selected_page}.html")

            html_content,sorted_line_boxes, img,im_np= Create_page(image,im_np,0,output_dir, output_file, selected_page)

            with open(output_file, "rb") as file:
                file_contents = file.read()

            file_name = f"{directory_name}_{selected_page}.html"
            b64 = base64.b64encode(file_contents).decode()
            href = f'<a href="data:file/html;base64,{b64}" download="{file_name}">Download File {selected_page}</a>'
            download_link = href
            st.markdown(download_link, unsafe_allow_html=True)
            if st.button('Download HTML'):
                with open(output_file, "w") as f:
                    f.write(html_content)
                #generate_html(sorted_line_boxes, output_dir, img, im_np, output_file, selected_page)
    else:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        im_np = cv2.imdecode(image, cv2.IMREAD_COLOR)
        #im_np = cv2.resize(im_np, None, fx=1.9, fy=1.9)
        file_details["NumPages"] = 1
        st.write(file_details)
        file_name_parts = uploaded_file.name.split('.')
        directory_name = '.'.join(file_name_parts[:-1])

        # Construct the directory path
        output_dir = rf"C:\Users\Brigosha_Guest\Desktop\streamlit_projects\Output_html\{directory_name}"

        # Ensure the directory exists
        os.makedirs(output_dir, exist_ok=True)

        output_file = os.path.join(output_dir, "page1.html")

        html_content,sorted_line_boxes, img ,im_np= Create_page(image,im_np,1,output_dir, output_file, 1)



        # Draw bounding box outlines with the assigned color
        #st.image(im_np, caption=f"Annotated Page {selected_page}", use_column_width=True)


        with open(output_file, "rb") as file:
            file_contents = file.read()

        file_name = f"{directory_name}_{1}.html"
        b64 = base64.b64encode(file_contents).decode()
        href = f'<a href="data:file/html;base64,{b64}" download="{file_name}">Download File {1}</a>'
        download_link = href
        st.markdown(download_link, unsafe_allow_html=True)
        if st.button('Download HTML'):
            with open(output_file, "w") as f:
                f.write(html_content)
            #generate_html(sorted_line_boxes, output_dir,img,im_np,output_file,1)

        #im_with_boxes = out.get_image()
        # Remove the temporary file if it exists
    if temp_pdf_path is not None:
        st.write(f"Removing temporary file: {temp_pdf_path}")
        os.remove(temp_pdf_path)
