import cv2

def get_fps_width_height(cap):
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return fps, width, height

def initialize_video_writer(cap, annotated_video_output_path):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps, width, height = get_fps_width_height(cap)
    out = cv2.VideoWriter(annotated_video_output_path, fourcc, fps, (width, height))
    return out

def sum_ov_value(results, class_value_dict):
    total_dice_value = 0
    class_name_dict = results[0].names
    predicted_classes = results[0].boxes.cls
    for prediction in predicted_classes:
        total_dice_value += class_value_dict[class_name_dict[int(prediction)]]
    return round(total_dice_value, 2)

def counts_of_classes(results, class_value_dict):
    class_name_dict = results[0].names
    predicted_classes = results[0].boxes.cls
    counts_dict = {}
    for key in class_name_dict:
        count = predicted_classes.cpu().tolist().count(key)
        if count == 0:
            continue
        counts_dict[class_name_dict[key]] = count
    ordered_counts_dict = {key: counts_dict[key] for key in class_value_dict.keys() if key in counts_dict}
    return ordered_counts_dict

def overlay_sum(frame, text, font, font_scale, color, thickness, text_location):
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_width, text_height = text_size
    if text_location == "topleft":
        text_x = 10 
        text_y = text_height + 10
    elif text_location == "bottomleft":
        text_x = 10 
        text_y = frame.shape[0] - 10 
    elif text_location == "topright":
        text_x = frame.shape[1] - text_width - 10 
        text_y = text_height + 10
    elif text_location == "bottomright":
        text_x = frame.shape[1] - text_width - 10 
        text_y = frame.shape[0] - 10 
    else: # Default to topleft if an invalid location is passed
        text_x = 10 
        text_y = text_height + 10
    cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

def overlay_counts(frame, dic, font, font_scale, color, thickness, text_location):
    x_offset, y_offset = 10, 10
    for i, key in enumerate(dic):
        text = f"{key}: {dic[key]}"
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_width, text_height = text_size
        if text_location == "topleft":
            text_x = x_offset
            text_y = y_offset + ((i + 1) * (text_height + 10))
        elif text_location == "bottomleft":
            text_x = x_offset
            text_y = frame.shape[0] - (len(dic) - i) * (text_height + 10)
        elif text_location == "topright":
            text_x = frame.shape[1] - text_width - x_offset
            text_y = y_offset + ((i + 1) * (text_height + 10))
        elif text_location == "bottomright":
            text_x = frame.shape[1] - text_width - x_offset
            text_y = frame.shape[0] - (len(dic) - i) * (text_height + 10)
        else: # Default to topright if an invalid location is passed
            text_x = frame.shape[1] - text_width - x_offset
            text_y = y_offset + ((i + 1) * (text_height + 10))
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, color, thickness)

def overlay_frames(cap, out, prediction_model, prediction_confidence, class_value_dict, 
                   font, font_scale, color, thickness, sum_text_location, counts_text_location):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = prediction_model.predict(frame, conf = prediction_confidence)
        annotated_frame = results[0].plot()  # Draw bounding boxes and labels on the frame
        overlay_counts(annotated_frame, counts_of_classes(results, class_value_dict), font=font, font_scale=font_scale, color=color, 
                       thickness=thickness, text_location=counts_text_location)
        overlay_sum(annotated_frame, f"Total Value: {sum_ov_value(results, class_value_dict)}", 
                    font=font, font_scale=font_scale, color=color, thickness=thickness, text_location=sum_text_location)
        out.write(annotated_frame) # Write the annotated frame to the output video

def release_capture_and_writer(cap, out):
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def create_annotated_video(video_input_path, annotated_video_output_path, prediction_model, class_value_dict, prediction_confidence = 0.8, 
                           font = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 3,
                           color = (0, 255, 0), thickness = 2, sum_text_location = "topleft", counts_text_location = "topright"):
    cap = cv2.VideoCapture(video_input_path)
    out = initialize_video_writer(cap, annotated_video_output_path)
    overlay_frames(cap, out, prediction_model, prediction_confidence, class_value_dict, 
                   font = font, font_scale = font_scale, color = color, thickness = thickness, 
                   sum_text_location = sum_text_location, counts_text_location = counts_text_location)
    release_capture_and_writer(cap, out)