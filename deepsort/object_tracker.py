import sys
import time

import cv2
import numpy as np
import tensorflow as tf

from absl import app, flags
from absl.flags import FLAGS

from deep.model import DeepAppearanceDescriptor

from sort.tracker import Tracker
from sort.utils import scale_bbox_to_original, to_detections, draw_track

from yolo.model import YOLOv3_Tiny
from yolo.utils import correct_boxes_and_scores, non_max_suppression, anchor_boxes


flags.DEFINE_string(
    "video", "./test.mp4", "path to input video or set to 0 for webcam"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")


def main(_argv):

    # Intitialize model
    deep_app_model = DeepAppearanceDescriptor()
    deep_app_model(tf.keras.layers.Input(shape=(64, 128, 3)))
    deep_app_model.load_weights("./deep/weights.e194-acc0.8881.h5")

    # Initialize tracker
    tracker = Tracker()

    # Initialize video
    video = cv2.VideoCapture(0)

    if FLAGS.output:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, fourcc, fps, (width, height))

    ### Load YOLOv3 tiny model
    yolo_model = YOLOv3_Tiny(
        input_size=416,
        anchor_boxes=anchor_boxes,
        n_classes=1,
        iou_threshold=0.5,
        score_threshold=0.5,
    )

    yolo_model(tf.keras.Input(shape=(416, 416, 3)))
    yolo_model.load_weights("./yolo/weights.e073-acc0.2822.h5")

    while True:
        # Read frame
        ret, frame = video.read()
        if not ret:
            raise RuntimeError("Could not read frame")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        img_resized = cv2.resize(frame, (416, 416))
        img_resized = img_resized / 255.0
        img_resized = np.expand_dims(img_resized, axis=0)
        img_resized = tf.constant(img_resized, tf.float32)

        # Get detections

        y_preds = yolo_model(img_resized, train=False)

        # Get boxes and scores
        boxes, scores = correct_boxes_and_scores(y_preds)

        # convert everything to numpy
        boxes = boxes[0]
        scores = scores[0]

        # non-max suppression
        boxes, scores = non_max_suppression(boxes, scores, 30, 0.5, 0.5)

        boxes = boxes.numpy()
        bboxes = scale_bbox_to_original(boxes, (frame.shape[0], frame.shape[1]))
        bboxes_original_scale = np.round(bboxes).astype(int)

        num_objs = boxes.shape[0]

        # draw the bounding boxes
        # for bbox in bboxes_original_scale:
        #     x1, y1, x2, y2 = bbox
        #     frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        detections = to_detections(frame, bboxes_original_scale, deep_app_model)

        # Update tracker
        tracker.predict()
        tracker.update(detections)

        # # Draw detections
        # for detection in detections:
        #     label = detection.label
        #     confidence = detection.confidence
        #     xmin = detection.xmin
        #     ymin = detection.ymin
        #     xmax = detection.xmax
        #     ymax = detection.ymax
        #     cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        #     cv2.putText(
        #         frame,
        #         f"{label}: {confidence:.2f}",
        #         (xmin, ymin),
        #         cv2.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (0, 255, 0),
        #         2,
        #     )

        # Draw tracker
        for track in tracker.tracks:
            if not track.is_tracked():
                continue

            frame = draw_track(frame, track)

        print(f"{num_objs} objects detected, {len(tracker.tracks)} tracks")

        # Display frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("frame", frame)
        if FLAGS.output:
            out.write(frame)

        # Exit if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

        # time.sleep(1000 // 30 / 1000)

    # Release resources
    video.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
