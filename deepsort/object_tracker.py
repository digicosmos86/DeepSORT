import sys
import time

import cv2
import numpy as np
import tensorflow as tf

from tensorflow.python.saved_model import tag_constants

from absl import app, flags, logging
from absl.flags import FLAGS

from deep.model import DeepAppearanceDescriptor

# from sort.tracker import Tracker
from sort.detection import Detection
from sort.utils import scale_bbox_to_original


flags.DEFINE_string(
    "video", "./data/video/test.mp4", "path to input video or set to 0 for webcam"
)
flags.DEFINE_string("output", None, "path to output video")
flags.DEFINE_float("iou", 0.45, "iou threshold")
flags.DEFINE_float("score", 0.50, "score threshold")


def main(_argv):
    max_cosine_distance = 0.4
    nn_budget = None
    nms_max_ovrerlap = 1.0

    # Intitialize model
    deep_app_model = DeepAppearanceDescriptor()
    deep_app_model(tf.keras.layers.Input(shape=(64, 128, 3)))
    deep_app_model.load_weights("./deep/weights.e194-acc0.8881.h5")

    # Initialize tracker
    # metric = nn_matching.NearestNeighborDistanceMetric(
    #     "cosine", max_cosine_distance, nn_budget
    # )
    # tracker = Tracker(metric)

    # Initialize video
    video = cv2.VideoCapture("./test.mp4")

    if FLAGS.output:
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(video.get(cv2.CAP_PROP_FPS))
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        out = cv2.VideoWriter(FLAGS.output, fourcc, fps, (width, height))

    ### Load YOLOv4 model
    yolo_model = tf.saved_model.load("yolov4-416/", tags=[tag_constants.SERVING])
    infer = yolo_model.signatures["serving_default"]

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

        # Detect objects
        pred_boxes = infer(img_resized)[
            "tf.concat_16"
        ]  # this outputs a dict with key "tf.concat16"

        boxes = tf.reshape(pred_boxes[:, :, :4], [1, -1, 1, 4])  # [1, n_boxes, 1, 4]
        pred_conf = pred_boxes[:, :, 4:]  # batch_size, num_boxes, num_classes

        (
            boxes,
            scores,
            classes,
            valid_detections,
        ) = tf.image.combined_non_max_suppression(
            boxes=boxes,
            scores=pred_conf,
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=FLAGS.iou,
            score_threshold=FLAGS.score,
        )

        # convert everything to numpy
        num_objs = int(valid_detections.numpy().squeeze())
        bboxes = boxes.numpy()[0]
        bboxes = bboxes[0:num_objs]
        scores = scores.numpy()[0]
        scores = scores[0:num_objs]
        classes = classes.numpy()[0]
        classes = classes[0:num_objs]

        # We only need the person class. We discard the rest.
        keep = np.where(classes == 0)
        bboxes = bboxes[keep]

        # the bounding boxes are in the format (ymin, xmin, ymax, xmax)
        # we need to convert them to (xmin, ymin, width, height)

        bboxes_original_scale = scale_bbox_to_original(
            bboxes, (frame.shape[0], frame.shape[1])
        )
        bboxes_original_scale = np.round(bboxes_original_scale).astype(int)

        # draw the bounding boxes
        for i, bbox in enumerate(bboxes_original_scale):
            x1, y1, x2, y2 = bbox
            frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("frame", frame)
        cv2.waitKey(0)

        sys.exit()

        # Convert to numpy
        objects = objects.numpy()

        # Convert to detection
        detections = []
        for object in objects:
            detection = Detection(object[0], object[1], object[2], object[3])
            detections.append(detection)

        # Update tracker
        tracker.process_detections(detections)

        # Draw detections
        for detection in detections:
            label = detection.label
            confidence = detection.confidence
            xmin = detection.xmin
            ymin = detection.ymin
            xmax = detection.xmax
            ymax = detection.ymax
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(
                frame,
                f"{label}: {confidence:.2f}",
                (xmin, ymin),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Draw tracker
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()

            # Draw bbox and label
            cv2.rectangle(
                frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 255, 0), 2
            )
            cv2.putText(
                frame,
                f"{track.id}",
                (bbox[0], bbox[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                2,
            )

        # Display frame
        cv2.imshow("frame", frame)
        if FLAGS.output:
            out.write(frame)

        # Exit if ESC is pressed
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # Release resources
    video.release()
    if FLAGS.output:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
