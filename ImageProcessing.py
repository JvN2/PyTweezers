import queue

import cv2
import numpy

FRAME_QUEUE_SIZE = 10
FRAME_HEIGHT = 480
FRAME_WIDTH = 480

def add_camera_id(frame, cam_id: str):
    # Helper function inserting 'cam_id' into given frame. This function
    # manipulates the original image buffer inside frame object.
    cv2.putText(frame.as_opencv_image(), 'Cam: {}'.format(cam_id), org=(0, 30), fontScale=1,
                color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)
    return frame



def create_dummy_frame(mean=0) -> numpy.ndarray:
    cv_frame = numpy.random.poisson(mean, (FRAME_HEIGHT, FRAME_WIDTH, 1)).astype(numpy.uint8)

    # cv2.putText(cv_frame, 'No Stream available. \nPlease connect a Camera.', org=(30, 30),
    #             fontScale=1, color=255, thickness=1, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL)

    return cv_frame

def resize_if_required(frame) -> numpy.ndarray:
    # Helper function resizing the given frame, if it has not the required dimensions.
    # On resizing, the image data is copied and resized, the image inside the frame object
    # is untouched.
    cv_frame = frame.as_opencv_image()

    if (frame.get_height() != FRAME_HEIGHT) or (frame.get_width() != FRAME_WIDTH):
        cv_frame = cv2.resize(cv_frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv2.INTER_AREA)
        cv_frame = cv_frame[..., numpy.newaxis]

    return cv_frame

class FrameConsumer:
    def __init__(self, frame_queue: queue.Queue):
        # self.log = Log.get_instance()
        self.frame_queue = frame_queue

    def stop(self):
        self.running = False    

    def run(self):
        IMAGE_CAPTION = 'Multithreading Example: Press <Enter> to exit'
        KEY_CODE_ENTER = 13

        frames = {}
        alive = True

        self.running = True  

        # self.log.info('\'FrameConsumer\' started.')

        while alive:
            # Update current state by dequeuing all currently available frames.
            frames_left = self.frame_queue.qsize()
            while frames_left:
                try:
                    cam_id, frame = self.frame_queue.get_nowait()

                except queue.Empty:
                    break

                # Add/Remove frame from current state.
                if frame:
                    frames[cam_id] = frame

                else:
                    frames.pop(cam_id, None)

                frames_left -= 1

            # Construct image by stitching frames together.
            if frames:
                cv_images = [resize_if_required(frames[cam_id]) for cam_id in sorted(frames.keys())]
                cv2.imshow(IMAGE_CAPTION, numpy.concatenate(cv_images, axis=1))

            # If there are no frames available, show dummy image instead
            else:
                cv2.imshow(IMAGE_CAPTION, create_dummy_frame())

            # Check for shutdown condition
            if KEY_CODE_ENTER == cv2.waitKey(10) or self.running is False:
                cv2.destroyAllWindows()
                alive = False

        self.log.info('\'FrameConsumer\' terminated.')