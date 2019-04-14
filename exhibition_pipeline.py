import sys
if sys.version_info.major < 3 or sys.version_info.minor < 4:
    print("Please using python3.4 or greater!")
    sys.exit(1)
import cv2, time, argparse
from time import sleep
import multiprocessing as mp
import threading
from models import FacePersonDetectionWorker
import traceback
import logging as log

pipeline = None
lastresults = None
threads = []
processes = []
frameBuffer = None
results = None
fps = ""
detectfps = ""
framecount = 0
detectframecount = 0
time1 = 0
time2 = 0
cam = None
camera_width = 640
camera_height = 480
window_name = ""
background_transparent_mode = 0
ssd_detection_mode = 1
face_detection_mode = 0
elapsedtime = 0.0
align_to = None
align = None


def camThread(settings, results, frameBuffer, camera_width, camera_height, vidfps, input=None, output=None):
    global fps
    global detectfps
    global lastresults
    global framecount
    global detectframecount
    global time1
    global time2
    global cam
    global window_name
    global align_to
    global align
    writer = None

    if input is None:
        cam = cv2.VideoCapture(0)
        if cam.isOpened() != True:
            logger.error("USB Camera Open Error!!!")
            sys.exit(0)
        window_name = "USB Camera"
    else:
        cam = cv2.VideoCapture(input)
        if cam.isOpened() != True:
            logger.error("File Open Error!!!")
            sys.exit(0)
        window_name = "File Stream"

    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    logger.debug("Initializing capture thread...")

    while True:
        t1 = time.perf_counter()

        s, color_image = cam.read()

        if input is not None:
            # Video File Stream Read
            # if we reached the end of file, break
            if not s:
                break
            # disable video resize for videos of width smaller than 1280px
            camera_width = None
            camera_height = None
            height = color_image.shape[0]
            width = color_image.shape[1]
            if width > 1280:
                camera_width = 1280
                resizing_factor = width*1.0/camera_width
                camera_height = int(height/resizing_factor)

        # if frame is not successfully captured from camera, retry
        if not s:
            continue

        if input is None:
            color_image = cv2.flip(color_image, 1)
        if camera_width is not None and camera_height is not None:
            color_image = cv2.resize(color_image, (camera_width, camera_height), interpolation=cv2.INTER_AREA)
        if frameBuffer.full():
            frameBuffer.get()
        frames = color_image

        height = color_image.shape[0]
        width = color_image.shape[1]
        frameBuffer.put(color_image.copy())
        res = None

        if output is not None and writer is None:
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(output, fourcc, vidfps,
                                     (width, height), True)

        if not results.empty():
            res = results.get(False)
            detectframecount += 1
            imdraw = overlay_on_image(frames, res, settings)
            lastresults = res
        else:
            imdraw = overlay_on_image(frames, lastresults, settings)

        cv2.imshow(window_name, cv2.resize(imdraw, (width, height)))

        # check to see if we should write the frame to disk
        if writer is not None:
            writer.write(imdraw)

        if cv2.waitKey(1)&0xFF == ord('q'):
            break

        ## Print FPS
        framecount += 1
        if framecount >= 15:
            fps = "(Playback) {:.1f} FPS".format(time1 / 15)
            detectfps = "(Detection) {:.1f} FPS".format(detectframecount / time2)
            framecount = 0
            detectframecount = 0
            time1 = 0
            time2 = 0
        t2 = time.perf_counter()
        elapsedTime = t2 - t1
        time1 += 1 / elapsedTime
        time2 += elapsedTime

    # Stop streaming
    if writer is not None:
        writer.release()
    if pipeline is not None:
        pipeline.stop()
    # If video is streamed from file, release the video file pointer
    if input is not None:
        cam.release()
    sys.exit(0)


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


def async_infer(ncsworker):
    while True:
        ncsworker.predict_async()


def inferencer(results, frameBuffer, enable_face_analysis, enable_reidentification, detected_people_history, camera_width, camera_height, number_of_ncs, vidfps, skpfrm):

    # Init infer threads
    threads = []
    for devid in range(number_of_ncs):
        thworker = threading.Thread(target=async_infer, args=(FacePersonDetectionWorker(devid, frameBuffer, detected_people_history, results, enable_face_analysis, enable_reidentification, camera_width, camera_height, number_of_ncs, vidfps, skpfrm),))
        thworker.start()
        threads.append(thworker)

    for th in threads:
        th.join()


def overlay_on_image(frames, object_infos, settings):
    try:
        color_image = frames

        if isinstance(object_infos, type(None)):
            return color_image

        # Show images
        height = color_image.shape[0]
        width = color_image.shape[1]

        if background_transparent_mode == 0:
            img_cp = color_image.copy()
        elif background_transparent_mode == 1:
            img_cp = background_img.copy()

        for person in object_infos:
            face = person.face
            pedestrian = person.pedestrian

            if face is not None:
                x_min_face, y_min_face, x_max_face, y_max_face = face.get_absolute_positions(width, height)

                label_text = face.get_label()

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (x_min_face, y_min_face), (x_max_face, y_max_face), box_color, box_thickness)
                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = x_min_face
                label_top = y_min_face - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                              label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            label_text_color, 1)

                # write emotional state and head pose on screen
                label_emotion = face.get_label_emotion()
                if label_emotion is not None:
                    y_emotion = y_max_face + 15 if y_max_face + 15 > 15 else y_max_face - 15
                    cv2.putText(img_cp, label_emotion, (x_min_face, y_emotion),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)
                label_head_pose = face.get_label_head_pose()
                if label_head_pose is not None:
                    y_head_pose = y_min_face - 15 if y_min_face - 15 > 15 else y_min_face + 15
                    cv2.putText(img_cp, label_head_pose, (x_min_face, y_head_pose),
                                cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)


            if pedestrian is not None:
                x_min_pedestrian, y_min_pedestrian, x_max_pedestrian, y_max_pedestrian = pedestrian.get_absolute_positions(width, height)

                drawing_initial_flag = True

                label_text = person.get_label()
                if label_text is None:
                    label_text = "Face"

                box_color = (255, 128, 0)
                box_thickness = 1
                cv2.rectangle(img_cp, (x_min_pedestrian, y_min_pedestrian), (x_max_pedestrian, y_max_pedestrian), box_color, box_thickness)
                label_background_color = (125, 175, 75)
                label_text_color = (255, 255, 255)
                label_size = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                label_left = x_min_pedestrian
                label_top = y_min_pedestrian - label_size[1]
                if (label_top < 1):
                    label_top = 1
                label_right = label_left + label_size[0]
                label_bottom = label_top + label_size[1]
                cv2.rectangle(img_cp, (label_left - 1, label_top - 1), (label_right + 1, label_bottom + 1),
                              label_background_color, -1)
                cv2.putText(img_cp, label_text, (label_left, label_bottom), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            label_text_color, 1)

        cv2.putText(img_cp, fps,       (width-160,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)
        cv2.putText(img_cp, detectfps, (width-160,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38,0,255), 1, cv2.LINE_AA)

        # display settings
        lbl_pedestrian_reidentification = "Pedestrian Reidentification: "
        if settings["pedestrian_reidentification_enabled"]:
            lbl_pedestrian_reidentification += "Enabled"
        else:
            lbl_pedestrian_reidentification += "Disabled"
        cv2.putText(img_cp, lbl_pedestrian_reidentification, (width - 285, height - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (38, 0, 255), 1, cv2.LINE_AA)

        lbl_face_analysis = "Face Analysis: "
        if settings["face_analysis_enabled"]:
            lbl_face_analysis += "Enabled"
        else:
            lbl_face_analysis += "Disabled"
        cv2.putText(img_cp, lbl_face_analysis, (width - 285, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (38, 0, 255), 1, cv2.LINE_AA)
        return img_cp

    except Exception as e:
        import traceback
        traceback.print_exc()

def update_history(detected_people_history, reported_people_history_ids, people_to_report):
    while True:
        # logger.info("Updating history...")
        ids_to_delete = []
        if len(detected_people_history) > 10:
            for id, face in detected_people_history.items():
                time_elapsed_seconds = face.get_time_elapsed_from_last_detection()
                time_present_milliseconds = face.milliseconds_present
                # for faces that have been present for at least 5 seconds, the limit is 30 seconds
                if time_present_milliseconds / 1000 > 5:
                    if time_elapsed_seconds > 30:
                        ids_to_delete.append(id)
                        if face.id not in reported_people_history_ids:
                            logger.debug(
                                "Face with ID {} has been idle for {} seconds and has not been reported yet,"
                                " therefore will be added to the list to report to the server".format(
                                    face.id, time_elapsed_seconds))
                            people_to_report.append(face)
                else:
                    if time_elapsed_seconds > 5:
                        ids_to_delete.append(id)
            for id in ids_to_delete:
                del detected_people_history[id]
        time.sleep(5)


if __name__ == '__main__':

    logger = log.getLogger("exhibition_pipeline")
    FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
    console_handler = log.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    logger.setLevel(log.DEBUG)
    logger.addHandler(console_handler)

    parser = argparse.ArgumentParser()
    parser.add_argument('-wd','--width',dest='camera_width',type=int,default=640,help='Width of the frames in the video stream. (USB Camera Mode Only. Default=640)')
    parser.add_argument('-ht','--height',dest='camera_height',type=int,default=480,help='Height of the frames in the video stream. (USB Camera Mode Only. Default=480)')
    parser.add_argument('-numncs','--numberofncs',dest='number_of_ncs',type=int,default=1,help='Number of NCS. (Default=1)')
    parser.add_argument('-vidfps','--fpsofvideo',dest='fps_of_video',type=int,default=30,help='FPS of Video. (USB Camera Mode Only. Default=30)')
    parser.add_argument('-skpfrm','--skipframe',dest='number_of_frame_skip',type=int,default=0,help='Number of frame skip. Default=0')
    parser.add_argument('-o', '--output', dest='output_file', type=str, help='Recording Output File')
    parser.add_argument('-i', '--input', dest='input_video_file', type=str, help='Input Video File')
    parser.add_argument('-r', '--reidentification', dest='reidentification_enabled', type=int, default=1,
                        help='Enable Pedestrian Re-Identification. Default=1 (1 - Enabled; 0 - Disabled)')
    parser.add_argument('-fa', '--face-analysis', dest='face_analysis_enabled', type=int, default=1,
                        help='Enable Face Analysis. Default=1 (1 - Enabled; 0 - Disabled)')

    args = parser.parse_args()

    camera_width = args.camera_width
    camera_height = args.camera_height
    number_of_ncs = args.number_of_ncs
    vidfps = args.fps_of_video
    skpfrm = args.number_of_frame_skip
    output = args.output_file
    input_video_file = args.input_video_file

    if input_video_file is None:
        lbl_source = "video camera stream"
    else:
        lbl_source = "video file {}".format(input_video_file)

    logger.info("Initializing analysis of {}".format(lbl_source))

    detected_people_history = {}
    reported_people_history_ids = []
    people_to_report = []
    last_person_id = 0

    if args.reidentification_enabled == 1:
        enable_reidentification = True
    else:
        enable_reidentification = False

    if args.face_analysis_enabled == 1:
        enable_face_analysis = True
    else:
        enable_face_analysis = False

    settings = {
        "pedestrian_reidentification_enabled": enable_reidentification,
        "face_analysis_enabled": enable_face_analysis
    }

    try:

        mp.set_start_method('fork')
        frameBuffer = mp.Queue(10)
        results = mp.Queue()

        # Start streaming
        p = mp.Process(target=camThread,
                       args=(settings, results, frameBuffer, camera_width, camera_height,
                             vidfps, input_video_file, output),
                       daemon=True)
        p.start()
        processes.append(p)

        # Start detection MultiStick
        # Activation of inferencer
        p = mp.Process(target=inferencer,
                       args=(results, frameBuffer, enable_face_analysis, enable_reidentification,
                             detected_people_history, camera_width, camera_height, number_of_ncs, vidfps, skpfrm),
                       daemon=True)
        p.start()
        processes.append(p)

        p = mp.Process(target=update_history, args=(detected_people_history, reported_people_history_ids, people_to_report))
        p.start()
        processes.append(p)

        while True:
            sleep(1)
            # check if camera thread has stopped
            if processes[0].exitcode is not None:
                # if camera thread has stopped, terminate all active threads in order to exit the program
                raise ValueError('Capture Thread Stopped', processes[0].exitcode)

    except Exception as e:
        if e.args[1] != 0:
            traceback.print_exc()
    finally:
        for p in range(len(processes)):
            processes[p].terminate()

            logger.info("Execution terminated")