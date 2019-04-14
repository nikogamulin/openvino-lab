from openvino.inference_engine import IENetwork, IEPlugin
import cv2
import detections_helper
import logging as log
import heapq
import sys
import time
import datetime


logger = log.getLogger("models")
FORMATTER = log.Formatter("%(asctime)s — %(name)s — %(levelname)s — %(message)s")
console_handler = log.StreamHandler(sys.stdout)
console_handler.setFormatter(FORMATTER)
logger.setLevel(log.DEBUG)
logger.addHandler(console_handler)


# l = Search list
# x = Search target value
def searchlist(l, x, notfoundvalue=-1):
    if x in l:
        return l.index(x)
    else:
        return notfoundvalue


class FaceDetection:
    def __init__(self, plugin, num_requests=2):
        net = IENetwork(model='models/face-detection-adas-0001-fp16.xml',
                        weights='models/face-detection-adas-0001-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        del net

    def detect(self, frame, cur_request_id, next_request_id, log=None):
        detected_people = []
        try:
            # Prepare input blob and perform an inference
            blob = cv2.dnn.blobFromImage(frame, size=(self.w, self.h), swapRB=False, crop=False)
            self.exec_net.start_async(
                request_id=next_request_id, inputs={self.input_blob: blob})
            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                # Parse detection results of the current request
                out = self.exec_net.requests[cur_request_id].outputs
                out_faces = out['detection_out']
                _, _, detections_count_faces, _ = out_faces.shape

                for box_index in range(detections_count_faces):
                    image_id_face, label_face, conf_face, x_min_face, y_min_face, x_max_face, y_max_face = out_faces[0,
                                                                                                           0,
                                                                                                           box_index, :]

                    if conf_face > 0.9:
                        face = detections_helper.Face(conf_face, x_min_face, y_min_face, x_max_face, y_max_face)
                        # logger.info("Detected face with confidence {}".format(conf_face))
                        detected_people.append(face)

        except Exception as e:
            log.error('An arror occurred while trying to make detections:', e)

        return detected_people


class FacePersonDetection:
    def __init__(self, plugin, results, conf_face_threshold=0.7, conf_person_threshold=0.7):
        logger.info("Initializing model for face detection")
        net = IENetwork(model='models/face-person-detection-retail-0002-fp16.xml',
                                weights='models/face-person-detection-retail-0002-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.debug("Loading face-person-detection-retail-0002-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=2)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.debug(
            "face-person-detection-retail-0002-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net
        self.conf_face_threshold = conf_face_threshold
        self.conf_pedestrian_threshold = conf_person_threshold
        self.results = results

    def detect(self, frame, cur_request_id, next_request_id, log=None):
        inf_start = time.time()
        try:
            blob = cv2.dnn.blobFromImage(frame, size=(self.w, self.h), swapRB=False, crop=False)
            self.exec_net.start_async(
                request_id=next_request_id, inputs={self.input_blob: blob})
            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                detections = []

                out = self.exec_net.requests[cur_request_id].outputs
                out_faces = out['detection_out_face']
                _, _, detections_count_faces, _ = out_faces.shape

                out_pedestrians = out['detection_out_pedestrian']
                _, _, detections_count_pedestrians, _ = out_pedestrians.shape

                for box_index in range(detections_count_faces):
                    image_id_face, label_face, conf_face, x_min_face, y_min_face, x_max_face, y_max_face = out_faces[0,
                                                                                                           0, box_index,
                                                                                                           :]
                    image_id_pedestrian, label_pedestrian, conf_pedestrian, x_min_pedestrian, y_min_pedestrian, \
                        x_max_pedestrian, y_max_pedestrian = out_pedestrians[0, 0, box_index, :]

                    if conf_face > self.conf_face_threshold or conf_pedestrian > self.conf_pedestrian_threshold:
                        person = detections_helper.Person()
                        if conf_face > self.conf_face_threshold:
                            person.set_face(conf_face, x_min_face, y_min_face, x_max_face, y_max_face)
                        if conf_pedestrian > self.conf_pedestrian_threshold:
                            person.set_pedestrian(conf_pedestrian, x_min_pedestrian, y_min_pedestrian, x_max_pedestrian,
                                                  y_max_pedestrian)
                        person.check_integrity()

                        detections.append(person)
                self.results.put(detections)

        except Exception as e:
            if logger is not None:
                logger.error('An arror occurred while trying to make detections:', e)

        inf_end = time.time()
        det_time = inf_end - inf_start
        logger.debug("Inference time: {}".format(det_time))

class FacePersonDetectionWorker:
    def __init__(self, devid, frameBuffer, detected_people_history, results, enable_face_analysis, enable_reidentification, camera_width, camera_height, number_of_ncs, vidfps, skip_frames, conf_face_threshold=0.7, conf_person_threshold=0.7):
        logger.info("Initializing model for person and face detection with face probability threshold {} and person probability threshold {}".format(conf_face_threshold, conf_person_threshold))
        self.devid = devid
        self.frameBuffer = frameBuffer
        self.model_xml = "models/face-person-detection-retail-0002-fp16.xml"
        self.model_bin = "models/face-person-detection-retail-0002-fp16.bin"
        self.camera_width = camera_width
        self.camera_height = camera_height
        self.num_requests = 4
        self.inferred_request = [0] * self.num_requests
        self.heap_request = []
        self.inferred_cnt = 0
        self.plugin = IEPlugin(device="MYRIAD")
        self.net = IENetwork(model=self.model_xml, weights=self.model_bin)
        self.input_blob = next(iter(self.net.inputs))
        self.exec_net = self.plugin.load(network=self.net, num_requests=self.num_requests)
        self.results = results
        self.number_of_ncs = number_of_ncs
        self.skip_frame = skip_frames
        self.roop_frame = 0
        self.vidfps = vidfps
        self.n, self.c, self.h, self.w = self.net.inputs[self.input_blob].shape
        self.conf_face_threshold = conf_face_threshold
        self.conf_pedestrian_threshold = conf_person_threshold
        self.ageGenderDetection = AgeGenderDetection(self.plugin)
        self.headPoseEstimation = HeadPoseDetection(self.plugin)
        self.personReidentification = PersonIdentification(self.plugin)
        self.emotionDetection = EmotionDetection(self.plugin)
        self.detected_people_history = detected_people_history
        self.last_person_id = 0
        self.enable_reidentification = enable_reidentification
        self.enable_face_analysis = enable_face_analysis

    def predict_async(self):
        try:

            if self.frameBuffer.empty():
                return

            self.roop_frame += 1
            if self.roop_frame <= self.skip_frame:
                self.frameBuffer.get()
                return
            self.roop_frame = 0
            frame = self.frameBuffer.get()
            h, w, c = frame.shape

            blob = cv2.dnn.blobFromImage(frame, size=(self.w, self.h), swapRB=False, crop=False)
            reqnum = searchlist(self.inferred_request, 0)

            if reqnum > -1:
                self.exec_net.start_async(request_id=reqnum, inputs={self.input_blob: blob})
                self.inferred_request[reqnum] = 1
                self.inferred_cnt += 1
                if self.inferred_cnt == sys.maxsize:
                    self.inferred_request = [0] * self.num_requests
                    self.heap_request = []
                    self.inferred_cnt = 0
                heapq.heappush(self.heap_request, (self.inferred_cnt, reqnum))

            cnt, dev = heapq.heappop(self.heap_request)

            if self.exec_net.requests[dev].wait(0) == 0:
                self.exec_net.requests[dev].wait(-1)

                detected_people_new = []
                out = self.exec_net.requests[dev].outputs
                out_faces = out['detection_out_face']
                _, _, detections_count_faces, _ = out_faces.shape

                out_pedestrians = out['detection_out_pedestrian']
                _, _, detections_count_pedestrians, _ = out_pedestrians.shape

                detections = []
                matched_ids = []
                for box_index in range(detections_count_faces):
                    image_id_face, label_face, conf_face, x_min_face, y_min_face, x_max_face, y_max_face = out_faces[0,
                                                                                                           0, box_index,
                                                                                                           :]
                    image_id_pedestrian, label_pedestrian, conf_pedestrian, x_min_pedestrian, y_min_pedestrian, \
                    x_max_pedestrian, y_max_pedestrian = out_pedestrians[0, 0, box_index, :]

                    if conf_face > self.conf_face_threshold or conf_pedestrian > self.conf_pedestrian_threshold:
                        person = detections_helper.Person()
                        if conf_face > self.conf_face_threshold:
                            person.set_face(conf_face, x_min_face, y_min_face, x_max_face, y_max_face)
                            if self.enable_face_analysis:
                                x_min_face, y_min_face, x_max_face, y_max_face = person.face.get_absolute_positions(w, h)
                                face_rect = frame[y_min_face:y_max_face, x_min_face:x_max_face]
                                age_gender = self.ageGenderDetection.detect(face_rect)
                                prob_male = age_gender['male']
                                prob_female = age_gender['female']
                                if prob_male > prob_female:
                                    gender = 'Male'
                                else:
                                    gender = 'Female'
                                age = age_gender['age']
                                pitch, yaw, roll = self.headPoseEstimation.detect(face_rect)
                                emotion = self.emotionDetection.detect(face_rect)
                                person.face.age = age
                                person.face.gender = gender
                                person.face.emotion = emotion
                                person.face.head_orientation = {"pitch": pitch, "yaw": yaw, "roll": roll}
                        if conf_pedestrian > self.conf_pedestrian_threshold:
                            matching_history_cosine_similarities = []
                            match_found = False
                            person.set_pedestrian(conf_pedestrian, x_min_pedestrian, y_min_pedestrian, x_max_pedestrian,
                                                  y_max_pedestrian)
                            x_min_pedestrian, y_min_pedestrian, x_max_pedestrian, y_max_pedestrian = person.pedestrian.get_absolute_positions(w, h)
                            pedestrian_rect = frame[y_min_pedestrian:y_max_pedestrian, x_min_pedestrian:x_max_pedestrian]
                            if self.enable_reidentification:
                                identity_vec = self.personReidentification.detect(pedestrian_rect)
                                person.pedestrian.descriptor = identity_vec
                                detected_people_history_count = len(self.detected_people_history.keys())
                                if detected_people_history_count > 0:
                                    for id, existing_person in self.detected_people_history.items():
                                        # prevent assigning same face from history to more than one newly detected faces
                                        if id in matched_ids:
                                            continue
                                        cosine_similarity = detections_helper.get_cosine_similarity(
                                            existing_person.pedestrian.descriptor, person.pedestrian.descriptor)
                                        matching_history_cosine_similarities.append((id, cosine_similarity))

                                    if len(matching_history_cosine_similarities) > 0:
                                        cosine_similarities = [float(item[1]) for item in
                                                               matching_history_cosine_similarities]
                                        # similarities_avg = statistics.mean(cosine_similarities)
                                        # similarities_stdev = statistics.stdev(cosine_similarities)
                                        # https://www.researchgate.net/post/Determination_of_threshold_for_cosine_similarity_score
                                        # threshold = similarities_avg + 0.75 * similarities_stdev
                                        threshold = 0.4
                                        # pair the detected face with most similar face from history
                                        most_similar_tuple = max(matching_history_cosine_similarities, key=lambda x: x[1])
                                        if most_similar_tuple[1] > threshold:
                                            matching_person_id = most_similar_tuple[0]
                                            matching_person = self.detected_people_history[matching_person_id]
                                            # matching_face.descriptor = face.descriptor
                                            # descriptor_new = (matching_person.pedestrian.descriptor + pedestrian.descriptor)/2
                                            matching_person.pedestrian.descriptor = person.pedestrian.descriptor

                                            time_current = datetime.datetime.now()
                                            # elapsed_time_milliseconds = (time_current - start).microseconds / 1000
                                            matching_person.milliseconds_present += (time_current - matching_person.last_seen).microseconds / 1000
                                            matching_person.update_last_seen()
                                            matching_person.detections_count += 1
                                            match_found = True
                                            matched_ids.append(matching_person_id)
                                            person = matching_person
                                if not match_found:
                                    self.last_person_id += 1
                                    person.id = self.last_person_id
                                    detected_people_new.append(person)


                        person.check_integrity()

                        detections.append(person)
                for person in detected_people_new:
                    self.detected_people_history[person.id] = person
                self.results.put(detections)

                self.inferred_request[dev] = 0
            else:
                heapq.heappush(self.heap_request, (cnt, dev))

        except:
            import traceback
            traceback.print_exc()


class AgeGenderDetection:
    def __init__(self, plugin, num_requests=1):
        net = IENetwork(model='models/age-gender-recognition-retail-0013-fp16.xml',
                        weights='models/age-gender-recognition-retail-0013-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.info("Loading age-gender-recognition-retail-0013-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.info(
            "age-gender-recognition-retail-0013-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net

    def detect(self, face_rect):
        blob = cv2.dnn.blobFromImage(face_rect, size=(self.w, self.h), swapRB=False, crop=False)
        inf_start = time.time()
        self.exec_net.infer({self.input_blob: blob})
        res = self.exec_net.requests[0].outputs
        age = int(res['age_conv3'] * 100)
        prob_female = res['prob'][0][0][0][0]
        prob_male = res['prob'][0][1][0][0]
        inf_end = time.time()
        det_time = inf_end - inf_start
        logger.debug("Age and gender estimation time: {}".format(det_time))
        return {'male': prob_male, 'female': prob_female, 'age': age}


class HeadPoseDetection:
    def __init__(self, plugin, num_requests=1):
        net = IENetwork(model='models/head-pose-estimation-adas-0001-fp16.xml',
                        weights='models/head-pose-estimation-adas-0001-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.info("Loading head-pose-estimation-adas-0001-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.info(
            "head-pose-estimation-adas-0001-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net

    def detect(self, face_rect):
        blob = cv2.dnn.blobFromImage(face_rect, size=(self.w, self.h), swapRB=False, crop=False)
        inf_start = time.time()
        self.exec_net.infer({self.input_blob: blob})
        res = self.exec_net.requests[0].outputs
        pitch = res['angle_p_fc'][0][0]
        yaw = res['angle_y_fc'][0][0]
        roll = res['angle_r_fc'][0][0]
        inf_end = time.time()
        det_time = inf_end - inf_start
        logger.debug("Head pose estimation time: {}".format(det_time))
        return pitch, yaw, roll


class EmotionDetection:
    def __init__(self, plugin, num_requests=1):
        net = IENetwork(model='models/emotions-recognition-retail-0003-fp16.xml',
                        weights='models/emotions-recognition-retail-0003-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.info("Loading emotions-recognition-retail-0003-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.info(
            "emotions-recognition-retail-0003-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net

    def detect(self, face_rect):
        blob = cv2.dnn.blobFromImage(face_rect, size=(self.w, self.h), swapRB=False, crop=False)
        inf_start = time.time()
        self.exec_net.infer({self.input_blob: blob})
        res = self.exec_net.requests[0].outputs
        neutral, happy, sad, surprise, anger = res['prob_emotion'][0, :, 0, 0]
        inf_end = time.time()
        det_time = inf_end - inf_start
        logger.debug("Emotion estimation time: {}".format(det_time))
        return {'neutral': neutral, 'happy': happy, 'sad': sad, 'surprise': surprise, 'anger': anger}


class PersonIdentification:
    def __init__(self, plugin, num_requests=1):
        net = IENetwork(model='models/person-reidentification-retail-0031-fp16.xml',
                        weights='models/person-reidentification-retail-0031-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.info("Loading person-reidentification-retail-0031-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.info(
            "person-reidentification-retail-0031-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net

    def detect(self, pedestrian_rect):
        blob = cv2.dnn.blobFromImage(pedestrian_rect, size=(self.w, self.h), swapRB=False, crop=False)
        inf_start = time.time()
        self.exec_net.infer({self.input_blob: blob})
        res = self.exec_net.requests[0].outputs
        inf_end = time.time()
        det_time = inf_end - inf_start
        logger.debug("Person identification time: {}".format(det_time))
        return res['ip_reid'][0]


class FaceIdentification:
    def __init__(self, plugin, num_requests=1):
        net = IENetwork(model='models/face-reidentification-retail-0071-fp16.xml',
                        weights='models/face-reidentification-retail-0071-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.info("Loading face-reidentification-retail-0071-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.info(
            "face-reidentification-retail-0071-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net

    def detect(self, face_rect):
        blob = cv2.dnn.blobFromImage(face_rect, size=(self.w, self.h), swapRB=False, crop=False)
        self.exec_net.infer({self.input_blob: blob})
        res = self.exec_net.requests[0].outputs
        result = None
        keys = []
        for k, v in res.items():
            keys.append(k)
            if result is not None:
                logger.warning('Warning, multiple keys detected!')
            result = v[0, :, 0, 0]
        return result





class VehicleLicencePlateDetection:
    def __init__(self, plugin, num_requests=2):
        net = IENetwork(model='models/vehicle-license-plate-detection-barrier-0106-fp16.xml',
                                weights='models/vehicle-license-plate-detection-barrier-0106-fp16.bin')
        self.input_blob = next(iter(net.inputs))
        logger.info("Loading vehicle-detection-adas-0002-fp16 IR to the plugin...")
        self.exec_net = plugin.load(network=net, num_requests=num_requests)
        # Read and pre-process input image
        self.n, self.c, self.h, self.w = net.inputs[self.input_blob].shape
        logger.info(
            "vehicle-detection-adas-0002-fp16 net.input.shape(n, c, h, w):{}".format(
                net.inputs[self.input_blob].shape))
        del net

    def detect(self, frame, cur_request_id, next_request_id, log=None):
        detections = []
        try:
            # Prepare input blob and perform an inference
            blob = cv2.dnn.blobFromImage(frame, size=(self.w, self.h), swapRB=False, crop=False)
            self.exec_net.start_async(
                request_id=next_request_id, inputs={self.input_blob: blob})
            if self.exec_net.requests[cur_request_id].wait(-1) == 0:
                # Parse detection results of the current request
                out = self.exec_net.requests[cur_request_id].outputs
                out_detections = out['DetectionOutput_']
                _, _, detections_count, _ = out_detections.shape

                for box_index in range(detections_count):
                    image_id, label, conf, x_min, y_min, x_max, y_max = out_detections[0, 0, box_index, :]
                    if label > 0:
                        detections.append([image_id, label, conf, x_min, y_min, x_max, y_max])
        except Exception as e:
            log.error('An arror occurred while trying to make detections:', e)

        return detections

    def detect_sync(self, frame):
        blob = cv2.dnn.blobFromImage(frame, size=(self.w, self.h), swapRB=False, crop=False)
        self.exec_net.infer({self.input_blob: blob})
        out = self.exec_net.requests[0].outputs
        self.exec_net.infer({self.input_blob: blob})
        out1 = self.exec_net.requests[0].outputs
        out_detections = out['DetectionOutput_']
        _, _, detections_count, _ = out_detections.shape

        for box_index in range(detections_count):
            image_id, label, conf, x_min, y_min, x_max, y_max = out_detections[0, 0, box_index, :]
            if label > 1:
                print(label)
