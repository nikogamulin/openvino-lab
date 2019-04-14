from numpy import dot
from numpy.linalg import norm
from datetime import datetime


def get_cosine_similarity(descriptor_1, descriptor_2):
    return dot(descriptor_1, descriptor_2) / (norm(descriptor_1) * norm(descriptor_2))


class Pedestrian:
    def __init__(self, confidence, min_x, min_y, max_x, max_y):
        self.descriptor = None
        self.confidence = confidence
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.x = (min_x + max_x) / 2
        self.y = (min_y + max_y) / 2

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def get_cosine_similarity(self, desc):
        return dot(self.descriptor, desc)/(norm(self.descriptor)*norm(desc))

    def get_absolute_positions(self, W, H):
        min_x = int(W * self.min_x)
        max_x =int(W * self.max_x)
        min_y =int(H * self.min_y)
        max_y = int(H * self.max_y)
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > W:
            max_x = W
        if max_y > H:
            max_y = H
        return min_x, min_y, max_x, max_y


class Face:
    def __init__(self, confidence, min_x, min_y, max_x, max_y):
        self.last_seen = datetime.now()
        self.first_seen = datetime.now()
        self.milliseconds_present = 0
        self.detections_count = 1
        self.descriptor = None
        self.id = None
        self.age = None
        self.gender = None
        self.emotion = None
        self.confidence = confidence
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y
        self.x = (min_x + max_x)/2
        self.y = (min_y + max_y)/2
        self.head_orientation = None

    def set_descriptor(self, descriptor):
        self.descriptor = descriptor

    def get_absolute_positions(self, W, H):
        min_x = int(W * self.min_x)
        max_x =int(W * self.max_x)
        min_y =int(H * self.min_y)
        max_y = int(H * self.max_y)
        if min_x < 0:
            min_x = 0
        if min_y < 0:
            min_y = 0
        if max_x > W:
            max_x = W
        if max_y > H:
            max_y = H
        return min_x, min_y, max_x, max_y

    def get_time_elapsed_from_last_detection(self):
        time_now = datetime.now()
        return (time_now - self.last_seen).total_seconds()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def get_label(self):
        if self.age is not None and self.gender is not None:
            label = " ({}, {})".format(self.gender, self.age)
            return label
        return None

    def get_label_head_pose(self):
        if self.head_orientation is None:
            return None
        label_head_pose = "Yaw: {}, Pitch: {}, Roll: {}".format(
            self.head_orientation["yaw"], self.head_orientation["pitch"], self.head_orientation["roll"])
        return label_head_pose

    def get_label_emotion(self):
        if self.emotion is None:
            return None
        itemEmotions = []
        for k, v in self.emotion.items():
            itemEmotions.append("{}: {}".format(k, format(v, '.2f')))
        label_emotion = ", ".join(itemEmotions)
        return label_emotion


class Person:
    def __init__(self):
        self.id = None
        self.pedestrian = None
        self.face = None
        self.last_seen = datetime.now()
        self.first_seen = datetime.now()
        self.milliseconds_present = 0
        self.detections_count = 1

    def set_face(self, confidence, x_min, y_min, x_max, y_max):
        self.face = Face(confidence, x_min, y_min, x_max, y_max)

    def set_pedestrian(self, confidence, x_min, y_min, x_max, y_max):
        self.pedestrian = Pedestrian(confidence, x_min, y_min, x_max, y_max)

    def get_time_elapsed_from_last_detection(self):
        time_now = datetime.now()
        return (time_now - self.last_seen).total_seconds()

    def update_last_seen(self):
        self.last_seen = datetime.now()

    def check_integrity(self):
        if self.face is not None and self.pedestrian is not None:
            if self.face.x < self.pedestrian.min_x or self.face.x > self.pedestrian.max_x \
                    or self.face.y < self.pedestrian.min_y or self.face.y > self.pedestrian.max_y:
                print("Face is out of body bounds!")

    def get_label(self):
        label = "ID: {}".format(self.id)
        label += " Present {} s".format(int(self.milliseconds_present / 1000))
        # print(label)
        return label