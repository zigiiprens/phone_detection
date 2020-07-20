from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import keras
import argparse
import numpy as np
from PIL import Image
from tflite_runtime.interpreter import Interpreter

def parser():
    parser = argparse.ArgumentParser(
        description='Phone detection system Hayai')
    
    parser.add_argument('--input', type=str, default='data/phones2.jpg',
                        help='Input image filepath')
    parser.add_argument('--model', type=str, default='models/detect.tflite',
                        help='Input model filepath')
    parser.add_argument('--labels', type=str, default='models/coco_labels.txt',
                        help='Input label filepath')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Min thresold acceptes')
    return parser.parse_args()

class PhoneDetection():
    """ 
    PhoneDetection class analyze the input image and find the phone found in the frame
        
        Parameters
        ----------
        image_path : string
            The path of the image.
        model_path : string
            The path of the model.
        label_path : string
            The path of the labels file.
        threshold : float
            The min threshold accepted.
        
        
        Information
        -----------
    """

    def __init__(self, image_path, model_path, label_path, threshold):
        self.image_path = image_path
        self.model_path = model_path
        self.label_path = label_path
        self.threshold = threshold
        self.image_width = 0
        self.image_height = 0
        self.average_time = None
        self.interpreter = None
        self.counter_cell = 0
        self.location_cell = []
        self.confidence_cell = []

        # Load phone detector model and labels in the __init__
        self.load_labels()
        self.label_id_cell_phone = 76
        
    
    def load_labels(self):
        """Loads the labels file. Supports files with or without index numbers."""
        with open(self.label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            self.labels = {}
            for row_number, content in enumerate(lines):
                pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
                if len(pair) == 2 and pair[0].strip().isdigit():
                    self.labels[int(pair[0])] = pair[1].strip()
                else:
                    self.labels[row_number] = pair[0].strip()
        print("==============> Labels {}" .format(self.labels))

    def load_model(self):
        interpreter = Interpreter(self.model_path)
        interpreter.allocate_tensors()
        _, self.input_height, self.input_width, _ = interpreter.get_input_details()[0]['shape']
        return interpreter

    def set_input_tensor(self, interpreter):
        """Sets the input tensor."""
        interpreter.allocate_tensors()
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:,:] = self.image


    def get_output_tensor(self, interpreter, index):
        """Returns the output tensor at the given index."""
        interpreter.allocate_tensors()
        output_details = interpreter.get_output_details()[index]
        tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
        return tensor

    def detect_objects(self, interpreter):
        """Returns a list of detection results, each a dictionary of object info."""

        self.set_input_tensor(interpreter)
        interpreter.allocate_tensors()
        interpreter.invoke()

        # Get all output details
        boxes = self.get_output_tensor(interpreter,0)
        classes = self.get_output_tensor(interpreter,1)
        scores = self.get_output_tensor(interpreter,2)
        count = int(self.get_output_tensor(interpreter,3))

        results = []
        for i in range(count):
            if scores[i] >= self.threshold:
                result = {
                    'bounding_box': boxes[i],
                    'class_id': classes[i],
                    'score': scores[i]
                }
                results.append(result)
        return results



    def prepocess_image(self, interpreter):
        _, self.input_height, self.input_width, _ = interpreter.get_input_details()[0]['shape']

        self.image = Image.open(self.image_path).convert('RGB').resize(
              (self.input_width, self.input_height), Image.ANTIALIAS)  # pylint: disable=maybe-no-member
        self.image_width, self.image_height = self.image.size

    def annotate_objects(self, results):
        """Draws the bounding box and label for each object in the results."""

        if self.image_height and self.image_width == 0:
            self.image_width, self.image_height = 1, 1
        
        for obj in results:
            # Convert the bounding box figures from relative coordinates
            # to absolute coordinates based on the original resolution
            ymin, xmin, ymax, xmax = obj['bounding_box']
            xmin = int(xmin * self.image_width)
            xmax = int(xmax * self.image_width)
            ymin = int(ymin * self.image_height)
            ymax = int(ymax * self.image_height)

            if obj['class_id'] == self.label_id_cell_phone:
                self.location_cell.append([xmin,ymin,xmax,ymax])
                self.confidence_cell.append(obj['score'])
                self.counter_cell += 1
        
        
        print("Labels {} {} {}" .format(self.labels[obj['class_id']], obj['score'], [xmin, xmax, ymin, ymax]))

    def process(self):
        # predicted = np.argmax(self.model.predict(self.image))
        # label_map = dict((v,k) for k,v in self.emotion_dict.items()) 
        # self.predicted_label = label_map[predicted]
        pass

    def average_time_calculator(self):
        """This function calculate the average time spended to get the results."""
        pass


def main(args):
    """
    Example of output of this main function:
    ----------------------------------------
    """
    phone_detector = PhoneDetection(args.input, args.model, args.labels, args.threshold)
    interpreter = phone_detector.load_model()
    phone_detector.prepocess_image(interpreter)
    ret = phone_detector.detect_objects(interpreter)
    phone_detector.annotate_objects(ret)

    print("Number of phone detected      => {}" .format(phone_detector.counter_cell))
    print("Phone Location                => {}" .format(phone_detector.location_cell))
    print("Phone confidence              => {}" .format(phone_detector.confidence_cell))
    print("Average Process Time          => {} seconds" .format(phone_detector.average_time))


if __name__ == "__main__":
    args = parser()
    main(args)