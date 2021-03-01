import tensorflow_hub as hub
# For measuring the inference time.
import time
from src.cv.image_preprocessor import tf, Image, load_img, display_image, download_and_resize_image, draw_boxes
from src.datacleaner import format_run_time


class ObjectDetector:

    def __init__(self, image_type='path', module='inception_resnet_v2'):
        self.image_type = image_type
        if module is 'inception_resnet_v2':
            self.module_handle = f"https://tfhub.dev/google/faster_rcnn/openimages_v4/{module}/1"
        elif module is 'mobilenet_v2':
            self.module_handle = f"https://tfhub.dev/google/openimages_v4/ssd/{module}/1"
        else:
            raise Exception(f'{module} not implemented')
        self.detector = hub.load(self.module_handle).signatures['default']
        self.results = None
        self.classes_detected = False
        # Main statistics
        self.detected_entities_counts = None

    def detect_objects(self,
                       image,
                       classes_kept=None,
                       threshold=0.1,
                       boxes=False,
                       crop=False,
                       crop_by_score='max',
                       display_original_image=False,
                       display=True,
                       verbose=True,
                       return_image=False):
        if type(classes_kept) in [list, tuple]:
            classes_kept = list(map(str.lower, classes_kept))
        else:
            classes_kept = classes_kept.lower()
        # Get image path
        if self.image_type is 'url':
            image_path = download_and_resize_image(image, display=display_original_image)
        else:
            image_path = image
        # Load image as Tensor
        img = load_img(image_path)
        converted_img = tf.image.convert_image_dtype(img, tf.float32)[tf.newaxis, ...]
        # Compute inference time
        start_time = time.time()
        self.results = self.detector(converted_img)
        end_time = time.time()
        # Get statistics from detected entitites
        self.get_count_by_detected_entities()
        result = {key: value.numpy() for key, value in self.results.items()}
        if verbose:
            print("Found {} objects.".format(len(result["detection_scores"])))
            print("Inference time: ", end_time - start_time)
        # Convert image tensor as numpy array
        image_as_array = img.numpy()
        # Enable drawing detection boxes
        if boxes:
            # Draw boxes
            image_as_array = draw_boxes(image_as_array,
                                        result["detection_boxes"],
                                        result["detection_class_entities"],
                                        result["detection_scores"],
                                        selected_classes=classes_kept,
                                        min_score=threshold)
        # Enable cropping image by bounding box (threshold from score)
        if crop:
            try:
                img_height, img_width, img_channel = image_as_array.shape
                detected_entities = enumerate(self.results['detection_class_entities'].numpy())
                detected_entities_scores = enumerate(self.results['detection_scores'].numpy())
                # Crop image by selected class
                if classes_kept:
                    idxs = [i for i, entity in detected_entities if entity.decode("utf-8").lower() in classes_kept]
                    scores = [(i, score) for i, score in detected_entities_scores if i in idxs]
                else:
                    scores = [(i, score) for i, score in detected_entities_scores]
                # Crop image by selecting best score
                if crop_by_score is 'max':
                    best_score_idx, best_score = max(scores, key=lambda item: item[1])
                    # Get detection box from best score index value
                    box = self.results['detection_boxes'][best_score_idx]
                ymin, xmin, ymax, xmax = box
                x_up = int(xmin * img_width)
                y_up = int(ymin * img_height)
                x_down = int(xmax * img_width)
                y_down = int(ymax * img_height)
                # Crop image by bounding box
                image_as_array = image_as_array[y_up:y_down, x_up:x_down, :]
                self.classes_detected = True
            except:
                if verbose:
                    print('{} class(es) not detected'.format(classes_kept))
                self.classes_detected = False
        # Display image
        if display:
            display_image(image_as_array)
        if return_image:
            return image_as_array

    def get_count_by_detected_entities(self):
        detected_entities_as_binary = self.results['detection_class_entities'].numpy().tolist()
        detected_entities = list(map(lambda binary_entity: binary_entity.decode('utf-8'), detected_entities_as_binary))
        detected_entities_counts = dict()
        for entity in detected_entities:
            detected_entities_counts[entity] = detected_entities_counts.get(entity, 0) + 1
        self.detected_entities_counts = detected_entities_counts

    def search_detected_entity(self, entity):
        detected_entities = enumerate(self.results['detection_class_entities'].numpy())
        detected_entities_scores = enumerate(self.results['detection_scores'].numpy())
        indexes = [i for i, detected_ent in detected_entities if detected_ent.decode("utf-8").lower() == entity.lower()]
        detected_entities_scores = [(i, score) for i, score in detected_entities_scores if i in indexes]
        return detected_entities_scores


def crop_images_with_object_detection(images, model='mobilenet_v2', img_class='dog', v=False):
    # Use Tensorflow Hub Object Detection API (default model is mobilenet_v2 which is small and fast)
    detector = ObjectDetector(module=model)
    classes_detected = 0
    images_classes_undetected = []
    # Compute inference time
    start_time = time.time()
    for img_path in images:
        cropped_img = detector.detect_objects(img_path,
                                              classes_kept=img_class,
                                              crop=True,
                                              display=False,
                                              verbose=v,
                                              return_image=True)
        if detector.classes_detected:
            classes_detected += int(detector.classes_detected)
        else:
            images_classes_undetected.append(img_path)
        cropped_img = Image.fromarray(cropped_img)
        cropped_img.save(img_path)
    print('{}s detected : {}/{} ({}%)'.format(img_class,
                                              classes_detected,
                                              len(images),
                                              round(classes_detected/len(images)*100, 1)))
    print("{} undetected {}s".format(len(images_classes_undetected), img_class))
    end_time = time.time()
    run_time = format_run_time(time.strftime('%H:%M:%S', time.gmtime(end_time-start_time)))
    print("Inference time: ", run_time)
    return images_classes_undetected
