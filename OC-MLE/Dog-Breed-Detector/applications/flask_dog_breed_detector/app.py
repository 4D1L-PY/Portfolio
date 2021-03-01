import atexit
import base64
import io
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# import flask_monitoringdashboard as dashboard
from flask import Flask
from flask import request, render_template, session, redirect, url_for, jsonify, send_file
from utils.dog_breeds_detection_utils import delete_image_files, schedule_image_files_deletion, get_dog_breed_detector_results
from config.app_config import *

########################################################################################################################
#                                               PREPROCESSING                                                          #
########################################################################################################################


# Main app configuration

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'dbd'
app.config['SESSION_TYPE'] = 'filesystem'

# Add Flask monitoring dashboard
# dashboard.bind(app)


########################################################################################################################
#                                               MAIN VIEW                                                              #
########################################################################################################################


@app.route('/', methods=['GET', 'POST'])
def upload_and_detect():
    """

    Route which detect a dog breed from an uploaded dog image and render main template (HTML view) 
    
    """
    if request.method == "POST":
        # Request image file
        image_file = request.files['image']
        if image_file:
            # Get main results from Dog Breed Detector
            results = get_dog_breed_detector_results(image_file)
            gram_cam_img_filename = 'grad_cam_' + image_file.filename
            results['grad-cam'].save(UPLOAD_FOLDER + gram_cam_img_filename)
            # Create session variables to display images based on switch button value
            session['original_img'] = image_file.filename
            session['grad_cam_img'] = gram_cam_img_filename
            return render_template('index.html',
                                    dog_image=image_file.filename,
                                    breed=results['breed'],
                                    score=int(round(results['accuracy'])), 
                                    app_logo=APP_LOGO_NAME, 
                                    upload_button_logo=UPLOAD_BUTTON_LOGO_NAME)

    return render_template('index.html',
                           dog_image=None,
                           app_logo=APP_LOGO_NAME,
                           upload_button_logo=UPLOAD_BUTTON_LOGO_NAME)


########################################################################################################################
#                                               GRAD CAM VIEW                                                          #
########################################################################################################################


@app.route('/_display_grad_cam', methods=['POST'])
def display_grad_cam():
    """
    
    Route which enable displaying dog image Grad CAM version
    
    """
    # Get image filenames from session
    grad_cam_img_filename = session.get('grad_cam_img', None)
    original_img_filename = session.get('original_img', None)
    return jsonify({'grad_cam': grad_cam_img_filename,
                    'original_img': original_img_filename})


########################################################################################################################
#                                                     API                                                              #
########################################################################################################################


@app.route('/api', methods=['POST'])
def api():
    """
    API route (used by Dog Breed Detector Android App version)

    """
    # Request image file
    image_file = request.files['file']
    # Get main results from Dog Breed Detector
    results = get_dog_breed_detector_results(image_file)
    # Buffer Grad-CAM PIL image as bytes 
    buffered = io.BytesIO()
    results['grad-cam'].save(buffered, format="PNG")
    # Encode Grad-CAM image as string
    grad_cam_img_str = base64.b64encode(buffered.getvalue()).decode('ascii')  
    return jsonify({'breed': results['breed'],
                    'accuracy': results['accuracy'],
                    'grad-cam': grad_cam_img_str})

    
########################################################################################################################
#                                                   RUN APP                                                            #
########################################################################################################################


if __name__ == '__main__':
    # Schedule dog image files auto-deletion at 00h
    # schedule_image_files_deletion(scheduled_time='00:00')
    # Delete all dog image files when the application is down
    atexit.register(delete_image_files, UPLOAD_FOLDER+'*')
    app.run(host=SERVER['HOST'],
            port=SERVER['PORT'],
            debug=True)
