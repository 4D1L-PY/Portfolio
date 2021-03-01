import os
import base64
import io
import requests
import threading
import PIL
import kivy
from kivymd.app import MDApp
from kivymd.uix.card import MDCard
from kivymd.uix.spinner import MDSpinner
from kivymd.uix.button import MDFloatingActionButton
from kivy.uix.image import AsyncImage
from kivy.metrics import dp
from kivy.utils import get_color_from_hex
from kivy.properties import ListProperty
from plyer import filechooser


###################################################################################################################################
#                                                   APP CONFIGURATION                                                             #
###################################################################################################################################

# Kivy environment variables

os.environ['KIVY_IMAGE'] = 'pil'
os.environ['KIVY_CAMERA'] = 'android'

# API URL

DBD_URL = "https://dog-breed-detector-306216.ew.r.appspot.com/api"

# Accuracy threshold colors

GREEN = {'rgba': (0, 177, 106, 1),
         'hex': '#33CC00'}

ORANGE = {'rgba': (230, 126, 34, 1),
          'hex': '#FF9900'}

RED = {'rgba': (150, 40, 27, 1),
       'hex': '#CC0000'}


def normalize_rgba(rgba_tuple):
	return tuple(map(lambda i: i/255 if i > 1 else i, rgba_tuple))


COLOR_CONVERTOR = {'rgba': normalize_rgba,
                   'hex': get_color_from_hex}


###################################################################################################################################
#                                             DOG BREED DETECTOR MAIN BUTTON                                                      #
###################################################################################################################################


class DogBreedDetectorButton(MDFloatingActionButton):
	"""
	Button that triggers 'filechooser.open_file()' and processes
	the data response from filechooser Activity.
	"""

	selection = ListProperty([])

	def choose(self):
		"""
		Call plyer filechooser API to run a filechooser Activity.
		"""
		filechooser.open_file(on_selection=self.handle_selection)

	def handle_selection(self, selection):
		"""
		Callback function for handling the selection response from Activity.
		"""
		self.selection = selection

	def on_selection(self, *a, **k):
		"""
		Update DogCard widget after FileChoose.selection is changed
		via FileChoose.handle_selection.
		"""
		mdapp = MDApp.get_running_app()
		mdapp.display_card(icon_color=(1 ,1, 1, 1))
		mdapp.root.ids.dog_card.upload_dog_image(self.selection[0])
        

###################################################################################################################################
#                                             DOG BREED DETECTOR RESULTS CARD                                                     #
###################################################################################################################################   


class DogCard(MDCard):
	"""
	Material design card which display main results from dog breed detection  
	"""

	# Original dog image
	dog_img_filename = None
	dog_async_img_widget = None
	# Grad-CAM dog image
	grad_CAM_img_as_bytes = None
	grad_CAM_img_filename = None
	grad_CAM_async_img_widget = None
	# Displayed image
	displayed_img = 'Original'
	# Spinner
	spinner_widget = None

	def display_spinner(self):
		self.spinner_widget = MDSpinner()
		self.spinner_widget.size_hint = None, None # .25, .5
		self.spinner_widget.size = (dp(250), dp(250))
		self.spinner_widget.pos_hint = {'center_x': .5, 'center_y': .5}
		self.spinner_widget.active = True
		#self.spinner_widget.opacity = 1
		self.spinner_widget.color = [0, 0, 0, 1]
		self.add_widget(self.spinner_widget)
		self.padding = "45dp"

	def delete_spinner(self):
		self.padding = 0
		self.remove_widget(self.spinner_widget)

	def build_async_image(self, data, filename, img_size=(1200, 700), rotation_angle=90):
		# Read uploaded dog image (+ resize & save locally)
		pil_img = PIL.Image.open(data)
		# pil_img_width, pil_img_height = pil_img.size
		# # Rotate image if picture was taken in portrait mode
		# if pil_img_width < pil_img_height:
		# 	pil_img = pil_img.rotate(rotation_angle)
		pil_img = pil_img.resize(img_size, PIL.Image.ANTIALIAS)
		pil_img.save(filename, format='JPEG', optimize=True, quality=90)
		return AsyncImage(source=filename,
		                  nocache=True,
		                  size_hint_x=1.,
		                  size_hint_y=1.,
		                  allow_stretch=False, # old value -> True
		                  keep_ratio=False) # old value -> False

	def remove_previous_dog_image_widgets(self):
		os.remove(self.dog_img_filename)
		self.remove_widget(self.dog_async_img_widget)
		if self.grad_CAM_async_img_widget is not None:
			self.remove_widget(self.grad_CAM_async_img_widget)

	def upload_dog_image(self, img_path):
		# Get running app
		mdapp = MDApp.get_running_app()
		# Handle next upload
		if self.dog_img_filename is not None:
			self.remove_previous_dog_image_widgets()
		self.dog_img_filename = mdapp.user_data_dir + '/' + img_path.split('/')[-1]
		# gc_filename_pattern = self.dog_img_filename.split('.')
		self.grad_CAM_img_filename = mdapp.user_data_dir + '/' + 'grad_CAM.jpg' # gc_filename_pattern[0] + '_gc.' + gc_filename_pattern[1]
		self.dog_async_img_widget = self.build_async_image(data=img_path, filename=self.dog_img_filename)
		# Display spinner
		mdapp.card_spinner()
		# Initiate dog breed detection
		self.detect_dog_breed_thread = threading.Thread(target=mdapp.detect_dog_breed)
		self.detect_dog_breed_thread.start()
				
	def display_grad_CAM_image(self):
		# Delete previous displayed image widget
		self.remove_widget(self.dog_async_img_widget)
		# Build Grad-CAM widget
		self.grad_CAM_async_img_widget = self.build_async_image(data=self.grad_CAM_img_as_bytes, filename=self.grad_CAM_img_filename)
		# Add Grad-CAM widget
		self.add_widget(self.grad_CAM_async_img_widget)

	def display_dog_image(self):
		# Delete Grad-CAM widget (process one image widget at a time)
		if self.grad_CAM_async_img_widget is not None:
			self.remove_widget(self.grad_CAM_async_img_widget)
		try:
			self.add_widget(self.dog_async_img_widget)
		except kivy.uix.widget.WidgetException:
			print('Widget already created !')


###################################################################################################################################
#                                            DOG BREED DETECTOR MAIN APP CLASS                                                    #
################################################################################################################################### 


class MainApp(MDApp):
	# API results
	breed_label = ""
	accuracy_value = ""
	# Define whitespace str in order to center displayed accuracy value (TO DO -> improve centering method)
	breed_spacing_center = '    '
	accuracy_spacing_center = '       '

	def display_card(self, icon_color, text_color=None):
		############################################################
		#                   BREED LABEL TOOLBAR                    #
		############################################################
		# Display/hide breed label icon
		self.change_icon_and_text_colors(toolbar_id='breed_label',
		                                  icon_color=icon_color)
		# Display breed label
		self.root.ids.breed_label.title = self.breed_label
		############################################################
		#                  BREED ACCURACY TOOLBAR                  #
		############################################################
		# Display/hide accuracy icon
		self.change_icon_and_text_colors(toolbar_id='accuracy_value',
		 	                             icon_color=icon_color,
		 	                             text_color=text_color)
		# Display accuracy value
		self.root.ids.accuracy_value.title = self.accuracy_value

	def card_spinner(self, disable=False):
		if disable:
			self.root.ids.app_title.title = "DOG BREED DETECTOR"
			self.root.ids.dog_card.delete_spinner()
			self.root.ids.dog_card.display_dog_image()
		else:
			# Display spinner on Card
			self.root.ids.app_title.title = "Detecting dog breed ... "
			self.root.ids.dog_card.display_spinner()

	def api_post_request(self):
		# Make API call
		image_filename = self.root.ids.dog_card.dog_img_filename
		results_json = requests.post(url=DBD_URL,
		                             files={'file': open(image_filename, 'rb')}).json()
		return results_json

	def detect_dog_breed(self):
		# Make API call
		results_json = self.api_post_request()
		# Processing API results
		self.breed_label = results_json['breed'] + self.breed_spacing_center
		# Round accuracy value
		accuracy = round(results_json['accuracy'])
		# Get accuracy color
		accuracy_color = self.get_accuracy_color(accuracy)
		# Convert accuracy value as str
		self.accuracy_value = str(accuracy) + ' %' + self.accuracy_spacing_center
		# Get Grad-CAM
		grad_cam_img_str = results_json['grad-cam']
		# Load Grad-CAM image as bytes
		self.root.ids.dog_card.grad_CAM_img_as_bytes = io.BytesIO(base64.b64decode(grad_cam_img_str))
		# Disable spinner and display API results
		self.card_spinner(disable=True)
		self.display_card(icon_color=(0, 0, 0, 1), text_color=accuracy_color)

	def get_accuracy_color(self, accuracy, lower_threshold=75, upper_threshold=90, mode='rgba'):
		if accuracy > upper_threshold:                                  # Green
			accuracy_color = GREEN[mode]
		elif lower_threshold < accuracy < upper_threshold:              # Orange
			accuracy_color = ORANGE[mode]
		else:                                                           # Red
			accuracy_color = RED[mode]
		accuracy_color = COLOR_CONVERTOR[mode](accuracy_color)
		return accuracy_color
	
	def update_icon_color(self, action_items, icon_color):
		# Update icon color
		if action_items is not None:
			for item in action_items:
				item.text_color = icon_color
		return
	
	def change_icon_and_text_colors(self, toolbar_id, icon_color, text_color=None):
		# If text color is None then icon color is also applied to text
		if text_color is None:
			text_color = icon_color
		# Get MDToolBar widget by id
		toolbar = self.root.ids[toolbar_id]
		# Define 'specific_text_color' which colorize icon & text
		toolbar.specific_text_color = text_color
		# Get icon item from MDToolBar
		left_action_items = toolbar.ids.left_actions.children
		right_action_items = toolbar.ids.right_actions.children
		all_action_items = [left_action_items, right_action_items]
		# Apply icon color to all action items
		for action_items in all_action_items:
			self.update_icon_color(action_items, icon_color)
		return

	def image_displayer_callback(self):
		# Display Grad-CAM
		if self.root.ids.dog_card.displayed_img is 'Grad-CAM':
			self.root.ids.dog_card.display_dog_image()
			self.root.ids.dog_card.displayed_img = 'Original'
		# Display original dog image
		elif self.root.ids.dog_card.displayed_img is 'Original':
			self.root.ids.dog_card.display_grad_CAM_image()
			self.root.ids.dog_card.displayed_img = 'Grad-CAM'
		return

	def on_start(self):
		# Get main Android permissions in order to upload an image file already stored
		from android.permissions import request_permissions, Permission
		def callback(permissions, results):
			print(permissions, results)
		request_permissions([Permission.INTERNET,
			                 Permission.CAMERA,
		                     Permission.WRITE_EXTERNAL_STORAGE,
		                     Permission.READ_EXTERNAL_STORAGE],
		                     callback)
	
	def build(self):
		return 


# Run main app
if __name__ == '__main__':
    MainApp().run()
