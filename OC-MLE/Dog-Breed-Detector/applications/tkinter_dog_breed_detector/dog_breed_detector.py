# GUI modules
import tkinter as tk
from tkinter import filedialog, ttk
# Image module
from PIL import Image, ImageTk
# Custom package
from dog_breed_detector_package.config import os, sys, IMG_SIZE, BUTTONS_FONT, PREDICTIONS_FONT
from dog_breed_detector_package.utils import get_dog_breed_detector_results


def detect_dog_breed():
	"""
	
	Browse dog image, then run dog breed detection inference and display original image
	
	"""
	
	# Get all widgets from dog breeds detector main window
	all_widgets = dog_breed_detector_window.winfo_children()
	# Delete previous breed label if it exists
	if len(all_widgets) > 4:
		for w in all_widgets[-4:]:
			w.destroy()
	# Load image file from local user data
	file = filedialog.askopenfilename(initialdir=os.getcwd(),
	                                  title="Select Dog Image File",
	                                  filetypes=(("JPG File", ["*.jpg", "*.jpeg"]),
												 ("PNG File", "*.png"),
												 ("All Files", "*.*")))
	img = Image.open(file)
	results = get_dog_breed_detector_results(file)
	# Prepare original image
	img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
	dog_breed_detector_window.img = ImageTk.PhotoImage(img)
	# Prepare Grad-CAM image
	grad_cam_img = results['grad-cam']
	grad_cam_img.thumbnail(IMG_SIZE, Image.ANTIALIAS)
	dog_breed_detector_window.grad_cam_img = ImageTk.PhotoImage(grad_cam_img)
	# Display original image
	main_label.configure(image=dog_breed_detector_window.img)
	main_label.image = dog_breed_detector_window.img
	# Display Grad CAM image
	def display_grad_cam(display_type='Original'):
		if display_type == 'Grad-CAM':
			main_label.configure(image=dog_breed_detector_window.grad_cam_img)
			main_label.image = dog_breed_detector_window.grad_cam_img
		elif display_type == 'Original':
			main_label.configure(image=dog_breed_detector_window.img)
			main_label.image = dog_breed_detector_window.img
	# Create original image & Grad-CAM display buttons
	original_image_button = tk.Button(dog_breed_detector_window,
	                                  text="Original",
	                                  font=BUTTONS_FONT,
	                                  command=lambda: display_grad_cam(display_type='Original'))
	grad_cam_image_button = tk.Button(dog_breed_detector_window,
		                              text="Grad-CAM",
		                              font=BUTTONS_FONT,
		                              command=lambda: display_grad_cam(display_type='Grad-CAM'))
	# Retrieve breed label & prediction accuracy
	breed, accuracy = results['breed'].upper(), results['accuracy']
	# Display predicted breed label
	breed_label = tk.Label(dog_breed_detector_window,
	                       text=f'BREED : {breed}',
	                       fg="black",
	                       font=PREDICTIONS_FONT)
	breed_label.pack(pady=10)
	# Display predicted accuracy
	accuracy_color = "dark green" if accuracy > 90 else 'dark orange' if 75 < accuracy < 90 else 'dark red'
	accuracy_label = tk.Label(dog_breed_detector_window,
	                          text=f'ACCURACY : {accuracy}%',
	                          fg=accuracy_color,
	                          font=PREDICTIONS_FONT)
	accuracy_label.pack()
	# Pack image displayer buttons
	original_image_button.pack(side=tk.LEFT,
	                           padx=(50, 0))
	grad_cam_image_button.pack(side=tk.LEFT,
		                       padx=25)
	return


# Run main program
if __name__ == "__main__":
	# Build main program window
	dog_breed_detector_window = tk.Tk()
	# Pack main frame to program window
	main_frame = tk.Frame(dog_breed_detector_window)
	main_frame.pack(side=tk.BOTTOM,
		            padx=15,
		            pady=15)
	# Pack main label
	main_label = tk.Label(dog_breed_detector_window)
	main_label.pack()
	# Pack browse button (which initiate dog breed detector)
	browse_button = tk.Button(main_frame,
	                          text="Browse dog image",
	                          font=BUTTONS_FONT,
	                          command=detect_dog_breed)
	browse_button.pack(side=tk.LEFT)
	# Pack exit program button
	exit_button = tk.Button(main_frame,
		                    text="Exit",
		                    font=BUTTONS_FONT,
		                    command=lambda: sys.exit())
	exit_button.pack(side=tk.LEFT,
		             padx=10)
	# Add program title, window size and run it
	dog_breed_detector_window.title("Dog Breed Detector")
	dog_breed_detector_window.geometry("300x400")
	dog_breed_detector_window.mainloop()