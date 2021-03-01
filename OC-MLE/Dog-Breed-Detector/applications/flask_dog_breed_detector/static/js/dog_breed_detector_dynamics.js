$(document).ready(function() {
	// Enable/disable spinner 
	display_spinner(input_form_id='#DogImageUploaderInput')
	// Activate Dog Image Zoom
	$('.materialboxed').materialbox();
	// Display original image file and Grad-CAM version based on switch options 
	display_switch_options(switch_class='.switch', option_id='#DogImage')
	// Color predicted breed accuracy based on threshold value
	color_accuracy(accuracy_id='#PredictedBreedAccuracy', optional_id='p')
});


function display_spinner(input_form_id,
						 upload_button_id='#DogImageUploader',
						 upload_button_logo_id='#DogImageUploaderButtonLogo',
						 button_logo_filename='dog_breeds_detector_button_logo.svg',
                         card_id='#DogBreedDetectedCard',
                         breed_label_id='#PredictedBreedLabel') {
	// Replace button logo with spinner
	$(input_form_id).click(function() {
		// Get upload button height
		var upload_button_height = $(upload_button_id).height();
		// Build spinner
		var spinner = '<div class="lds-ring"><div></div><div></div><div></div><div></div></div>'
		// Replace upload button with spinner
		$(upload_button_logo_id).replaceWith(spinner);
		// Update spinner height
		$('.lds-ring').css('height', upload_button_height + 'px');
        // Hide predicted results card
        $(card_id).hide();
    });
	// Replace spinner with button logo
    if ($(breed_label_id).text() != null) {
    	// Display dog breed card
    	$(card_id).show()
    	// Display uploading button button (replace spinner with button)
    	var button_logo = '<img id="'+upload_button_logo_id+'" src="{{ url_for(\'static/img\', filename=\''+button_logo_filename+'\') }}"/>'
    	$('.lds-ring').replaceWith(button_logo)
    }	
}


function display_switch_options(switch_class,
                                option_id,
                                ajax_url='/_display_grad_cam',
                                first_option='grad_cam', 
                                second_option='original_img', 
                                data_folder='static/img/dogs/') {
	// Update switch options with AJAX
	$(switch_class).find("input[type=checkbox]").on("change",function() {
		// Get switch status
		var switch_status = $(this).prop('checked');
	    // AJAX POST request
	    $.ajax({
	    	method: 'POST',
	        url: ajax_url,
	        dataType: "json",
	        success: function(data){
	        	// Update displayed image based on switch status
	        	if (switch_status) {
	            	$(option_id).attr('src', data_folder + data[first_option]);
	        	}
        		else {
        			$(option_id).attr('src', data_folder + data[second_option]);
        		}
        	}
	    });
	});
}


function color_accuracy(accuracy_id,
	                    optional_id='',
	                    upper_thr=90,
	                    lower_thr=75, 
	                    high_accuracy_color='green',
	                    medium_accuracy_color='orange',
	                    low_accuracy_color='red') {
	// Get predicted breed accuracy
	var accuracy = $(accuracy_id).text().match(/\d+/)[0];
    // Color predicted breed accuracy value by thresholds
    if (parseInt(accuracy) > upper_thr) {
       $(accuracy_id + ' ' + optional_id).css('color', high_accuracy_color);
    } else if (parseInt(accuracy) >= lower_thr && parseInt(accuracy) <= upper_thr) {
       $(accuracy_id + ' ' + optional_id).css('color', medium_accuracy_color);
    } else {
       $(accuracy_id + ' ' + optional_id).css('color', low_accuracy_color);
    }
}
