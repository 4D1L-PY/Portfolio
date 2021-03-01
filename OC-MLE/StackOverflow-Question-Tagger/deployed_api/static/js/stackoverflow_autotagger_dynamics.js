function start_tagging_question(event) {
    // Clean previous tagging results
    $("#Results").html("");
    // Stop submit the form, we will post it manually.
    event.preventDefault();
    // Get form
    var form = $('#TextForm')[0];
    // Create an FormData object
    var data = new FormData(form);
    // Replace the submit button by loader
    var tagging_button_height = $('#start-tagging').height();
    $("#start-tagging" ).replaceWith('<div class="lds-ring"><div></div><div></div><div></div><div></div></div>');
    $('.lds-ring').css('height', tagging_button_height+'px');
    // send ajax POST request to start tagging question
    $.ajax({
        type: 'POST',
        enctype: 'multipart/form-data',
        data: data,
        url: '/_tag_question',
        processData: false,
        contentType: false,
        cache: false,
        timeout: 600000,
        success: function(data) {
                    parse_tags(data);
        },
        error: function() {
                    alert('Unexpected error');
        }
    });


}

function parse_tags(data) {
    // Build tracing map card for each location task <span class="new badge">4</span>
    var tagging_results =  $('<table id="TaggingResults" class="centered">'+
                                '<thead>'+
                                    '<tr>'+
                                        '<th>Model</th>'+
                                        '<th>Predicted tags</th>'+
                                    '</tr>'+
                                '</thead>'+
                                '<tbody>'+
                                '</tbody>'+
                              '</table>');
    // Add tagging results table to div
    $('#Results').append(tagging_results);
    // Iterate over JSON data which contains model names & predicted tags
    $.each(data, function(model, tags) {
        // Build model result
        var model_result = $('<tr>'+
                                '<td>'+model+'</td>'+
                                '<td id="'+model+'">'+'</td>'+
                             '</tr>');
        // Add model result to tagging results table
        $('#TaggingResults > tbody').append(model_result);
        // Iterate over model predicted tags
        $.each(tags, function(index) {
            // Create badged tag
            // Materialize.css badge: $('<span class="new badge blue"'+ 'data-badge-caption="'+tags[index]+'"></span>');
            var tag_badge = $('<span class="TagBadge">'+tags[index]+'</span>');
            // Add badged tag to model result
            $('#'+model).append(tag_badge);
        });
    });
    // Display reset button
    $(".lds-ring").replaceWith('<button id="reset-question" class="btn waves-effect waves-light" '+ 
                               'onclick="window.location.href="{{ url_for(\'index\') }}"">Reset'+
                               '<i class="material-icons right">clear</i></button></a>');
    
}

// Main function
$(function() {
        // Show pulsed submit button when a question is already written
        $('#QuestionTextArea').on('blur', function(e) {
            if ($("#QuestionTextArea").val()) {
                $("#start-tagging").attr('class', 'btn waves-effect waves-light pulse');
            }
            else {
                $("#start-tagging").attr('class', 'btn waves-effect waves-light');
            }
        });
        // Start background jobs
        $('#start-tagging').click(start_tagging_question);
});


