$(document).ready(function () {
    var cameraButton = $('#cameraButton');
    var sessionButton = $('#sessionButton');

    var cameraState = localStorage.getItem('cameraState') || "Open Video Camera";
    var sessionState = localStorage.getItem('sessionState') || "Start Session";

    cameraButton.val(cameraState);
    sessionButton.val(sessionState);

    cameraButton.click(function () {
        if (cameraState === 'Open Video Camera') {
            cameraState = 'Close Video Camera';
        } else {
            cameraState = 'Open Video Camera';
        }
        cameraButton.val(cameraState);

        localStorage.setItem('cameraState', cameraState);

        $('#tasksForm').submit();
    });

    sessionButton.click(function () {
        if (sessionState === 'Start Session') {
            sessionState = 'Stop Session';
        } else {
            sessionState = 'Start Session';
        }
        sessionButton.val(sessionState);

        localStorage.setItem('sessionState', sessionState);

        $('#tasksForm').submit();
    });

    // Populate JSON file selector
    $.getJSON($('#jsonFileSelector').attr('data-files-url'), function (data) {
        var jsonFileSelector = $('#jsonFileSelector');
        jsonFileSelector.empty(); // Clear existing options
        $.each(data.json_files, function (index, filename) {
            jsonFileSelector.append($('<option>', {
                value: filename,
                text: filename
            }));
        });
    });
});
