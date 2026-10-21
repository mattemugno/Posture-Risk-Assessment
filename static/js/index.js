$(document).ready(function () {
    const tasksForm = $('#tasksForm');

    function bindToggle(button, storageKey, initialValue, toggledValue) {
        let state = localStorage.getItem(storageKey) || initialValue;
        button.val(state);

        button.click(function () {
            state = state === initialValue ? toggledValue : initialValue;
            button.val(state);
            localStorage.setItem(storageKey, state);
            tasksForm.submit();
        });
    }

    bindToggle($('#cameraButton'), 'cameraState', 'Open Video Camera', 'Close Video Camera');
    bindToggle($('#sessionButton'), 'sessionState', 'Start Session', 'Stop Session');

    const jsonFileSelector = $('#jsonFileSelector');
    $.getJSON(jsonFileSelector.attr('data-files-url'), function (data) {
        jsonFileSelector.empty();
        $.each(data.json_files, function (index, filename) {
            jsonFileSelector.append($('<option>', {
                value: filename,
                text: filename
            }));
        });
    });
});
