(async function () {
    // Camera/session forms submit once, with labels rendered from server state.
    const selector = document.getElementById('jsonFileSelector');
    try {
        const response = await fetch(selector.dataset.filesUrl);
        if (!response.ok) throw new Error('Unable to load saved sessions');
        const data = await response.json();
        const options = data.json_files.map(filename => new Option(filename, filename));
        selector.replaceChildren(...(options.length ? options : [new Option('No saved sessions', '')]));
    } catch (error) {
        selector.replaceChildren(new Option('Unable to load saved sessions', ''));
        console.warn('Unable to load saved sessions.', error);
    }
})();
