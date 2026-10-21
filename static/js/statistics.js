document.getElementById('back-button').addEventListener('click', function () {
    window.history.back();
});

(async function () {
    const report = document.getElementById('session-charts');
    const message = document.getElementById('chart-message');
    const controls = document.getElementById('chart-controls');
    const form = document.getElementById('range-form');
    const startInput = document.getElementById('range-start');
    const endInput = document.getElementById('range-end');
    const charts = ['angles-chart', 'status-chart', 'heatmap-chart', 'risk-chart'].map(
        id => document.getElementById(id));
    const narrowScreen = window.matchMedia('(max-width: 600px)');
    let syncingRange = false;
    message.hidden = false;

    function baseLayout(style) {
        return {
            autosize: true,
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            font: {family: 'Arial, Helvetica, sans-serif', size: 13, color: style.ink},
            hoverlabel: {bgcolor: '#ffffff', bordercolor: style.muted, font: {size: 14}},
            modebar: {bgcolor: '#ffffff', color: style.muted, activecolor: style.acceptable},
            dragmode: 'zoom',
            hovermode: 'x unified',
            showlegend: false,
        };
    }

    function timeAxis(range, style) {
        return {
            type: 'date', range: range.map(value => new Date(value).toISOString()),
            tickformat: '%H:%M:%S', hoverformat: '%H:%M:%S.%L',
            nticks: narrowScreen.matches ? 3 : 6,
            showgrid: false, zeroline: false, linecolor: style.grid,
            tickfont: {color: style.muted, size: 12},
        };
    }

    function timeline(data, times, range, status) {
        const layout = baseLayout(data.style);
        const domains = [[0.72, 0.96], [0.38, 0.62], [0.04, 0.28]];
        layout.margin = {l: status ? 110 : 58, r: 18, t: 58, b: 56};
        layout.annotations = [];
        layout.shapes = [];
        const traces = data.series.map((series, index) => {
            const suffix = index === 0 ? '' : String(index + 1);
            const axis = timeAxis(range, data.style);
            axis.anchor = 'y' + suffix;
            axis.showticklabels = index === 2;
            if (index > 0) axis.matches = 'x';
            if (index === 2) axis.title = {text: 'Time (H:M:S)', standoff: 14};
            layout['xaxis' + suffix] = axis;
            layout['yaxis' + suffix] = {
                domain: domains[index], anchor: 'x' + suffix,
                range: status ? [-0.18, 2.18] : [series.angles.reduce((min, value) => Math.min(min, value), 0), 180],
                fixedrange: true, zeroline: false,
                gridcolor: data.style.grid, linecolor: data.style.grid,
                tickfont: {color: data.style.muted, size: 12},
            };
            if (status) {
                layout['yaxis' + suffix].tickvals = [0, 1, 2];
                layout['yaxis' + suffix].ticktext = ['Acceptable', 'Checking', 'Not<br>acceptable'];
            } else {
                layout['yaxis' + suffix].dtick = 60;
                data.angle_guides.forEach((guide, guideIndex) => {
                    layout.shapes.push({
                        type: 'line', xref: 'paper', x0: 0, x1: 1,
                        yref: 'y' + suffix, y0: guide, y1: guide,
                        line: {color: guideIndex === 0 ? data.style.acceptable : data.style.not_acceptable,
                            width: 1.5, dash: 'dash'},
                    });
                });
            }
            layout.annotations.push({
                text: '<b>' + series.label + '</b>', xref: 'paper', yref: 'paper',
                x: 0, y: domains[index][1] + 0.025, xanchor: 'left', yanchor: 'bottom',
                showarrow: false, font: {size: 15},
            });
            return {
                type: 'scatter', mode: times.length === 1 ? 'lines+markers' : 'lines',
                name: series.label, x: times, y: status ? series.status : series.angles,
                xaxis: 'x' + suffix, yaxis: 'y' + suffix,
                line: {color: series.color, width: 2.5}, marker: {size: 7},
                customdata: status ? series.status.map(value => data.status_labels[value]) : undefined,
                hovertemplate: status ? '%{customdata}<extra>%{fullData.name}</extra>'
                    : '%{y:.2f}°<extra>%{fullData.name}</extra>',
            };
        });
        if (!status) layout.yaxis2.title = {text: 'Angle (°)', standoff: 10};
        return {traces, layout};
    }

    function heatmap(data, times, range) {
        const series = [data.series[1], data.series[0], data.series[2]];
        const layout = baseLayout(data.style);
        layout.margin = {l: narrowScreen.matches ? 92 : 130, r: 64, t: 40, b: 60};
        layout.hovermode = 'closest';
        layout.xaxis = {...timeAxis(range, data.style), title: {text: 'Time (H:M:S)', standoff: 14}};
        layout.yaxis = {autorange: 'reversed', fixedrange: true, tickfont: {size: 13},
            tickvals: series.map(item => item.label),
            ticktext: narrowScreen.matches ? ['Left<br>shoulder', 'Right<br>shoulder', 'Trunk']
                : series.map(item => item.label)};
        return {layout, traces: [{
            type: 'heatmap', x: times, y: series.map(item => item.label),
            z: series.map(item => item.angles),
            colorscale: data.style.heatmap.map((color, index, colors) => [index / (colors.length - 1), color]),
            colorbar: {title: {text: 'Angle (°)', side: 'right'}, thickness: 12, len: 0.85, tickfont: {size: 12}},
            hovertemplate: '%{y}<br>%{x|%H:%M:%S.%L}<br>%{z:.2f}°<extra></extra>',
        }]};
    }

    function riskLayout(data) {
        const layout = baseLayout(data.style);
        layout.margin = {l: 18, r: 18, t: 24, b: 52};
        layout.showlegend = true;
        layout.legend = {orientation: 'h', x: 0.5, xanchor: 'center', y: -0.08,
            itemclick: false, itemdoubleclick: false, font: {size: 13}};
        layout.annotations = [];
        const domains = [];
        data.risk.forEach((item, index) => {
            const domain = narrowScreen.matches
                ? {x: [0.15, 0.85], y: [0.7 - index * 0.34, 0.94 - index * 0.34]}
                : {x: [index / 3 + 0.025, (index + 1) / 3 - 0.025], y: [0.12, 0.85]};
            domains.push(domain);
            const x = (domain.x[0] + domain.x[1]) / 2;
            const y = (domain.y[0] + domain.y[1]) / 2;
            layout.annotations.push(
                {text: '<b>' + item.label + '</b>', x, y: domain.y[1] + 0.03, yanchor: 'bottom',
                    showarrow: false, font: {size: 16}},
                {text: item.not_acceptable.toFixed(1) + '%', x, y: y + 0.025,
                    showarrow: false, font: {size: 25}},
                {text: 'not acceptable', x, y: y - (narrowScreen.matches ? 0.025 : 0.07),
                    showarrow: false, font: {size: 12, color: data.style.muted}},
            );
        });
        layout.annotations = layout.annotations.map(annotation => ({
            xref: 'paper', yref: 'paper', xanchor: 'center', yanchor: 'middle', ...annotation,
        }));
        return {layout, domains};
    }

    async function draw(element, traces, layout, config) {
        element.hidden = false;
        await Plotly.newPlot(element, traces, layout, config);
        element.classList.add('is-ready');
        element.parentElement.querySelector('.chart-fallback').hidden = true;
    }

    function milliseconds(value) {
        if (typeof value === 'number') return value;
        const iso = value.replace(' ', 'T');
        return Date.parse(/[zZ]$|[+-]\d\d:\d\d$/.test(iso) ? iso : iso + 'Z');
    }

    function eventRange(event, fullRange) {
        for (const axis of ['xaxis', 'xaxis2', 'xaxis3']) {
            if (event[axis + '.autorange']) return fullRange;
            const range = event[axis + '.range'];
            if (range) return range.map(milliseconds);
            if (event[axis + '.range[0]'] !== undefined && event[axis + '.range[1]'] !== undefined) {
                return [milliseconds(event[axis + '.range[0]']), milliseconds(event[axis + '.range[1]'])];
            }
        }
        return null;
    }

    try {
        if (!window.Plotly) throw new Error('Chart library unavailable');
        const response = await fetch(report.dataset.chartUrl);
        if (!response.ok) throw new Error('Session data unavailable');
        const data = await response.json();
        if (data.timestamps.length === 0) {
            message.textContent = 'This session has no complete posture samples to display.';
            charts.forEach(element => {
                element.hidden = false;
                element.classList.add('is-ready', 'is-empty');
                element.textContent = 'No samples available.';
                element.parentElement.querySelector('.chart-fallback').hidden = true;
            });
            return;
        }

        const numericTimes = data.timestamps.map(value => value * 1000);
        const times = numericTimes.map(value => new Date(value).toISOString());
        const first = numericTimes.reduce((min, value) => Math.min(min, value));
        const last = numericTimes.reduce((max, value) => Math.max(max, value));
        const fullRange = first === last ? [first - 500, last + 500] : [first, last];
        const config = {
            responsive: true, displaylogo: false, displayModeBar: true, scrollZoom: false,
            doubleClick: 'reset', showTips: false,
            modeBarButtons: [['zoom2d', 'pan2d', 'zoomIn2d', 'zoomOut2d', 'resetScale2d'], ['toImage']],
            toImageButtonOptions: {format: 'png', filename: 'posture-session', scale: 2},
        };
        const angles = timeline(data, times, fullRange, false);
        const status = timeline(data, times, fullRange, true);
        const map = heatmap(data, times, fullRange);
        const risk = riskLayout(data);
        const pies = data.risk.map((item, index) => ({
            type: 'pie', labels: ['Acceptable', 'Not acceptable'],
            values: [item.acceptable, item.not_acceptable], name: item.label,
            hole: 0.76, sort: false, direction: 'clockwise', textinfo: 'none',
            marker: {colors: [data.style.acceptable, data.style.not_acceptable], line: {color: '#ffffff', width: 2}},
            domain: risk.domains[index], showlegend: index === 0,
            hovertemplate: '%{label}: %{value:.1f}%<extra>%{fullData.name}</extra>',
        }));
        await Promise.all([
            draw(charts[0], angles.traces, angles.layout, config),
            draw(charts[1], status.traces, status.layout, config),
            draw(charts[2], map.traces, map.layout, config),
            draw(charts[3], pies, risk.layout, {...config, displayModeBar: false}),
        ]);

        const maxSeconds = Math.ceil(last - first) / 1000;
        startInput.max = endInput.max = String(maxSeconds);
        function updateInputs(range) {
            startInput.value = Math.max(0, (range[0] - first) / 1000).toFixed(3);
            endInput.value = Math.min(maxSeconds, (range[1] - first) / 1000).toFixed(3);
            startInput.setCustomValidity('');
        }
        async function applyRange(range) {
            if (syncingRange || !range.every(Number.isFinite) || range[0] >= range[1]) return;
            const width = Math.min(range[1] - range[0], fullRange[1] - fullRange[0]);
            const start = Math.max(fullRange[0], Math.min(range[0], fullRange[1] - width));
            range = [start, start + width];
            syncingRange = true;
            try {
                const update = {'xaxis.range': range.map(value => new Date(value).toISOString()),
                    'xaxis.autorange': false};
                await Promise.all(charts.slice(0, 3).map(element => Plotly.relayout(element, update)));
                updateInputs(range);
            } finally {
                syncingRange = false;
            }
        }
        charts.slice(0, 3).forEach(element => {
            element.on('plotly_relayout', event => {
                if (syncingRange) return;
                const range = eventRange(event, fullRange);
                if (range) applyRange(range);
            });
        });
        form.addEventListener('submit', event => {
            event.preventDefault();
            startInput.setCustomValidity('');
            if (Number(startInput.value) > Number(endInput.value)) {
                startInput.setCustomValidity('Start must be before the end of the interval.');
            }
            if (form.reportValidity()) {
                const range = [startInput, endInput].map(input => first + Number(input.value) * 1000);
                applyRange(range[0] === range[1] ? [range[0] - 500, range[1] + 500] : range);
            }
        });
        startInput.addEventListener('input', () => startInput.setCustomValidity(''));
        endInput.addEventListener('input', () => startInput.setCustomValidity(''));
        document.getElementById('reset-zoom').addEventListener('click', () => applyRange(fullRange));
        narrowScreen.addEventListener('change', () => {
            const updated = riskLayout(data);
            Plotly.restyle(charts[3], {domain: updated.domains});
            Plotly.relayout(charts[3], {annotations: updated.layout.annotations});
            charts.slice(0, 3).forEach(element => {
                const update = {};
                for (const axis of ['xaxis', 'xaxis2', 'xaxis3']) {
                    if (element.layout[axis]) update[axis + '.nticks'] = narrowScreen.matches ? 3 : 6;
                }
                Plotly.relayout(element, update);
            });
            Plotly.relayout(charts[2], {'margin.l': narrowScreen.matches ? 92 : 130,
                'yaxis.ticktext': narrowScreen.matches ? ['Left<br>shoulder', 'Right<br>shoulder', 'Trunk']
                    : [data.series[1].label, data.series[0].label, data.series[2].label]});
        });
        updateInputs([first, last]);
        controls.hidden = false;
        message.hidden = true;
        report.dataset.interactive = 'ready';
    } catch (error) {
        console.warn('Unable to load interactive charts.', error);
        charts.forEach(element => {
            if (window.Plotly) Plotly.purge(element);
            element.hidden = true;
            element.classList.remove('is-ready');
            const fallback = element.parentElement.querySelector('.chart-fallback');
            fallback.hidden = !fallback.hasAttribute('src');
        });
        if (charts[0].parentElement.querySelector('.chart-fallback').hasAttribute('src')) {
            message.textContent = 'Interactive charts could not be loaded. The static reports are available below.';
        }
        report.dataset.interactive = 'unavailable';
    }
})();
