/**
 * HUMAnN3 Pathway Abundance Viewer
 * Chart Utilities
 * Helper functions for chart generation and configuration
 */

app.service('ChartUtils', ['FormattersService', function(FormattersService) {
    var service = this;
    
    /**
     * Initialize Chart.js with global defaults
     */
    service.initializeChartDefaults = function() {
        if (!window.Chart) {
            console.warn('Chart.js not loaded yet');
            return false;
        }
        
        // Set global defaults
        Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
        Chart.defaults.color = '#2c3e50';
        Chart.defaults.scale.grid.color = 'rgba(0,0,0,0.05)';
        
        // Custom tooltip styling
        Chart.defaults.plugins.tooltip.backgroundColor = 'rgba(44, 62, 80, 0.9)';
        Chart.defaults.plugins.tooltip.titleFont = {
            size: 14,
            weight: 'bold'
        };
        Chart.defaults.plugins.tooltip.bodyFont = {
            size: 13
        };
        Chart.defaults.plugins.tooltip.padding = 10;
        Chart.defaults.plugins.tooltip.cornerRadius = 4;
        
        // Responsiveness
        Chart.defaults.maintainAspectRatio = false;
        Chart.defaults.responsive = true;
        
        return true;
    };
    
    /**
     * Create a chart configuration object
     * @param {String} type - Chart type ('bar', 'line', etc.)
     * @param {Array} labels - X-axis labels
     * @param {Array} datasets - Chart datasets
     * @param {Object} options - Additional chart options
     * @returns {Object} - Chart configuration object
     */
    service.createChartConfig = function(type, labels, datasets, options) {
        options = options || {};
        
        // Basic configuration
        const config = {
            type: type,
            data: {
                labels: labels,
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: datasets.length > 1,
                        position: 'top',
                        labels: {
                            boxWidth: 12,
                            usePointStyle: true,
                            padding: 15
                        }
                    },
                    title: {
                        display: options.title ? true : false,
                        text: options.title || '',
                        font: {
                            size: 16,
                            weight: 'bold'
                        },
                        padding: {
                            top: 10,
                            bottom: 20
                        }
                    },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                let label = context.dataset.label || '';
                                if (label) {
                                    label += ': ';
                                }
                                if (context.parsed.y !== null) {
                                    label += FormattersService.formatNumber(context.parsed.y, options.precision || 4);
                                }
                                return label;
                            }
                        }
                    }
                },
                scales: {}
            }
        };
        
        // Apply chart type specific options
        switch (type) {
            case 'bar':
                config.options.scales = service.getBarScaleOptions(options);
                break;
                
            case 'line':
                config.options.scales = service.getLineScaleOptions(options);
                
                // Set line styling
                datasets.forEach(dataset => {
                    dataset.tension = 0.1;
                    dataset.pointRadius = 3;
                    dataset.pointHoverRadius = 5;
                    
                    // If not specified
                    if (dataset.fill === undefined) {
                        dataset.fill = false;
                    }
                });
                break;
                
            case 'radar':
                // Remove scales for radar chart
                config.options.scales = {};
                
                // Styling for radar
                config.options.elements = {
                    line: {
                        tension: 0.1,
                        borderWidth: 2
                    }
                };
                
                // Limit datasets for radar to avoid clutter
                if (labels.length > 12) {
                    config.data.labels = labels.slice(0, 12);
                    datasets.forEach(dataset => {
                        dataset.data = dataset.data.slice(0, 12);
                    });
                    
                    // Add note about limiting
                    if (config.options.title && config.options.title.text) {
                        config.options.title.text += ' (Limited to 12 items)';
                    }
                }
                break;
                
            case 'polarArea':
                // Adjust radius for better display
                config.options.plugins.legend.position = 'right';
                config.options.scales = {
                    r: {
                        ticks: {
                            backdropColor: 'transparent'
                        }
                    }
                };
                break;
                
            case 'doughnut':
            case 'pie':
                // Adjust legend for pie/doughnut
                config.options.plugins.legend.position = 'right';
                
                // Custom tooltip for pie/doughnut charts
                config.options.plugins.tooltip.callbacks.label = function(context) {
                    const label = context.label || '';
                    const value = FormattersService.formatNumber(context.raw, options.precision || 4);
                    const percentage = ((context.parsed / context.dataset.data.reduce((a, b) => a + b, 0)) * 100).toFixed(1);
                    return `${label}: ${value} (${percentage}%)`;
                };
                
                // Center text if specified
                if (options.centerText) {
                    if (!config.options.plugins.doughnutLabel) {
                        config.options.plugins.doughnutLabel = {};
                    }
                    config.options.plugins.doughnutLabel.text = options.centerText;
                }
                break;
        }
        
        // Apply provided options
        if (options.scales) {
            config.options.scales = Object.assign(config.options.scales, options.scales);
        }
        
        if (options.plugins) {
            config.options.plugins = Object.assign(config.options.plugins, options.plugins);
        }
        
        if (options.animation !== undefined) {
            config.options.animation = options.animation;
        }
        
        return config;
    };
    
    /**
     * Get scale options for bar charts
     * @param {Object} options - Options for scales
     * @returns {Object} - Configured scales object
     */
    service.getBarScaleOptions = function(options) {
        return {
            x: {
                title: {
                    display: options.xTitle ? true : false,
                    text: options.xTitle || '',
                    font: {
                        weight: 'bold'
                    }
                },
                ticks: {
                    maxRotation: 45,
                    minRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: options.maxXTicksLimit || 20
                }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: options.yTitle ? true : false,
                    text: options.yTitle || '',
                    font: {
                        weight: 'bold'
                    }
                },
                ticks: {
                    callback: function(value) {
                        return FormattersService.formatNumber(value, options.precision || 2);
                    }
                }
            }
        };
    };
    
    /**
     * Get scale options for line charts
     * @param {Object} options - Options for scales
     * @returns {Object} - Configured scales object
     */
    service.getLineScaleOptions = function(options) {
        return {
            x: {
                title: {
                    display: options.xTitle ? true : false,
                    text: options.xTitle || '',
                    font: {
                        weight: 'bold'
                    }
                },
                ticks: {
                    autoSkip: true,
                    maxTicksLimit: options.maxXTicksLimit || 10
                }
            },
            y: {
                beginAtZero: true,
                title: {
                    display: options.yTitle ? true : false,
                    text: options.yTitle || '',
                    font: {
                        weight: 'bold'
                    }
                },
                ticks: {
                    callback: function(value) {
                        return FormattersService.formatNumber(value, options.precision || 2);
                    }
                }
            }
        };
    };
    
    /**
     * Create datasets for a chart
     * @param {Array} data - Data array
     * @param {Object} options - Dataset options
     * @returns {Array} - Array of dataset objects
     */
    service.createDatasets = function(data, options) {
        options = options || {};
        const datasets = [];
        
        // Single dataset case
        if (Array.isArray(data) && (!Array.isArray(data[0]) || typeof data[0] === 'number')) {
            const dataset = {
                label: options.label || 'Data',
                data: data,
                backgroundColor: options.backgroundColor || '#3498db',
                borderColor: options.borderColor || '#2980b9',
                borderWidth: options.borderWidth || 1
            };
            
            // Additional properties
            if (options.fill !== undefined) dataset.fill = options.fill;
            if (options.tension !== undefined) dataset.tension = options.tension;
            if (options.pointRadius !== undefined) dataset.pointRadius = options.pointRadius;
            
            datasets.push(dataset);
        } 
        // Multiple datasets case
        else if (Array.isArray(data) && Array.isArray(data[0])) {
            const labels = options.labels || [];
            const colors = service.generateColors(data.length, options.colorScheme);
            
            data.forEach((dataSet, i) => {
                const dataset = {
                    label: labels[i] || `Dataset ${i + 1}`,
                    data: dataSet,
                    backgroundColor: colors[i],
                    borderColor: service.adjustColor(colors[i], -0.2),
                    borderWidth: options.borderWidth || 1
                };
                
                // Additional properties
                if (options.fill !== undefined) dataset.fill = options.fill;
                if (options.tension !== undefined) dataset.tension = options.tension;
                if (options.pointRadius !== undefined) dataset.pointRadius = options.pointRadius;
                
                datasets.push(dataset);
            });
        }
        // Object-based datasets
        else if (typeof data === 'object' && !Array.isArray(data)) {
            // Convert object to datasets
            Object.keys(data).forEach((key, i) => {
                const colors = service.generateColors(Object.keys(data).length, options.colorScheme);
                
                const dataset = {
                    label: key,
                    data: data[key],
                    backgroundColor: colors[i],
                    borderColor: service.adjustColor(colors[i], -0.2),
                    borderWidth: options.borderWidth || 1
                };
                
                // Additional properties
                if (options.fill !== undefined) dataset.fill = options.fill;
                if (options.tension !== undefined) dataset.tension = options.tension;
                if (options.pointRadius !== undefined) dataset.pointRadius = options.pointRadius;
                
                datasets.push(dataset);
            });
        }
        
        return datasets;
    };
    
    /**
     * Generate chart colors
     * @param {Number} count - Number of colors needed
     * @param {String} scheme - Color scheme name
     * @returns {Array} - Array of color strings
     */
    service.generateColors = function(count, scheme) {
        return FormattersService.generateColors(count, scheme);
    };
    
    /**
     * Adjust color lightness
     * @param {String} color - Color to adjust (hex or rgb)
     * @param {Number} amount - Amount to lighten (positive) or darken (negative)
     * @returns {String} - Adjusted color
     */
    service.adjustColor = function(color, amount) {
        const rgb = FormattersService.parseColor(color);
        
        // Adjust each component
        const adjusted = rgb.map(value => {
            // Lighten or darken
            const newValue = amount > 0 ?
                Math.round(value + (255 - value) * amount) :
                Math.round(value + value * amount);
            
            // Clamp to valid range
            return Math.max(0, Math.min(255, newValue));
        });
        
        return `rgb(${adjusted[0]}, ${adjusted[1]}, ${adjusted[2]})`;
    };
    
    /**
     * Create a heatmap chart
     * @param {Array} data - 2D array of values
     * @param {Array} xLabels - X-axis labels
     * @param {Array} yLabels - Y-axis labels
     * @param {Object} options - Chart options
     * @returns {Object} - Chart configuration object
     */
    service.createHeatmap = function(data, xLabels, yLabels, options) {
        options = options || {};
        
        // Create a structured dataset for the heatmap
        const datasets = [];
        const colorRange = options.colorRange || ['#f7fbff', '#2171b5'];
        
        // Find min and max values for color scaling
        let minValue = Infinity;
        let maxValue = -Infinity;
        
        data.forEach(row => {
            row.forEach(value => {
                if (value < minValue) minValue = value;
                if (value > maxValue) maxValue = value;
            });
        });
        
        // Ensure min/max are different
        if (minValue === maxValue) {
            minValue = minValue > 0 ? 0 : minValue - 1;
        }
        
        // Create a dataset for each row
        yLabels.forEach((yLabel, i) => {
            const rowData = [];
            
            // Transform data into format needed for heatmap
            xLabels.forEach((xLabel, j) => {
                const value = data[i][j];
                
                rowData.push({
                    x: xLabel,
                    y: yLabel,
                    v: value // Original value for tooltip
                });
            });
            
            datasets.push({
                label: yLabel,
                data: rowData,
                backgroundColor: function(context) {
                    const value = context.raw.v;
                    const percent = (value - minValue) / (maxValue - minValue);
                    return service.getGradientColor(colorRange[0], colorRange[1], percent);
                }
            });
        });
        
        // Create heatmap configuration
        const config = {
            type: 'matrix',
            data: {
                datasets: datasets
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        display: false
                    },
                    tooltip: {
                        callbacks: {
                            title: function(context) {
                                return `${context[0].raw.y} / ${context[0].raw.x}`;
                            },
                            label: function(context) {
                                return `Value: ${FormattersService.formatNumber(context.raw.v, options.precision || 4)}`;
                            }
                        }
                    },
                    colorScale: {
                        display: true,
                        position: 'right',
                        width: 20,
                        height: '80%',
                        min: minValue,
                        max: maxValue,
                        colors: colorRange
                    }
                }
            }
        };
        
        return config;
    };
    
    /**
     * Get color from gradient by percentage
     * @param {String} color1 - Start color
     * @param {String} color2 - End color
     * @param {Number} percent - Percentage (0-1)
     * @returns {String} - Interpolated color
     */
    service.getGradientColor = function(color1, color2, percent) {
        return FormattersService.interpolateColors(color1, color2, percent);
    };
    
    return service;
}]);