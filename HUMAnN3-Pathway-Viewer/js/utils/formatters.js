/**
 * HUMAnN3 Pathway Abundance Viewer
 * Formatters Utility
 * Helper functions for formatting data
 */

app.service('FormattersService', [function() {
    var service = this;
    
    /**
     * Format a number with appropriate units and precision
     * @param {Number} num - Number to format
     * @param {Number} precision - Number of decimal places (default: 2)
     * @returns {String} - Formatted number string
     */
    service.formatNumber = function(num, precision) {
        if (num === undefined || num === null) return '0';
        precision = precision || 2;
        
        if (num >= 1000000) {
            return (num / 1000000).toFixed(precision) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(precision) + 'K';
        } else if (num >= 1) {
            return num.toFixed(precision);
        } else if (num === 0) {
            return '0';
        } else {
            // For very small numbers, use scientific notation
            return num.toExponential(precision);
        }
    };
    
    /**
     * Format a file size with appropriate units
     * @param {Number} bytes - Size in bytes
     * @param {Number} precision - Number of decimal places (default: 1)
     * @returns {String} - Formatted file size string
     */
    service.formatFileSize = function(bytes, precision) {
        if (bytes === 0) return '0 Bytes';
        precision = precision || 1;
        
        const units = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
        const unitIndex = Math.floor(Math.log(bytes) / Math.log(1024));
        
        return (bytes / Math.pow(1024, unitIndex)).toFixed(precision) + ' ' + units[unitIndex];
    };
    
    /**
     * Format a date to a readable string
     * @param {Date|String|Number} date - Date to format
     * @param {Boolean} includeTime - Whether to include time (default: false)
     * @returns {String} - Formatted date string
     */
    service.formatDate = function(date, includeTime) {
        if (!date) return '';
        
        const dateObj = new Date(date);
        if (isNaN(dateObj.getTime())) return '';
        
        const options = {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        };
        
        if (includeTime) {
            options.hour = '2-digit';
            options.minute = '2-digit';
        }
        
        return dateObj.toLocaleDateString(undefined, options);
    };
    
    /**
     * Truncate a string with ellipsis if it exceeds max length
     * @param {String} str - String to truncate
     * @param {Number} maxLength - Maximum length (default: 50)
     * @returns {String} - Truncated string
     */
    service.truncate = function(str, maxLength) {
        if (!str) return '';
        maxLength = maxLength || 50;
        
        if (str.length <= maxLength) return str;
        return str.substring(0, maxLength) + '...';
    };
    
    /**
     * Capitalize first letter of a string
     * @param {String} str - String to capitalize
     * @returns {String} - Capitalized string
     */
    service.capitalize = function(str) {
        if (!str) return '';
        return str.charAt(0).toUpperCase() + str.slice(1);
    };
    
    /**
     * Format a percentage
     * @param {Number} value - Value to format as percentage
     * @param {Number} precision - Number of decimal places (default: 1)
     * @returns {String} - Formatted percentage string
     */
    service.formatPercent = function(value, precision) {
        if (value === undefined || value === null) return '0%';
        precision = precision || 1;
        
        return value.toFixed(precision) + '%';
    };
    
    /**
     * Convert a CSV string to a downloadable file
     * @param {String} csvContent - CSV content
     * @param {String} filename - Filename without extension
     */
    service.downloadCSV = function(csvContent, filename) {
        if (!csvContent) return;
        
        // Add BOM for Excel compatibility
        const bom = new Uint8Array([0xEF, 0xBB, 0xBF]);
        const blob = new Blob([bom, csvContent], { type: 'text/csv;charset=utf-8;' });
        
        // Create download link
        const link = document.createElement('a');
        
        // Support for browsers with URL.createObjectURL
        if (window.URL && window.URL.createObjectURL) {
            const url = URL.createObjectURL(blob);
            link.setAttribute('href', url);
            link.setAttribute('download', filename + '.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } else {
            // Fallback for older browsers
            link.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent));
            link.setAttribute('download', filename + '.csv');
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    };
    
    /**
     * Clean up a string for use in filenames
     * @param {String} str - String to clean
     * @returns {String} - Filename-safe string
     */
    service.sanitizeFilename = function(str) {
        if (!str) return '';
        return str.replace(/[/\\?%*:|"<>]/g, '_').trim();
    };
    
    /**
     * Format a sample name for display
     * @param {String} sampleName - Sample name
     * @param {Number} maxLength - Maximum length for display (default: 25)
     * @returns {String} - Formatted sample name
     */
    service.formatSampleName = function(sampleName, maxLength) {
        if (!sampleName) return '';
        maxLength = maxLength || 25;
        
        // Remove common suffixes
        let display = sampleName.replace(/_Abundance$/, '').replace(/\.abundance$/, '');
        
        // Truncate if necessary
        if (display.length > maxLength) {
            display = display.substring(0, maxLength) + '...';
        }
        
        return display;
    };
    
    /**
     * Convert Canvas to image data URL
     * @param {HTMLCanvasElement} canvas - Canvas element
     * @param {String} type - Image type (default: 'image/png')
     * @param {Number} quality - Image quality for JPEG (default: 0.95)
     * @returns {String} - Data URL
     */
    service.canvasToDataURL = function(canvas, type, quality) {
        type = type || 'image/png';
        quality = quality || 0.95;
        
        return canvas.toDataURL(type, quality);
    };
    
    /**
     * Download canvas as image
     * @param {HTMLCanvasElement} canvas - Canvas element
     * @param {String} filename - Filename without extension
     * @param {String} type - Image type (default: 'image/png')
     */
    service.downloadCanvasAsImage = function(canvas, filename, type) {
        if (!canvas) return;
        
        type = type || 'image/png';
        const extension = type.split('/')[1] || 'png';
        const dataURL = service.canvasToDataURL(canvas, type);
        
        // Create download link
        const link = document.createElement('a');
        link.setAttribute('href', dataURL);
        link.setAttribute('download', filename + '.' + extension);
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    };
    
    /**
     * Generate color array for visualizations
     * @param {Number} count - Number of colors needed
     * @param {String} scheme - Color scheme name (default: 'default')
     * @returns {Array} - Array of color strings
     */
    service.generateColors = function(count, scheme) {
        scheme = scheme || 'default';
        
        // Base color sets
        const colorSchemes = {
            default: [
                '#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6',
                '#1abc9c', '#f1c40f', '#e67e22', '#34495e', '#2980b9'
            ],
            viridis: [
                '#440154', '#414487', '#2a788e', '#22a884', '#7ad151',
                '#fde725', '#5ec962', '#3b528b', '#21918c', '#5ec962'
            ],
            inferno: [
                '#000004', '#420a68', '#932667', '#dd513a', '#fca50a',
                '#fcffa4', '#fb8861', '#d8576b', '#7c02a8', '#b73779'
            ],
            plasma: [
                '#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636',
                '#f0f921', '#cc4778', '#9c179e', '#d8576b', '#fb9e3a'
            ],
            cool: [
                '#6e40aa', '#4776ff', '#10a4db', '#36c956', '#eff953',
                '#aff05b', '#65d2e4', '#1390e8', '#5954cc', '#8e2cad'
            ],
            warm: [
                '#6e40aa', '#be3caf', '#fe4b83', '#ff7847', '#e2b72f',
                '#b8dd2c', '#cb9d3c', '#ff5e6b', '#d073ed', '#8a99fe'
            ]
        };
        
        // Get base colors for selected scheme
        const baseColors = colorSchemes[scheme] || colorSchemes.default;
        const result = [];
        
        // If we need more colors than available, we'll interpolate
        if (count <= baseColors.length) {
            // Return subset of colors
            for (let i = 0; i < count; i++) {
                result.push(baseColors[i]);
            }
        } else {
            // Interpolate colors
            for (let i = 0; i < count; i++) {
                const index = (i * baseColors.length) / count;
                const lowerIndex = Math.floor(index);
                const upperIndex = Math.ceil(index) % baseColors.length;
                
                if (lowerIndex === upperIndex) {
                    result.push(baseColors[lowerIndex]);
                } else {
                    const weight = index - lowerIndex;
                    result.push(service.interpolateColors(
                        baseColors[lowerIndex],
                        baseColors[upperIndex],
                        weight
                    ));
                }
            }
        }
        
        return result;
    };
    
    /**
     * Interpolate between two colors
     * @param {String} color1 - First color (hex or rgb)
     * @param {String} color2 - Second color (hex or rgb)
     * @param {Number} weight - Weight of second color (0-1)
     * @returns {String} - Interpolated color in rgb format
     */
    service.interpolateColors = function(color1, color2, weight) {
        // Convert hex to rgb if necessary
        const rgb1 = service.parseColor(color1);
        const rgb2 = service.parseColor(color2);
        
        // Interpolate
        const result = [
            Math.round(rgb1[0] + (rgb2[0] - rgb1[0]) * weight),
            Math.round(rgb1[1] + (rgb2[1] - rgb1[1]) * weight),
            Math.round(rgb1[2] + (rgb2[2] - rgb1[2]) * weight)
        ];
        
        return `rgb(${result[0]}, ${result[1]}, ${result[2]})`;
    };
    
    /**
     * Parse color string to RGB array
     * @param {String} color - Color string (hex or rgb)
     * @returns {Array} - RGB values as array [r, g, b]
     */
    service.parseColor = function(color) {
        // Check if already rgb format
        let rgbMatch = color.match(/^rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)$/i);
        if (rgbMatch) {
            return [parseInt(rgbMatch[1]), parseInt(rgbMatch[2]), parseInt(rgbMatch[3])];
        }
        
        // Convert hex to rgb
        // Remove # if present
        color = color.replace(/^#/, '');
        
        // Handle shorthand hex
        if (color.length === 3) {
            color = color[0] + color[0] + color[1] + color[1] + color[2] + color[2];
        }
        
        // Convert to rgb
        const r = parseInt(color.substring(0, 2), 16);
        const g = parseInt(color.substring(2, 4), 16);
        const b = parseInt(color.substring(4, 6), 16);
        
        return [r, g, b];
    };
    
    return service;
}]);