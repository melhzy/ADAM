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
     * Format a sample name (truncate if too long)
     * @param {String} name - Sample name
     * @param {Number} maxLength - Maximum length before truncation
     * @returns {String} - Formatted sample name
     */
    service.formatSampleName = function(name, maxLength) {
        if (!name) return '';
        maxLength = maxLength || 15;
        
        if (name.length <= maxLength) return name;
        
        // If name is too long, truncate with ellipsis
        return name.substring(0, maxLength - 3) + '...';
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
     * Sanitize a string for use as a filename
     * @param {String} str - String to sanitize
     * @returns {String} - Sanitized string
     */
    service.sanitizeFilename = function(str) {
        if (!str) return 'file';
        
        // Replace invalid filename characters with underscores
        return str.replace(/[\\/:*?"<>|]/g, '_').replace(/\s+/g, '_').substring(0, 100);
    };
    
    /**
     * Download a CSV string as a file
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
            
            // Trigger download
            link.click();
            
            // Clean up
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        } else {
            // Fallback for browsers without URL.createObjectURL support
            const encodedUri = 'data:text/csv;charset=utf-8,' + encodeURIComponent(csvContent);
            link.setAttribute('href', encodedUri);
            link.setAttribute('download', filename + '.csv');
            link.click();
        }
    };
    
    /**
     * Download a canvas as an image
     * @param {HTMLCanvasElement} canvas - Canvas to download
     * @param {String} filename - Filename without extension
     */
    service.downloadCanvasAsImage = function(canvas, filename) {
        if (!canvas) return;
        
        // Create download link
        const link = document.createElement('a');
        
        // Convert canvas to PNG data URL
        const dataURL = canvas.toDataURL('image/png');
        
        // Set link attributes
        link.setAttribute('href', dataURL);
        link.setAttribute('download', filename + '.png');
        link.style.visibility = 'hidden';
        document.body.appendChild(link);
        
        // Trigger download
        link.click();
        
        // Clean up
        document.body.removeChild(link);
    };
    
    /**
     * Parse color from various formats to RGB array
     * @param {String} color - Color in hex, rgb, or rgba format
     * @returns {Array} - [r, g, b] array
     */
    service.parseColor = function(color) {
        // Default if parsing fails
        if (!color) return [0, 0, 0];
        
        // Handle hex colors
        if (color.startsWith('#')) {
            const hex = color.substring(1);
            const r = parseInt(hex.substring(0, 2), 16);
            const g = parseInt(hex.substring(2, 4), 16);
            const b = parseInt(hex.substring(4, 6), 16);
            return [r, g, b];
        }
        
        // Handle rgb/rgba colors
        if (color.startsWith('rgb')) {
            const values = color.match(/\d+/g);
            if (values && values.length >= 3) {
                return [parseInt(values[0]), parseInt(values[1]), parseInt(values[2])];
            }
        }
        
        // Default fallback
        return [0, 0, 0];
    };
    
    /**
     * Interpolate between two colors
     * @param {String} color1 - First color (hex or rgb)
     * @param {String} color2 - Second color (hex or rgb)
     * @param {Number} ratio - Interpolation ratio (0-1)
     * @returns {String} - Interpolated color as hex
     */
    service.interpolateColors = function(color1, color2, ratio) {
        const rgb1 = service.parseColor(color1);
        const rgb2 = service.parseColor(color2);
        
        const r = Math.round(rgb1[0] + (rgb2[0] - rgb1[0]) * ratio);
        const g = Math.round(rgb1[1] + (rgb2[1] - rgb1[1]) * ratio);
        const b = Math.round(rgb1[2] + (rgb2[2] - rgb1[2]) * ratio);
        
        return '#' + 
            r.toString(16).padStart(2, '0') + 
            g.toString(16).padStart(2, '0') + 
            b.toString(16).padStart(2, '0');
    };
    
    /**
     * Generate colors for charts
     * @param {Number} count - Number of colors needed
     * @param {String} scheme - Color scheme name
     * @returns {Array} - Array of color strings
     */
    service.generateColors = function(count, scheme) {
        // Base color sets for different schemes
        const colorSets = {
            default: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#f1c40f', '#e67e22'],
            viridis: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
            inferno: ['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4'],
            plasma: ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
            cool: ['#6e40aa', '#4776ff', '#10a4db', '#36c956', '#eff953'],
            warm: ['#6e40aa', '#be3caf', '#fe4b83', '#ff7847', '#e2b72f']
        };
        
        const colors = colorSets[scheme] || colorSets.default;
        
        // Generate color array
        const result = [];
        for (let i = 0; i < count; i++) {
            result.push(colors[i % colors.length]);
        }
        
        return result;
    };
    
    return service;
}]);