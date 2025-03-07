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
     * Parse a query string into an object
     * @param {String} queryString - The query string to parse
     * @returns {Object} - Object containing the query parameters
     */
    service.parseQueryString = function(queryString) {
        if (!queryString) return {};
        
        const params = {};
        const queries = queryString.substring(1).split('&');
        
        for (let i = 0; i < queries.length; i++) {
            const pair = queries[i].split('=');
            params[decodeURIComponent(pair[0])] = decodeURIComponent(pair[1] || '');
        }
        
        return params;
    };
    
    return service;
}]);