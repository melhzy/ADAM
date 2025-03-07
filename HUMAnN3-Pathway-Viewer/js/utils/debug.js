/**
 * HUMAnN3 Pathway Abundance Viewer
 * Debug Utility
 * Helper functions for debugging the application
 */

app.service('DebugService', ['$window', function($window) {
    var service = this;
    
    // Debug mode flag (can be enabled via URL parameter ?debug=true)
    var debugMode = false;
    
    /**
     * Initialize the debug service
     */
    service.init = function() {
        // Check if debug mode is enabled via URL parameter
        const urlParams = new URLSearchParams(window.location.search);
        debugMode = urlParams.get('debug') === 'true';
        
        // Add global error handler
        $window.onerror = function(message, source, lineno, colno, error) {
            service.logError('Global error: ' + message, error);
            return false;
        };
        
        service.log('Debug mode initialized');
    };
    
    /**
     * Log a message to the console if debug mode is enabled
     * @param {String} message - Message to log
     */
    service.log = function(message) {
        if (debugMode) {
            console.log('[HUMAnN3 Debug]:', message);
        }
    };
    
    /**
     * Log an error to the console
     * @param {String} message - Error message
     * @param {Error} error - Error object
     */
    service.logError = function(message, error) {
        console.error('[HUMAnN3 Error]:', message, error || '');
    };
    
    /**
     * Check if the application is properly initialized
     * @returns {Boolean} - True if properly initialized
     */
    service.checkAppInit = function() {
        // Check if key services are available
        const missingServices = [];
        
        if (!app) missingServices.push('Angular app');
        if (!app.service) missingServices.push('Angular services');
        
        try {
            // Check for formatters service
            const formattersInjector = angular.element(document.body).injector();
            const formatters = formattersInjector.get('FormattersService');
            if (!formatters) missingServices.push('FormattersService');
        } catch (e) {
            missingServices.push('FormattersService (error: ' + e.message + ')');
        }
        
        if (missingServices.length > 0) {
            service.logError('App initialization issue: Missing ' + missingServices.join(', '));
            return false;
        }
        
        return true;
    };
    
    return service;
}]);

// Auto-initialize the debug service when included
app.run(['DebugService', function(DebugService) {
    DebugService.init();
}]);
