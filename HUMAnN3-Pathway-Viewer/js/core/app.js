/**
 * HUMAnN3 Pathway Abundance Viewer
 * Core Application Module
 * Initializes the Angular application and sets up global configurations
 */

// Initialize the main Angular application module
var app = angular.module('humannPathwayViewer', []);

// Application configuration
app.config(['$compileProvider', function($compileProvider) {
    // Disable debug info for production (better performance)
    // Enable for debug mode
    $compileProvider.debugInfoEnabled(false);
    
    // Allow blob URLs for downloads
    $compileProvider.aHrefSanitizationWhitelist(/^\s*(https?|ftp|mailto|tel|file|blob):/);
}]);

// Global error handling
app.factory('$exceptionHandler', ['$log', function($log) {
    return function(exception, cause) {
        // Log to console
        $log.error(exception, cause);
        
        // If debug service is available, log there too
        try {
            const debugService = angular.element(document.body).injector().get('DebugService');
            if (debugService && debugService.logError) {
                debugService.logError('Angular exception: ' + exception.message, exception);
            }
        } catch (e) {
            // Debug service not available yet
            console.error('Error logging to DebugService:', e);
        }
    };
}]);

// Application initialization
app.run(['$rootScope', '$window', '$timeout', function($rootScope, $window, $timeout) {
    // Hide loading indicator when Angular has bootstrapped
    $timeout(function() {
        var loadingIndicator = document.getElementById('app-loading-status');
        if (loadingIndicator) {
            loadingIndicator.style.display = 'none';
        }
    }, 500);
    
    // Detect if debug mode is enabled
    const urlParams = new URLSearchParams($window.location.search);
    $rootScope.debugMode = urlParams.get('debug') === 'true';
    
    // Application version
    $rootScope.appVersion = '1.0.0';
    
    // Expose app status to root scope
    $rootScope.appStatus = {
        initialized: true,
        dataLoaded: false,
        error: null
    };
    
    // Log initialization
    console.log('HUMAnN3 Pathway Viewer initialized (v' + $rootScope.appVersion + ')');
}]);

// Main Controller
app.controller('MainController', ['$scope', 'DataManager', function($scope, DataManager) {
    // Provide access to DataManager across application
    $scope.dataManager = DataManager;
}]);

// File upload directives
app.directive('fileSelect', function() {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            element.on('change', function(event) {
                const files = event.target.files;
                scope.$apply(function() {
                    scope.$eval(attrs.fileSelect, { $event: { files: files } });
                });
            });
        }
    };
});

app.directive('fileDrop', function() {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            element.on('drop', function(event) {
                event.preventDefault();
                event.stopPropagation();
                
                const files = event.dataTransfer.files;
                scope.$apply(function() {
                    scope.$eval(attrs.fileDrop, { $event: { files: files } });
                });
            });
            
            // Prevent default to allow drop
            element.on('dragover', function(event) {
                event.preventDefault();
                event.stopPropagation();
            });
        }
    };
});

app.directive('fileDragOver', function() {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            element.on('dragover', function(event) {
                event.preventDefault();
                event.stopPropagation();
                
                scope.$apply(function() {
                    scope.$eval(attrs.fileDragOver);
                });
            });
        }
    };
});

app.directive('fileDragLeave', function() {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            element.on('dragleave', function(event) {
                event.preventDefault();
                event.stopPropagation();
                
                scope.$apply(function() {
                    scope.$eval(attrs.fileDragLeave);
                });
            });
        }
    };
});

// Formatter filters
app.filter('formatNumber', ['FormattersService', function(FormattersService) {
    return function(number, precision) {
        return FormattersService.formatNumber(number, precision);
    };
}]);

app.filter('capitalize', function() {
    return function(input) {
        if (!input) return '';
        return input.charAt(0).toUpperCase() + input.slice(1).toLowerCase();
    };
});