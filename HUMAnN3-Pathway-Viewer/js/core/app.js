/**
 * HUMAnN3 Pathway Abundance Viewer
 * Core Application Module
 */

// Initialize Angular module
var app = angular.module('humannPathwayViewer', []);

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