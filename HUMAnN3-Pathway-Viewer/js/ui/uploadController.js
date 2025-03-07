/**
 * HUMAnN3 Pathway Abundance Viewer
 * Upload Controller
 * Manages file upload features
 */

angular.module('humanPathwayViewer')
.controller('UploadController', ['$scope', 'DataManager', 'EventHandler', function($scope, dataService, eventHandler) {
    $scope.dataService = dataService;
    
    // Initialize controller state
    $scope.isLoading = false;
    $scope.errorMessage = '';
    
    // Handle file selection
    $scope.handleFileSelect = function(fileInput) {
        if (fileInput.files && fileInput.files[0]) {
            var file = fileInput.files[0];
            $scope.isLoading = true;
            $scope.errorMessage = '';
            
            // Apply scope to update UI immediately
            $scope.$apply();
            
            // Process the file
            var reader = new FileReader();
            reader.onload = function(e) {
                try {
                    var data = e.target.result;
                    dataService.loadData(data, file.name);
                    eventHandler.trigger('dataLoaded');
                    $scope.isLoading = false;
                } catch(err) {
                    $scope.errorMessage = 'Error parsing file: ' + err.message;
                    $scope.isLoading = false;
                }
                // Apply scope to update UI with results
                $scope.$apply();
            };
            
            reader.onerror = function() {
                $scope.errorMessage = 'Error reading file';
                $scope.isLoading = false;
                $scope.$apply();
            };
            
            reader.readAsText(file);
        }
    };
    
    // Additional file upload methods can go here
}]);
