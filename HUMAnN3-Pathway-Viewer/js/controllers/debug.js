/**
 * HUMAnN3 Pathway Abundance Viewer
 * Debug Controller
 * Handles debug panel functionality
 */

app.controller('DebugController', ['$scope', 'DebugService', 'FormattersService', '$http', 
function($scope, DebugService, FormattersService, $http) {
    
    // Initialize debug variables
    $scope.debugEnabled = false;
    $scope.logEntries = [];
    
    // Check URL for debug parameter
    function init() {
        const urlParams = new URLSearchParams(window.location.search);
        $scope.debugEnabled = urlParams.get('debug') === 'true';
        
        if ($scope.debugEnabled) {
            $scope.logEntries.push('Debug mode activated');
            console.log('Debug controller initialized');
        }
    }
    
    // Run diagnostic tests
    $scope.runDiagnostics = function() {
        $scope.logEntries.push('Running diagnostics...');
        
        // Check Angular app
        $scope.logEntries.push('Angular version: ' + angular.version.full);
        
        // Check services
        $scope.logEntries.push('FormattersService: ' + 
            (FormattersService.checkService() ? 'OK' : 'FAIL'));
        
        // Check app initialization
        $scope.logEntries.push('App initialization: ' + 
            (DebugService.checkAppInit() ? 'OK' : 'FAIL'));
            
        // Check API connection (if applicable)
        $http.get('api/status')
            .then(function(response) {
                $scope.logEntries.push('API connection: OK');
            })
            .catch(function(error) {
                $scope.logEntries.push('API connection: FAIL - ' + error.message);
            });
    };
    
    // Clear the debug log
    $scope.clearLog = function() {
        $scope.logEntries = [];
        $scope.logEntries.push('Log cleared');
    };
    
    // Initialize the controller
    init();
}]);
