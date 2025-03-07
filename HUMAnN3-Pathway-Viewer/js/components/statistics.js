/**
 * HUMAnN3 Pathway Abundance Viewer
 * Statistics Controller
 * Handles statistics panel functionality
 */

app.controller('StatisticsController', [
    '$scope', 'DataManager', 'EventService',
    function($scope, DataManager, EventService) {
        // Initialize statistics
        $scope.stats = {
            totalPathways: 0,
            totalSamples: 0,
            metacycPathways: 0,
            unmappedPercent: '0.00'
        };
        
        /**
         * Update statistics from DataManager
         */
        function updateStats() {
            $scope.stats = DataManager.getStats();
        }
        
        /**
         * Initialize controller
         */
        function init() {
            // Subscribe to events
            EventService.on('data:loaded', function() {
                $scope.$apply(function() {
                    updateStats();
                });
            });
            
            EventService.on('data:reset', function() {
                $scope.$apply(function() {
                    $scope.stats = {
                        totalPathways: 0,
                        totalSamples: 0,
                        metacycPathways: 0,
                        unmappedPercent: '0.00'
                    };
                });
            });
            
            // Check if data is already loaded
            if (DataManager.hasData) {
                updateStats();
            }
        }
        
        // Initialize controller
        init();
    }
]);