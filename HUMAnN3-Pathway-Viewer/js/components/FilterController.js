/**
 * HUMAnN3 Pathway Abundance Viewer
 * Filter Controller
 * Handles pathway filtering and sorting
 */

app.controller('FilterController', [
    '$scope', 'DataManager', 'EventService', 'OptimizationService',
    function($scope, DataManager, EventService, OptimizationService) {
        // Filter state
        $scope.filters = {
            searchTerm: '',
            pathwayType: 'all',
            sortField: 'id',
            selectedSample: ''
        };
        
        /**
         * Apply current filters
         */
        $scope.applyFilters = function() {
            // Apply filters via DataManager
            DataManager.applyFilters($scope.filters);
        };
        
        /**
         * Reset filters to default values
         */
        $scope.resetFilters = function() {
            $scope.filters = {
                searchTerm: '',
                pathwayType: 'all',
                sortField: 'id',
                selectedSample: ''
            };
            
            // Apply the reset filters
            $scope.applyFilters();
        };
        
        /**
         * Initialize controller
         */
        function init() {
            // Create debounced search function for better performance
            const debouncedSearch = OptimizationService.performance.debounce(function() {
                $scope.applyFilters();
            }, 300);
            
            // Watch for search term changes
            $scope.$watch('filters.searchTerm', function(newVal, oldVal) {
                if (newVal !== oldVal) {
                    debouncedSearch();
                }
            });
            
            // Subscribe to events
            EventService.on('data:loaded', function() {
                // Apply default filters
                $scope.applyFilters();
            });
            
            EventService.on('data:reset', function() {
                $scope.$apply(function() {
                    $scope.resetFilters();
                });
            });
            
            // Check if data is already loaded
            if (DataManager.hasData) {
                $scope.applyFilters();
            }
        }
        
        // Initialize controller
        init();
    }
]);