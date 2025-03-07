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
            console.log('Applying filters:', JSON.stringify($scope.filters));
            
            // Apply filters via DataManager and get filtered results
            const filteredPathways = DataManager.applyFilters($scope.filters);
            
            // Emit event with filtered pathways - we're passing this directly
            // rather than having the controller call DataManager.getFilteredPathways()
            EventService.emit('filters:applied', {
                filters: $scope.filters,
                pathways: filteredPathways
            });
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
            
            // Watch for pathway type changes
            $scope.$watch('filters.pathwayType', function(newVal, oldVal) {
                if (newVal !== oldVal) {
                    $scope.applyFilters();
                }
            });
            
            // Watch for sort field changes
            $scope.$watch('filters.sortField', function(newVal, oldVal) {
                if (newVal !== oldVal) {
                    $scope.applyFilters();
                }
            });
            
            // Watch for sample filter changes
            $scope.$watch('filters.selectedSample', function(newVal, oldVal) {
                if (newVal !== oldVal) {
                    $scope.applyFilters();
                }
            });
            
            // Subscribe to events
            EventService.on('data:loaded', function() {
                console.log('Received data:loaded event in FilterController');
                // Apply default filters when data is initially loaded
                $scope.applyFilters();
            });
            
            EventService.on('data:reset', function() {
                $scope.$apply(function() {
                    $scope.resetFilters();
                });
            });
            
            // Check if data is already loaded
            if (DataManager.hasData) {
                console.log('Data already loaded, applying initial filters');
                $scope.applyFilters();
            }
        }
        
        // Initialize controller
        init();
    }
]);