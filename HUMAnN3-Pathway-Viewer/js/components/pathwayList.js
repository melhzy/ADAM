/**
 * HUMAnN3 Pathway Abundance Viewer
 * Pathway List Controller
 * Handles pathway list display and interactions
 */

app.controller('PathwayListController', [
    '$scope', '$timeout', 'DataManager', 'EventService', 'OptimizationService',
    function($scope, $timeout, DataManager, EventService, OptimizationService) {
        // Controller state
        $scope.isLoading = false;
        $scope.isLoadingMore = false;
        $scope.filteredPathways = [];
        $scope.displayedPathways = [];
        $scope.selectedPathwayId = null;
        $scope.currentPage = 1;
        $scope.totalPages = 1;
        $scope.itemsPerPage = 50;
        $scope.viewMode = 'normal'; // 'normal' or 'compact'
        
        /**
         * Select a pathway
         * @param {Object} pathway - Pathway to select
         */
        $scope.selectPathway = function(pathway) {
            if (!pathway) return;
            
            $scope.selectedPathwayId = pathway.id;
            DataManager.selectPathway(pathway);
        };
        
        /**
         * Toggle view mode between normal and compact
         */
        $scope.toggleViewMode = function() {
            $scope.viewMode = $scope.viewMode === 'normal' ? 'compact' : 'normal';
            
            // Save preference in localStorage
            try {
                localStorage.setItem('pathwayViewMode', $scope.viewMode);
            } catch (e) {
                // Ignore storage errors
                console.warn('Failed to save view mode preference:', e);
            }
        };
        
        /**
         * Go to previous page
         */
        $scope.prevPage = function() {
            if ($scope.currentPage > 1) {
                $scope.currentPage--;
                updateDisplayedPathways();
                scrollToTop();
            }
        };
        
        /**
         * Go to next page
         */
        $scope.nextPage = function() {
            if ($scope.currentPage < $scope.totalPages) {
                $scope.currentPage++;
                updateDisplayedPathways();
                scrollToTop();
            }
        };
        
        /**
         * Go to specific page
         * @param {Number} page - Page number
         */
        $scope.goToPage = function(page) {
            if (page >= 1 && page <= $scope.totalPages && page !== $scope.currentPage) {
                $scope.currentPage = page;
                updateDisplayedPathways();
                scrollToTop();
            }
        };
        
        /**
         * Scroll pathway list to top
         */
        function scrollToTop() {
            const pathwayList = document.getElementById('pathway-list');
            if (pathwayList) {
                pathwayList.scrollTop = 0;
            }
        }
        
        /**
         * Update displayed pathways based on current page
         */
        function updateDisplayedPathways() {
            const startIndex = ($scope.currentPage - 1) * $scope.itemsPerPage;
            const endIndex = Math.min(startIndex + $scope.itemsPerPage, $scope.filteredPathways.length);
            
            $scope.displayedPathways = $scope.filteredPathways.slice(startIndex, endIndex);
        }
        
        /**
         * Update pagination based on filtered pathways
         */
        function updatePagination() {
            $scope.totalPages = Math.ceil($scope.filteredPathways.length / $scope.itemsPerPage);
            
            // Ensure current page is valid
            if ($scope.currentPage > $scope.totalPages) {
                $scope.currentPage = Math.max(1, $scope.totalPages);
            }
        }
        
        /**
         * Find the page containing a specific pathway
         * @param {String} pathwayId - Pathway ID to find
         */
        function findPathwayPage(pathwayId) {
            const index = $scope.filteredPathways.findIndex(p => p.id === pathwayId);
            
            if (index >= 0) {
                const page = Math.floor(index / $scope.itemsPerPage) + 1;
                $scope.goToPage(page);
            }
        }
        
        /**
         * Initialize controller
         */
        function init() {
            // Try to restore view mode preference
            try {
                const savedViewMode = localStorage.getItem('pathwayViewMode');
                if (savedViewMode === 'compact' || savedViewMode === 'normal') {
                    $scope.viewMode = savedViewMode;
                }
            } catch (e) {
                // Ignore storage errors
            }
            
            // Subscribe to events
            EventService.on('data:loaded', function() {
                console.log('Received data:loaded event in PathwayListController');
                
                // Update with an $apply since this is an async event
                $scope.$apply(function() {
                    $scope.isLoading = false;
                    $scope.filteredPathways = DataManager.getFilteredPathways();
                    console.log('Filtered pathways count:', $scope.filteredPathways.length);
                    
                    updatePagination();
                    updateDisplayedPathways();
                });
            });
            
            EventService.on('filters:applied', function(data) {
                console.log('Received filters:applied event with', (data && data.pathways) ? data.pathways.length : 0, 'pathways');
                
                $scope.$apply(function() {
                    $scope.isLoading = false;
                    // Important: take the filtered pathways from the event data rather than calling getFilteredPathways
                    $scope.filteredPathways = data && data.pathways ? data.pathways : [];
                    
                    updatePagination();
                    
                    // If we have a selected pathway, try to keep it in view
                    if ($scope.selectedPathwayId) {
                        findPathwayPage($scope.selectedPathwayId);
                    } else {
                        // Otherwise go to first page
                        $scope.currentPage = 1;
                    }
                    
                    updateDisplayedPathways();
                });
            });
            
            EventService.on('pathway:selected', function(pathway) {
                $scope.$apply(function() {
                    $scope.selectedPathwayId = pathway.id;
                    
                    // Ensure selected pathway is visible
                    const isPathwayDisplayed = $scope.displayedPathways.some(p => p.id === pathway.id);
                    
                    if (!isPathwayDisplayed) {
                        findPathwayPage(pathway.id);
                    }
                });
            });
            
            EventService.on('data:reset', function() {
                $scope.$apply(function() {
                    $scope.isLoading = false;
                    $scope.filteredPathways = [];
                    $scope.displayedPathways = [];
                    $scope.selectedPathwayId = null;
                    $scope.currentPage = 1;
                    $scope.totalPages = 1;
                });
            });
            
            // Check if data is already loaded and apply filters to initialize the list
            if (DataManager.hasData) {
                console.log('Data already loaded, initializing pathway list');
                $scope.filteredPathways = DataManager.getFilteredPathways();
                updatePagination();
                updateDisplayedPathways();
            }
            
            // Set up virtual scrolling for large pathway lists
            const pathwayList = document.getElementById('pathway-list');
            
            if (pathwayList) {
                // Optimized scroll handler
                const handleScroll = OptimizationService.performance.throttle(function() {
                    // Check if we're near the bottom for infinite scrolling
                    const scrollPos = pathwayList.scrollTop + pathwayList.clientHeight;
                    const scrollMax = pathwayList.scrollHeight - 50; // 50px threshold
                    
                    if (scrollPos >= scrollMax && !$scope.isLoadingMore && $scope.currentPage < $scope.totalPages) {
                        // Load more items
                        $scope.$apply(function() {
                            $scope.isLoadingMore = true;
                        });
                        
                        $timeout(function() {
                            $scope.nextPage();
                            $scope.isLoadingMore = false;
                        }, 100);
                    }
                }, 200);
                
                pathwayList.addEventListener('scroll', handleScroll);
                
                // Clean up event listener when scope is destroyed
                $scope.$on('$destroy', function() {
                    pathwayList.removeEventListener('scroll', handleScroll);
                });
            }
        }
        
        // Initialize controller
        init();
    }
]);