/**
 * HUMAnN3 Pathway Abundance Viewer
 * Pathway Detail Controller
 * Handles pathway detail view and basic visualizations
 */

app.controller('PathwayDetailController', [
    '$scope', '$timeout', 'DataManager', 'EventService', 'FormattersService', 'ChartUtils',
    function($scope, $timeout, DataManager, EventService, FormattersService, ChartUtils) {
        // Controller state
        $scope.selectedPathway = null;
        $scope.topSamples = [];
        $scope.sampleDisplayLimit = 20;
        $scope.quickChart = null;
        
        /**
         * Initialize quick visualization chart
         */
        function initQuickChart() {
            if (!$scope.selectedPathway) return;
            
            // Wait for Chart.js to load
            if (!window.Chart) {
                loadChartJS().then(() => {
                    ChartUtils.initializeChartDefaults();
                    createQuickChart();
                });
            } else {
                createQuickChart();
            }
        }
        
        /**
         * Create quick overview chart
         */
        function createQuickChart() {
            if (!$scope.selectedPathway) return;
            
            // Destroy previous chart if exists
            if ($scope.quickChart) {
                $scope.quickChart.destroy();
            }
            
            // Get top 5 samples for quick view
            const topSamples = $scope.topSamples.slice(0, 5);
            
            // Chart data
            const labels = topSamples.map(sample => FormattersService.formatSampleName(sample.name, 15));
            const data = topSamples.map(sample => sample.abundance);
            
            // Chart configuration
            const ctx = document.getElementById('quick-chart').getContext('2d');
            
            // Create datasets
            const datasets = [{
                label: 'Abundance',
                data: data,
                backgroundColor: 'rgba(52, 152, 219, 0.6)',
                borderColor: 'rgb(41, 128, 185)',
                borderWidth: 1
            }];
            
            // Create chart config
            const config = ChartUtils.createChartConfig('bar', labels, datasets, {
                animation: {
                    duration: 500
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                },
                plugins: {
                    legend: {
                        display: false
                    }
                }
            });
            
            // Create chart
            $scope.quickChart = new Chart(ctx, config);
        }
        
        /**
         * Initialize visualization tab
         */
        $scope.initVisualization = function() {
            // This is a placeholder for the tab activation
            // Actual visualization is handled by VisualizationController
        };
        
        /**
         * Prepare comparison tab
         */
        $scope.prepareComparison = function() {
            // This is a placeholder for the tab activation
            // Actual comparison is handled by ComparisonController
        };
        
        /**
         * Export pathway data to CSV
         */
        $scope.exportData = function() {
            if (!$scope.selectedPathway) return;
            
            // Get CSV content from DataManager
            const csvContent = DataManager.exportPathwayData($scope.selectedPathway);
            
            if (csvContent) {
                // Clean up pathway name for filename
                const filename = FormattersService.sanitizeFilename($scope.selectedPathway.name);
                
                // Download as CSV
                FormattersService.downloadCSV(csvContent, filename + '_abundance');
            }
        };
        
        /**
         * Update pathway details
         * @param {Object} pathway - Selected pathway
         */
        function updatePathwayDetails(pathway) {
            $scope.selectedPathway = pathway;
            
            if (pathway) {
                // Get top samples for this pathway
                $scope.topSamples = DataManager.getTopSamples(pathway, $scope.sampleDisplayLimit);
                
                // Initialize quick chart with a small delay to ensure DOM is ready
                $timeout(initQuickChart, 100);
            } else {
                $scope.topSamples = [];
            }
        }
        
        /**
         * Load Chart.js dynamically
         * @returns {Promise} - Promise that resolves when Chart.js is loaded
         */
        function loadChartJS() {
            if (window.Chart) return Promise.resolve();
            
            return new Promise((resolve, reject) => {
                const script = document.createElement('script');
                script.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js";
                script.onload = () => {
                    console.log("Chart.js loaded successfully");
                    resolve();
                };
                script.onerror = (error) => {
                    console.error("Failed to load Chart.js:", error);
                    reject(error);
                };
                document.head.appendChild(script);
            });
        }
        
        /**
         * Initialize controller
         */
        function init() {
            // Subscribe to events
            EventService.on('pathway:selected', function(pathway) {
                $scope.$apply(function() {
                    updatePathwayDetails(pathway);
                });
            });
            
            EventService.on('data:reset', function() {
                $scope.$apply(function() {
                    updatePathwayDetails(null);
                });
            });
            
            // Check for any existing selected pathway
            const currentPathway = DataManager.selectedPathway;
            if (currentPathway) {
                updatePathwayDetails(currentPathway);
            }
            
            // Preload Chart.js for better user experience
            $timeout(function() {
                loadChartJS().then(() => {
                    ChartUtils.initializeChartDefaults();
                });
            }, 3000); // Delay loading to prioritize initial app display
        }
        
        // Initialize controller
        init();
    }
]);