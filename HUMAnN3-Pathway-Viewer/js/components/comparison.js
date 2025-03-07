/**
 * HUMAnN3 Pathway Abundance Viewer
 * Comparison Controller
 * Handles pathway comparisons across samples or with other pathways
 */

app.controller('ComparisonController', [
    '$scope', '$timeout', 'DataManager', 'EventService', 'FormattersService', 'ChartUtils',
    function($scope, $timeout, DataManager, EventService, FormattersService, ChartUtils) {
        // Controller state
        $scope.comparisonOptions = {
            type: 'samples',  // 'samples' or 'pathways'
            pathways: [],     // Selected pathways for comparison
            samples: []       // Selected samples for comparison
        };
        $scope.similarPathways = [];
        $scope.chart = null;
        
        /**
         * Update comparison visualization
         */
        $scope.updateComparison = function() {
            // Ensure we have a selected pathway
            if (!$scope.$parent.selectedPathway) return;
            
            // Ensure Chart.js is loaded
            if (!window.Chart) {
                loadChartJS().then(() => {
                    ChartUtils.initializeChartDefaults();
                    createComparisonChart();
                });
            } else {
                createComparisonChart();
            }
        };
        
        /**
         * Create comparison chart based on current options
         */
        function createComparisonChart() {
            // Get selected pathway from parent scope
            const pathway = $scope.$parent.selectedPathway;
            if (!pathway) return;
            
            // Destroy previous chart if exists
            if ($scope.chart) {
                $scope.chart.destroy();
            }
            
            // Create chart based on comparison type
            if ($scope.comparisonOptions.type === 'samples') {
                createSampleComparisonChart(pathway);
            } else {
                createPathwayComparisonChart(pathway);
            }
        }
        
        /**
         * Create chart comparing pathway across selected samples
         * @param {Object} pathway - The selected pathway
         */
        function createSampleComparisonChart(pathway) {
            // Get selected samples or use top samples if none selected
            let samples = [];
            
            if ($scope.comparisonOptions.samples && $scope.comparisonOptions.samples.length > 0) {
                // Use selected samples
                samples = $scope.comparisonOptions.samples.map(sampleName => {
                    return {
                        name: sampleName,
                        abundance: pathway.getAbundanceForSample(sampleName)
                    };
                });
            } else {
                // Use top samples
                samples = DataManager.getTopSamples(pathway, 10);
            }
            
            // Sort samples by abundance
            samples.sort((a, b) => b.abundance - a.abundance);
            
            // Chart data
            const labels = samples.map(sample => FormattersService.formatSampleName(sample.name, 15));
            const data = samples.map(sample => sample.abundance);
            
            // Create datasets
            const datasets = [{
                label: pathway.name,
                data: data,
                backgroundColor: 'rgba(52, 152, 219, 0.7)',
                borderColor: 'rgb(41, 128, 185)',
                borderWidth: 1
            }];
            
            // Create chart config
            const config = ChartUtils.createChartConfig('bar', labels, datasets, {
                title: `${pathway.name} - Sample Comparison`,
                xTitle: 'Sample',
                yTitle: 'Abundance',
                animation: {
                    duration: 750
                }
            });
            
            // Create chart
            const ctx = document.getElementById('comparison-chart').getContext('2d');
            $scope.chart = new Chart(ctx, config);
        }
        
        /**
         * Create chart comparing selected pathway with other pathways
         * @param {Object} pathway - The selected pathway
         */
        function createPathwayComparisonChart(pathway) {
            // Get comparison pathways
            let comparisonPathways = [];
            
            if ($scope.comparisonOptions.pathways && $scope.comparisonOptions.pathways.length > 0) {
                // Get selected pathways
                comparisonPathways = $scope.comparisonOptions.pathways.map(pathwayId => {
                    return DataManager.getPathways().find(p => p.id === pathwayId);
                }).filter(p => p); // Remove null/undefined
            } else {
                // Use similar pathways
                comparisonPathways = $scope.similarPathways.slice(0, 5);
            }
            
            // Add the selected pathway as first item
            comparisonPathways = [pathway, ...comparisonPathways.filter(p => p.id !== pathway.id)];
            
            // Limit to 6 pathways for readability
            if (comparisonPathways.length > 6) {
                comparisonPathways = comparisonPathways.slice(0, 6);
            }
            
            // Get all samples
            const samples = DataManager.getSamples();
            
            // For samples comparison, we need to limit to top samples by average abundance
            // across all selected pathways
            const sampleScores = {};
            
            samples.forEach(sample => {
                sampleScores[sample] = comparisonPathways.reduce((sum, p) => {
                    return sum + p.getAbundanceForSample(sample);
                }, 0) / comparisonPathways.length;
            });
            
            // Sort samples by score
            const topSamples = Object.keys(sampleScores)
                .sort((a, b) => sampleScores[b] - sampleScores[a])
                .slice(0, 8); // Limit to 8 samples
            
            // Create datasets for chart
            const datasets = comparisonPathways.map((p, i) => {
                // Get abundance values for top samples
                const data = topSamples.map(sample => p.getAbundanceForSample(sample));
                
                // Use different colors for each pathway
                const colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c'];
                
                return {
                    label: p.id === pathway.id ? `${p.name} (selected)` : p.name,
                    data: data,
                    backgroundColor: `rgba(${hexToRgb(colors[i % colors.length])}, 0.7)`,
                    borderColor: colors[i % colors.length],
                    borderWidth: 1
                };
            });
            
            // Chart labels
            const labels = topSamples.map(sample => FormattersService.formatSampleName(sample, 15));
            
            // Create chart config
            const config = ChartUtils.createChartConfig('bar', labels, datasets, {
                title: 'Pathway Comparison',
                xTitle: 'Sample',
                yTitle: 'Abundance',
                animation: {
                    duration: 750
                }
            });
            
            // Create chart
            const ctx = document.getElementById('comparison-chart').getContext('2d');
            $scope.chart = new Chart(ctx, config);
        }
        
        /**
         * Convert hex color to RGB values
         * @param {String} hex - Hex color string
         * @returns {String} - Comma-separated RGB values
         */
        function hexToRgb(hex) {
            // Remove # if present
            hex = hex.replace('#', '');
            
            // Parse RGB values
            const r = parseInt(hex.substring(0, 2), 16);
            const g = parseInt(hex.substring(2, 4), 16);
            const b = parseInt(hex.substring(4, 6), 16);
            
            return `${r}, ${g}, ${b}`;
        }
        
        /**
         * Find similar pathways for comparison
         * @param {Object} pathway - The selected pathway
         * @param {Number} limit - Maximum number of similar pathways to find
         */
        function findSimilarPathways(pathway, limit) {
            limit = limit || 10;
            
            if (!pathway) {
                $scope.similarPathways = [];
                return;
            }
            
            // Get similar pathways from DataManager
            $scope.similarPathways = DataManager.findSimilarPathways(pathway, limit);
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
                script.onload = resolve;
                script.onerror = reject;
                document.head.appendChild(script);
            });
        }
        
        /**
         * Initialize controller
         */
        function init() {
            // Try to restore comparison preferences
            try {
                const savedOptions = localStorage.getItem('comparisonOptions');
                if (savedOptions) {
                    $scope.comparisonOptions = JSON.parse(savedOptions);
                }
            } catch (e) {
                // Ignore storage errors
            }
            
            // Watch for option changes to save preferences
            $scope.$watch('comparisonOptions', function(newVal) {
                try {
                    localStorage.setItem('comparisonOptions', JSON.stringify(newVal));
                } catch (e) {
                    // Ignore storage errors
                }
            }, true);
            
            // Watch for parent's selectedPathway changes
            $scope.$watch('$parent.selectedPathway', function(newVal) {
                if (newVal) {
                    // Find similar pathways for comparison
                    findSimilarPathways(newVal);
                    
                    // Update visualization when tab is active
                    if (document.getElementById('comparison-tab').classList.contains('active')) {
                        $timeout($scope.updateComparison, 100);
                    }
                }
            });
            
            // Subscribe to events
            EventService.on('data:reset', function() {
                if ($scope.chart) {
                    $scope.chart.destroy();
                    $scope.chart = null;
                }
                $scope.similarPathways = [];
            });
            
            // Load Chart.js in the background
            $timeout(function() {
                loadChartJS().then(() => {
                    ChartUtils.initializeChartDefaults();
                });
            }, 2000);
        }
        
        // Initialize controller
        init();
    }
]);