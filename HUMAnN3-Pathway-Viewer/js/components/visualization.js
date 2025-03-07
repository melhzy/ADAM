/**
 * HUMAnN3 Pathway Abundance Viewer
 * Visualization Controller
 * Handles advanced pathway visualizations
 */

app.controller('VisualizationController', [
    '$scope', '$timeout', 'DataManager', 'EventService', 'FormattersService', 'ChartUtils',
    function($scope, $timeout, DataManager, EventService, FormattersService, ChartUtils) {
        // Controller state
        $scope.visualOptions = {
            chartType: 'bar',
            sampleLimit: '20',
            colorScheme: 'default'
        };
        $scope.chart = null;
        
        /**
         * Update visualization based on current options
         */
        $scope.updateVisualization = function() {
            // Ensure we have a selected pathway
            if (!$scope.$parent.selectedPathway) return;
            
            // Ensure Chart.js is loaded
            if (!window.Chart) {
                loadChartJS().then(() => {
                    ChartUtils.initializeChartDefaults();
                    createVisualization();
                });
            } else {
                createVisualization();
            }
        };
        
        /**
         * Create pathway visualization
         */
        function createVisualization() {
            // Get selected pathway from parent scope
            const pathway = $scope.$parent.selectedPathway;
            if (!pathway) return;
            
            // Destroy previous chart if exists
            if ($scope.chart) {
                $scope.chart.destroy();
            }
            
            // Get top samples based on sample limit option
            const limit = $scope.visualOptions.sampleLimit === 'all' ? 
                DataManager.getSamples().length : 
                parseInt($scope.visualOptions.sampleLimit) || 20;
            
            const topSamples = DataManager.getTopSamples(pathway, limit);
            
            // Process chart data based on chart type
            const chartType = $scope.visualOptions.chartType;
            const labels = topSamples.map(sample => FormattersService.formatSampleName(sample.name, 15));
            const data = topSamples.map(sample => sample.abundance);
            
            // Create dataset with appropriate styling
            const datasets = [{
                label: 'Abundance',
                data: data,
                backgroundColor: generateColors(data.length, true),
                borderColor: generateColors(data.length, false),
                borderWidth: 1
            }];
            
            // Chart options based on chart type
            const options = {
                title: pathway.name,
                xTitle: 'Sample',
                yTitle: 'Abundance',
                animation: {
                    duration: 750
                }
            };
            
            // Special handling for specific chart types
            if (chartType === 'heatmap') {
                createHeatmapChart(pathway, topSamples);
                return;
            }
            
            // Create chart configuration
            const config = ChartUtils.createChartConfig(chartType, labels, datasets, options);
            
            // Create chart
            const ctx = document.getElementById('abundance-chart').getContext('2d');
            $scope.chart = new Chart(ctx, config);
            
            // Log chart creation
            console.log(`Created ${chartType} chart for ${pathway.name} with ${topSamples.length} samples`);
        }
        
        /**
         * Create heatmap visualization (special case)
         * @param {Object} pathway - Selected pathway
         * @param {Array} topSamples - Top samples for the pathway
         */
        function createHeatmapChart(pathway, topSamples) {
            const canvas = document.getElementById('abundance-chart');
            const ctx = canvas.getContext('2d');
            
            // Clear canvas
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            
            // Get color scheme
            const colorScheme = $scope.visualOptions.colorScheme || 'default';
            const colorMap = getColorSchemeGradient(colorScheme);
            
            // Calculate dimensions
            const cellWidth = Math.min(50, canvas.width / topSamples.length);
            const cellHeight = 50;
            const padding = 5;
            
            // Get max value for color scaling
            const maxValue = Math.max(...topSamples.map(s => s.abundance));
            
            // Draw cells
            topSamples.forEach((sample, i) => {
                // Calculate color based on value
                const value = sample.abundance;
                const normalizedValue = maxValue > 0 ? value / maxValue : 0;
                const color = getColorForValue(normalizedValue, colorMap);
                
                // Calculate position
                const x = padding + i * cellWidth;
                const y = padding;
                
                // Draw cell
                ctx.fillStyle = color;
                ctx.fillRect(x, y, cellWidth - padding, cellHeight);
                
                // Add value text
                ctx.fillStyle = normalizedValue > 0.5 ? 'white' : 'black';
                ctx.font = '10px sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(FormattersService.formatNumber(value, 2), x + (cellWidth - padding) / 2, y + cellHeight / 2);
                
                // Add sample name below (rotated)
                ctx.save();
                ctx.translate(x + (cellWidth - padding) / 2, y + cellHeight + 5);
                ctx.rotate(Math.PI / 4); // 45 degrees
                ctx.fillStyle = 'black';
                ctx.textAlign = 'left';
                ctx.fillText(FormattersService.formatSampleName(sample.name, 10), 0, 0);
                ctx.restore();
            });
            
            // Draw legend
            drawHeatmapLegend(ctx, canvas.width - 100, padding, 100, 20, colorMap, maxValue);
            
            // No chart object for heatmap
            $scope.chart = null;
        }
        
        /**
         * Draw heatmap legend
         * @param {CanvasRenderingContext2D} ctx - Canvas context
         * @param {Number} x - X position
         * @param {Number} y - Y position
         * @param {Number} width - Legend width
         * @param {Number} height - Legend height
         * @param {Object} colorMap - Color map object
         * @param {Number} maxValue - Maximum value
         */
        function drawHeatmapLegend(ctx, x, y, width, height, colorMap, maxValue) {
            // Draw gradient
            const gradient = ctx.createLinearGradient(x, y, x + width, y);
            
            Object.keys(colorMap).forEach(position => {
                gradient.addColorStop(parseFloat(position), colorMap[position]);
            });
            
            ctx.fillStyle = gradient;
            ctx.fillRect(x, y, width, height);
            
            // Draw border
            ctx.strokeStyle = 'black';
            ctx.lineWidth = 1;
            ctx.strokeRect(x, y, width, height);
            
            // Draw min/max labels
            ctx.fillStyle = 'black';
            ctx.font = '10px sans-serif';
            ctx.textAlign = 'left';
            ctx.fillText('0', x, y + height + 10);
            ctx.textAlign = 'right';
            ctx.fillText(FormattersService.formatNumber(maxValue, 2), x + width, y + height + 10);
        }
        
        /**
         * Get color scheme gradient
         * @param {String} scheme - Color scheme name
         * @returns {Object} - Color map object
         */
        function getColorSchemeGradient(scheme) {
            const colorSchemes = {
                default: {
                    '0': '#f7fbff',
                    '0.4': '#c6dbef',
                    '0.7': '#6baed6',
                    '0.9': '#2171b5',
                    '1': '#08306b'
                },
                viridis: {
                    '0': '#440154',
                    '0.25': '#414487',
                    '0.5': '#2a788e',
                    '0.75': '#22a884',
                    '1': '#7ad151'
                },
                inferno: {
                    '0': '#000004',
                    '0.25': '#420a68',
                    '0.5': '#932667',
                    '0.75': '#dd513a',
                    '1': '#fca50a'
                },
                plasma: {
                    '0': '#0d0887',
                    '0.25': '#6a00a8',
                    '0.5': '#b12a90',
                    '0.75': '#e16462',
                    '1': '#fca636'
                },
                cool: {
                    '0': '#6e40aa',
                    '0.25': '#4776ff',
                    '0.5': '#10a4db',
                    '0.75': '#36c956',
                    '1': '#eff953'
                },
                warm: {
                    '0': '#6e40aa',
                    '0.25': '#be3caf',
                    '0.5': '#fe4b83',
                    '0.75': '#ff7847',
                    '1': '#e2b72f'
                }
            };
            
            return colorSchemes[scheme] || colorSchemes.default;
        }
        
        /**
         * Get color for a normalized value from a color map
         * @param {Number} normalizedValue - Normalized value (0-1)
         * @param {Object} colorMap - Color map object
         * @returns {String} - Color value
         */
        function getColorForValue(normalizedValue, colorMap) {
            const positions = Object.keys(colorMap).map(parseFloat).sort((a, b) => a - b);
            
            // Find the two positions that our value falls between
            let pos1 = positions[0];
            let pos2 = positions[positions.length - 1];
            
            for (let i = 0; i < positions.length - 1; i++) {
                if (normalizedValue >= positions[i] && normalizedValue <= positions[i + 1]) {
                    pos1 = positions[i];
                    pos2 = positions[i + 1];
                    break;
                }
            }
            
            // If value exactly matches a position, return that color
            if (normalizedValue === pos1) return colorMap[pos1];
            if (normalizedValue === pos2) return colorMap[pos2];
            
            // Interpolate between the two colors
            const ratio = (normalizedValue - pos1) / (pos2 - pos1);
            return interpolateColor(colorMap[pos1], colorMap[pos2], ratio);
        }
        
        /**
         * Interpolate between two colors
         * @param {String} color1 - First color
         * @param {String} color2 - Second color
         * @param {Number} ratio - Interpolation ratio (0-1)
         * @returns {String} - Interpolated color
         */
        function interpolateColor(color1, color2, ratio) {
            // Parse colors
            const r1 = parseInt(color1.substring(1, 3), 16);
            const g1 = parseInt(color1.substring(3, 5), 16);
            const b1 = parseInt(color1.substring(5, 7), 16);
            
            const r2 = parseInt(color2.substring(1, 3), 16);
            const g2 = parseInt(color2.substring(3, 5), 16);
            const b2 = parseInt(color2.substring(5, 7), 16);
            
            // Interpolate
            const r = Math.round(r1 + (r2 - r1) * ratio);
            const g = Math.round(g1 + (g2 - g1) * ratio);
            const b = Math.round(b1 + (b2 - b1) * ratio);
            
            // Convert back to hex
            return `#${r.toString(16).padStart(2, '0')}${g.toString(16).padStart(2, '0')}${b.toString(16).padStart(2, '0')}`;
        }
        
        /**
         * Generate colors for chart elements
         * @param {Number} count - Number of colors needed
         * @param {Boolean} transparent - Whether to use transparency
         * @returns {Array|String} - Colors or color array
         */
        function generateColors(count, transparent) {
            const scheme = $scope.visualOptions.colorScheme || 'default';
            
            // Get base colors based on scheme
            const colorSets = {
                default: ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#1abc9c', '#f1c40f', '#e67e22'],
                viridis: ['#440154', '#414487', '#2a788e', '#22a884', '#7ad151', '#fde725'],
                inferno: ['#000004', '#420a68', '#932667', '#dd513a', '#fca50a', '#fcffa4'],
                plasma: ['#0d0887', '#6a00a8', '#b12a90', '#e16462', '#fca636', '#f0f921'],
                cool: ['#6e40aa', '#4776ff', '#10a4db', '#36c956', '#eff953'],
                warm: ['#6e40aa', '#be3caf', '#fe4b83', '#ff7847', '#e2b72f']
            };
            
            const colors = colorSets[scheme] || colorSets.default;
            
            // Generate color array
            const result = [];
            for (let i = 0; i < count; i++) {
                const color = colors[i % colors.length];
                
                if (transparent) {
                    // Add transparency
                    const r = parseInt(color.slice(1, 3), 16);
                    const g = parseInt(color.slice(3, 5), 16);
                    const b = parseInt(color.slice(5, 7), 16);
                    result.push(`rgba(${r}, ${g}, ${b}, 0.7)`);
                } else {
                    result.push(color);
                }
            }
            
            return count === 1 ? result[0] : result;
        }
        
        /**
         * Download chart as image
         */
        $scope.downloadChart = function() {
            if (!$scope.$parent.selectedPathway) return;
            
            const canvas = document.getElementById('abundance-chart');
            const pathwayName = FormattersService.sanitizeFilename($scope.$parent.selectedPathway.name);
            
            // Download chart image
            FormattersService.downloadCanvasAsImage(canvas, pathwayName + '_chart');
        };
        
        /**
         * Export data to CSV
         */
        $scope.exportData = function() {
            $scope.$parent.exportData();
        };
        
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
            // Try to restore visualization preferences
            try {
                const savedOptions = localStorage.getItem('visualizationOptions');
                if (savedOptions) {
                    $scope.visualOptions = JSON.parse(savedOptions);
                }
            } catch (e) {
                // Ignore storage errors
            }
            
            // Watch for option changes to save preferences
            $scope.$watch('visualOptions', function(newVal) {
                try {
                    localStorage.setItem('visualizationOptions', JSON.stringify(newVal));
                } catch (e) {
                    // Ignore storage errors
                }
            }, true);
            
            // Watch for parent's selectedPathway changes
            $scope.$watch('$parent.selectedPathway', function(newVal) {
                if (newVal) {
                    // Update visualization when tab is active
                    if (document.getElementById('visualization-tab').classList.contains('active')) {
                        $timeout($scope.updateVisualization, 100);
                    }
                }
            });
            
            // Subscribe to events
            EventService.on('data:reset', function() {
                if ($scope.chart) {
                    $scope.chart.destroy();
                    $scope.chart = null;
                }
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