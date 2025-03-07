/**
 * HUMAnN3 Pathway Abundance Viewer
 * File Upload Controller
 * Handles file selection, parsing and processing
 */

app.controller('FileUploadController', [
    '$scope', '$timeout', 'DataManager', 'EventService', 'CSVParser', 'FormattersService', 'OptimizationService',
    function($scope, $timeout, DataManager, EventService, CSVParser, FormattersService, OptimizationService) {
        // Controller state
        $scope.fileName = '';
        $scope.fileStatus = '';
        $scope.fileStatusClass = '';
        $scope.fileStatusIcon = '';
        $scope.isProcessing = false;
        $scope.processingProgress = 0;
        $scope.isDragging = false;

        /**
         * Handle file selected via input
         * @param {FileList} files - Selected files
         */
        $scope.handleFileSelect = function(files) {
            if (!files || !files.length) return;
            
            // Get first file
            const file = files[0];
            processFile(file);
        };
        
        /**
         * Handle file dropped on drop zone
         * @param {FileList} files - Dropped files
         */
        $scope.handleFileDrop = function(files) {
            if (!files || !files.length) return;
            
            // Get first file
            const file = files[0];
            $scope.isDragging = false;
            processFile(file);
        };
        
        /**
         * Handle drag over event
         */
        $scope.handleDragOver = function() {
            $scope.isDragging = true;
        };
        
        /**
         * Handle drag leave event
         */
        $scope.handleDragLeave = function() {
            $scope.isDragging = false;
        };
        
        /**
         * Process the selected file
         * @param {File} file - The file to process
         */
        function processFile(file) {
            // Check file type
            const fileExtension = file.name.split('.').pop().toLowerCase();
            const isValidType = ['csv', 'tsv', 'txt'].includes(fileExtension);
            
            if (!isValidType) {
                setFileStatus('error', 'Invalid file type. Please select a CSV or TSV file.', 'fas fa-exclamation-circle');
                return;
            }
            
            // Check file size
            const maxSize = 500 * 1024 * 1024; // 500MB
            if (file.size > maxSize) {
                setFileStatus('warning', 'File is very large (> 500MB). Processing may take a long time or fail.', 'fas fa-exclamation-triangle');
            }
            
            // Update UI
            $scope.fileName = file.name;
            $scope.isProcessing = true;
            $scope.processingProgress = 0;
            
            setFileStatus('info', 'Reading file...', 'fas fa-spinner fa-spin');
            
            // Update DataManager with file info
            DataManager.setFileInfo(file);
            
            // Detect file type (tab-delimited or comma-delimited)
            const isTabDelimited = fileExtension === 'tsv' || fileExtension === 'txt';
            
            // Parse options
            const parseOptions = {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true,
                isTabDelimited: isTabDelimited,
                comments: '#',
                // For extremely large files, set a maximum row limit
                maxRows: file.size > 100 * 1024 * 1024 ? 10000 : 0, // Limit to 10K rows for files > 100MB
                onProgress: function(progress) {
                    $scope.$apply(function() {
                        $scope.processingProgress = progress.progress;
                        
                        // Update status message for long-running parsing
                        if ($scope.processingProgress < 100) {
                            setFileStatus('info', `Parsing file: ${$scope.processingProgress}% complete...`, 'fas fa-spinner fa-spin');
                        }
                    });
                }
            };
            
            // Parse the file
            CSVParser.parse(file, parseOptions)
                .then(function(result) {
                    // Check for empty data
                    if (!result.data || !result.data.length) {
                        setFileStatus('error', 'No data found in file.', 'fas fa-exclamation-circle');
                        $scope.isProcessing = false;
                        return;
                    }
                    
                    // Check for truncated data
                    if (result.truncated) {
                        setFileStatus('warning', 
                            `File is very large. Showing first ${FormattersService.formatNumber(result.totalRows)} rows.`, 
                            'fas fa-exclamation-triangle');
                    }
                    
                    // Log headers and row count
                    console.log(`Parsed ${result.data.length} rows with ${result.meta.fields ? result.meta.fields.length : 0} columns`);
                    
                    // Start data processing
                    setFileStatus('info', 'Processing data...', 'fas fa-cog fa-spin');
                    
                    // Process the data
                    $timeout(function() {
                        processData(result.data, result.meta.fields);
                    }, 100);
                })
                .catch(function(error) {
                    console.error('Error parsing file:', error);
                    setFileStatus('error', 'Error parsing file: ' + error.message, 'fas fa-exclamation-circle');
                    $scope.isProcessing = false;
                });
        }
        
        /**
         * Process parsed data
         * @param {Array} data - Parsed data rows
         * @param {Array} headers - Column headers
         */
        function processData(data, headers) {
            try {
                // Subscribe to data processing progress updates
                const progressListener = EventService.on('data:processing', function(info) {
                    $scope.$apply(function() {
                        $scope.processingProgress = info.progress;
                    });
                });
                
                // Process the data in DataManager
                DataManager.processData(data, headers);
                
                // Remove progress listener
                progressListener();
                
                // Update UI
                $timeout(function() {
                    $scope.isProcessing = false;
                    $scope.processingProgress = 100;
                    
                    const stats = DataManager.getStats();
                    setFileStatus('success', 
                        `Successfully loaded ${FormattersService.formatNumber(stats.totalPathways)} pathways across ${stats.totalSamples} samples.`, 
                        'fas fa-check-circle');
                    
                    // Clean up memory
                    OptimizationService.memory.triggerCleanup();
                }, 500);
            } catch (error) {
                console.error('Error processing data:', error);
                setFileStatus('error', 'Error processing data: ' + error.message, 'fas fa-exclamation-circle');
                $scope.isProcessing = false;
            }
        }
        
        /**
         * Set file status message and style
         * @param {String} type - Status type ('info', 'success', 'warning', 'error')
         * @param {String} message - Status message
         * @param {String} icon - Icon class
         */
        function setFileStatus(type, message, icon) {
            $scope.$apply(function() {
                $scope.fileStatus = message;
                $scope.fileStatusClass = 'file-' + type;
                $scope.fileStatusIcon = icon;
            });
        }
        
        // Initialize controller
        function init() {
            // Subscribe to events
            EventService.on('data:reset', function() {
                $scope.$apply(function() {
                    $scope.fileName = '';
                    $scope.fileStatus = '';
                    $scope.fileStatusClass = '';
                    $scope.fileStatusIcon = '';
                    $scope.isProcessing = false;
                    $scope.processingProgress = 0;
                });
            });
        }
        
        // Initialize controller
        init();
    }
]);