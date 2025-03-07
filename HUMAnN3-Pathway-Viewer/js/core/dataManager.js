/**
 * HUMAnN3 Pathway Abundance Viewer
 * Data Manager Service
 * Central service for data handling and manipulation
 */

app.service('DataManagerService', ['$q', 'EventHandlerService', 'CSVParserService', 'FormattersService', 
function($q, EventHandlerService, CSVParserService, FormattersService) {
    var service = this;
    
    // Data storage
    service.pathways = [];
    service.samples = [];
    service.hasData = false;
    service.dataStats = {
        totalPathways: 0,
        totalSamples: 0,
        metacycPathways: 0,
        unmappedPercent: 0,
        maxAbundance: 0,
        avgAbundance: 0
    };
    
    // Keep track of file metadata
    var fileInfo = {
        name: null,
        size: 0,
        lastModified: null
    };
    
    /**
     * Load data from a file
     * @param {File} file - File to load
     * @returns {Promise} - Promise resolving when data is loaded
     */
    service.loadFromFile = function(file) {
        const deferred = $q.defer();
        
        // Validate file input
        if (!file || !(file instanceof File)) {
            console.error('Invalid file object provided:', file);
            deferred.reject('Invalid file object. Please select a valid file.');
            
            // Broadcast error event
            EventHandlerService.broadcast(EventHandlerService.events.DATA_ERROR, {
                error: 'Invalid file object. Please select a valid file.'
            });
            
            return deferred.promise;
        }
        
        console.log('File loading started:', file.name, file.size);
        
        // Store file info
        fileInfo = {
            name: file.name,
            size: file.size,
            lastModified: new Date(file.lastModified)
        };
        
        // Broadcast file upload start event
        EventHandlerService.broadcast(EventHandlerService.events.FILE_UPLOAD_START, {
            file: fileInfo
        });
        
        // Parse the file
        CSVParserService.parseFile(file)
            .then(
                function(result) {
                    console.log('File parsed successfully, processing data...');
                    // Process the data and prepare it for use
                    return processData(result);
                },
                function(parseError) {
                    console.error('Error parsing file:', parseError);
                    throw new Error('Could not parse file: ' + (parseError.message || parseError));
                }
            )
            .then(
                function(data) {
                    console.log('Data processed successfully, updating service data...');
                    // Update the service data
                    updateServiceData(data);
                    
                    // Broadcast data loaded event
                    EventHandlerService.broadcast(EventHandlerService.events.DATA_LOADED, {
                        pathways: service.pathways,
                        samples: service.samples,
                        stats: service.dataStats
                    });
                    
                    // Resolve the promise
                    deferred.resolve(service.pathways);
                    console.log('File loading completed successfully.');
                },
                function(processError) {
                    console.error('Error processing data:', processError);
                    throw new Error('Error processing data: ' + (processError.message || processError));
                }
            )
            .catch(function(error) {
                const errorMsg = error instanceof Error ? error.message : error.toString();
                console.error('Error loading data from file:', errorMsg);
                
                // Broadcast error event
                EventHandlerService.broadcast(EventHandlerService.events.DATA_ERROR, {
                    error: errorMsg
                });
                
                // Reject the promise
                deferred.reject(errorMsg);
            });
        
        // Return the promise
        return deferred.promise;
    };
    
    /**
     * Process parsed data
     * @param {Object} parseResult - Result from CSV parser
     * @returns {Object} - Processed data
     */
    function processData(parseResult) {
        return $q(function(resolve, reject) {
            try {
                // Validate parse result
                if (!parseResult || !parseResult.data || !Array.isArray(parseResult.data.pathways)) {
                    console.error('Invalid parse result structure:', parseResult);
                    reject('Invalid data format. Expected pathway data is missing.');
                    return;
                }
                
                console.log('Processing data, found', (parseResult.data.pathways || []).length, 'pathways');
                
                // Extract basic info
                const pathways = parseResult.data.pathways || [];
                const sampleCount = parseResult.sampleCount || 0;
                
                // Calculate statistics
                let metacycCount = 0;
                let unmappedAbundance = 0;
                let totalAbundance = 0;
                let maxAbundance = 0;
                
                // Process pathways
                for (let i = 0; i < pathways.length; i++) {
                    const pathway = pathways[i];
                    
                    // Count MetaCyc pathways
                    if (pathway.type === 'metacyc') {
                        metacycCount++;
                    }
                    
                    // Track unmapped abundance
                    if (pathway.type === 'unmapped') {
                        unmappedAbundance += pathway.avgAbundance;
                    }
                    
                    // Track total abundance
                    totalAbundance += pathway.avgAbundance;
                    
                    // Track maximum abundance
                    maxAbundance = Math.max(maxAbundance, pathway.maxAbundance);
                }
                
                // Extract all sample names
                const sampleNames = [];
                if (pathways.length > 0) {
                    const firstPathway = pathways[0];
                    Object.keys(firstPathway.abundances).forEach(function(sampleName) {
                        sampleNames.push(sampleName);
                    });
                }
                
                // Calculate unmapped percentage
                const unmappedPercent = totalAbundance > 0 ? 
                    (unmappedAbundance / totalAbundance * 100).toFixed(1) : 0;
                
                // Prepare result
                const result = {
                    pathways: pathways,
                    samples: sampleNames,
                    stats: {
                        totalPathways: pathways.length,
                        totalSamples: sampleNames.length,
                        metacycPathways: metacycCount,
                        unmappedPercent: unmappedPercent,
                        maxAbundance: maxAbundance,
                        avgAbundance: totalAbundance / (pathways.length || 1)
                    }
                };
                
                // Resolve the promise with the processed data
                resolve(result);
            } catch (e) {
                console.error('Error in processData:', e);
                reject('Error processing data: ' + e.message);
            }
        });
    }
    
    /**
     * Update service data with processed data
     * @param {Object} data - Processed data
     */
    function updateServiceData(data) {
        // Update pathways array
        service.pathways = data.pathways;
        
        // Update samples array
        service.samples = data.samples;
        
        // Update statistics
        service.dataStats = data.stats;
        
        // Set the hasData flag
        service.hasData = true;
    }
    
    /**
     * Get list of all sample names
     * @returns {Array} - Array of sample names
     */
    service.getSamples = function() {
        return service.samples;
    };
    
    /**
     * Get a pathway by ID
     * @param {String} id - Pathway ID
     * @returns {Object|null} - Pathway object or null if not found
     */
    service.getPathwayById = function(id) {
        if (!id) return null;
        
        for (let i = 0; i < service.pathways.length; i++) {
            if (service.pathways[i].id === id) {
                return service.pathways[i];
            }
        }
        return null;
    };
    
    /**
     * Filter pathways based on criteria
     * @param {Object} filters - Filter criteria
     * @returns {Array} - Filtered pathways
     */
    service.filterPathways = function(filters) {
        if (!filters) return service.pathways;
        
        // Start with all pathways
        let result = service.pathways.slice();
        
        // Apply search term filter
        if (filters.searchTerm) {
            const searchTerm = filters.searchTerm.toLowerCase();
            result = result.filter(function(pathway) {
                return pathway.id.toLowerCase().includes(searchTerm) || 
                       pathway.name.toLowerCase().includes(searchTerm);
            });
        }
        
        // Apply pathway type filter
        if (filters.pathwayType && filters.pathwayType !== 'all') {
            const typeFilter = filters.pathwayType.toLowerCase();
            result = result.filter(function(pathway) {
                return pathway.type === typeFilter || 
                       (typeFilter === 'pwy' && pathway.type === 'metacyc');
            });
        }
        
        // Apply sorting
        if (filters.sortField) {
            result.sort(function(a, b) {
                if (filters.sortField === 'id') {
                    return a.id.localeCompare(b.id);
                } else if (filters.sortField === 'name') {
                    return a.name.localeCompare(b.name);
                } else if (filters.sortField === 'abundance') {
                    return b.avgAbundance - a.avgAbundance;
                }
                return 0;
            });
        }
        
        return result;
    };
    
    /**
     * Clear all data
     */
    service.clearData = function() {
        service.pathways = [];
        service.samples = [];
        service.hasData = false;
        service.dataStats = {
            totalPathways: 0,
            totalSamples: 0,
            metacycPathways: 0,
            unmappedPercent: 0,
            maxAbundance: 0,
            avgAbundance: 0
        };
        
        // Reset file info
        fileInfo = {
            name: null,
            size: 0,
            lastModified: null
        };
    };
    
    /**
     * Get file information
     * @returns {Object} - File information
     */
    service.getFileInfo = function() {
        return fileInfo;
    };
    
    // Add utility method to get pathway display name
    service.getPathwayDisplayName = function(pathway) {
        if (!pathway) return '';
        
        // For many MetaCyc pathways, the name might include the ID
        // This extracts just the descriptive part
        const name = pathway.name;
        if (name.includes(pathway.id)) {
            return name.replace(pathway.id, '').trim();
        }
        
        return name;
    };
    
    return service;
}]);