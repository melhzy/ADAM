/**
 * HUMAnN3 Pathway Abundance Viewer
 * Data Manager Service
 * Responsible for data processing, storage, and manipulation
 */

app.service('DataManager', ['$rootScope', 'EventService', 'PathwayModel', 'SampleModel',
    function($rootScope, EventService, PathwayModel, SampleModel) {
        var service = this;
        
        // Service properties
        service.hasData = false;
        service.rawData = null;
        service.pathways = [];
        service.samples = [];
        service.headerInfo = null;
        service.filteredPathways = [];
        service.selectedPathway = null;
        
        // File information
        service.fileInfo = {
            name: null,
            size: 0,
            type: null,
            lastModified: null
        };
        
        /**
         * Set file information
         * @param {File} file - The uploaded file object
         */
        service.setFileInfo = function(file) {
            service.fileInfo = {
                name: file.name,
                size: file.size,
                type: file.type || 'unknown',
                lastModified: new Date(file.lastModified)
            };
            
            $rootScope.$broadcast('fileInfo:updated', service.fileInfo);
        };
        
        /**
         * Process raw data from CSV/TSV
         * @param {Array} rawData - The parsed data from PapaParse
         * @param {Array} headers - The headers from the CSV/TSV
         * @param {Object} options - Processing options
         */
        service.processData = function(rawData, headers, options) {
            console.time('Data Processing');
            service.rawData = rawData;
            service.headerInfo = headers;
            
            // Reset collections
            service.pathways = [];
            service.samples = [];
            service.filteredPathways = [];
            service.selectedPathway = null;
            
            // Clean up headers and identify non-sample columns
            const nonSampleHeaders = [
                "# Pathway", "Pathway", "0", "#Pathway", "pathway_name", 
                "pathway", "description", "id", "name"
            ];
            
            // Extract sample names (columns except non-sample headers)
            service.samples = headers.filter(header => {
                return header && !nonSampleHeaders.includes(header);
            });
            
            console.log(`Found ${service.samples.length} sample columns`);
            
            // Find pathway name/id columns
            let pathwayIdColumn = null;
            let pathwayNameColumn = null;
            
            // Try to identify pathway ID and name columns
            for (const header of headers) {
                if (header === "# Pathway" || header === "#Pathway" || header === "id") {
                    pathwayIdColumn = header;
                }
                if (header === "Pathway" || header === "pathway_name" || header === "pathway" || header === "name") {
                    pathwayNameColumn = header;
                }
            }
            
            // If not found, use first column as pathway identifier
            if (!pathwayIdColumn && !pathwayNameColumn && headers.length > 0) {
                pathwayNameColumn = headers[0];
            }
            
            console.log(`Using pathway ID column: ${pathwayIdColumn}`);
            console.log(`Using pathway name column: ${pathwayNameColumn}`);
            
            // Process data rows in batches
            const batchSize = 1000;
            let processedCount = 0;
            
            const processBatch = (startIndex, endIndex) => {
                for (let i = startIndex; i < endIndex && i < rawData.length; i++) {
                    const row = rawData[i];
                    
                    // Skip empty rows
                    if (!row || Object.keys(row).length === 0) continue;
                    
                    // Get pathway ID and name
                    let pathwayId = '';
                    let pathwayName = '';
                    
                    if (pathwayIdColumn && row[pathwayIdColumn] !== undefined) {
                        pathwayId = row[pathwayIdColumn].toString();
                    }
                    
                    if (pathwayNameColumn && row[pathwayNameColumn] !== undefined) {
                        pathwayName = row[pathwayNameColumn].toString();
                    }
                    
                    // Use name as ID if no ID column
                    if (!pathwayId && pathwayName) {
                        pathwayId = pathwayName;
                    }
                    
                    // Skip row if we can't identify the pathway
                    if (!pathwayId) {
                        continue;
                    }
                    
                    // Determine pathway type
                    let pathwayType = 'other';
                    const upperName = pathwayName.toUpperCase();
                    
                    if (upperName.includes("UNMAPPED")) {
                        pathwayType = 'unmapped';
                    } else if (upperName.includes("UNINTEGRATED")) {
                        pathwayType = 'unintegrated';
                    } else if (upperName.includes("PWY") || pathwayId.includes("PWY")) {
                        pathwayType = 'metacyc';
                    }
                    
                    // Extract abundance values across samples
                    const abundanceValues = {};
                    
                    service.samples.forEach(sample => {
                        let value = row[sample];
                        // Convert to number
                        if (value === undefined || value === null) {
                            value = 0;
                        } else if (typeof value === 'string') {
                            value = parseFloat(value.replace(/,/g, '')) || 0;
                        } else if (typeof value !== 'number') {
                            value = 0;
                        }
                        abundanceValues[sample] = value;
                    });
                    
                    // Create pathway model
                    const pathway = new PathwayModel({
                        id: pathwayId,
                        name: pathwayName,
                        type: pathwayType,
                        abundanceValues: abundanceValues,
                        index: i
                    });
                    
                    service.pathways.push(pathway);
                    processedCount++;
                }
                
                // If more rows to process, schedule next batch
                if (endIndex < rawData.length) {
                    // Update progress
                    const progress = Math.round((endIndex / rawData.length) * 100);
                    EventService.emit('data:processing', { progress: progress });
                    
                    // Process next batch
                    setTimeout(() => {
                        processBatch(endIndex, endIndex + batchSize);
                    }, 0);
                } else {
                    // All data processed
                    finishProcessing();
                }
            };
            
            // Start batch processing from the beginning
            processBatch(0, batchSize);
            
            // Function to finalize processing
            function finishProcessing() {
                // Apply initial filters (show all pathways)
                service.filteredPathways = [...service.pathways];
                service.hasData = true;
                
                console.timeEnd('Data Processing');
                console.log(`Processed ${service.pathways.length} pathways`);
                
                // Notify application that data is loaded
                EventService.emit('data:loaded', {
                    pathwayCount: service.pathways.length,
                    sampleCount: service.samples.length
                });
            }
        };
        
        /**
         * Get all samples
         * @returns {Array} - List of sample names
         */
        service.getSamples = function() {
            return service.samples;
        };
        
        /**
         * Get all pathways
         * @returns {Array} - List of pathway models
         */
        service.getPathways = function() {
            return service.pathways;
        };
        
        /**
         * Get filtered pathways
         * @returns {Array} - List of filtered pathway models
         */
        service.getFilteredPathways = function() {
            return service.filteredPathways;
        };
        
        /**
         * Apply filters to pathways
         * @param {Object} filters - Filter criteria
         */
        service.applyFilters = function(filters) {
            console.time('Filter Application');
            
            // Default filters
            filters = filters || {};
            filters.searchTerm = filters.searchTerm || '';
            filters.pathwayType = filters.pathwayType || 'all';
            filters.sortField = filters.sortField || 'id';
            filters.selectedSample = filters.selectedSample || '';
            
            // Apply filters
            service.filteredPathways = service.pathways.filter(pathway => {
                // Search term filter
                const searchMatches = !filters.searchTerm || 
                    pathway.name.toLowerCase().includes(filters.searchTerm.toLowerCase()) ||
                    pathway.id.toLowerCase().includes(filters.searchTerm.toLowerCase());
                
                // Pathway type filter
                let typeMatches = true;
                if (filters.pathwayType !== 'all') {
                    if (filters.pathwayType === 'PWY') {
                        typeMatches = pathway.type === 'metacyc';
                    } else {
                        typeMatches = pathway.name.includes(filters.pathwayType);
                    }
                }
                
                return searchMatches && typeMatches;
            });
            
            // Apply sorting
            if (filters.sortField === 'id') {
                service.filteredPathways.sort((a, b) => a.id.localeCompare(b.id));
            } else if (filters.sortField === 'name') {
                service.filteredPathways.sort((a, b) => a.name.localeCompare(b.name));
            } else if (filters.sortField === 'abundance') {
                // Sort by sample-specific abundance or average
                if (filters.selectedSample) {
                    service.filteredPathways.sort((a, b) => 
                        b.abundanceValues[filters.selectedSample] - a.abundanceValues[filters.selectedSample]
                    );
                } else {
                    service.filteredPathways.sort((a, b) => b.avgAbundance - a.avgAbundance);
                }
            }
            
            console.timeEnd('Filter Application');
            
            // Notify application that filters have been applied
            EventService.emit('filters:applied', {
                count: service.filteredPathways.length,
                filters: filters
            });
            
            return service.filteredPathways;
        };
        
        /**
         * Select a pathway
         * @param {String|Object} pathwayIdOrObj - Pathway ID or pathway object
         */
        service.selectPathway = function(pathwayIdOrObj) {
            let pathway = null;
            
            if (typeof pathwayIdOrObj === 'string') {
                // Find by ID
                pathway = service.pathways.find(p => p.id === pathwayIdOrObj);
            } else if (pathwayIdOrObj && pathwayIdOrObj.id) {
                // Use provided pathway object
                pathway = pathwayIdOrObj;
            }
            
            service.selectedPathway = pathway;
            
            if (pathway) {
                // Notify application that a pathway has been selected
                EventService.emit('pathway:selected', pathway);
            }
            
            return service.selectedPathway;
        };
        
        /**
         * Get statistics about the loaded data
         * @returns {Object} - Statistics about the data
         */
        service.getStats = function() {
            const stats = {
                totalPathways: service.pathways.length,
                totalSamples: service.samples.length,
                metacycPathways: service.pathways.filter(p => p.type === 'metacyc').length,
                unmappedPercent: '0.00'
            };
            
            // Calculate percentage of unmapped reads if available
            const unmappedItems = service.pathways.filter(p => p.type === 'unmapped');
            if (unmappedItems.length > 0) {
                // Calculate average percentage across samples
                const totalAbundance = service.samples.reduce((total, sample) => {
                    return total + service.pathways.reduce((sum, pathway) => 
                        sum + pathway.abundanceValues[sample], 0);
                }, 0);
                
                const unmappedAbundance = service.samples.reduce((total, sample) => {
                    return total + unmappedItems.reduce((sum, pathway) => 
                        sum + pathway.abundanceValues[sample], 0);
                }, 0);
                
                if (totalAbundance > 0) {
                    stats.unmappedPercent = (unmappedAbundance / totalAbundance * 100).toFixed(2);
                }
            }
            
            return stats;
        };
        
        /**
         * Reset all data
         */
        service.reset = function() {
            service.hasData = false;
            service.rawData = null;
            service.pathways = [];
            service.samples = [];
            service.headerInfo = null;
            service.filteredPathways = [];
            service.selectedPathway = null;
            service.fileInfo = {
                name: null,
                size: 0,
                type: null,
                lastModified: null
            };
            
            // Notify application that data has been reset
            EventService.emit('data:reset');
        };
        
        /**
         * Find similar pathways
         * @param {Object} pathway - The reference pathway
         * @param {Number} limit - Maximum number of similar pathways to return
         * @returns {Array} - List of similar pathways
         */
        service.findSimilarPathways = function(pathway, limit) {
            if (!pathway || !service.pathways.length) return [];
            
            limit = limit || 5;
            
            // For MetaCyc pathways, find pathways in the same category
            if (pathway.type === 'metacyc') {
                // Extract main category from PWY ID
                const match = pathway.id.match(/^(PWY-\d+)/);
                const mainCategory = match ? match[1] : null;
                
                if (mainCategory) {
                    // Find pathways in the same category
                    const sameCategory = service.pathways.filter(p => 
                        p.id !== pathway.id && p.id.includes(mainCategory)
                    );
                    
                    if (sameCategory.length > 0) {
                        return sameCategory.slice(0, limit);
                    }
                }
            }
            
            // For other pathways, find pathways of the same type
            const sameType = service.pathways.filter(p => 
                p.id !== pathway.id && p.type === pathway.type
            );
            
            if (sameType.length > 0) {
                return sameType.slice(0, limit);
            }
            
            // Return some other pathways if no similar ones found
            return service.pathways.filter(p => p.id !== pathway.id).slice(0, limit);
        };
        
        /**
         * Get top samples for a pathway
         * @param {Object} pathway - The pathway
         * @param {Number} limit - Maximum number of samples to return
         * @returns {Array} - List of sample objects with name, abundance, and relative percentage
         */
        service.getTopSamples = function(pathway, limit) {
            if (!pathway || !service.samples.length) return [];
            
            limit = limit || 20;
            
            // Create sample objects with name and abundance
            const sampleObjects = service.samples.map(sampleName => {
                return {
                    name: sampleName,
                    abundance: pathway.abundanceValues[sampleName] || 0
                };
            });
            
            // Sort by abundance (descending)
            sampleObjects.sort((a, b) => b.abundance - a.abundance);
            
            // Calculate relative percentage based on highest abundance
            const maxAbundance = sampleObjects[0].abundance;
            sampleObjects.forEach(sample => {
                sample.relativePercent = maxAbundance > 0 ? 
                    (sample.abundance / maxAbundance * 100).toFixed(2) : 0;
            });
            
            // Return top N samples
            return sampleObjects.slice(0, limit);
        };
        
        /**
         * Export pathway data to CSV
         * @param {Object} pathway - The pathway to export
         * @returns {String} - CSV content
         */
        service.exportPathwayData = function(pathway) {
            if (!pathway) return null;
            
            let csvContent = "Sample,Abundance\n";
            
            // Get samples sorted by abundance
            const topSamples = service.getTopSamples(pathway, service.samples.length);
            
            // Add data rows
            topSamples.forEach(sample => {
                csvContent += `${sample.name},${sample.abundance}\n`;
            });
            
            return csvContent;
        };
    }
]);