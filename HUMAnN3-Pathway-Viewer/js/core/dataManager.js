/**
 * HUMAnN3 Pathway Abundance Viewer
 * Data Manager Service
 * Central service for data handling and manipulation
 */

app.service('DataManager', ['$q', 'EventService', 'CSVParser', 'FormattersService', 'PathwayModel',
    function($q, EventService, CSVParser, FormattersService, PathwayModel) {
        var service = this;
        
        // Data storage
        service.pathways = [];
        service.samples = [];
        service.hasData = false;
        service.selectedPathway = null;
        service.dataStats = {
            totalPathways: 0,
            totalSamples: 0,
            metacycPathways: 0,
            unmappedPercent: 0,
            maxAbundance: 0,
            avgAbundance: 0
        };
        
        // Cache for filtered pathways
        service.filterCache = null;
        
        // Keep track of file metadata
        var fileInfo = {
            name: null,
            size: 0,
            lastModified: null
        };
        
        /**
         * Set file info
         * @param {File} file - File object
         */
        service.setFileInfo = function(file) {
            fileInfo = {
                name: file.name,
                size: file.size,
                lastModified: new Date(file.lastModified)
            };
            
            // Broadcast file info updated event
            EventService.emit('fileInfo:updated', fileInfo);
        };
        
        /**
         * Process parsed data
         * @param {Array} rawData - Parsed CSV/TSV data
         * @param {Array} headers - Column headers
         */
        service.processData = function(rawData, headers) {
            // Validate input
            if (!rawData || !rawData.length || !headers || !headers.length) {
                throw new Error('Invalid data format. Missing data or headers.');
            }
            
            try {
                console.log('Processing data, rows:', rawData.length, 'columns:', headers.length);
                
                // First column should be pathway IDs
                const pathwayIdColumn = headers[0];
                
                // For HUMAnN3 pathabundance files, the first column is often named '#SampleID'
                // or something similar. Let's handle variations.
                let sampleColumns = [];
                
                // Check if there's a 'NAME' column which indicates a specific format
                const hasNameColumn = headers.includes('NAME');
                
                if (hasNameColumn) {
                    // Format where first column is ID and there's a separate NAME column
                    sampleColumns = headers.filter(h => 
                        h !== pathwayIdColumn && 
                        h !== 'NAME' && 
                        !h.startsWith('# ') &&
                        !h.startsWith('#'));
                } else {
                    // Standard format where all non-first columns are samples
                    // Filter out any metadata columns (commonly start with # in HUMAnN3 files)
                    sampleColumns = headers.slice(1).filter(h => 
                        !h.startsWith('# ') && 
                        !h.startsWith('#') &&
                        h !== 'NAME');
                }
                
                console.log('Found', sampleColumns.length, 'samples');
                
                // Keep track of samples
                service.samples = sampleColumns;
                
                // Process data into pathway objects
                const pathways = [];
                let processedRows = 0;
                
                // Update progress
                EventService.emit('data:processing', { progress: 0, total: rawData.length });
                
                // Process each row (one pathway per row)
                rawData.forEach((row, index) => {
                    // Get pathway ID and type
                    let pathwayId = row[pathwayIdColumn];
                    
                    // Skip rows with empty pathway IDs
                    if (!pathwayId || pathwayId.trim() === '') {
                        return;
                    }
                    
                    // For HUMAnN3 files, sometimes the pathway name is included with the ID
                    // or in a separate NAME column
                    let pathwayName = '';
                    
                    if (hasNameColumn && row['NAME']) {
                        pathwayName = row['NAME'];
                    } else {
                        // If no NAME column, use the ID as the name
                        pathwayName = pathwayId;
                    }
                    
                    // Detect pathway type based on ID patterns
                    let pathwayType = 'other';
                    
                    if (pathwayId.includes('UNMAPPED')) {
                        pathwayType = 'unmapped';
                    } else if (pathwayId.includes('UNINTEGRATED')) {
                        pathwayType = 'unintegrated';
                    } else if (pathwayId.includes('PWY') || pathwayId.includes('MetaCyc')) {
                        pathwayType = 'metacyc';
                    }
                    
                    // Create abundance map for this pathway
                    const abundanceValues = {};
                    
                    sampleColumns.forEach(sample => {
                        // Convert to number and handle missing values
                        let value = row[sample];
                        if (value === undefined || value === null || value === '') {
                            value = 0;
                        } else if (typeof value === 'string') {
                            value = parseFloat(value) || 0;
                        }
                        
                        abundanceValues[sample] = value;
                    });
                    
                    // Create pathway object
                    const pathway = new PathwayModel({
                        id: pathwayId,
                        name: pathwayName,
                        type: pathwayType,
                        abundanceValues: abundanceValues,
                        index: index
                    });
                    
                    pathways.push(pathway);
                    processedRows++;
                    
                    // Update progress periodically
                    if (index % 100 === 0 || index === rawData.length - 1) {
                        const progress = Math.round((index / rawData.length) * 100);
                        EventService.emit('data:processing', { progress: progress, total: rawData.length });
                    }
                });
                
                // Calculate statistics
                let metacycCount = 0;
                let unmappedAbundance = 0;
                let totalAbundance = 0;
                let maxAbundance = 0;
                
                pathways.forEach(pathway => {
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
                });
                
                // Calculate unmapped percentage
                const unmappedPercent = totalAbundance > 0 ? 
                    (unmappedAbundance / totalAbundance * 100).toFixed(1) : 0;
                
                // Update service state
                service.pathways = pathways;
                service.hasData = true;
                service.dataStats = {
                    totalPathways: pathways.length,
                    totalSamples: sampleColumns.length,
                    metacycPathways: metacycCount,
                    unmappedPercent: unmappedPercent,
                    maxAbundance: maxAbundance,
                    avgAbundance: totalAbundance / (pathways.length || 1)
                };
                
                // Clear filter cache
                service.filterCache = null;
                
                // Broadcast data loaded event
                EventService.emit('data:loaded', {
                    pathways: service.pathways,
                    samples: service.samples,
                    stats: service.dataStats
                });
                
                console.log('Data processing complete:', service.dataStats);
                
            } catch (error) {
                console.error('Error processing data:', error);
                EventService.emit('data:error', { error: error.message });
                throw error;
            }
        };
        
        /**
         * Apply filters to pathways
         * @param {Object} filters - Filter criteria
         */
        service.applyFilters = function(filters) {
            // Start with all pathways
            let filtered = service.pathways.slice();
            
            // Apply search term filter
            if (filters.searchTerm) {
                const searchTerm = filters.searchTerm.toLowerCase();
                filtered = filtered.filter(function(pathway) {
                    return pathway.id.toLowerCase().includes(searchTerm) || 
                           pathway.name.toLowerCase().includes(searchTerm);
                });
            }
            
            // Apply pathway type filter
            if (filters.pathwayType && filters.pathwayType !== 'all') {
                const typeFilter = filters.pathwayType.toLowerCase();
                filtered = filtered.filter(function(pathway) {
                    if (typeFilter === 'metacyc' || typeFilter === 'pwy') {
                        return pathway.type === 'metacyc';
                    }
                    return pathway.type === typeFilter;
                });
            }
            
            // Apply sort
            if (filters.sortField) {
                filtered.sort(function(a, b) {
                    if (filters.sortField === 'id') {
                        return a.id.localeCompare(b.id);
                    } else if (filters.sortField === 'name') {
                        return a.name.localeCompare(b.name);
                    } else if (filters.sortField === 'abundance') {
                        // If a specific sample is selected, sort by that sample's abundance
                        if (filters.selectedSample) {
                            return b.getAbundanceForSample(filters.selectedSample) - 
                                   a.getAbundanceForSample(filters.selectedSample);
                        }
                        // Otherwise sort by average abundance
                        return b.avgAbundance - a.avgAbundance;
                    }
                    return 0;
                });
            }
            
            // Store filtered result in cache
            service.filterCache = filtered;
            
            // Broadcast filters applied event
            EventService.emit('filters:applied', {
                filters: filters,
                pathways: filtered
            });
            
            return filtered;
        };
        
        /**
         * Get filtered pathways
         * @returns {Array} - Filtered pathway objects
         */
        service.getFilteredPathways = function() {
            console.log('getFilteredPathways called, pathways count:', service.pathways.length);
            // If we have a filterCache, use it, otherwise return all pathways
            const result = service.filterCache || service.pathways;
            console.log('Returning filtered pathways, count:', result.length);
            return result;
        };

        /**
         * Get all samples
         * @returns {Array} - Sample names
         */
        service.getSamples = function() {
            return service.samples;
        };
        
        /**
         * Get statistics
         * @returns {Object} - Stats object
         */
        service.getStats = function() {
            return service.dataStats;
        };
        
        /**
         * Select a pathway
         * @param {Object} pathway - Pathway to select
         */
        service.selectPathway = function(pathway) {
            service.selectedPathway = pathway;
            EventService.emit('pathway:selected', pathway);
        };
        
        /**
         * Get top samples for a pathway
         * @param {Object} pathway - Pathway object
         * @param {Number} limit - Maximum samples to return
         * @returns {Array} - Array of sample objects
         */
        service.getTopSamples = function(pathway, limit) {
            if (!pathway) return [];
            
            limit = limit || 10;
            
            // Get samples with abundance
            const samples = Object.keys(pathway.abundanceValues)
                .map(sample => ({
                    name: sample,
                    abundance: pathway.abundanceValues[sample]
                }))
                .sort((a, b) => b.abundance - a.abundance)
                .slice(0, limit);
            
            // Calculate relative percentage
            if (samples.length > 0) {
                const maxAbundance = samples[0].abundance;
                samples.forEach(sample => {
                    sample.relativePercent = (sample.abundance / maxAbundance) * 100;
                });
            }
            
            return samples;
        };
        
        /**
         * Find similar pathways based on abundance patterns
         * @param {Object} pathway - Reference pathway
         * @param {Number} limit - Maximum pathways to return
         * @returns {Array} - Array of similar pathways
         */
        service.findSimilarPathways = function(pathway, limit) {
            if (!pathway) return [];
            
            limit = limit || 10;
            
            // Create a copy of pathways that excludes the reference pathway
            const otherPathways = service.pathways.filter(p => p.id !== pathway.id);
            
            // Calculate similarity scores (using correlation of abundance patterns)
            const pathwaysWithScores = otherPathways.map(p => {
                const score = calculateSimilarity(pathway, p);
                return {
                    pathway: p,
                    score: score
                };
            });
            
            // Sort by similarity score (descending)
            pathwaysWithScores.sort((a, b) => b.score - a.score);
            
            // Return top matches
            return pathwaysWithScores.slice(0, limit).map(item => item.pathway);
        };
        
        /**
         * Calculate similarity between two pathways
         * @param {Object} pathway1 - First pathway
         * @param {Object} pathway2 - Second pathway
         * @returns {Number} - Similarity score (0-1)
         */
        function calculateSimilarity(pathway1, pathway2) {
            // Use all samples that exist in both pathways
            const commonSamples = service.samples.filter(sample => 
                pathway1.abundanceValues[sample] !== undefined && 
                pathway2.abundanceValues[sample] !== undefined
            );
            
            if (commonSamples.length === 0) return 0;
            
            // Use simple dot product similarity for quick calculation
            let dotProduct = 0;
            let magnitude1 = 0;
            let magnitude2 = 0;
            
            commonSamples.forEach(sample => {
                const value1 = pathway1.abundanceValues[sample];
                const value2 = pathway2.abundanceValues[sample];
                
                dotProduct += value1 * value2;
                magnitude1 += value1 * value1;
                magnitude2 += value2 * value2;
            });
            
            // Prevent division by zero
            if (magnitude1 === 0 || magnitude2 === 0) return 0;
            
            // Return cosine similarity
            return dotProduct / (Math.sqrt(magnitude1) * Math.sqrt(magnitude2));
        }
        
        /**
         * Export pathway data to CSV
         * @param {Object} pathway - Pathway to export
         * @returns {String} - CSV content
         */
        service.exportPathwayData = function(pathway) {
            if (!pathway) return null;
            
            // Create CSV content
            let csvContent = 'Sample,Abundance\n';
            
            // Add sample data
            service.samples.forEach(sample => {
                const abundance = pathway.abundanceValues[sample] || 0;
                csvContent += `${sample},${abundance}\n`;
            });
            
            return csvContent;
        };
        
        /**
         * Reset data
         */
        service.resetData = function() {
            service.pathways = [];
            service.samples = [];
            service.hasData = false;
            service.selectedPathway = null;
            service.filterCache = null;
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
            
            // Broadcast reset event
            EventService.emit('data:reset');
        };
        
        return service;
    }]);