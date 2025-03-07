/**
 * HUMAnN3 Pathway Abundance Viewer
 * CSV Parser Utility
 * Enhanced CSV/TSV parsing with optimizations for large files
 */

app.factory('CSVParser', ['$q', function($q) {
    var service = {};
    
    /**
     * Parse CSV/TSV data with optimizations for large files
     * @param {String|File} input - CSV/TSV string or File object
     * @param {Object} options - Parser options
     * @returns {Promise} - Promise resolving to parsed data
     */
    service.parse = function(input, options) {
        var deferred = $q.defer();
        var defaultOptions = {
            delimiter: null,          // Auto-detect
            header: true,             // First row is header
            dynamicTyping: true,      // Convert numbers and booleans
            skipEmptyLines: true,     // Skip empty lines
            comments: '#',            // Treat lines starting with # as comments
            chunk: null,              // Process in chunks (for large files)
            preview: 0,               // Number of rows to preview (0 = all)
            isTabDelimited: false,    // Whether input is tab-delimited
            maxRows: 0,               // Maximum rows to process (0 = all)
            onProgress: null          // Progress callback function
        };
        
        // Merge options
        options = Object.assign({}, defaultOptions, options);
        
        // Determine how to process the input
        if (input instanceof File) {
            // Process file
            _parseFile(input, options, deferred);
        } else if (typeof input === 'string') {
            // Process string
            _parseString(input, options, deferred);
        } else {
            deferred.reject(new Error('Invalid input type. Expected File or String.'));
        }
        
        return deferred.promise;
    };
    
    /**
     * Parse CSV/TSV from file
     * @param {File} file - CSV/TSV file
     * @param {Object} options - Parser options
     * @param {Object} deferred - Promise deferred object
     * @private
     */
    function _parseFile(file, options, deferred) {
        console.time('File Parsing');
        const fileSize = file.size;
        const fileName = file.name;
        
        // Detect file type from extension
        const fileExtension = fileName.split('.').pop().toLowerCase();
        const isTabDelimited = options.isTabDelimited || 
                              fileExtension === 'tsv' || 
                              fileExtension === 'txt';
        
        options.isTabDelimited = isTabDelimited;
        
        // For large files, use streaming approach with chunks
        if (fileSize > 10 * 1024 * 1024) { // > 10MB
            console.log(`Large file detected (${(fileSize / (1024 * 1024)).toFixed(2)} MB). Using chunked parsing.`);
            
            // Set up chunk processing
            if (!options.chunk) {
                options.chunk = {
                    size: 1024 * 1024, // 1MB chunks
                    processed: 0
                };
            }
            
            // Configure Papa Parse for streaming
            const papaConfig = {
                delimiter: options.isTabDelimited ? '\t' : options.delimiter,
                header: options.header,
                dynamicTyping: options.dynamicTyping,
                skipEmptyLines: options.skipEmptyLines,
                comments: options.comments,
                delimitersToGuess: [',', '\t', '|', ';'],
                chunk: function(results, parser) {
                    // Track progress
                    options.chunk.processed += results.data.length;
                    const progress = Math.min(100, Math.round((parser.streamer._input.length / fileSize) * 100));
                    
                    if (options.onProgress) {
                        options.onProgress({
                            progress: progress,
                            processed: options.chunk.processed,
                            total: fileSize
                        });
                    }
                    
                    // Check if we've reached the maximum rows
                    if (options.maxRows > 0 && options.chunk.processed >= options.maxRows) {
                        parser.abort();
                        
                        // Return processed data so far
                        deferred.resolve({
                            data: results.data,
                            meta: results.meta,
                            truncated: true,
                            totalRows: options.chunk.processed
                        });
                    }
                },
                complete: function(results) {
                    console.timeEnd('File Parsing');
                    console.log(`Completed parsing file with ${results.data.length} rows and ${results.meta.fields ? results.meta.fields.length : 0} columns.`);
                    
                    // Resolve with results
                    deferred.resolve({
                        data: results.data,
                        meta: results.meta,
                        truncated: false,
                        totalRows: options.chunk.processed
                    });
                },
                error: function(error) {
                    console.error('Error parsing file:', error);
                    deferred.reject(error);
                }
            };
            
            // Start streaming parse
            Papa.parse(file, papaConfig);
        } else {
            // For smaller files, parse all at once
            const reader = new FileReader();
            
            reader.onload = function(event) {
                const fileData = event.target.result;
                _parseString(fileData, options, deferred);
            };
            
            reader.onerror = function(error) {
                console.error('Error reading file:', error);
                deferred.reject(error);
            };
            
            reader.readAsText(file);
        }
    }
    
    /**
     * Parse CSV/TSV from string
     * @param {String} data - CSV/TSV string
     * @param {Object} options - Parser options
     * @param {Object} deferred - Promise deferred object
     * @private
     */
    function _parseString(data, options, deferred) {
        // Check for BOM and remove if present
        if (data.charCodeAt(0) === 0xFEFF) {
            data = data.slice(1);
        }
        
        // For tab-delimited files, try manual parsing for better performance
        if (options.isTabDelimited && data.length < 5 * 1024 * 1024) { // < 5MB
            try {
                const result = _parseTabDelimited(data, options);
                deferred.resolve(result);
                return;
            } catch (error) {
                console.warn('Manual TSV parsing failed, falling back to PapaParse:', error);
                // Fall back to PapaParse if manual parsing fails
            }
        }
        
        // Configure Papa Parse
        const papaConfig = {
            delimiter: options.isTabDelimited ? '\t' : options.delimiter,
            header: options.header,
            dynamicTyping: options.dynamicTyping,
            skipEmptyLines: options.skipEmptyLines,
            comments: options.comments,
            delimitersToGuess: [',', '\t', '|', ';'],
            preview: options.preview > 0 ? options.preview : undefined,
            complete: function(results) {
                if (results.errors && results.errors.length > 0) {
                    console.warn('Parsing warnings:', results.errors);
                }
                
                deferred.resolve({
                    data: results.data,
                    meta: results.meta,
                    truncated: options.preview > 0,
                    totalRows: results.data.length
                });
            },
            error: function(error) {
                console.error('Error parsing data:', error);
                deferred.reject(error);
            }
        };
        
        // Parse the data
        Papa.parse(data, papaConfig);
    }
    
    /**
     * Parse tab-delimited data manually for better performance
     * @param {String} data - Tab-delimited string
     * @param {Object} options - Parser options
     * @returns {Object} - Parsed data result
     * @private
     */
    function _parseTabDelimited(data, options) {
        console.time('Manual TSV Parsing');
        
        // Split into lines
        const lines = data.split(/\r?\n/);
        let headerLine = null;
        let headers = [];
        const parsedData = [];
        
        // Find header line (skip comments)
        for (let i = 0; i < Math.min(20, lines.length); i++) {
            if (lines[i] && !lines[i].trim().startsWith('#')) {
                headerLine = lines[i];
                break;
            }
        }
        
        if (!headerLine) {
            throw new Error('Could not find header row');
        }
        
        // Process headers
        headers = headerLine.split('\t').map(h => h.trim());
        
        // Process data rows
        const startRow = options.header ? 1 : 0;
        const maxRows = options.maxRows > 0 ? 
                       Math.min(lines.length, startRow + options.maxRows) : 
                       lines.length;
        
        let rowsProcessed = 0;
        
        for (let i = startRow; i < maxRows; i++) {
            const line = lines[i];
            if (!line || line.trim() === '' || (options.comments && line.trim().startsWith(options.comments))) {
                continue;
            }
            
            const values = line.split('\t');
            const row = {};
            
            if (options.header) {
                // Map values to headers
                headers.forEach((header, j) => {
                    if (values[j] !== undefined) {
                        let value = values[j].trim();
                        
                        // Apply dynamic typing
                        if (options.dynamicTyping && value !== '') {
                            if (!isNaN(value)) {
                                value = parseFloat(value);
                            } else if (value.toLowerCase() === 'true') {
                                value = true;
                            } else if (value.toLowerCase() === 'false') {
                                value = false;
                            }
                        }
                        
                        row[header] = value;
                    } else {
                        row[header] = null;
                    }
                });
            } else {
                // Use array for row
                values.forEach((value, j) => {
                    let processedValue = value.trim();
                    
                    // Apply dynamic typing
                    if (options.dynamicTyping && processedValue !== '') {
                        if (!isNaN(processedValue)) {
                            processedValue = parseFloat(processedValue);
                        } else if (processedValue.toLowerCase() === 'true') {
                            processedValue = true;
                        } else if (processedValue.toLowerCase() === 'false') {
                            processedValue = false;
                        }
                    }
                    
                    row[j] = processedValue;
                });
            }
            
            parsedData.push(row);
            rowsProcessed++;
            
            // Call progress callback
            if (options.onProgress && rowsProcessed % 1000 === 0) {
                options.onProgress({
                    progress: Math.round((i / maxRows) * 100),
                    processed: rowsProcessed,
                    total: maxRows - startRow
                });
            }
        }
        
        console.timeEnd('Manual TSV Parsing');
        
        return {
            data: parsedData,
            meta: {
                fields: options.header ? headers : null,
                delimiter: '\t'
            },
            truncated: maxRows < lines.length,
            totalRows: rowsProcessed
        };
    }
    
    return service;
}]);