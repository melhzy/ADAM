/**
 * HUMAnN3 Pathway Abundance Viewer
 * Optimization Utilities
 * Performance optimizations for handling large datasets
 */

app.service('OptimizationService', ['$window', '$timeout', function($window, $timeout) {
    var service = this;
    
    /**
     * Memory management helper functions
     */
    service.memory = {
        /**
         * Trigger garbage collection (hint)
         * Note: This doesn't directly trigger GC, but hints that it could be a good time
         */
        triggerCleanup: function() {
            // Clear any large objects no longer needed
            if (window.gc) window.gc(); // Only works in debug mode
            
            // Force browser to consider releasing memory
            const memoryHog = [];
            try {
                // Fill and clear an array to hint the garbage collector
                for (let i = 0; i < 10000; i++) {
                    memoryHog.push({});
                }
            } catch (e) {
                // Ignore any errors, this is just a GC hint
            }
            
            // Clear the reference
            while (memoryHog.length > 0) {
                memoryHog.pop();
            }
        },
        
        /**
         * Break large arrays into chunks for processing
         * @param {Array} array - Array to chunk
         * @param {Number} chunkSize - Size of each chunk
         * @returns {Array} - Array of chunks
         */
        chunkArray: function(array, chunkSize) {
            if (!array || !array.length) return [];
            
            const chunks = [];
            chunkSize = chunkSize || 1000;
            
            for (let i = 0; i < array.length; i += chunkSize) {
                chunks.push(array.slice(i, i + chunkSize));
            }
            
            return chunks;
        },
        
        /**
         * Release memory from objects
         * @param {Object} obj - Object to clean up
         */
        releaseObject: function(obj) {
            if (!obj) return;
            
            // Clear all properties
            for (const prop in obj) {
                if (obj.hasOwnProperty(prop)) {
                    obj[prop] = null;
                }
            }
        }
    };
    
    /**
     * Worker management for background processing
     */
    service.workers = {
        pool: {},
        
        /**
         * Create a web worker for a task
         * @param {String} taskName - Identifier for the worker
         * @param {Function} workerFunc - Function to execute in worker
         * @returns {Worker} - Web worker instance
         */
        create: function(taskName, workerFunc) {
            // Convert function to string
            const funcStr = workerFunc.toString();
            
            // Create worker URL
            const blob = new Blob([
                `self.onmessage = function(e) {
                    const workerFunc = ${funcStr};
                    const result = workerFunc(e.data);
                    self.postMessage(result);
                }`
            ], { type: 'application/javascript' });
            
            const url = URL.createObjectURL(blob);
            
            // Create worker
            const worker = new Worker(url);
            
            // Store worker reference
            this.pool[taskName] = {
                worker: worker,
                url: url
            };
            
            return worker;
        },
        
        /**
         * Execute a task with a worker
         * @param {String} taskName - Task identifier
         * @param {Function} workerFunc - Function to execute in worker
         * @param {Object} data - Data to pass to worker
         * @returns {Promise} - Promise resolving with worker result
         */
        execute: function(taskName, workerFunc, data) {
            return new Promise((resolve, reject) => {
                // Create or reuse worker
                let worker;
                
                if (this.pool[taskName]) {
                    worker = this.pool[taskName].worker;
                } else {
                    worker = this.create(taskName, workerFunc);
                }
                
                // Set up handlers
                worker.onmessage = function(e) {
                    resolve(e.data);
                };
                
                worker.onerror = function(error) {
                    reject(error);
                };
                
                // Start worker
                worker.postMessage(data);
            });
        },
        
        /**
         * Terminate a worker
         * @param {String} taskName - Task identifier
         */
        terminate: function(taskName) {
            if (this.pool[taskName]) {
                this.pool[taskName].worker.terminate();
                URL.revokeObjectURL(this.pool[taskName].url);
                delete this.pool[taskName];
            }
        },
        
        /**
         * Terminate all workers
         */
        terminateAll: function() {
            for (const taskName in this.pool) {
                this.terminate(taskName);
            }
        }
    };
    
    /**
     * Browser performance optimizations
     */
    service.performance = {
        /**
         * Check if browser is likely to handle large datasets well
         * @returns {Boolean} - True if browser is high-performance
         */
        isHighPerformanceBrowser: function() {
            // Check for modern browser features that indicate good performance
            return !!(
                window.requestIdleCallback &&
                window.requestAnimationFrame &&
                window.performance &&
                window.Worker &&
                navigator.hardwareConcurrency > 1
            );
        },
        
        /**
         * Get recommended chunk size based on device capabilities
         * @returns {Number} - Recommended chunk size
         */
        getRecommendedChunkSize: function() {
            if (!navigator.hardwareConcurrency) return 1000;
            
            // Base chunk size on number of cores
            const cores = navigator.hardwareConcurrency;
            
            if (cores >= 8) return 5000;
            if (cores >= 4) return 2500;
            if (cores >= 2) return 1000;
            
            return 500;
        },
        
        /**
         * Run a task during browser idle time
         * @param {Function} callback - Function to execute
         * @param {Object} options - requestIdleCallback options
         */
        runWhenIdle: function(callback, options) {
            if (window.requestIdleCallback) {
                window.requestIdleCallback(callback, options);
            } else {
                // Fallback for browsers without requestIdleCallback
                setTimeout(callback, 1);
            }
        },
        
        /**
         * Batch DOM updates for better performance
         * @param {Function} updateFunc - DOM update function
         */
        batchDomUpdates: function(updateFunc) {
            if (window.requestAnimationFrame) {
                window.requestAnimationFrame(updateFunc);
            } else {
                updateFunc();
            }
        },
        
        /**
         * Measure execution time of a function
         * @param {String} label - Label for the measurement
         * @param {Function} func - Function to measure
         * @param {Array} args - Arguments to pass to function
         * @returns {*} - Function result
         */
        measure: function(label, func, args) {
            console.time(label);
            const result = func.apply(null, args || []);
            console.timeEnd(label);
            return result;
        },
        
        /**
         * Debounce a function call
         * @param {Function} func - Function to debounce
         * @param {Number} wait - Wait time in milliseconds
         * @returns {Function} - Debounced function
         */
        debounce: function(func, wait) {
            let timeout;
            return function() {
                const context = this;
                const args = arguments;
                
                const later = function() {
                    timeout = null;
                    func.apply(context, args);
                };
                
                clearTimeout(timeout);
                timeout = setTimeout(later, wait);
            };
        },
        
        /**
         * Throttle a function call
         * @param {Function} func - Function to throttle
         * @param {Number} limit - Limit in milliseconds
         * @returns {Function} - Throttled function
         */
        throttle: function(func, limit) {
            let lastCall = 0;
            return function() {
                const now = Date.now();
                if (now - lastCall >= limit) {
                    lastCall = now;
                    return func.apply(this, arguments);
                }
            };
        }
    };
    
    /**
     * Data processing optimizations
     */
    service.dataProcessing = {
        /**
         * Process large data arrays in chunks with progress tracking
         * @param {Array} data - Data array to process
         * @param {Function} processFunc - Processing function for each item
         * @param {Function} progressCallback - Progress callback
         * @param {Number} chunkSize - Size of chunks to process
         * @returns {Promise} - Promise that resolves when processing is complete
         */
        processInChunks: function(data, processFunc, progressCallback, chunkSize) {
            return new Promise((resolve, reject) => {
                if (!data || !data.length) {
                    resolve([]);
                    return;
                }
                
                chunkSize = chunkSize || service.performance.getRecommendedChunkSize();
                const chunks = service.memory.chunkArray(data, chunkSize);
                const result = [];
                let processed = 0;
                
                function processNextChunk(index) {
                    if (index >= chunks.length) {
                        resolve(result);
                        return;
                    }
                    
                    // Process current chunk
                    try {
                        const chunkResult = chunks[index].map(processFunc);
                        result.push(...chunkResult);
                        
                        // Update progress
                        processed += chunks[index].length;
                        const progress = Math.round((processed / data.length) * 100);
                        
                        if (progressCallback) {
                            progressCallback({
                                processed: processed,
                                total: data.length,
                                progress: progress
                            });
                        }
                        
                        // Schedule next chunk with delay to allow UI updates
                        $timeout(() => {
                            processNextChunk(index + 1);
                        }, 0);
                    } catch (error) {
                        reject(error);
                    }
                }
                
                // Start processing the first chunk
                processNextChunk(0);
            });
        },
        
        /**
         * Index data for faster searching and filtering
         * @param {Array} data - Data array to index
         * @param {Array} fields - Fields to create indices for
         * @returns {Object} - Indexing result
         */
        createIndices: function(data, fields) {
            if (!data || !data.length || !fields || !fields.length) {
                return {};
            }
            
            const indices = {};
            
            fields.forEach(field => {
                indices[field] = {};
                
                // Create index for field
                data.forEach((item, index) => {
                    const value = item[field];
                    
                    if (value !== undefined && value !== null) {
                        if (!indices[field][value]) {
                            indices[field][value] = [];
                        }
                        
                        indices[field][value].push(index);
                    }
                });
            });
            
            return {
                indices: indices,
                
                // Find items by field value
                find: function(field, value) {
                    if (!this.indices[field] || !this.indices[field][value]) {
                        return [];
                    }
                    
                    return this.indices[field][value].map(index => data[index]);
                },
                
                // Check if value exists in field
                exists: function(field, value) {
                    return !!(this.indices[field] && this.indices[field][value]);
                },
                
                // Get unique values for a field
                getUniqueValues: function(field) {
                    if (!this.indices[field]) {
                        return [];
                    }
                    
                    return Object.keys(this.indices[field]);
                }
            };
        },
        
        /**
         * Lazy load data when needed
         * @param {Array} data - Complete data array
         * @param {Number} pageSize - Items per page
         * @returns {Object} - Pagination controller
         */
        createLazyLoader: function(data, pageSize) {
            if (!data) {
                data = [];
            }
            
            pageSize = pageSize || 50;
            
            return {
                data: data,
                pageSize: pageSize,
                currentPage: 1,
                totalPages: Math.ceil(data.length / pageSize),
                
                // Get current page of data
                getCurrentPage: function() {
                    const start = (this.currentPage - 1) * this.pageSize;
                    const end = start + this.pageSize;
                    return this.data.slice(start, end);
                },
                
                // Go to specific page
                goToPage: function(page) {
                    if (page < 1 || page > this.totalPages) {
                        return false;
                    }
                    
                    this.currentPage = page;
                    return true;
                },
                
                // Go to next page
                nextPage: function() {
                    return this.goToPage(this.currentPage + 1);
                },
                
                // Go to previous page
                prevPage: function() {
                    return this.goToPage(this.currentPage - 1);
                }
            };
        }
    };
    
    return service;
}]);