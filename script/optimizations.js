// Memory and loading optimizations for HUMAnN3 Pathway Abundance Viewer
// Add this to a new file 'script/optimizations.js' and include it in your HTML

// Global memory management
window.memoryManagement = {
    // Garbage collection helper - call at strategic points
    triggerCleanup: function() {
        // Clear any large objects that are no longer needed
        if (window.gc) window.gc(); // Only works in some debug environments
        
        // Force browser to consider releasing memory
        const memoryHog = [];
        try {
            // Fill and clear an array to hint the garbage collector
            for (let i = 0; i < 1000000; i++) {
                memoryHog.push({});
            }
        } catch (e) {
            // Ignore any errors, this is just a GC hint
        }
        // Clear the reference
        memoryHog.length = 0;
    },
    
    // Break large arrays into chunks
    chunkArray: function(array, chunkSize) {
        const chunks = [];
        for (let i = 0; i < array.length; i += chunkSize) {
            chunks.push(array.slice(i, i + chunkSize));
        }
        return chunks;
    },
    
    // Release memory from large objects
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

// Lazy-loading manager
window.lazyLoader = {
    loadedResources: {},
    
    // Load Chart.js only when needed
    loadChartJS: function() {
        if (this.loadedResources.chartjs) return Promise.resolve();
        
        return new Promise((resolve, reject) => {
            console.log("Dynamically loading Chart.js");
            const script = document.createElement('script');
            script.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/3.9.1/chart.min.js";
            script.onload = () => {
                console.log("Chart.js loaded successfully");
                this.loadedResources.chartjs = true;
                resolve();
            };
            script.onerror = (error) => {
                console.error("Failed to load Chart.js:", error);
                reject(error);
            };
            document.head.appendChild(script);
        });
    },
    
    // Pre-initialize Chart.js with optimized defaults
    initChartJS: async function() {
        await this.loadChartJS();
        
        // Apply global Chart.js optimizations
        if (window.Chart && window.Chart.defaults) {
            // Reduce animation duration for better performance
            Chart.defaults.animation = {
                duration: 500,
                easing: 'easeOutQuart'
            };
            
            // Optimize tooltips
            Chart.defaults.plugins.tooltip = {
                ...Chart.defaults.plugins.tooltip,
                enabled: true,
                mode: 'nearest',
                intersect: false,
                animation: {
                    duration: 100
                }
            };
            
            // Optimize rendering
            Chart.defaults.responsive = true;
            Chart.defaults.maintainAspectRatio = false;
            
            console.log("Applied Chart.js optimizations");
        }
    }
};

// Virtual rendering for pathway list (improves performance with large datasets)
class VirtualList {
    constructor(container, itemHeight = 53) {
        this.container = typeof container === 'string' ? document.getElementById(container) : container;
        this.itemHeight = itemHeight;
        this.items = [];
        this.visibleItems = [];
        this.scrollTop = 0;
        this.viewportHeight = 0;
        this.totalHeight = 0;
        this.renderedRange = { start: 0, end: 0 };
        
        // Create necessary DOM structure
        this.createDOMStructure();
        
        // Bind methods
        this.onScroll = this.onScroll.bind(this);
        this.render = this.render.bind(this);
        
        // Add event listeners
        this.container.addEventListener('scroll', this.onScroll);
        window.addEventListener('resize', this.render);
    }
    
    createDOMStructure() {
        // Create a content container with the full height
        this.content = document.createElement('div');
        this.content.className = 'virtual-list-content';
        
        // Create a viewport for visible items
        this.viewport = document.createElement('div');
        this.viewport.className = 'virtual-list-viewport';
        
        // Add to DOM
        this.content.appendChild(this.viewport);
        this.container.appendChild(this.content);
    }
    
    setItems(items) {
        this.items = items;
        this.totalHeight = items.length * this.itemHeight;
        this.content.style.height = `${this.totalHeight}px`;
        this.render();
    }
    
    onScroll() {
        this.scrollTop = this.container.scrollTop;
        this.render();
    }
    
    render() {
        this.viewportHeight = this.container.clientHeight;
        
        // Calculate visible range
        const startIndex = Math.floor(this.scrollTop / this.itemHeight);
        const endIndex = Math.min(
            this.items.length,
            Math.ceil((this.scrollTop + this.viewportHeight) / this.itemHeight) + 1
        );
        
        // Only re-render if necessary
        if (startIndex !== this.renderedRange.start || endIndex !== this.renderedRange.end) {
            this.renderedRange = { start: startIndex, end: endIndex };
            this.renderItems(startIndex, endIndex);
        }
    }
    
    renderItems(startIndex, endIndex) {
        // Clear current items
        this.viewport.innerHTML = '';
        
        // Position viewport
        this.viewport.style.transform = `translateY(${startIndex * this.itemHeight}px)`;
        
        // Render visible items
        for (let i = startIndex; i < endIndex; i++) {
            if (i >= this.items.length) break;
            
            const item = this.items[i];
            const element = this.createItemElement(item, i);
            this.viewport.appendChild(element);
        }
    }
    
    createItemElement(item, index) {
        const isSelected = item.id === selectedPathwayId;
        const typeClass = `type-${item.type}`;
        const typeLabel = item.type.charAt(0).toUpperCase() + item.type.slice(1);
        
        const div = document.createElement('div');
        div.className = `pathway-item ${isSelected ? 'selected' : ''}`;
        div.dataset.id = item.id;
        div.dataset.index = item.index;
        div.style.height = `${this.itemHeight}px`;
        
        div.innerHTML = `
            <div class="pathway-header">
                <span class="pathway-id">${item.id}</span>
                <span class="pathway-name">${item.name}</span>
            </div>
            <div class="pathway-meta">
                <span class="pathway-type ${typeClass}">${typeLabel}</span>
            </div>
        `;
        
        // Add click event listener
        div.addEventListener('click', function() {
            // Call the global showPathwayDetail function
            if (typeof window.showPathwayDetail === 'function') {
                window.showPathwayDetail(item.id, item.index);
            }
        });
        
        return div;
    }
    
    // Clean up when done
    destroy() {
        this.container.removeEventListener('scroll', this.onScroll);
        window.removeEventListener('resize', this.render);
        this.container.innerHTML = '';
    }
}

// Initialize virtual list when appropriate
window.initVirtualList = function() {
    // Only use virtual list for large datasets
    if (allData && allData.length > 200) {
        console.log("Initializing virtual list for better performance");
        window.virtualList = new VirtualList('pathway-list');
        
        // Override renderPathwaysList to use virtual list
        window.originalRenderPathwaysList = window.renderPathwaysList;
        window.renderPathwaysList = function() {
            const startIndex = (currentPage - 1) * itemsPerPage;
            const endIndex = Math.min(startIndex + itemsPerPage, filteredData.length);
            const currentPageData = filteredData.slice(startIndex, endIndex);
            
            if (filteredData.length === 0) {
                $('#pathway-list').html('<div class="no-results">No matching pathways found</div>');
                return;
            }
            
            // Use virtual list
            if (window.virtualList) {
                window.virtualList.setItems(currentPageData);
            } else {
                // Fall back to original method if something went wrong
                if (window.originalRenderPathwaysList) {
                    window.originalRenderPathwaysList();
                }
            }
        };
    }
};

// File chunking for extremely large files
window.fileChunker = {
    chunkSize: 10 * 1024 * 1024, // 10MB chunks
    
    readLargeFile: function(file, onChunkRead, onComplete, onError) {
        const totalChunks = Math.ceil(file.size / this.chunkSize);
        let currentChunk = 0;
        const fileReader = new FileReader();
        
        const readNextChunk = () => {
            const start = currentChunk * this.chunkSize;
            const end = Math.min(start + this.chunkSize, file.size);
            
            const blob = file.slice(start, end);
            fileReader.readAsArrayBuffer(blob);
        };
        
        fileReader.onload = (e) => {
            // Process this chunk
            const chunk = new Uint8Array(e.target.result);
            const progress = Math.round(((currentChunk + 1) / totalChunks) * 100);
            
            // Call callback with chunk data
            onChunkRead(chunk, currentChunk, totalChunks, progress);
            
            // Move to next chunk or finalize
            currentChunk++;
            if (currentChunk < totalChunks) {
                // Process next chunk
                setTimeout(readNextChunk, 0);
            } else {
                // Complete
                if (onComplete) onComplete();
            }
        };
        
        fileReader.onerror = (error) => {
            if (onError) onError(error);
        };
        
        // Start reading the first chunk
        readNextChunk();
    }
};

// Add to document ready function
$(document).ready(function() {
    // Add CSS for virtual list
    const style = document.createElement('style');
    style.textContent = `
        .virtual-list-content {
            position: relative;
            width: 100%;
        }
        .virtual-list-viewport {
            position: absolute;
            width: 100%;
        }
    `;
    document.head.appendChild(style);
    
    // Listen for visualization panel visibility
    const observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            if (mutation.attributeName === 'style' && 
                $('#visualization-panel').is(':visible') && 
                !window.lazyLoader.loadedResources.chartjs) {
                
                // Load Chart.js when visualization panel becomes visible
                window.lazyLoader.initChartJS().catch(err => {
                    console.error("Error initializing Chart.js:", err);
                    $('#abundance-chart').closest('.chart-container')
                        .html('<div class="error">Error loading chart library</div>');
                });
            }
        });
    });
    
    // Start observing visualization panel for display changes
    observer.observe(document.getElementById('visualization-panel'), { 
        attributes: true 
    });
});