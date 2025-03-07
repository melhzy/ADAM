// optimizations.js - Performance optimizations for HUMAnN3 data viewer with Alzheimer's focus

// Virtual List implementation for efficient rendering of large lists
class VirtualList {
    constructor(containerId, itemHeight = 53) {
        this.container = document.getElementById(containerId);
        this.itemHeight = itemHeight;
        this.items = [];
        this.scrollTop = 0;
        this.viewportHeight = 0;
        this.totalHeight = 0;
        this.renderedRange = { start: 0, end: 0 };
        
        // Create DOM structure
        this.createDOMStructure();
        
        // Bind methods
        this.onScroll = this.onScroll.bind(this);
        this.render = this.render.bind(this);
        
        // Add event listeners
        this.container.addEventListener('scroll', this.onScroll);
        window.addEventListener('resize', this.render);
    }
    
    createDOMStructure() {
        // Create a content container with full height
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
        // Force requestAnimationFrame to limit rendering frequency
        if (this.renderPending) return;
        
        this.renderPending = true;
        requestAnimationFrame(() => {
            this.renderPending = false;
            this.renderItems();
        });
    }
    
    renderItems() {
        this.viewportHeight = this.container.clientHeight;
        
        // Calculate visible range with buffer
        const buffer = 5; // Extra items to render above/below viewport
        const startIndex = Math.max(0, Math.floor(this.scrollTop / this.itemHeight) - buffer);
        const endIndex = Math.min(
            this.items.length,
            Math.ceil((this.scrollTop + this.viewportHeight) / this.itemHeight) + buffer
        );
        
        // Only re-render if necessary
        if (startIndex !== this.renderedRange.start || endIndex !== this.renderedRange.end) {
            this.renderedRange = { start: startIndex, end: endIndex };
            this.updateDOM(startIndex, endIndex);
        }
    }
    
    updateDOM(startIndex, endIndex) {
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
        const isSelected = item.id === window.selectedPathwayId;
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
                <span class="pathway-abundance">Avg: ${window.formatNumber(item.avgAbundance)}</span>
            </div>
        `;
        
        // Add click event listener
        div.addEventListener('click', function() {
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

// Memory management utilities
const memoryOptimization = {
    // Trigger garbage collection hint
    triggerCleanup: function() {
        const memoryHog = [];
        try {
            // Fill and clear an array to hint the garbage collector
            for (let i = 0; i < 10000; i++) {
                memoryHog.push({});
            }
        } catch (e) {
            // Ignore any errors
        }
        // Clear the reference
        memoryHog.length = 0;
    },
    
    // Chunk large data processing operations
    chunkProcess: function(items, processFn, chunkSize = 1000, onComplete) {
        let index = 0;
        
        function processNextChunk() {
            const end = Math.min(index + chunkSize, items.length);
            
            for (let i = index; i < end; i++) {
                processFn(items[i], i);
            }
            
            index = end;
            
            if (index < items.length) {
                setTimeout(processNextChunk, 0);
            } else if (onComplete) {
                onComplete();
            }
        }
        
        processNextChunk();
    },
    
    // Release large objects from memory
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

// Efficient file reading for large files
const fileReader = {
    // Read a large file in chunks
    readLargeFile: function(file, onProgress, onComplete, onError) {
        const chunkSize = 10 * 1024 * 1024; // 10MB chunks
        const totalChunks = Math.ceil(file.size / chunkSize);
        let currentChunk = 0;
        let result = '';
        
        const reader = new FileReader();
        
        reader.onload = function(e) {
            // Concatenate chunk to result
            result += e.target.result;
            
            // Update progress
            const progress = Math.round(((currentChunk + 1) / totalChunks) * 100);
            onProgress && onProgress(progress);
            
            // Move to next chunk or complete
            currentChunk++;
            if (currentChunk < totalChunks) {
                readNextChunk();
            } else {
                onComplete && onComplete(result);
                // Help GC
                result = null;
            }
        };
        
        reader.onerror = function(e) {
            onError && onError(e);
        };
        
        function readNextChunk() {
            const start = currentChunk * chunkSize;
            const end = Math.min(start + chunkSize, file.size);
            const slice = file.slice(start, end);
            reader.readAsText(slice);
        }
        
        // Start reading
        readNextChunk();
    }
};

// Chart.js optimizations
const chartOptimizations = {
    // Apply global optimizations to Chart.js
    applyGlobalOptimizations: function() {
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

// Apply optimizations when document is ready
document.addEventListener('DOMContentLoaded', function() {
    // Apply Chart.js optimizations if loaded
    if (window.Chart) {
        chartOptimizations.applyGlobalOptimizations();
    }
    
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
});