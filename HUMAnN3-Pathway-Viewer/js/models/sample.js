/**
 * HUMAnN3 Pathway Abundance Viewer
 * Sample Model
 * Represents a biological sample with its metadata
 */

app.factory('SampleModel', [function() {
    /**
     * Sample Model Constructor
     * @param {Object} data - Initialization data
     */
    function Sample(data) {
        // Initialize with default values
        this.name = '';
        this.metadata = {};
        this.pathwayAbundances = {};  // Map of pathway ID to abundance
        this.totalAbundance = 0;
        
        // Apply provided data
        if (data) {
            this.update(data);
        }
        
        // Calculate derived properties
        this.calculateTotalAbundance();
    }
    
    /**
     * Update sample data
     * @param {Object} data - New data to apply
     */
    Sample.prototype.update = function(data) {
        if (!data) return;
        
        if (data.name !== undefined) this.name = data.name;
        if (data.metadata !== undefined) this.metadata = Object.assign({}, data.metadata);
        if (data.pathwayAbundances !== undefined) this.pathwayAbundances = Object.assign({}, data.pathwayAbundances);
        
        // Recalculate total abundance
        this.calculateTotalAbundance();
    };
    
    /**
     * Calculate total abundance across all pathways
     */
    Sample.prototype.calculateTotalAbundance = function() {
        const pathwayIds = Object.keys(this.pathwayAbundances);
        
        if (pathwayIds.length === 0) {
            this.totalAbundance = 0;
            return;
        }
        
        this.totalAbundance = pathwayIds.reduce((sum, id) => sum + this.pathwayAbundances[id], 0);
    };
    
    /**
     * Get abundance for a specific pathway
     * @param {String} pathwayId - Pathway ID
     * @returns {Number} - Abundance value
     */
    Sample.prototype.getAbundanceForPathway = function(pathwayId) {
        return this.pathwayAbundances[pathwayId] || 0;
    };
    
    /**
     * Set abundance for a specific pathway
     * @param {String} pathwayId - Pathway ID
     * @param {Number} value - Abundance value
     */
    Sample.prototype.setAbundanceForPathway = function(pathwayId, value) {
        if (typeof value !== 'number') {
            value = parseFloat(value) || 0;
        }
        
        this.pathwayAbundances[pathwayId] = value;
        this.calculateTotalAbundance();
    };
    
    /**
     * Get top pathways by abundance
     * @param {Number} limit - Maximum number of pathways to return
     * @returns {Array} - Array of objects with pathway ID and abundance
     */
    Sample.prototype.getTopPathways = function(limit) {
        limit = limit || 10;
        
        const pathways = Object.keys(this.pathwayAbundances)
            .map(id => ({
                id: id,
                abundance: this.pathwayAbundances[id]
            }))
            .sort((a, b) => b.abundance - a.abundance)
            .slice(0, limit);
        
        return pathways;
    };
    
    /**
     * Get normalized pathway abundances as percentages
     * @returns {Object} - Map of pathway ID to percentage (0-100)
     */
    Sample.prototype.getAbundancePercentages = function() {
        const result = {};
        
        if (this.totalAbundance <= 0) {
            return result;
        }
        
        Object.keys(this.pathwayAbundances).forEach(id => {
            result[id] = (this.pathwayAbundances[id] / this.totalAbundance) * 100;
        });
        
        return result;
    };
    
    /**
     * Get metadata value
     * @param {String} key - Metadata key
     * @param {*} defaultValue - Default value if key not found
     * @returns {*} - Metadata value
     */
    Sample.prototype.getMetadata = function(key, defaultValue) {
        return this.metadata[key] !== undefined ? this.metadata[key] : defaultValue;
    };
    
    /**
     * Set metadata value
     * @param {String} key - Metadata key
     * @param {*} value - Metadata value
     */
    Sample.prototype.setMetadata = function(key, value) {
        this.metadata[key] = value;
    };
    
    /**
     * Convert to simple object (for export/serialization)
     * @returns {Object} - Simple object representation
     */
    Sample.prototype.toObject = function() {
        return {
            name: this.name,
            metadata: Object.assign({}, this.metadata),
            pathwayAbundances: Object.assign({}, this.pathwayAbundances),
            totalAbundance: this.totalAbundance
        };
    };
    
    /**
     * Create sample from simple object
     * @param {Object} obj - Object representation
     * @returns {Sample} - New sample instance
     */
    Sample.fromObject = function(obj) {
        return new Sample(obj);
    };
    
    // Return the constructor function
    return Sample;
}]);