/**
 * HUMAnN3 Pathway Abundance Viewer
 * Pathway Model
 * Represents a metabolic pathway with its abundance data
 */

app.factory('PathwayModel', [function() {
    /**
     * Pathway Model Constructor
     * @param {Object} data - Initialization data
     */
    function Pathway(data) {
        // Initialize with default values
        this.id = '';
        this.name = '';
        this.type = 'other';  // 'metacyc', 'unmapped', 'unintegrated', 'other'
        this.abundanceValues = {};
        this.index = -1;
        this.avgAbundance = 0;
        this.maxAbundance = 0;
        this.maxSample = '';
        
        // Apply provided data
        if (data) {
            this.update(data);
        }
        
        // Calculate derived properties
        this.calculateStatistics();
    }
    
    /**
     * Update pathway data
     * @param {Object} data - New data to apply
     */
    Pathway.prototype.update = function(data) {
        if (!data) return;
        
        if (data.id !== undefined) this.id = data.id;
        if (data.name !== undefined) this.name = data.name;
        if (data.type !== undefined) this.type = data.type;
        if (data.abundanceValues !== undefined) this.abundanceValues = data.abundanceValues;
        if (data.index !== undefined) this.index = data.index;
        
        // Recalculate statistics after update
        this.calculateStatistics();
    };
    
    /**
     * Calculate statistics from abundance values
     */
    Pathway.prototype.calculateStatistics = function() {
        const samples = Object.keys(this.abundanceValues);
        
        if (samples.length === 0) {
            this.avgAbundance = 0;
            this.maxAbundance = 0;
            this.maxSample = '';
            return;
        }
        
        // Calculate average abundance
        const total = samples.reduce((sum, sample) => sum + this.abundanceValues[sample], 0);
        this.avgAbundance = total / samples.length;
        
        // Find maximum abundance and its sample
        let maxValue = -Infinity;
        let maxSample = '';
        
        samples.forEach(sample => {
            const value = this.abundanceValues[sample];
            if (value > maxValue) {
                maxValue = value;
                maxSample = sample;
            }
        });
        
        this.maxAbundance = maxValue;
        this.maxSample = maxSample;
    };
    
    /**
     * Get abundance for a specific sample
     * @param {String} sample - Sample name
     * @returns {Number} - Abundance value
     */
    Pathway.prototype.getAbundanceForSample = function(sample) {
        return this.abundanceValues[sample] || 0;
    };
    
    /**
     * Get top samples by abundance
     * @param {Number} limit - Maximum number of samples to return
     * @returns {Array} - Array of objects with sample name and abundance
     */
    Pathway.prototype.getTopSamples = function(limit) {
        limit = limit || 10;
        
        const samples = Object.keys(this.abundanceValues)
            .map(sample => ({
                name: sample,
                abundance: this.abundanceValues[sample]
            }))
            .sort((a, b) => b.abundance - a.abundance)
            .slice(0, limit);
        
        return samples;
    };
    
    /**
     * Get normalized abundance values
     * @returns {Object} - Map of sample to normalized abundance (0-1)
     */
    Pathway.prototype.getNormalizedValues = function() {
        const samples = Object.keys(this.abundanceValues);
        const result = {};
        
        if (samples.length === 0 || this.maxAbundance <= 0) {
            return result;
        }
        
        samples.forEach(sample => {
            result[sample] = this.abundanceValues[sample] / this.maxAbundance;
        });
        
        return result;
    };
    
    /**
     * Get display-friendly abundance value
     * @param {String} sample - Optional sample name to get specific sample value
     * @returns {Number} - Abundance value (average if no sample specified)
     */
    Pathway.prototype.getDisplayAbundance = function(sample) {
        if (sample) {
            return this.getAbundanceForSample(sample);
        }
        return this.avgAbundance;
    };
    
    /**
     * Convert to simple object (for export/serialization)
     * @returns {Object} - Simple object representation
     */
    Pathway.prototype.toObject = function() {
        return {
            id: this.id,
            name: this.name,
            type: this.type,
            abundanceValues: Object.assign({}, this.abundanceValues),
            avgAbundance: this.avgAbundance,
            maxAbundance: this.maxAbundance,
            maxSample: this.maxSample,
            index: this.index
        };
    };
    
    /**
     * Create pathway from simple object
     * @param {Object} obj - Object representation
     * @returns {Pathway} - New pathway instance
     */
    Pathway.fromObject = function(obj) {
        return new Pathway(obj);
    };
    
    // Return the constructor function
    return Pathway;
}]);