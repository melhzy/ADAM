/**
 * HUMAnN3 Pathway Abundance Viewer
 * Event Service
 * Centralized event handling for application-wide communication
 */

app.service('EventService', ['$rootScope', function($rootScope) {
    // Event registry for tracking listeners
    var eventRegistry = {};
    
    /**
     * Register an event listener
     * @param {String} eventName - Name of the event to listen for
     * @param {Function} callback - Function to call when event is emitted
     * @returns {Function} - Deregistration function
     */
    this.on = function(eventName, callback) {
        // Create event in registry if it doesn't exist
        if (!eventRegistry[eventName]) {
            eventRegistry[eventName] = [];
        }
        
        // Add callback to registry
        eventRegistry[eventName].push(callback);
        
        // Return deregistration function
        return function() {
            const index = eventRegistry[eventName].indexOf(callback);
            if (index !== -1) {
                eventRegistry[eventName].splice(index, 1);
            }
        };
    };
    
    /**
     * Emit an event
     * @param {String} eventName - Name of the event to emit
     * @param {Object} data - Data to pass to event handlers
     */
    this.emit = function(eventName, data) {
        // Check if event has listeners
        if (eventRegistry[eventName]) {
            // Call all registered callbacks
            eventRegistry[eventName].forEach(function(callback) {
                try {
                    callback(data);
                } catch (error) {
                    console.error(`Error in event handler for ${eventName}:`, error);
                }
            });
        }
        
        // Also broadcast through Angular's event system
        $rootScope.$broadcast(eventName, data);
    };
    
    /**
     * Check if an event has listeners
     * @param {String} eventName - Name of the event to check
     * @returns {Boolean} - True if event has listeners, false otherwise
     */
    this.hasListeners = function(eventName) {
        return eventRegistry[eventName] && eventRegistry[eventName].length > 0;
    };
    
    /**
     * Remove all listeners for an event
     * @param {String} eventName - Name of the event to clear
     */
    this.clearListeners = function(eventName) {
        if (eventRegistry[eventName]) {
            eventRegistry[eventName] = [];
        }
    };
    
    /**
     * Remove all event listeners
     */
    this.clearAllListeners = function() {
        eventRegistry = {};
    };
    
    // Application-wide events reference
    this.events = {
        // Data events
        DATA_LOADED: 'data:loaded',
        DATA_PROCESSING: 'data:processing',
        DATA_RESET: 'data:reset',
        
        // Filter events
        FILTERS_APPLIED: 'filters:applied',
        
        // Pathway events
        PATHWAY_SELECTED: 'pathway:selected',
        
        // Visualization events
        VISUALIZATION_UPDATED: 'visualization:updated',
        
        // File events
        FILE_INFO_UPDATED: 'fileInfo:updated'
    };
}]);