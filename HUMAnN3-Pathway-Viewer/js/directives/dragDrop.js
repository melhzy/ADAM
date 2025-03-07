/**
 * HUMAnN3 Pathway Abundance Viewer
 * Drag and Drop Directives
 * Provides directives for drag-and-drop file uploads
 */

app.directive('ngDrop', function() {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            var dropHandler = scope.$eval(attrs.ngDrop);
            
            element[0].addEventListener('drop', function(event) {
                if (typeof dropHandler === 'function') {
                    scope.$apply(function() {
                        dropHandler(event);
                    });
                }
            }, false);
            
            // Prevent default to allow drop
            element[0].addEventListener('dragover', function(event) {
                event.preventDefault();
            }, false);
        }
    };
});

app.directive('ngDragOver', function() {
    return {
        restrict: 'A',
        link: function(scope, element, attrs) {
            var dragOverHandler = scope.$eval(attrs.ngDragOver);
            
            element[0].addEventListener('dragover', function(event) {
                if (typeof dragOverHandler === 'function') {
                    scope.$apply(function() {
                        dragOverHandler(event);
                    });
                }
            }, false);
        }
    };
});
