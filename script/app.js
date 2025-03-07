// Download data function - updated to work with Blob API
function downloadData(pathway) {
    if (!pathway) {
        console.error("Cannot download: no pathway selected");
        return;
    }
    
    console.log("Generating CSV for download");
    
    try {
        // Create CSV header
        const csvHeader = "Sample,Patient,Day,Site,Abundance\n";
        
        // Sort samples by abundance
        const sortedSamples = [...allSamples].sort((a, b) => {
            return pathway.abundanceValues[b] - pathway.abundanceValues[a];
        });
        
        // Generate CSV rows
        const csvRows = sortedSamples.map(sample => {
            const sampleName = sample.replace(/_Abundance$/, '');
            const meta = sampleMetadata[sampleName] || {};
            const abundance = pathway.abundanceValues[sample] || 0;
            
            return `${sampleName},${meta.patient || ''},${meta.day || ''},${meta.site || ''},${abundance}`;
        }).join('\n');
        
        // Create CSV content
        const csvContent = csvHeader + csvRows;
        
        // Create blob and download link
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const link = document.createElement("a");
        // Sanitize filename
        const filename = pathway.name.replace(/[/\\?%*:|"<>]/g, '_');
        link.setAttribute("href", url);
        link.setAttribute("download", `${filename}_abundance.csv`);
        link.style.visibility = 'hidden';
        
        // Append to document, trigger click, and remove
        document.body.appendChild(link);
        link.click();
        
        // Clean up
        setTimeout(() => {
            document.body.removeChild(link);
            URL.revokeObjectURL(url);
        }, 100);
    } catch (err) {
        console.error("Error downloading data:", err);
        alert("Error downloading data: " + err.message);
    }
}

// Export chart function - improved to work in all browsers
function exportChart() {
    if (!abundanceChart) {
        console.error("No chart to export");
        return;
    }
    
    try {
        const canvas = document.getElementById('abundance-chart');
        
        // For better quality, use a higher resolution when converting to image
        const context = canvas.getContext('2d');
        const originalRatio = context.backingStorePixelRatio || 1;
        const pixelRatio = window.devicePixelRatio || 1;
        const ratio = pixelRatio / originalRatio;
        
        // Get the original dimensions
        const originalWidth = canvas.width;
        const originalHeight = canvas.height;
        
        // Create a high-resolution copy of the canvas
        const tempCanvas = document.createElement('canvas');
        const tempContext = tempCanvas.getContext('2d');
        
        // Set dimensions to high resolution
        tempCanvas.width = originalWidth * ratio;
        tempCanvas.height = originalHeight * ratio;
        tempCanvas.style.width = originalWidth + 'px';
        tempCanvas.style.height = originalHeight + 'px';
        
        // Scale the context
        tempContext.scale(ratio, ratio);
        
        // Copy the chart to the high-res canvas
        tempContext.drawImage(canvas, 0, 0, originalWidth, originalHeight);
        
        // Convert to image
        const imageData = tempCanvas.toDataURL('image/png');
        
        // Create download link
        const downloadLink = document.createElement('a');
        downloadLink.href = imageData;
        downloadLink.download = `pathway_chart_${selectedPathwayId || 'export'}.png`;
        document.body.appendChild(downloadLink);
        downloadLink.click();
        document.body.removeChild(downloadLink);
        
        console.log("Chart exported successfully");
    } catch (err) {
        console.error("Error exporting chart:", err);
        alert("Error exporting chart: " + err.message);
    }
}