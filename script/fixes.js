// Optimized fixes.js for the HUMAnN3 Pathway Abundance Viewer

// Update chart visualization with optimizations for large datasets
function updateChart(pathway) {
    const chartType = $('#chart-type').val() || 'bar';
    // Parse the sample limit value as an integer or use 'all' string
    const sampleLimitVal = $('#sample-limit').val();
    const sampleLimit = sampleLimitVal === 'all' ? allSamples.length : (parseInt(sampleLimitVal) || 20);
    
    console.log(`Updating chart: type=${chartType}, sampleLimit=${sampleLimit}`);
    
    // OPTIMIZATION: Use indices for sorting instead of copying the full array
    // Create an array of indices
    const indices = Array.from(Array(allSamples.length).keys());
    // Sort indices by abundance values (descending)
    indices.sort((a, b) => pathway.abundanceValues[allSamples[b]] - pathway.abundanceValues[allSamples[a]]);
    
    // Limit samples to improve performance
    const limitedIndices = indices.slice(0, Math.min(sampleLimit, indices.length));
    
    // Extract data for the chart using the sorted indices
    const chartLabels = limitedIndices.map(i => {
        // Shorten sample names for display
        return allSamples[i].replace(/_Abundance$/, '').substring(0, 15);
    });
    
    const chartData = limitedIndices.map(i => pathway.abundanceValues[allSamples[i]]);
    
    // Define chart colors
    const backgroundColor = 'rgba(52, 152, 219, 0.5)';
    const borderColor = 'rgba(52, 152, 219, 1)';
    
    // Destroy previous chart if it exists
    if (abundanceChart) {
        abundanceChart.destroy();
    }
    
    // Create a new chart
    const ctx = document.getElementById('abundance-chart').getContext('2d');
    
    // Configure chart options based on the selected type
    let chartConfig = {
        type: chartType,
        data: {
            labels: chartLabels,
            datasets: [{
                label: 'Abundance',
                data: chartData,
                backgroundColor: backgroundColor,
                borderColor: borderColor,
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            animation: {
                // OPTIMIZATION: Disable animation for large datasets
                duration: limitedIndices.length > 30 ? 0 : 750
            },
            plugins: {
                title: {
                    display: true,
                    text: `${pathway.name.substring(0, 50)}${pathway.name.length > 50 ? '...' : ''} - Abundance`
                },
                tooltip: {
                    // OPTIMIZATION: Simpler tooltip callback
                    callbacks: {
                        label: (context) => `Abundance: ${formatNumber(context.raw)}`
                    }
                },
                legend: {
                    display: false // Hide legend for better performance
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Abundance'
                    }
                },
                x: {
                    title: {
                        display: true,
                        text: 'Sample'
                    },
                    ticks: {
                        // OPTIMIZATION: Reduce max rotation for better readability
                        maxRotation: 60,
                        minRotation: 30,
                        // OPTIMIZATION: Limit the number of ticks for better performance
                        autoSkip: true,
                        maxTicksLimit: 20
                    }
                }
            }
        }
    };
    
    // Special configurations for specific chart types
    if (chartType === 'radar') {
        chartConfig.options.scales = {}; // Remove scales for radar chart
        // Additional radar chart options
        chartConfig.options.elements = {
            line: {
                tension: 0.2,
                borderWidth: 2
            }
        };
        
        // Limit datasets for radar chart to avoid performance issues
        if (limitedIndices.length > 12) {
            chartConfig.data.labels = chartLabels.slice(0, 12);
            chartConfig.data.datasets[0].data = chartData.slice(0, 12);
            
            // Add note about limiting
            chartConfig.options.plugins.title.text += ' (Limited to top 12 samples)';
        }
    }
    
    // OPTIMIZATION: Use try-catch for chart creation to handle any errors gracefully
    try {
        // Create the chart with optimized settings
        abundanceChart = new Chart(ctx, chartConfig);
        console.log("Chart created successfully");
    } catch (err) {
        console.error("Error creating chart:", err);
        $('#abundance-chart').closest('.chart-container').html('<div class="error">Error creating chart: ' + err.message + '</div>');
    }
}

// Generate CSV for download - optimized for memory usage with large datasets
function downloadData(pathway) {
    if (!pathway) {
        console.error("Cannot download: no pathway selected");
        return;
    }
    
    console.log("Generating CSV for download");
    
    // OPTIMIZATION: Create a blob with streaming generation instead of building a large string
    try {
        // Sort samples by abundance first
        const indices = Array.from(Array(allSamples.length).keys());
        indices.sort((a, b) => pathway.abundanceValues[allSamples[b]] - pathway.abundanceValues[allSamples[a]]);
        
        // Create a BlobBuilder-like approach for memory efficiency
        const chunks = ["Sample,Abundance\n"];
        
        // Add data in chunks
        for (let i = 0; i < indices.length; i++) {
            const sample = allSamples[indices[i]];
            const abundance = pathway.abundanceValues[sample];
            chunks.push(`${sample},${abundance}\n`);
            
            // Flush every 1000 rows
            if (i % 1000 === 999) {
                console.log(`Processed ${i+1} rows for download`);
            }
        }
        
        // Create blob from chunks for better memory handling
        const blob = new Blob(chunks, { type: 'text/csv;charset=utf-8;' });
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
        
        console.log("Triggering download for:", filename);
        link.click();
        
        // Clean up
        document.body.removeChild(link);
        setTimeout(() => {
            URL.revokeObjectURL(url);
        }, 100);
    } catch (err) {
        console.error("Error downloading data:", err);
        alert("Error downloading data: " + err.message);
    }
}

// OPTIMIZATION: Use event delegation for improved event handling
$(document).ready(function() {
    // Chart type change - use one event handler
    $(document).on('change', '#chart-type, #sample-limit', function() {
        console.log(`${this.id} changed to:`, $(this).val());
        if (selectedPathwayId) {
            const pathway = allData.find(item => item.id === selectedPathwayId);
            if (pathway) {
                updateChart(pathway);
            }
        }
    });
    
    // Download button click
    $(document).on('click', '#download-btn', function(e) {
        e.preventDefault();
        console.log("Download button clicked");
        if (selectedPathwayId) {
            const pathway = allData.find(item => item.id === selectedPathwayId);
            if (pathway) {
                downloadData(pathway);
            } else {
                console.error("Could not find pathway for download");
            }
        } else {
            console.log("No pathway selected for download");
            alert("Please select a pathway first");
        }
    });
});