// Fix for the Chart Type, Sample Limit, and Download Data button

// Update chart visualization with optimizations for large datasets
function updateChart(pathway) {
    const chartType = $('#chart-type').val() || 'bar';
    // Parse the sample limit value as an integer or use 'all' string
    const sampleLimitVal = $('#sample-limit').val();
    const sampleLimit = sampleLimitVal === 'all' ? allSamples.length : (parseInt(sampleLimitVal) || 20);
    
    console.log(`Updating chart: type=${chartType}, sampleLimit=${sampleLimit}`);
    
    // Sort samples by abundance for the chart
    let sortedSamples = [...allSamples].sort((a, b) => {
        return pathway.abundanceValues[b] - pathway.abundanceValues[a];
    });
    
    // Limit samples to improve performance
    const samplesToDisplay = sortedSamples.slice(0, Math.min(sampleLimit, sortedSamples.length));
    
    // Extract data for the chart
    const chartLabels = samplesToDisplay.map(sample => {
        // Shorten sample names for display
        return sample.replace(/_Abundance$/, '').substring(0, 15);
    });
    
    const chartData = samplesToDisplay.map(sample => pathway.abundanceValues[sample]);
    
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
                duration: samplesToDisplay.length > 50 ? 0 : 1000 // Disable animation for large datasets
            },
            plugins: {
                title: {
                    display: true,
                    text: `${pathway.name.substring(0, 50)}${pathway.name.length > 50 ? '...' : ''} - Abundance`
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            return `Abundance: ${formatNumber(context.raw)}`;
                        }
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
                        maxRotation: 90,
                        minRotation: 45
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
        if (samplesToDisplay.length > 12) {
            chartConfig.data.labels = chartLabels.slice(0, 12);
            chartConfig.data.datasets[0].data = chartData.slice(0, 12);
            
            // Add note about limiting
            chartConfig.options.plugins.title.text += ' (Limited to top 12 samples)';
        }
    }
    
    // Create the chart with optimized settings
    try {
        abundanceChart = new Chart(ctx, chartConfig);
        console.log("Chart created successfully");
    } catch (err) {
        console.error("Error creating chart:", err);
        $('#abundance-chart').closest('.chart-container').html('<div class="error">Error creating chart</div>');
    }
}

// Generate CSV for download
function downloadData(pathway) {
    if (!pathway) {
        console.error("Cannot download: no pathway selected");
        return;
    }
    
    console.log("Generating CSV for download");
    
    let csvContent = "Sample,Abundance\n";
    
    // Sort samples by abundance
    const sortedSamples = [...allSamples].sort((a, b) => {
        return pathway.abundanceValues[b] - pathway.abundanceValues[a];
    });
    
    // Add data rows
    sortedSamples.forEach(sample => {
        const abundance = pathway.abundanceValues[sample];
        csvContent += `${sample},${abundance}\n`;
    });
    
    try {
        // Create blob instead of a direct data URL for better handling of large files
        const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
        const url = URL.createObjectURL(blob);
        
        // Create download link
        const link = document.createElement("a");
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

// Fix the event handlers section to ensure proper functionality
$(document).ready(function() {
    // Chart type change
    $(document).on('change', '#chart-type', function() {
        console.log("Chart type changed to:", $(this).val());
        if (selectedPathwayId) {
            const pathway = allData.find(item => item.id === selectedPathwayId);
            if (pathway) {
                updateChart(pathway);
            }
        }
    });
    
    // Sample limit change
    $(document).on('change', '#sample-limit', function() {
        console.log("Sample limit changed to:", $(this).val());
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