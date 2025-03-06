// jquery.js for HUMAnN3 Pathway Abundance Viewer
$(document).ready(function() {
    // Initialize variables
    let allData = [];
    let allSamples = [];
    let filteredData = [];
    let currentPage = 1;
    const itemsPerPage = 50;
    let selectedPathwayId = null;
    let abundanceChart = null;
    
    // File upload handling with added memory optimizations
    $('#csv-file-upload').on('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Update the displayed filename
        $('#file-upload-text').text(file.name);
        $('#file-status').html('<div class="loading">Reading file...</div>');
        
        // Display file size
        const fileSizeMB = (file.size / (1024 * 1024)).toFixed(2);
        console.log(`File size: ${fileSizeMB} MB`);
        
        // For extremely large files, warn user and use chunked processing
        if (file.size > 150 * 1024 * 1024) { // Over 150MB
            $('#file-status').html(`<div class="file-warning">Large file detected (${fileSizeMB} MB). Processing may take several minutes.</div>`);
        }
        
        // Check file extension
        const fileExtension = file.name.split('.').pop().toLowerCase();
        let isTabDelimited = fileExtension === 'tsv' || fileExtension === 'txt';
        
        // Use a more efficient reading approach
        const reader = new FileReader();
        
        reader.onload = function(event) {
            const fileData = event.target.result;
            
            // Try to detect if file is tab-delimited by examining first few lines
            if (!isTabDelimited) {
                const firstLines = fileData.split('\n').slice(0, 5).join('\n');
                isTabDelimited = firstLines.includes('\t');
            }
            
            // Clean up potential BOM and other problematic characters
            let cleanData = fileData;
            if (cleanData.charCodeAt(0) === 0xFEFF) {
                // Remove BOM
                cleanData = cleanData.slice(1);
            }
            
            console.log(`Processing file: ${file.name} (${isTabDelimited ? 'tab-delimited' : 'comma-delimited'})`);
            parseCSV(cleanData, isTabDelimited);
            
            // Help free up memory
            cleanData = null;
        };
        
        reader.onerror = function() {
            $('#file-status').html('<div class="file-error">Error reading file</div>');
        };
        
        // For very large files, consider using readAsArrayBuffer for better performance
        if (file.size > 200 * 1024 * 1024) { // Over 200MB
            $('#file-status').html(`<div class="file-warning">Extra large file detected (${fileSizeMB} MB). Only the first 10,000 pathways will be loaded.</div>`);
        }
        
        reader.readAsText(file);
    });
    
    // Parse CSV/TSV data with specific handling for HUMAnN3 data format
    function parseCSV(csvData, isTabDelimited = true) {
        console.time('File Parsing');
        const fileSize = (csvData.length / 1024 / 1024).toFixed(2);
        console.log(`Starting to parse file of size ${fileSize} MB`);
        
        // For very large files, use a streaming approach
        if (isTabDelimited && csvData.length > 50 * 1024 * 1024) { // Over 50MB
            console.log("Using streaming approach for large file");
            streamParseTabFile(csvData);
            return;
        }

// Try a manual approach if the file is tab-delimited
        if (isTabDelimited) {
            try {
                console.log("Using manual parsing for tab-delimited file");
                const lines = csvData.split('\n');
                let headers = [];
                const data = [];
                
                // Find the header line (skip comment lines)
                let headerLine = null;
                for (let i = 0; i < Math.min(20, lines.length); i++) {
                    if (lines[i] && !lines[i].trim().startsWith('#')) {
                        headerLine = lines[i];
                        break;
                    }
                }
                
                if (!headerLine) {
                    $('#file-status').html(`<div class="file-error">Could not find header row in file</div>`);
                    console.timeEnd('File Parsing');
                    return;
                }
                
                // Process headers - split by tab
                headers = headerLine.split('\t').map(h => h.trim());
                console.log("Found headers:", headers);
                
                // For very large files, limit the number of rows processed initially
                const rowLimit = lines.length > 10000 ? 10000 : lines.length;
                console.log(`Processing first ${rowLimit} rows of ${lines.length} total rows`);
                
                // Process data rows in batches
                const batchSize = 1000;
                let rowsProcessed = 0;
                
                function processRowBatch(startIndex) {
                    const endIndex = Math.min(startIndex + batchSize, rowLimit);
                    
                    for (let i = startIndex; i < endIndex; i++) {
                        if (i === 0) continue; // Skip header row
                        
                        const line = lines[i];
                        if (!line || line.trim() === '' || line.trim().startsWith('#')) continue;
                        
                        const values = line.split('\t');
                        const row = {};
                        
                        // Assign values to headers
                        headers.forEach((header, j) => {
                            if (values[j] !== undefined) {
                                // Try to convert numeric values
                                const value = values[j].trim();
                                if (!isNaN(value) && value !== '') {
                                    row[header] = parseFloat(value);
                                } else {
                                    row[header] = value;
                                }
                            } else {
                                row[header] = null;
                            }
                        });
                        
                        data.push(row);
                        rowsProcessed++;
                    }
                    
                    // Update progress
                    if (endIndex < rowLimit) {
                        $('#file-status').html(`<div class="loading">Parsing file: ${Math.round(endIndex/rowLimit*100)}%</div>`);
                        setTimeout(() => processRowBatch(endIndex), 0);
                    } else {
                        // Complete
                        console.log(`Manually parsed ${data.length} rows with ${headers.length} columns`);
                        console.timeEnd('File Parsing');
                        
                        // For very large files, inform user that we're only showing a subset
                        if (lines.length > rowLimit) {
                            $('#file-status').html(`<div class="file-warning">Showing first ${data.length} pathways. For best performance with large files, consider filtering your dataset.</div>`);
                        }
                        
                        processData(data, headers);
                        $('#app-content').show();
                    }
                }
                
                // Start processing the first batch
                processRowBatch(1); // Start from 1 to skip header
                
            } catch (err) {
                console.error("Manual parsing failed:", err);
                $('#file-status').html(`<div class="file-warning">Manual parsing failed. Trying fallback method...</div>`);
                // Fall back to PapaParse if manual parsing fails
                tryPapaParse(csvData);
            }
        } else {
            // Use PapaParse for non-tab-delimited files
            tryPapaParse(csvData);
        }
    }
    
    // Stream-based parsing for very large tab-delimited files
    function streamParseTabFile(fileData) {
        console.log("Starting stream parsing of large file");
        $('#file-status').html(`<div class="loading">Parsing large file...</div>`);
        
        // Split the file into lines
        const lines = fileData.split('\n');
        const totalLines = lines.length;
        console.log(`File contains ${totalLines} lines`);
        
        // Find header line (skip comments)
        let headerLine = null;
        let headerIndex = 0;
        
        for (let i = 0; i < Math.min(20, lines.length); i++) {
            if (lines[i] && !lines[i].trim().startsWith('#')) {
                headerLine = lines[i];
                headerIndex = i;
                break;
            }
        }

if (!headerLine) {
            $('#file-status').html(`<div class="file-error">Could not find header row in file</div>`);
            return;
        }
        
        // Process headers
        const headers = headerLine.split('\t').map(h => h.trim());
        console.log(`Found ${headers.length} columns in header`);
        
        // Determine how many rows to process
        // For extremely large files, limit the initial load
        const maxInitialRows = 5000; // Adjust based on browser performance
        const rowsToProcess = Math.min(totalLines - headerIndex - 1, maxInitialRows);
        
        // Prepare data array
        const data = [];
        
        // Process rows in chunks to prevent UI freezing
        const chunkSize = 500;
        let currentRow = headerIndex + 1; // Start after header
        
        function processChunk() {
            const endRow = Math.min(currentRow + chunkSize, headerIndex + 1 + rowsToProcess);
            const percentComplete = Math.round(((currentRow - headerIndex - 1) / rowsToProcess) * 100);
            
            $('#file-status').html(`<div class="loading">Loading data: ${percentComplete}% complete</div>`);
            
            // Process this chunk of rows
            for (let i = currentRow; i < endRow; i++) {
                const line = lines[i];
                if (!line || line.trim() === '' || line.trim().startsWith('#')) continue;
                
                try {
                    const values = line.split('\t');
                    const row = {};
                    
                    // Only process essential columns to save memory
                    headers.forEach((header, j) => {
                        if (values[j] !== undefined) {
                            row[header] = values[j].trim();
                        } else {
                            row[header] = null;
                        }
                    });
                    
                    data.push(row);
                } catch (e) {
                    console.error(`Error processing row ${i}:`, e);
                }
            }
            
            currentRow = endRow;
            
            if (currentRow < headerIndex + 1 + rowsToProcess) {
                // More rows to process
                setTimeout(processChunk, 0);
            } else {
                // Done processing
                console.log(`Stream processed ${data.length} rows`);
                
                if (rowsToProcess < totalLines - headerIndex - 1) {
                    $('#file-status').html(`<div class="file-warning">Loaded first ${data.length} pathways. File contains ${totalLines - headerIndex - 1} total pathways.</div>`);
                }
                
                processData(data, headers);
                $('#app-content').show();
            }
        }
        
        // Start processing the first chunk
        processChunk();
    }
    
    // Fallback to PapaParse for parsing
    function tryPapaParse(csvData) {
        Papa.parse(csvData, {
            header: true,
            skipEmptyLines: true,
            dynamicTyping: true,
            quoteChar: '"',   // Specify quote character
            escapeChar: '\\', // Specify escape character
            delimitersToGuess: ['\t', ','], // Try to guess delimiter
            comments: "#",    // Treat lines starting with # as comments
            error: function(error) {
                $('#file-status').html(`<div class="file-error">Error parsing file: ${error.message}</div>`);
            },
            complete: function(results) {
                // Check for and handle errors
                if (results.errors && results.errors.length > 0) {
                    console.log("Parsing errors:", results.errors);
                    
                    // Try to continue anyway if we have data despite errors
                    if (results.data && results.data.length > 0 && results.meta && results.meta.fields) {
                        console.log("Attempting to process data despite errors");
                        $('#file-status').html(`<div class="file-warning">Warning: File had parsing issues but attempting to display data</div>`);
                        processData(results.data, results.meta.fields);
                        $('#app-content').show();
                    } else {
                        $('#file-status').html(`<div class="file-error">Error parsing file: ${results.errors[0].message}</div>`);
                        return;
                    }
                } else {
                    // Normal success case
                    processData(results.data, results.meta.fields);
                    $('#file-status').html(`<div class="file-success">Successfully loaded ${results.data.length} pathways</div>`);
                    $('#app-content').show();
                }
            }
        });
    }
    
    // Format numbers for display
    function formatNumber(num) {
        if (num === null || num === undefined) return "0";
        
        if (num >= 1000000) {
            return (num / 1000000).toFixed(2) + "M";
        } else if (num >= 1000) {
            return (num / 1000).toFixed(2) + "K";
        } else if (num >= 1) {
            return num.toFixed(2);
        } else {
            return num.toExponential(2);
        }
    }

// Process the loaded data more efficiently for large files
    function processData(data, headers) {
        console.time('Data Processing');
        
        // Reset current state
        allData = [];
        filteredData = [];
        currentPage = 1;
        selectedPathwayId = null;
        
        console.log("Processing data with headers:", headers);
        
        // Clean up headers and identify non-sample columns
        const nonSampleColumns = ["# Pathway", "Pathway", "0", "#Pathway", "pathway_name", "pathway", "description"];
        
        // Extract all sample names (columns except the non-sample ones)
        allSamples = headers.filter(header => {
            // Skip the non-sample columns and any empty headers
            if (!header || nonSampleColumns.includes(header)) return false;
            return true;
        });
        
        console.log(`Found ${allSamples.length} sample columns`);
        
        // Find pathway column index
        let pathwayColName = null;
        for (const colName of ["# Pathway", "Pathway", "#Pathway", "pathway_name", "pathway"]) {
            if (headers.includes(colName)) {
                pathwayColName = colName;
                break;
            }
        }
        
        // If not found, use the first column that's not a sample column
        if (!pathwayColName && headers.length > 0) {
            pathwayColName = headers[0];
        }
        
        console.log("Using pathway column:", pathwayColName);
        
        // Process in batches to improve performance with large files
        const batchSize = 1000;
        let batchCount = 0;
        
        function processBatch(startIndex) {
            const endIndex = Math.min(startIndex + batchSize, data.length);
            console.log(`Processing batch ${batchCount++}: rows ${startIndex} to ${endIndex-1}`);
            
            for (let i = startIndex; i < endIndex; i++) {
                try {
                    const row = data[i];
                    
                    // Skip invalid rows
                    let pathwayCol = null;
                    if (pathwayColName && row[pathwayColName] !== undefined && row[pathwayColName] !== null) {
                        pathwayCol = row[pathwayColName];
                    } else {
                        // Try all possible column names as a fallback
                        for (const colName of nonSampleColumns) {
                            if (row[colName] !== undefined && row[colName] !== null) {
                                pathwayCol = row[colName];
                                break;
                            }
                        }
                    }
                    
                    // Skip row if we can't identify the pathway
                    if (!pathwayCol) {
                        continue;
                    }
                    
                    // For rows with ID and name in separate fields
                    let pathwayId = i.toString();
                    let pathwayName = pathwayCol.toString();
                    
                    // Some formats include row number and pathway in separate columns
                    if (row["0"] !== undefined && pathwayCol) {
                        pathwayId = row["0"].toString();
                    }
                    
                    // Determine pathway type
                    let pathwayType = "other";
                    pathwayName = pathwayName.toString(); // Ensure it's a string
                    
                    if (pathwayName.toUpperCase().includes("UNMAPPED")) {
                        pathwayType = "unmapped";
                    } else if (pathwayName.toUpperCase().includes("UNINTEGRATED")) {
                        pathwayType = "unintegrated";
                    } else if (pathwayName.includes("PWY") || pathwayName.includes("pwy")) {
                        pathwayType = "metacyc";
                    }
                    
                    // Extract abundance values - use a more efficient approach
                    const abundanceValues = {};
                    let sumAbundance = 0;
                    
                    allSamples.forEach(sample => {
                        try {
                            let value = row[sample];
                            // Handle various formats and convert to number
                            if (value === undefined || value === null) {
                                value = 0;
                            } else if (typeof value === 'string') {
                                value = parseFloat(value.replace(/,/g, '')) || 0;
                            } else if (typeof value !== 'number') {
                                value = 0; // Default for unexpected types
                            }
                            abundanceValues[sample] = value;
                            sumAbundance += value;
                        } catch (err) {
                            abundanceValues[sample] = 0;
                        }
                    });
                    
                    // Create processed data item with minimal properties to save memory
                    const dataItem = {
                        index: i,
                        id: pathwayId,
                        name: pathwayName,
                        type: pathwayType,
                        abundanceValues: abundanceValues,
                        avgAbundance: sumAbundance / (allSamples.length || 1)
                    };
                    
                    allData.push(dataItem);
                } catch (err) {
                    console.error(`Error processing row ${i}:`, err);
                }
            }
            
            // Update UI after batch processing
            if (endIndex < data.length) {
                // Update processing status
                $('#file-status').html(`<div class="loading">Processing data: ${Math.round(endIndex/data.length*100)}%</div>`);
                
                // Schedule next batch with a small delay to allow UI updates
                setTimeout(() => processBatch(endIndex), 10);
            } else {
                // All batches completed
                finishProcessing();
            }
        }
        
        // Start batch processing from the beginning
        processBatch(0);
        
        function finishProcessing() {
            console.timeEnd('Data Processing');
            
            // Update statistics
            updateStats();
            
            // Initialize the view
            filteredData = [...allData];
            renderPathwaysList();
            updatePagination();
            
            // Reset detail view
            $('#pathway-detail').html('<div class="no-selection"><p>Select a pathway to view abundance across samples</p></div>');
            $('#visualization-panel').hide();
            
            // Update status
            $('#file-status').html(`<div class="file-success">Successfully loaded ${allData.length} pathways</div>`);
        }
    }

// Update dashboard statistics
    function updateStats() {
        $('#total-pathways').text(allData.length);
        $('#total-samples').text(allSamples.length);
        
        const metacycCount = allData.filter(item => item.type === "metacyc").length;
        $('#metacyc-pathways').text(metacycCount);
        
        // Calculate percentage of unmapped reads (if available)
        const unmappedItems = allData.filter(item => item.type === "unmapped");
        if (unmappedItems.length > 0) {
            // Calculate average percentage across samples
            const totalAbundance = allSamples.reduce((total, sample) => {
                return total + allData.reduce((sum, pathway) => sum + pathway.abundanceValues[sample], 0);
            }, 0);
            
            const unmappedAbundance = allSamples.reduce((total, sample) => {
                return total + unmappedItems.reduce((sum, pathway) => sum + pathway.abundanceValues[sample], 0);
            }, 0);
            
            const unmappedPercent = (unmappedAbundance / totalAbundance * 100).toFixed(2);
            $('#unmapped-percent').text(`${unmappedPercent}%`);
        } else {
            $('#unmapped-percent').text("N/A");
        }
    }
    
    // Render the pathways list
    function renderPathwaysList() {
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = Math.min(startIndex + itemsPerPage, filteredData.length);
        const currentPageData = filteredData.slice(startIndex, endIndex);
        
        if (filteredData.length === 0) {
            $('#pathway-list').html('<div class="no-results">No matching pathways found</div>');
            return;
        }
        
        let html = '';
        currentPageData.forEach(item => {
            const isSelected = item.id === selectedPathwayId;
            const typeClass = `type-${item.type}`;
            const typeLabel = item.type.charAt(0).toUpperCase() + item.type.slice(1);
            
            html += `
                <div class="pathway-item ${isSelected ? 'selected' : ''}" data-id="${item.id}" data-index="${item.index}">
                    <div class="pathway-header">
                        <span class="pathway-id">${item.id}</span>
                        <span class="pathway-name">${item.name}</span>
                    </div>
                    <div class="pathway-meta">
                        <span class="pathway-type ${typeClass}">${typeLabel}</span>
                    </div>
                </div>
            `;
        });
        
        $('#pathway-list').html(html);
    }
    
    // Update pagination controls
    function updatePagination() {
        const totalPages = Math.ceil(filteredData.length / itemsPerPage);
        
        if (totalPages <= 1) {
            $('#pagination').empty();
            return;
        }
        
        let paginationHtml = `
            <button id="prev-page" ${currentPage === 1 ? 'disabled' : ''}>
                &laquo;
            </button>
        `;
        
        // Display limited number of page buttons with ellipsis
        const maxVisibleButtons = 5;
        let startPage = Math.max(1, currentPage - Math.floor(maxVisibleButtons / 2));
        let endPage = Math.min(totalPages, startPage + maxVisibleButtons - 1);
        
        if (endPage - startPage + 1 < maxVisibleButtons) {
            startPage = Math.max(1, endPage - maxVisibleButtons + 1);
        }
        
        if (startPage > 1) {
            paginationHtml += `<button class="page-number" data-page="1">1</button>`;
            if (startPage > 2) {
                paginationHtml += `<button disabled>...</button>`;
            }
        }
        
        for (let i = startPage; i <= endPage; i++) {
            paginationHtml += `
                <button class="page-number ${i === currentPage ? 'active' : ''}" data-page="${i}">
                    ${i}
                </button>
            `;
        }
        
        if (endPage < totalPages) {
            if (endPage < totalPages - 1) {
                paginationHtml += `<button disabled>...</button>`;
            }
            paginationHtml += `<button class="page-number" data-page="${totalPages}">${totalPages}</button>`;
        }
        
        paginationHtml += `
            <button id="next-page" ${currentPage === totalPages ? 'disabled' : ''}>
                &raquo;
            </button>
        `;
        
        $('#pagination').html(paginationHtml);
    }

// Display pathway details
    function showPathwayDetail(pathwayId, pathwayIndex) {
        console.log(`Showing details for pathway: ${pathwayId}, index: ${pathwayIndex}`);
        selectedPathwayId = pathwayId;
        
        // Find pathway by ID or by index if ID lookup fails
        let pathway = allData.find(item => item.id === pathwayId);
        
        // Fallback to index if ID lookup fails (this is crucial)
        if (!pathway && pathwayIndex !== undefined) {
            pathway = allData.find(item => item.index == pathwayIndex);
        }
        
        // If still not found, show error
        if (!pathway) {
            console.error(`Could not find pathway with id ${pathwayId} or index ${pathwayIndex}`);
            $('#pathway-detail').html('<div class="error">Error: Could not find pathway details</div>');
            return;
        }
        
        console.log("Found pathway:", pathway);
        
        // Create abundance table
        let tableHtml = `
            <table class="abundance-table">
                <thead>
                    <tr>
                        <th>Sample</th>
                        <th>Abundance</th>
                    </tr>
                </thead>
                <tbody>
        `;
        
        // Sort samples by abundance to show highest values first
        const sortedSamples = [...allSamples].sort((a, b) => {
            return pathway.abundanceValues[b] - pathway.abundanceValues[a];
        }).slice(0, 100); // Limit to top 100 samples for performance
        
        sortedSamples.forEach(sample => {
            const abundance = pathway.abundanceValues[sample];
            tableHtml += `
                <tr>
                    <td>${sample}</td>
                    <td>${formatNumber(abundance)}</td>
                </tr>
            `;
        });
        
        tableHtml += `</tbody></table>`;
        
        // Create detailed view
        const typeLabel = pathway.type.charAt(0).toUpperCase() + pathway.type.slice(1);
        const detailHtml = `
            <div class="detail-header">
                <h2>${pathway.name}</h2>
                <span class="pathway-type type-${pathway.type}">${typeLabel}</span>
            </div>
            <div class="detail-info">
                <h3>Pathway Details</h3>
                <p><strong>ID:</strong> ${pathway.id}</p>
                <p><strong>Type:</strong> ${typeLabel}</p>
                <p><strong>Average Abundance:</strong> ${formatNumber(pathway.avgAbundance)}</p>
            </div>
            <div class="detail-info">
                <h3>Abundance Across Samples (Top 100)</h3>
                ${tableHtml}
            </div>
        `;
        
        $('#pathway-detail').html(detailHtml);
        
        // Show and update chart
        $('#visualization-panel').show();
        updateChart(pathway);
        
        // Update selected state in list
        $('.pathway-item').removeClass('selected');
        $(`.pathway-item[data-id="${pathwayId}"]`).addClass('selected');
    }
    
    // Update chart visualization with optimizations for large datasets
    function updateChart(pathway) {
        const chartType = $('#chart-type').val();
        const sampleLimit = parseInt($('#sample-limit').val()) || 20;
        
        // Sort samples by abundance for the chart
        let sortedSamples = [...allSamples].sort((a, b) => {
            return pathway.abundanceValues[b] - pathway.abundanceValues[a];
        });
        
        // Limit samples to improve performance
        const samplesToDisplay = sampleLimit === 0 ? 
            sortedSamples.slice(0, Math.min(20, sortedSamples.length)) : 
            sortedSamples.slice(0, Math.min(sampleLimit, sortedSamples.length));
        
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
        abundanceChart = new Chart(ctx, chartConfig);
    }
    
    // Filter data based on search and filter criteria
    function filterData() {
        const searchTerm = $('#search-input').val().toLowerCase().trim();
        const pathwayFilter = $('#pathway-filter').val();
        const sortBy = $('#sort-by').val();
        
        // Apply filters
        filteredData = allData.filter(item => {
            const matchesSearch = !searchTerm || 
                                  item.name.toLowerCase().includes(searchTerm) || 
                                  item.id.toString().includes(searchTerm);
            
            const matchesPathwayType = pathwayFilter === 'all' || 
                                       (pathwayFilter === 'PWY' && item.type === 'metacyc') ||
                                       item.name.includes(pathwayFilter);
            
            return matchesSearch && matchesPathwayType;
        });
        
        // Apply sorting
        if (sortBy === 'index') {
            filteredData.sort((a, b) => a.index - b.index);
        } else if (sortBy === 'name') {
            filteredData.sort((a, b) => a.name.localeCompare(b.name));
        } else if (sortBy === 'abundance') {
            filteredData.sort((a, b) => b.avgAbundance - a.avgAbundance);
        }
        
        // Reset to first page when filtering
        currentPage = 1;
        renderPathwaysList();
        updatePagination();
        
        // Clear selection if it's no longer in filtered results
        if (selectedPathwayId) {
            const stillExists = filteredData.some(item => item.id === selectedPathwayId);
            if (!stillExists) {
                $('#pathway-detail').html('<div class="no-selection"><p>Select a pathway to view details</p></div>');
                $('#visualization-panel').hide();
                selectedPathwayId = null;
            }
        }
    }
    
    // Generate CSV for download
    function downloadData(pathway) {
        if (!pathway) return;
        
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
        
        // Create download link
        const encodedUri = encodeURI("data:text/csv;charset=utf-8," + csvContent);
        const link = document.createElement("a");
        link.setAttribute("href", encodedUri);
        link.setAttribute("download", `${pathway.name.replace(/[/\\?%*:|"<>]/g, '_')}_abundance.csv`);
        document.body.appendChild(link);
        
        // Trigger download
        link.click();
        document.body.removeChild(link);
    }
    
    // Event Handlers
    
    // Pathway item click
    $(document).on('click', '.pathway-item', function() {
        const pathwayId = $(this).data('id');
        const pathwayIndex = $(this).data('index');
        showPathwayDetail(pathwayId, pathwayIndex);
    });
    
    // Search button click
    $('#search-btn').click(function() {
        filterData();
    });
    
    // Search input enter key
    $('#search-input').on('keyup', function(e) {
        if (e.key === 'Enter') {
            filterData();
        }
    });
    
    // Reset button click
    $('#reset-btn').click(function() {
        $('#search-input').val('');
        $('#pathway-filter').val('all');
        $('#sort-by').val('index');
        filterData();
    });
    
    // Pathway type filter change
    $('#pathway-filter').change(function() {
        filterData();
    });
    
    // Sort by filter change
    $('#sort-by').change(function() {
        filterData();
    });
    
    // Chart type change
    $('#chart-type').change(function() {
        if (selectedPathwayId) {
            const pathway = allData.find(item => item.id === selectedPathwayId);
            if (pathway) {
                updateChart(pathway);
            }
        }
    });
    
    // Sample limit change
    $('#sample-limit').change(function() {
        if (selectedPathwayId) {
            const pathway = allData.find(item => item.id === selectedPathwayId);
            if (pathway) {
                updateChart(pathway);
            }
        }
    });
    
    // Download button click
    $('#download-btn').click(function() {
        if (selectedPathwayId) {
            const pathway = allData.find(item => item.id === selectedPathwayId);
            if (pathway) {
                downloadData(pathway);
            }
        }
    });
    
    // Pagination navigation
    $(document).on('click', '.page-number', function() {
        currentPage = parseInt($(this).data('page'));
        renderPathwaysList();
        updatePagination();
        window.scrollTo(0, 0);
    });
    
    $(document).on('click', '#prev-page:not([disabled])', function() {
        currentPage--;
        renderPathwaysList();
        updatePagination();
        window.scrollTo(0, 0);
    });
    
    $(document).on('click', '#next-page:not([disabled])', function() {
        currentPage++;
        renderPathwaysList();
        updatePagination();
        window.scrollTo(0, 0);
    });
    
    // Initialize the file upload interface
    $('#file-upload-text').text('Choose CSV file');
});
