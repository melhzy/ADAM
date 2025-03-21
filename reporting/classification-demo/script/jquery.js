// jquery.js
$(document).ready(function() {
    // Initialize variables
    let allData = [];
    let filteredData = [];
    let currentPage = 1;
    const itemsPerPage = 10;
    let selectedSampleId = null;
    
    // Initialize markdown parser
    const md = window.markdownit();
    
    // File upload handling
    $('#csv-file-upload').on('change', function(e) {
        const file = e.target.files[0];
        if (!file) return;
        
        // Update the displayed filename
        $('#file-upload-text').text(file.name);
        $('#file-status').html('<div class="loading">Processing file...</div>');
        
        // Read and parse the CSV file
        const reader = new FileReader();
        reader.onload = function(event) {
            const csvData = event.target.result;
            parseCSV(csvData);
        };
        reader.onerror = function() {
            $('#file-status').html('<div class="file-error">Error reading file</div>');
        };
        reader.readAsText(file);
    });
    
    // Automatically load the CSV file on page load
    loadDefaultCSV();
    
    function loadDefaultCSV() {
        $('#file-status').html('<div class="loading">Loading default data file...</div>');
        
        // Use fetch to get the CSV file
        fetch('classification_agent_output_gpt4omini20240718.csv')
            .then(response => {
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                return response.text();
            })
            .then(csvData => {
                parseCSV(csvData);
                $('#file-upload-text').text('classification_agent_output_gpt4omini20240718.csv');
            })
            .catch(error => {
                console.error('Error loading CSV:', error);
                $('#file-status').html('<div class="file-error">Error loading default CSV file. Please upload manually.</div>');
            });
    }
    
    // Parse CSV data
    function parseCSV(csvData) {
        Papa.parse(csvData, {
            header: true,
            skipEmptyLines: true,
            complete: function(results) {
                if (results.errors && results.errors.length > 0) {
                    $('#file-status').html(`<div class="file-error">Error parsing CSV: ${results.errors[0].message}</div>`);
                    return;
                }
                
                processData(results.data);
                $('#file-status').html(`<div class="file-success">Successfully loaded ${results.data.length} records</div>`);
                $('#app-content').show();
            },
            error: function(error) {
                $('#file-status').html(`<div class="file-error">Error parsing CSV: ${error.message}</div>`);
            }
        });
    }
    
    // Process the loaded data
    function processData(data) {
        // Reset current state
        allData = [];
        filteredData = [];
        currentPage = 1;
        selectedSampleId = null;
        
        // Check if data exists and has at least one row
        if (!data || data.length === 0) {
            $('#file-status').html(`<div class="file-error">No data found in the CSV file</div>`);
            return;
        }
        
        // Get the first data row to identify available columns
        const firstRow = data[0];
        const availableColumns = Object.keys(firstRow);
        
        console.log("Available columns:", availableColumns);
        
        // Required columns for our application
        const requiredColumns = ["Sample ID", "Formatted Summary"];
        const missingColumns = requiredColumns.filter(col => !availableColumns.includes(col));
        
        if (missingColumns.length > 0) {
            $('#file-status').html(`<div class="file-error">Missing required columns: ${missingColumns.join(", ")}</div>`);
            return;
        }
        
        // Filter out rows with missing required data
        allData = data.filter(item => item["Sample ID"] && item["Formatted Summary"]);
        console.log(`Loaded ${allData.length} valid records from CSV`);
        
        if (allData.length === 0) {
            $('#samples-list').html('<div class="no-results">No valid data found in the CSV file</div>');
            return;
        }
        
        // Update filter options for Dataset
        if (allData[0]["Dataset"]) {
            updateDatasetFilter(allData);
        }
        
        filteredData = [...allData];
        renderSamplesList();
        updatePagination();
        
        // Reset detail view
        $('#sample-detail').html('<div class="no-selection"><p>Select a sample to view details</p></div>');
    }
    
    // Update Dataset filter options
    function updateDatasetFilter(data) {
        // Get unique dataset values
        const datasets = [...new Set(data.map(item => item["Dataset"]))].filter(Boolean);
        
        if (datasets.length > 0) {
            let options = '<option value="all">All</option>';
            datasets.forEach(dataset => {
                options += `<option value="${dataset}">${dataset}</option>`;
            });
            
            // Check if filter already exists
            if ($('#dataset-filter').length === 0) {
                // Create dataset filter
                $('.filter-options').append(`
                    <div class="filter-group">
                        <label for="dataset-filter">Dataset:</label>
                        <select id="dataset-filter">
                            ${options}
                        </select>
                    </div>
                `);
                
                // Add event listener
                $('#dataset-filter').change(function() {
                    filterData();
                });
            } else {
                // Just update options
                $('#dataset-filter').html(options);
            }
        }
    }
    
    // Render the samples list
    function renderSamplesList() {
        const startIndex = (currentPage - 1) * itemsPerPage;
        const endIndex = Math.min(startIndex + itemsPerPage, filteredData.length);
        const currentPageData = filteredData.slice(startIndex, endIndex);
        
        if (filteredData.length === 0) {
            $('#samples-list').html('<div class="no-results">No matching samples found</div>');
            return;
        }
        
        let html = '';
        currentPageData.forEach(item => {
            const isSelected = item["Sample ID"] === selectedSampleId;
            const dataset = item["Dataset"] ? `<span class="dataset-label">${item["Dataset"]}</span>` : '';
            const prediction = item["Prediction"] ? `<span class="prediction-badge prediction-${item["Prediction"] === "Yes" ? 'yes' : (item["Prediction"] === "No" ? 'no' : 'unknown')}">${item["Prediction"]}</span>` : '';
            const groundTruth = item["Ground Truth"] ? `<span class="truth-badge truth-${item["Ground Truth"] === "Yes" ? 'yes' : (item["Ground Truth"] === "No" ? 'no' : 'unknown')}">GT: ${item["Ground Truth"]}</span>` : '';
            
            html += `
                <div class="sample-item ${isSelected ? 'selected' : ''}" data-id="${item["Sample ID"]}">
                    <div class="sample-header">
                        <span class="sample-id">${item["Sample ID"]}</span>
                        ${dataset}
                    </div>
                    <div class="sample-badges">
                        ${prediction}
                        ${groundTruth}
                    </div>
                    <div class="sample-preview">
                        ${getPreviewText(item["Formatted Summary"])}
                    </div>
                </div>
            `;
        });
        
        $('#samples-list').html(html);
    }
    
    // Get a short preview of the content
    function getPreviewText(content) {
        if (!content) return '';
        
        const strippedText = content.replace(/#{1,6}\s?[^#\n]+/g, '')  // Remove markdown headers
                                    .replace(/\*\*([^*]+)\*\*/g, '$1')   // Remove bold
                                    .replace(/\*([^*]+)\*/g, '$1')       // Remove italic
                                    .replace(/\n/g, ' ')                 // Replace newlines with spaces
                                    .replace(/\s+/g, ' ')                // Normalize whitespace
                                    .trim();
        
        return strippedText.length > 100 ? strippedText.substring(0, 100) + '...' : strippedText;
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
    
    // Display sample details
    function showSampleDetail(sampleId) {
        selectedSampleId = sampleId;
        
        const sample = filteredData.find(item => item["Sample ID"] === sampleId);
        if (!sample) return;
        
        // Render detail view with markdown content
        const renderedContent = md.render(sample["Formatted Summary"]);
        
        let detailHtml = `<div class="detail-header">
            <h2>Sample ID: ${sample["Sample ID"]}</h2>`;
            
        if (sample["Dataset"]) {
            detailHtml += `<span class="dataset-badge">Dataset: ${sample["Dataset"]}</span>`;
        }
        
        detailHtml += `</div>`;
        
        // Add prediction information
        if (sample["Prediction"] || sample["Ground Truth"]) {
            detailHtml += `<div class="classification-info">`;
            
            if (sample["Prediction"]) {
                const predictionClass = sample["Prediction"] === "Yes" ? "prediction-yes" : 
                                      (sample["Prediction"] === "No" ? "prediction-no" : "prediction-unknown");
                detailHtml += `
                    <div class="info-item">
                        <span class="info-label">Prediction:</span>
                        <span class="info-value ${predictionClass}">${sample["Prediction"]}</span>
                    </div>`;
            }
            
            if (sample["Ground Truth"]) {
                const truthClass = sample["Ground Truth"] === "Yes" ? "truth-yes" : 
                                 (sample["Ground Truth"] === "No" ? "truth-no" : "truth-unknown");
                detailHtml += `
                    <div class="info-item">
                        <span class="info-label">Ground Truth:</span>
                        <span class="info-value ${truthClass}">${sample["Ground Truth"]}</span>
                    </div>`;
            }
            
            // Add conclusion if available
            if (sample["Conclusion"]) {
                detailHtml += `
                    <div class="info-item conclusion">
                        <span class="info-label">Conclusion:</span>
                        <span class="info-value">${sample["Conclusion"]}</span>
                    </div>`;
            }
            
            detailHtml += `</div>`;
        }
        
        detailHtml += `
            <div class="detail-content markdown-content">
                ${renderedContent}
            </div>
        `;
        
        $('#sample-detail').html(detailHtml);
        
        // Update selected state in list
        $('.sample-item').removeClass('selected');
        $(`.sample-item[data-id="${sampleId}"]`).addClass('selected');
    }
    
    // Filter data based on search and filter criteria
    function filterData() {
        const searchTerm = $('#search-input').val().toLowerCase().trim();
        const predictionFilter = $('#prediction-filter').val();
        const datasetFilter = $('#dataset-filter').val();
        
        filteredData = allData.filter(item => {
            const matchesSearch = !searchTerm || 
                                 (item["Sample ID"] && item["Sample ID"].toLowerCase().includes(searchTerm));
            
            const matchesPrediction = !predictionFilter || predictionFilter === 'all' || 
                                     (item["Prediction"] && item["Prediction"] === predictionFilter);
            
            const matchesDataset = !datasetFilter || datasetFilter === 'all' || 
                                  (item["Dataset"] && item["Dataset"] === datasetFilter);
            
            return matchesSearch && matchesPrediction && matchesDataset;
        });
        
        // Reset to first page when filtering
        currentPage = 1;
        renderSamplesList();
        updatePagination();
        
        // Clear selection if it's no longer in filtered results
        if (selectedSampleId) {
            const stillExists = filteredData.some(item => item["Sample ID"] === selectedSampleId);
            if (!stillExists) {
                $('#sample-detail').html('<div class="no-selection"><p>Select a sample to view details</p></div>');
                selectedSampleId = null;
            }
        }
    }
    
    // Event Handlers
    
    // Sample item click
    $(document).on('click', '.sample-item', function() {
        const sampleId = $(this).data('id');
        showSampleDetail(sampleId);
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
        if ($('#prediction-filter').length) $('#prediction-filter').val('all');
        if ($('#dataset-filter').length) $('#dataset-filter').val('all');
        filterData();
    });
    
    // Filter changes
    $(document).on('change', '#prediction-filter', function() {
        filterData();
    });
    
    // Pagination navigation
    $(document).on('click', '.page-number', function() {
        currentPage = parseInt($(this).data('page'));
        renderSamplesList();
        updatePagination();
        window.scrollTo(0, 0);
    });
    
    $(document).on('click', '#prev-page:not([disabled])', function() {
        currentPage--;
        renderSamplesList();
        updatePagination();
        window.scrollTo(0, 0);
    });
    
    $(document).on('click', '#next-page:not([disabled])', function() {
        currentPage++;
        renderSamplesList();
        updatePagination();
        window.scrollTo(0, 0);
    });
});