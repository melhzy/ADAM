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
        
        // Validate the required columns exist
        const requiredColumns = ["Sample ID", "Alzheimers", "Formatted Summary"];
        const missingColumns = requiredColumns.filter(col => !data[0] || typeof data[0][col] === 'undefined');
        
        if (missingColumns.length > 0) {
            $('#file-status').html(`<div class="file-error">Missing required columns: ${missingColumns.join(", ")}</div>`);
            return;
        }
        
        // Filter out rows with missing required data
        allData = data.filter(item => item["Sample ID"] && item["Alzheimers"] && item["Formatted Summary"]);
        console.log(`Loaded ${allData.length} valid records from CSV`);
        
        if (allData.length === 0) {
            $('#samples-list').html('<div class="no-results">No valid data found in the CSV file</div>');
            return;
        }
        
        filteredData = [...allData];
        renderSamplesList();
        updatePagination();
        
        // Reset detail view
        $('#sample-detail').html('<div class="no-selection"><p>Select a sample to view details</p></div>');
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
            
            html += `
                <div class="sample-item ${isSelected ? 'selected' : ''}" data-id="${item["Sample ID"]}">
                    <div class="sample-header">
                        <span class="sample-id">${item["Sample ID"]}</span>
                        <span class="alzheimers-badge alzheimers-${item["Alzheimers"] === "Yes" ? 'yes' : 'no'}">
                            ${item["Alzheimers"]}
                        </span>
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
        
        const detailHtml = `
            <div class="detail-header">
                <h2>Sample ID: ${sample["Sample ID"]}</h2>
                <span class="alzheimers-badge alzheimers-${sample["Alzheimers"] === "Yes" ? 'yes' : 'no'}">
                    Alzheimers: ${sample["Alzheimers"]}
                </span>
            </div>
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
        const alzheimersFilter = $('#alzheimers-filter').val();
        
        filteredData = allData.filter(item => {
            const matchesSearch = !searchTerm || 
                                 item["Sample ID"].toLowerCase().includes(searchTerm);
            
            const matchesAlzheimers = alzheimersFilter === 'all' || 
                                     item["Alzheimers"] === alzheimersFilter;
            
            return matchesSearch && matchesAlzheimers;
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
        $('#alzheimers-filter').val('all');
        filterData();
    });
    
    // Alzheimers filter change
    $('#alzheimers-filter').change(function() {
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
    
    // Initialize the file upload interface
    $('#file-upload-text').text('Choose CSV file');
});