from flask import Flask, render_template, request, jsonify, send_file
from flask_sqlalchemy import SQLAlchemy
import os
import hashlib
from collections import defaultdict
import logging
import pathlib
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import uuid
from urllib.parse import urlparse

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:Chinu%40248@localhost/carbon_footprint'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Analysis Report Model
class AnalysisReport(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(500), nullable=False)
    domain = db.Column(db.String(255), nullable=False)
    data_size_mb = db.Column(db.Float, nullable=False)
    green_hosted = db.Column(db.Boolean, nullable=False)
    co2_emission = db.Column(db.Float, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<AnalysisReport {self.url}>'

# Create database tables
with app.app_context():
    db.create_all()

# Store generated reports
reports = {}

def get_file_size(file_path):
    try:
        return os.path.getsize(file_path)
    except (OSError, PermissionError) as e:
        logger.error(f"Error getting file size for {file_path}: {str(e)}")
        return 0

def normalize_path(path):
    if not path:
        return ''
    return os.path.normpath(path)

def check_directory(directory):
    if not directory:
        return False
    return os.path.exists(directory) and os.path.isdir(directory)

def calculate_file_hash(file_path):
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except (OSError, PermissionError) as e:
        logger.error(f"Error calculating hash for {file_path}: {str(e)}")
        return None

def calculate_folder_hash(folder_path):
    try:
        hash_md5 = hashlib.md5()
        for root, _, files in os.walk(folder_path):
            for file in sorted(files):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'rb') as f:
                        for chunk in iter(lambda: f.read(4096), b''):
                            hash_md5.update(chunk)
                except (OSError, PermissionError):
                    continue
        return hash_md5.hexdigest()
    except (OSError, PermissionError) as e:
        logger.error(f"Error calculating hash for folder {folder_path}: {str(e)}")
        return None

def find_duplicates(directory):
    try:
        file_hashes = {}
        folder_hashes = {}
        total_size = 0
        potential_savings = 0
        
        for root, dirs, files in os.walk(directory):
            # Process files
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    file_size = get_file_size(file_path)
                    total_size += file_size
                    file_hash = calculate_file_hash(file_path)
                    if file_hash:
                        if file_hash in file_hashes:
                            file_hashes[file_hash].append(file_path)
                            potential_savings += file_size
                        else:
                            file_hashes[file_hash] = [file_path]
                except (OSError, PermissionError):
                    continue
            
            # Process folders
            for folder in dirs:
                folder_path = os.path.join(root, folder)
                try:
                    folder_hash = calculate_folder_hash(folder_path)
                    if folder_hash:
                        if folder_hash in folder_hashes:
                            folder_hashes[folder_hash].append(folder_path)
                        else:
                            folder_hashes[folder_hash] = [folder_path]
                except (OSError, PermissionError):
                    continue
        
        # Filter out non-duplicates
        duplicates = {
            'files': {h: paths for h, paths in file_hashes.items() if len(paths) > 1},
            'folders': {h: paths for h, paths in folder_hashes.items() if len(paths) > 1}
        }
        
        return duplicates, potential_savings, total_size
    except Exception as e:
        logger.error(f"Error finding duplicates: {str(e)}")
        return None, 0, 0

def calculate_carbon_emissions(file_size_mb, file_type):
    """Calculate carbon emissions based on file size and type."""
    # Average carbon emissions per GB of storage (in kg CO2e)
    STORAGE_EMISSIONS = 0.2  # kg CO2e per GB per year
    
    # Additional emissions factors for different file types
    EMISSIONS_FACTORS = {
        'images': 1.0,      # Base factor
        'videos': 1.5,      # Higher due to larger size and processing needs
        'documents': 0.8,   # Lower due to smaller size
        'audio': 1.2,       # Moderate due to streaming potential
        'archives': 0.9,    # Moderate due to compression
        'code': 0.7,        # Lower due to text-based nature
        'other': 1.0        # Base factor
    }
    
    # Convert MB to GB
    size_gb = file_size_mb / 1024
    
    # Calculate base emissions from storage
    base_emissions = size_gb * STORAGE_EMISSIONS
    
    # Apply file type factor
    factor = EMISSIONS_FACTORS.get(file_type, 1.0)
    total_emissions = base_emissions * factor
    
    return total_emissions

def analyze_file_types(directory):
    """Analyze files in a directory and categorize them by type."""
    file_types = {
        'images': {'extensions': ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp', '.svg'],
                  'count': 0, 'total_size': 0, 'carbon_emissions': 0},
        'documents': {'extensions': ['.pdf', '.doc', '.docx', '.txt', '.rtf', '.odt', '.xls', '.xlsx', '.ppt', '.pptx'],
                     'count': 0, 'total_size': 0, 'carbon_emissions': 0},
        'videos': {'extensions': ['.mp4', '.avi', '.mov', '.wmv', '.flv', '.mkv', '.webm'],
                  'count': 0, 'total_size': 0, 'carbon_emissions': 0},
        'audio': {'extensions': ['.mp3', '.wav', '.ogg', '.flac', '.aac', '.wma'],
                 'count': 0, 'total_size': 0, 'carbon_emissions': 0},
        'archives': {'extensions': ['.zip', '.rar', '.7z', '.tar', '.gz'],
                    'count': 0, 'total_size': 0, 'carbon_emissions': 0},
        'code': {'extensions': ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.php', '.rb', '.go'],
                'count': 0, 'total_size': 0, 'carbon_emissions': 0},
        'other': {'extensions': [], 'count': 0, 'total_size': 0, 'carbon_emissions': 0}
    }
    
    try:
        for root, _, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                try:
                    if os.path.isfile(file_path):
                        file_size = get_file_size(file_path)
                        file_ext = os.path.splitext(file)[1].lower()
                        
                        categorized = False
                        for category, data in file_types.items():
                            if category != 'other' and file_ext in data['extensions']:
                                data['count'] += 1
                                data['total_size'] += file_size
                                data['carbon_emissions'] += calculate_carbon_emissions(file_size / (1024 * 1024), category)
                                categorized = True
                                break
                        
                        if not categorized:
                            file_types['other']['count'] += 1
                            file_types['other']['total_size'] += file_size
                            file_types['other']['carbon_emissions'] += calculate_carbon_emissions(file_size / (1024 * 1024), 'other')
                except (IOError, OSError) as e:
                    print(f"Error processing file {file_path}: {str(e)}")
                    continue
        
        # Calculate percentages and convert sizes to MB
        total_files = sum(category['count'] for category in file_types.values())
        total_size = sum(category['total_size'] for category in file_types.values())
        total_emissions = sum(category['carbon_emissions'] for category in file_types.values())
        
        for category in file_types.values():
            if total_files > 0:
                category['percentage'] = (category['count'] / total_files) * 100
            else:
                category['percentage'] = 0
            category['size_mb'] = category['total_size'] / (1024 * 1024)  # Convert to MB
            if total_emissions > 0:
                category['emissions_percentage'] = (category['carbon_emissions'] / total_emissions) * 100
            else:
                category['emissions_percentage'] = 0
        
        return {
            'file_types': file_types,
            'total_files': total_files,
            'total_size_mb': total_size / (1024 * 1024),  # Convert to MB
            'total_emissions': total_emissions
        }
    except Exception as e:
        print(f"Error analyzing directory {directory}: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/history')
def history():
    """Display analysis history with optional filters, pagination, and sorting."""
    # Get filter parameters from request
    domain_filter = request.args.get('domain', '')
    start_date = request.args.get('start_date', '')
    end_date = request.args.get('end_date', '')
    page = request.args.get('page', 1, type=int)
    per_page = 10
    
    # Get sorting parameters
    sort_by = request.args.get('sort', 'timestamp')
    sort_order = request.args.get('order', 'desc')
    
    # Start with base query
    query = AnalysisReport.query
    
    # Apply domain filter if provided
    if domain_filter:
        query = query.filter(AnalysisReport.domain.ilike(f'%{domain_filter}%'))
    
    # Apply date range filter if provided
    if start_date:
        try:
            start_date = datetime.strptime(start_date, '%Y-%m-%d')
            query = query.filter(AnalysisReport.timestamp >= start_date)
        except ValueError:
            pass
    
    if end_date:
        try:
            end_date = datetime.strptime(end_date, '%Y-%m-%d')
            # Add one day to include the end date
            end_date = end_date + timedelta(days=1)
            query = query.filter(AnalysisReport.timestamp < end_date)
        except ValueError:
            pass
    
    # Apply sorting
    sort_column = {
        'url': AnalysisReport.url,
        'domain': AnalysisReport.domain,
        'data_size': AnalysisReport.data_size_mb,
        'green_hosted': AnalysisReport.green_hosted,
        'co2_emission': AnalysisReport.co2_emission,
        'timestamp': AnalysisReport.timestamp
    }.get(sort_by, AnalysisReport.timestamp)
    
    if sort_order == 'asc':
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())
    
    # Paginate results
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)
    
    return render_template('history.html', 
                         reports=pagination.items,
                         pagination=pagination,
                         domain_filter=domain_filter,
                         start_date=start_date,
                         end_date=end_date,
                         sort_by=sort_by,
                         sort_order=sort_order)

@app.route('/file-manager')
def file_manager():
    return render_template('file_manager.html')

@app.route('/verify-directory', methods=['POST'])
def verify_directory():
    data = request.get_json()
    directory = data.get('directory')
    
    if not directory or not check_directory(directory):
        return jsonify({'valid': False})
    
    return jsonify({'valid': True})

@app.route('/scan-duplicates', methods=['POST'])
def scan_duplicates():
    try:
        data = request.get_json()
        directory = data.get('directory', '')
        logger.debug(f"Scanning directory: {directory}")
        
        if not directory:
            return jsonify({'error': 'No directory provided'}), 400
        
        if not os.path.exists(directory):
            logger.error(f"Directory does not exist: {directory}")
            return jsonify({'error': 'Directory does not exist'}), 400
        
        if not os.path.isdir(directory):
            logger.error(f"Path is not a directory: {directory}")
            return jsonify({'error': 'Path is not a directory'}), 400
        
        duplicates, potential_savings, total_size = find_duplicates(directory)
        
        if duplicates is None:
            return jsonify({'error': 'Error scanning directory'}), 500
        
        return jsonify({
            'duplicates': duplicates,
            'total_duplicates': sum(len(files) - 1 for files in duplicates['files'].values()) + 
                              sum(len(folders) - 1 for folders in duplicates['folders'].values()),
            'potential_savings': potential_savings,
            'total_size': total_size
        })
    except Exception as e:
        logger.error(f"Error in scan_duplicates: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-folder', methods=['POST'])
def analyze_folder():
    try:
        data = request.get_json()
        directory = data.get('directory', '')
        
        if not directory:
            return jsonify({'error': 'No directory provided'}), 400
        
        if not os.path.exists(directory):
            return jsonify({'error': 'Directory does not exist'}), 400
        
        if not os.path.isdir(directory):
            return jsonify({'error': 'Path is not a directory'}), 400

        # Get file type analysis
        file_analysis = analyze_file_types(directory)
        if not file_analysis:
            return jsonify({'error': 'Error analyzing file types'}), 500

        # Get duplicate files
        duplicates, potential_savings, total_size = find_duplicates(directory)
        if duplicates is None:
            return jsonify({'error': 'Error finding duplicates'}), 500

        # Build folder structure
        folder_structure = {'name': os.path.basename(directory), 'files': 0, 'size': 0, 'subfolders': {}}
        total_files = 0
        total_folders = 0

        def build_folder_structure(current_path, current_structure):
            nonlocal total_files, total_folders
            try:
                for item in os.listdir(current_path):
                    item_path = os.path.join(current_path, item)
                    if os.path.isdir(item_path):
                        total_folders += 1
                        subfolder = {'name': item, 'files': 0, 'size': 0, 'subfolders': {}}
                        current_structure['subfolders'][item] = subfolder
                        build_folder_structure(item_path, subfolder)
                    else:
                        try:
                            size = get_file_size(item_path)
                            total_files += 1
                            current_structure['files'] += 1
                            current_structure['size'] += size
                        except (OSError, PermissionError):
                            continue
            except (OSError, PermissionError):
                pass

        build_folder_structure(directory, folder_structure)

        # Prepare file type counts for frontend
        file_types = {}
        for category, data in file_analysis['file_types'].items():
            file_types[category] = data['count']

        # Prepare duplicate sizes
        duplicate_sizes = {}
        for hash_value, files in duplicates['files'].items():
            if len(files) > 1:
                duplicate_sizes[hash_value] = get_file_size(files[0])

        return jsonify({
            'total_files': total_files,
            'total_folders': total_folders,
            'total_size': total_size,
            'file_types': file_types,
            'duplicates': duplicates,
            'duplicate_sizes': duplicate_sizes,
            'folder_structure': folder_structure
        })

    except Exception as e:
        logger.error(f"Error in analyze_folder: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/delete-file', methods=['POST'])
def delete_file():
    try:
        data = request.get_json()
        file_path = data.get('file_path')
        
        if not file_path:
            return jsonify({'error': 'No file path provided'}), 400
            
        if not os.path.exists(file_path):
            return jsonify({'error': 'File does not exist'}), 404
            
        if not os.path.isfile(file_path):
            return jsonify({'error': 'Path is not a file'}), 400
            
        try:
            os.remove(file_path)
            return jsonify({'success': True, 'message': 'File deleted successfully'})
        except Exception as e:
            logger.error(f'Error deleting file {file_path}: {str(e)}')
            return jsonify({'error': str(e)}), 500
            
    except Exception as e:
        logger.error(f'Error in delete_file: {str(e)}')
        return jsonify({'error': str(e)}), 500

@app.route('/email-cleanup')
def email_cleanup():
    return render_template('email_cleanup.html')

def calculate_email_carbon_impact(email_size_mb, attachment_size_mb):
    """Calculate carbon emissions for email storage and processing."""
    # Average carbon emissions per GB of email storage (in kg CO2e)
    STORAGE_EMISSIONS = 0.2  #  kg CO2e per GB per year
    
    # Additional emissions for email processing and transmission
    PROCESSING_EMISSIONS = 0.0004  # kg CO2e per MB processed
    
    # Energy consumption per email (in kWh)
    ENERGY_PER_EMAIL = 0.0003  # kWh per email
    
    # Carbon intensity of electricity (kg CO2e per kWh)
    CARBON_INTENSITY = 0.475  # kg CO2e per kWh
    
    # Convert MB to GB
    total_size_gb = (email_size_mb + attachment_size_mb) / 1024
    
    # Calculate storage emissions
    storage_emissions = total_size_gb * STORAGE_EMISSIONS
    
    # Calculate processing emissions
    processing_emissions = (email_size_mb + attachment_size_mb) * PROCESSING_EMISSIONS
    
    # Calculate energy consumption
    energy_consumption = ENERGY_PER_EMAIL * CARBON_INTENSITY
    
    # Total emissions
    total_emissions = storage_emissions + processing_emissions + energy_consumption
    
    return {
        'storage_emissions': storage_emissions,
        'processing_emissions': processing_emissions,
        'energy_emissions': energy_consumption,
        'total_emissions': total_emissions
    }

def get_emissions_equivalent(emissions_kg):
    """Convert carbon emissions to real-world equivalents."""
    # Average car emissions per km
    CAR_EMISSIONS = 0.2  # kg CO2e per km
    
    # Average tree absorption per year
    TREE_ABSORPTION = 21.77  # kg CO2e per tree per year
    
    # Average smartphone charging emissions
    SMARTPHONE_CHARGING = 0.0005  # kg CO2e per charge
    
    # Average light bulb emissions per hour
    LIGHT_BULB = 0.0004  # kg CO2e per hour
    
    car_km = emissions_kg / CAR_EMISSIONS
    trees_needed = emissions_kg / TREE_ABSORPTION
    smartphone_charges = emissions_kg / SMARTPHONE_CHARGING
    light_bulb_hours = emissions_kg / LIGHT_BULB
    
    return {
        'car_km': car_km,
        'trees_needed': trees_needed,
        'smartphone_charges': smartphone_charges,
        'light_bulb_hours': light_bulb_hours
    }

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        url = request.form.get('url')
        if not url:
            return jsonify({'error': 'URL is required'}), 400

        # Perform website analysis
        analysis_data = {
            'url': url,
            'page_size': 2.5,  # Example value
            'is_green': True,  # Example value
            'co2_emissions': 0.1234,  # Example value
            'suggestions': [
                'Compress images to reduce page size',
                'Minify CSS and JavaScript files',
                'Enable browser caching',
                'Use a content delivery network (CDN)',
                'Optimize server response time'
            ]
        }

        # Save to database
        report = AnalysisReport(
            url=url,
            domain=extract_domain(url),
            data_size_mb=analysis_data['page_size'],
            green_hosted=analysis_data['is_green'],
            co2_emission=analysis_data['co2_emissions']
        )
        db.session.add(report)
        db.session.commit()

        # Generate PDF report
        report_id = generate_pdf_report(analysis_data)
        analysis_data['report_id'] = report_id

        return render_template('result.html', analysis=analysis_data)

    except Exception as e:
        logger.error(f'Error analyzing website: {str(e)}')
        return jsonify({'error': str(e)}), 500

def extract_domain(url):
    """Extract domain from URL."""
    parsed = urlparse(url)
    return parsed.netloc

def generate_pdf_report(analysis_data):
    """Generate a PDF report for the website analysis."""
    # Create a unique ID for the report
    report_id = str(uuid.uuid4())
    
    # Create the PDF file
    filename = f"report_{report_id}.pdf"
    doc = SimpleDocTemplate(filename, pagesize=letter)
    
    # Container for the PDF elements
    elements = []
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=18,
        spaceAfter=20
    )
    normal_style = styles['Normal']
    
    # Add title
    elements.append(Paragraph("Carbon Footprint Analysis Report", title_style))
    elements.append(Spacer(1, 20))
    
    # Add website information
    elements.append(Paragraph("Website Information", heading_style))
    website_data = [
        ["Website URL:", analysis_data['url']],
        ["Page Size:", f"{analysis_data['page_size']:.2f} MB"],
        ["Hosting Status:", "Green" if analysis_data['is_green'] else "Not Green"],
        ["CO₂ Emissions:", f"{analysis_data['co2_emissions']:.4f} g CO₂ per page load"]
    ]
    
    # Create table for website information
    table = Table(website_data, colWidths=[2*inch, 4*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1) , 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 12),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('BACKGROUND', (1, 0), (1, -1), colors.white),
    ]))
    elements.append(table)
    elements.append(Spacer(1, 20))
    
    # Add optimization suggestions
    elements.append(Paragraph("Optimization Suggestions", heading_style))
    for suggestion in analysis_data['suggestions']:
        elements.append(Paragraph(f"• {suggestion}", normal_style))
        elements.append(Spacer(1, 5))
    
    # Build the PDF
    doc.build(elements)
    
    # Store the report
    reports[report_id] = filename
    
    return report_id

@app.route('/download/<report_id>')
def download_report(report_id):
    """Download the generated PDF report."""
    if report_id not in reports:
        return jsonify({'error': 'Report not found'}), 404
    
    filename = reports[report_id]
    try:
        return send_file(
            filename,
            as_attachment=True,
            download_name=f"carbon_footprint_report_{report_id}.pdf",
            mimetype='application/pdf'
        )
    finally:
        # Clean up the file after sending
        try:
            import time
            time.sleep(1)  # Wait for 1 second before attempting to delete
            if os.path.exists(filename):
                os.remove(filename)
            if report_id in reports:
                del reports[report_id]
        except Exception as e:
            logger.error(f"Error cleaning up report file: {str(e)}")
            # Don't raise the error, just log it

@app.route('/analyze-email', methods=['POST'])
def analyze_email():
    try:
        data = request.get_json()
        email_size = data.get('email_size', 0)
        attachment_size = data.get('attachment_size', 0)

        # Calculate carbon impact
        emissions = calculate_email_carbon_impact(email_size, attachment_size)
        
        # Calculate real-world equivalents
        equivalents = get_emissions_equivalent(emissions['total_emissions'])

        return jsonify({
            'emissions': emissions,
            'equivalents': equivalents
        })

    except Exception as e:
        logger.error(f'Error analyzing email: {str(e)}')
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5002)
