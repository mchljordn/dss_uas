#!/usr/bin/env python3
"""
STREAMLIT DEPLOYMENT READINESS CHECKLIST
Sales Forecasting Decision Support System
"""

import os
from pathlib import Path

def check_files():
    """Verify all necessary files exist"""
    base_path = Path(__file__).parent
    
    required_files = {
        'Python Files': [
            'app.py',
            'run_cells.py',
        ],
        'Data Files': [
            'Sales Transaction v.4a(modified).csv',
            'sales_forecast_30days.csv',
            'product_classification_abc.csv',
            'seasonal_analysis.csv',
        ],
        'Jupyter': [
            'mining.ipynb',
        ],
        'Configuration': [
            'requirements.txt',
            '.gitignore',
        ],
        'Streamlit Config': [
            '.streamlit/config.toml',
        ],
        'Documentation': [
            'README.md',
            'POSTER_ILMIAH.md',
            'RINGKASAN_EKSEKUTIF.md',
            'POSTER_VISUAL_LAYOUT.md',
            'CHECKLIST_FINAL.md',
            'TALKING_POINTS.md',
            'INDEX.md',
            'DEPLOYMENT_GUIDE.md',
            'STREAMLIT_FIX_SUMMARY.md',
        ],
    }
    
    print("=" * 70)
    print("🚀 STREAMLIT DEPLOYMENT READINESS CHECK")
    print("=" * 70)
    print()
    
    all_good = True
    total_files = 0
    found_files = 0
    
    for category, files in required_files.items():
        print(f"\n📁 {category}:")
        print("-" * 70)
        
        for file in files:
            filepath = base_path / file
            exists = filepath.exists()
            status = "✅" if exists else "❌"
            
            if exists:
                found_files += 1
                try:
                    size = filepath.stat().st_size
                    size_str = f"({size:,} bytes)"
                except:
                    size_str = "(size unknown)"
                print(f"  {status} {file:<40} {size_str}")
            else:
                print(f"  {status} {file:<40} MISSING!")
                all_good = False
            
            total_files += 1
    
    print("\n" + "=" * 70)
    print(f"SUMMARY: {found_files}/{total_files} files found")
    print("=" * 70)
    
    # Check models folder
    models_path = base_path / 'models'
    print(f"\n🤖 MODELS FOLDER:")
    if models_path.exists():
        model_files = list(models_path.glob('*.pkl'))
        print(f"  ✅ models/ folder exists")
        if model_files:
            print(f"  ✅ Found {len(model_files)} trained models:")
            for mf in model_files:
                size = mf.stat().st_size
                print(f"     - {mf.name} ({size:,} bytes)")
        else:
            print(f"  ⚠️  models/ folder is empty (will be auto-trained on first run)")
    else:
        print(f"  ⚠️  models/ folder not found (will be created on first run)")
    
    # Check dependencies
    print(f"\n📦 DEPENDENCIES:")
    req_file = base_path / 'requirements.txt'
    if req_file.exists():
        print(f"  ✅ requirements.txt exists")
        with open(req_file) as f:
            deps = f.read().strip().split('\n')
            print(f"  ✅ {len(deps)} dependencies specified:")
            for dep in deps:
                print(f"     - {dep}")
    else:
        print(f"  ❌ requirements.txt missing!")
        all_good = False
    
    print("\n" + "=" * 70)
    
    if all_good:
        print("✅ ALL CHECKS PASSED - READY FOR DEPLOYMENT!")
    else:
        print("⚠️  SOME FILES MISSING - CHECK ABOVE")
    
    print("=" * 70)
    print()
    
    return all_good

if __name__ == "__main__":
    check_files()
