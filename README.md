# 📊 Sales Forecasting Decision Support System

## 📖 Ringkasan Proyek

Sistem Decision Support System (DSS) untuk prediksi penjualan menggunakan **Time Series Forecasting** dengan tiga metode machine learning. Proyek ini menganalisis data transaksi penjualan untuk memberikan prediksi revenue 30 hari ke depan, rekomendasi stok inventory, dan insights bisnis melalui dashboard interaktif.

**Dataset**: 527,765 transaksi penjualan (374 hari data harian)  
**Output**: Prediksi revenue, klasifikasi produk ABC, rekomendasi stok, dan 5 file CSV untuk integrasi sistem

---

## 🔧 Metode yang Digunakan

### **1. Time Series Forecasting Models**

| Model | Metode | Hasil |
|-------|--------|-------|
| **Moving Average (MA)** | Rata-rata bergerak 7 hari sebagai baseline | MAE: 89,016.62 |
| **Exponential Smoothing** | Holt-Winters dengan trend + seasonal (periode 7 hari) | **MAE: 83,300.74** ✅ (Terbaik) |
| **ARIMA(1,1,2)** | Autoregressive Integrated Moving Average | MAE: 100,160.50 |

**Model Terpilih**: **Exponential Smoothing** (6.42% lebih baik dari baseline)

### **2. Analisis Data**

- **Data Cleaning**: Menghapus transaksi cancelled dan nilai negatif
- **Time Series Decomposition**: Memisahkan trend, seasonal, dan residual
- **Stationarity Test**: Augmented Dickey-Fuller test untuk validasi data
- **ACF/PACF Analysis**: Menentukan parameter optimal ARIMA

### **3. Evaluasi & Validasi**

- **Train-Test Split**: 80% training, 20% testing
- **Metrics**: MAE (Mean Absolute Error) dan RMSE (Root Mean Squared Error)
- **Accuracy**: 71.7% (model terbaik)

### **4. Business Intelligence**

- **ABC Classification**: Kategorisasi produk berdasarkan Pareto principle (80/20)
- **Seasonal Analysis**: Identifikasi hari terbaik/terburuk untuk penjualan
- **Demand Forecasting**: Prediksi high/low demand periods
- **Inventory Recommendations**: Safety stock dengan buffer 20%

### **5. Visualisasi**

- **Static Charts**: Matplotlib & Seaborn untuk trend analysis
- **Interactive Dashboards**: Plotly dengan 3 dashboard utama
  - Dashboard 1: Historical & forecast dengan 4 panel
  - Dashboard 2: ABC analysis matrix
  - Dashboard 3: Time series dengan confidence interval 95%

---

## 🎯 Hasil & Manfaat

### **Prediksi 30 Hari**
- **Total Revenue**: $9,363,812.85
- **Daily Average**: $312,127.10
- **Growth Rate**: Stabil (±2%)

### **Business Benefits**
✅ **Efisiensi Stok**: Hindari overstock 15-20%  
✅ **Peningkatan Profit**: Fokus pada 20% produk top (A category)  
✅ **Strategi Promosi**: Target hari peak (data-driven)  
✅ **Data-Driven Decision**: Dashboard real-time untuk monitoring

---

## 📂 Struktur Notebook

### **1. Import Libraries** 
📌 **Penjelasan**: Memuat semua library yang diperlukan untuk analisis
- Data manipulation (pandas, numpy)
- Visualisasi (matplotlib, seaborn, plotly)
- Time series models (ARIMA, Exponential Smoothing)
- Evaluasi (MAE, RMSE)

---

### **2. Load and Explore Data**

#### **Cell: Load Data**
📌 **Penjelasan**: Membaca file CSV `Sales Transaction v.4a.csv`
- Menampilkan shape dataset
- Preview 5 baris pertama data

#### **Cell: Check Cancelled Transactions**
📌 **Penjelasan**: Identifikasi dan hapus transaksi cancelled
- TransactionNo yang dimulai dengan "C" = cancelled
- Menghitung jumlah transaksi cancelled
- Membersihkan dataset dari transaksi tidak valid

#### **Cell: Data Quality Check**
📌 **Penjelasan**: Validasi kualitas data
- Hapus nilai negatif pada Quantity/Price
- Hapus missing values di kolom kunci
- Pastikan data siap untuk analisis

#### **Cell: Data Info**
📌 **Penjelasan**: Eksplorasi struktur data
- Tipe data setiap kolom
- Missing values
- Statistik deskriptif (mean, std, min, max)

---

### **3. Data Preprocessing for Time Series**

#### **Cell: Create Daily Aggregation**
📌 **Penjelasan**: Transform data transaksi menjadi time series harian
- Convert Date column ke datetime
- Hitung Revenue = Price × Quantity
- Agregasi per hari: Total Transactions, Quantity, Revenue
- Sort by date

#### **Cell: Prepare Time Series**
📌 **Penjelasan**: Persiapan data untuk modeling
- Set Date sebagai index
- Fill missing dates dengan forward fill
- Pastikan frekuensi harian konsisten

---

### **4. Exploratory Data Analysis (EDA)**

#### **📈 Grafik: Daily Revenue/Quantity/Transactions Over Time**
📌 **Penjelasan**: Visualisasi tren penjualan harian
- **Panel 1**: Total Revenue per hari → lihat tren pendapatan
- **Panel 2**: Total Quantity per hari → volume penjualan
- **Panel 3**: Number of Transactions → aktivitas transaksi

**Insight yang dicari:**
- Apakah ada tren naik/turun?
    - Hitung rolling mean 30 & 90 hari pada daily revenue dan plot bersama data asli.
    - Fit OLS (revenue ~ time) pada seluruh periode dan pada subperiode (mis. awal, tengah, akhir) jika perlu.
    - Laporkan: arah tren (naik / turun / flat), slope (unit revenue per day), p-value regresi. Keputusan: p-value ≤ 0.05 berarti slope signifikan.
    - Hitung growth rate (%) = (mean_revenue_akhir_period - mean_revenue_awal_period) / mean_revenue_awal_period × 100. Sertakan periode yang dipakai untuk "awal" dan "akhir" (mis. rata-rata 30 hari pertama vs 30 hari terakhir).

- Apakah ada pola berulang (seasonal)?
    - Lakukan time series decomposition (additive). Plot komponen: observed, trend, seasonal, residual.
    - Analisis pola mingguan: agregasi revenue rata-rata per DayOfWeek; analisis bulanan: rata-rata per Month.
    - Laporkan periode seasonality yang dominan (mis. weekly) dan ukuran amplitudo relatif:
        - Amplitudo relatif = Var(seasonal) / Var(observed) (laporkan sebagai persentase).
    - Jika seasonal amplitude > 10% dianggap material; sebutkan day(s) dengan peak/low konsisten.

- Apakah ada outlier/anomali?
    - Identifikasi dengan dua metode:
        - IQR: nilai < Q1 - 1.5*IQR atau > Q3 + 1.5*IQR.
        - Z-score: |z| > 3 pada revenue harian.
    - Identifikasi lonjakan/kejatuhan ekstrem dari residual model (residual > 3*std(residual) atau < -3*std).
    - Untuk setiap outlier catat: tanggal, revenue, deviasi vs median (absolute & %), metode deteksi (IQR/z/residual).
    - Berikan kemungkinan penyebab singkat (promosi, holiday, pengembalian massal, data error) dan rekomendasi:
        - Verifikasi transaksi pada tanggal tersebut (cek notes, promo calendar, logs).
        - Jika data error → perbaiki/hapus sebelum training.
        - Jika event valid → tambahkan flag/event feature untuk modelling.

- Output yang diharapkan di laporan EDA:
    - Tren: slope, p-value, growth rate % (awal vs akhir), plot rolling mean & OLS line.
    - Seasonality: periode dominan, plot seasonal component, amplitudo relatif (%).
    - Outliers: tabel (Date, Revenue, Deviation_abs, Deviation_pct, Detection_method, Likely_cause, Action).
    - Kesimpulan singkat: rekomendasi preprocessing (differencing, event flags, removal/adjustment outliers) dan langkah selanjutnya untuk modelling.

---

#### **📈 Grafik: Time Series Decomposition**
📌 **Penjelasan**: Memecah time series menjadi komponen:
- **Observed**: Data asli
- **Trend**: Pola jangka panjang (naik/turun)
- **Seasonal**: Pola berulang (mingguan/bulanan)
- **Residual**: Noise/error yang tidak dapat dijelaskan

**Kegunaan:**
- Memahami komponen utama yang mempengaruhi penjualan
- Deteksi seasonality untuk modeling

---

### **5. Stationarity Test**

#### **Cell: Augmented Dickey-Fuller (ADF) Test**
📌 **Penjelasan**: Test statistik untuk cek stationarity
- **H0**: Data non-stationary (ada trend/seasonal)
- **H1**: Data stationary (tidak ada trend)
- **p-value ≤ 0.05**: Data stationary ✓
- **p-value > 0.05**: Perlu differencing

**Kenapa penting?**
ARIMA membutuhkan data stationary untuk akurasi maksimal.

---

#### **📈 Grafik: ACF & PACF**
📌 **Penjelasan**: Autocorrelation & Partial Autocorrelation Function
- **ACF**: Korelasi antara data dengan lag-nya
- **PACF**: Korelasi setelah remove pengaruh lag sebelumnya

**Kegunaan:**
- Menentukan parameter ARIMA (p, d, q)
- p = order AR (dari PACF)
- q = order MA (dari ACF)

---

### **6. Train-Test Split**

#### **Cell: Split Data**
📌 **Penjelasan**: Membagi data untuk validasi
- **80% Training**: Untuk melatih model
- **20% Testing**: Untuk evaluasi akurasi

#### **📈 Grafik: Train-Test Split Visualization**
📌 **Penjelasan**: Visualisasi pembagian data
- Garis merah = batas split
- Biru = training data
- Orange = test data

**Kegunaan:**
Memastikan model dilatih pada historical data dan ditest pada data "masa depan"

---

### **7. Model 1: Moving Average (MA)**

#### **Cell: Calculate Moving Average**
📌 **Penjelasan**: Model baseline sederhana
- Window = 7 hari (weekly average)
- Prediksi = rata-rata 7 hari terakhir

**Karakteristik:**
- ✓ Sederhana dan cepat
- ✗ Tidak dapat menangkap trend kompleks

#### **Cell: Evaluate MA Model**
📌 **Penjelasan**: Hitung MAE dan RMSE
- MAE: Rata-rata error absolut
- RMSE: Error dengan penalti untuk outlier

---

### **8. Model 2: Exponential Smoothing**

#### **Cell: Train Exponential Smoothing**
📌 **Penjelasan**: Model Holt-Winters
- **Trend**: Menangkap pola naik/turun
- **Seasonal**: Menangkap pola berulang (7 hari)
- **Additive**: Model penjumlahan komponen

**Karakteristik:**
- ✓ Menangkap seasonality
- ✓ Adaptive terhadap perubahan
- ✗ Membutuhkan data cukup panjang

#### **Cell: Evaluate ES Model**
📌 **Penjelasan**: Hitung MAE dan RMSE untuk Exponential Smoothing

---

### **9. Model 3: ARIMA**

#### **Cell: ARIMA Grid Search**
📌 **Penjelasan**: Mencoba berbagai konfigurasi ARIMA(p,d,q)
- Test 7 kombinasi berbeda
- Pilih model dengan AIC terendah
- AIC = Akaike Information Criterion (goodness of fit)

**Parameter ARIMA:**
- **p**: Autoregressive order (lag observations)
- **d**: Differencing order (stationarity)
- **q**: Moving average order

#### **Cell: Generate ARIMA Forecast**
📌 **Penjelasan**: Prediksi menggunakan model terbaik
- Forecast untuk periode test
- Display model summary

#### **Cell: Evaluate ARIMA Model**
📌 **Penjelasan**: Hitung MAE dan RMSE untuk ARIMA

---

### **10. Model Comparison**

#### **Cell: Compare All Models**
📌 **Penjelasan**: Tabel perbandingan performa
- MAE dan RMSE untuk 3 model
- Improvement % terhadap baseline (MA)
- **Model terbaik** = MAE terendah

#### **📈 Grafik: MAE & RMSE Comparison**
📌 **Penjelasan**: Bar chart perbandingan
- Visual comparison antar model
- Nilai ditampilkan di atas bar

**Insight:**
Model mana yang paling akurat untuk forecasting?

---

#### **📈 Grafik: Forecast vs Actual (Full View)**
📌 **Penjelasan**: Visualisasi prediksi semua model
- Biru = Training data
- Hitam = Actual test data
- Garis putus-putus = Prediksi model

**Kegunaan:**
Lihat visual model mana yang paling mendekati data aktual

---

#### **📈 Grafik: Forecast Comparison (Zoomed)**
📌 **Penjelasan**: Zoom pada periode test
- Detail perbandingan prediksi vs aktual
- Marker untuk setiap titik data

---

### **11. Future Forecast (30 Days)**

#### **Cell: Retrain on Full Data**
📌 **Penjelasan**: Latih model terbaik menggunakan SEMUA data
- Model dilatih ulang dengan full dataset
- Forecast 30 hari ke depan
- Generate confidence interval (95%)

**Output:**
- Tanggal forecast
- Prediksi revenue per hari
- Total revenue 30 hari

---

#### **📈 Grafik: 30-Day Forecast Visualization**
📌 **Penjelasan**: Visualisasi prediksi masa depan
- Biru = Historical (90 hari terakhir)
- Hijau = Forecast 30 hari
- Area hijau muda = Confidence interval 95%
- Garis merah = Start forecast

**Insight:**
Berapa ekspektasi revenue 30 hari ke depan?

---

### **12. Residual Analysis**

#### **📈 Grafik: Residual Diagnostics (4 Panel)**
📌 **Penjelasan**: Validasi kualitas model
1. **Residuals Over Time**: Error seharusnya random around 0
2. **Histogram**: Error seharusnya normal distribution
3. **Q-Q Plot**: Normalitas error (points harus di garis diagonal)
4. **ACF of Residuals**: Tidak ada autocorrelation (dalam confidence band)

**Model bagus jika:**
- Residuals random (no pattern)
- Normally distributed
- No autocorrelation

---

### **13. Decision Support Summary**

#### **Cell: DSS Report**
📌 **Penjelasan**: Laporan komprehensif untuk business
- Data overview (total transactions, revenue, dates)
- Model performance comparison
- 30-day forecast summary
- Key insights (growth rate, trend)
- Business recommendations

**Output:**
Report lengkap untuk stakeholder/management

---

### **14. Product-Level Analysis**

#### **Cell: Product Performance Analysis**
📌 **Penjelasan**: Analisis performa per produk
- Top 10 products by revenue
- Total quantity, transactions, avg price

---

#### **Cell: ABC Classification**
📌 **Penjelasan**: Pareto Analysis (80/20 rule)
- **Category A**: Top products (80% revenue) → Prioritas tinggi
- **Category B**: Moderate sellers (15% revenue) → Monitor
- **Category C**: Slow movers (5% revenue) → Review/diskon

**Kegunaan:**
Fokus inventory management pada produk penting

---

#### **📈 Grafik: Product Performance (4 Panel)**
📌 **Penjelasan**: Visualisasi performa produk
1. **Top 15 by Revenue**: Produk dengan pendapatan tertinggi
2. **Top 15 by Quantity**: Produk dengan volume terbanyak
3. **Pareto Chart**: 80/20 rule visualization
4. **Category Distribution**: Pie chart ABC classification

**Insight:**
- Produk mana yang paling menguntungkan?
- Apakah 80% revenue dari sedikit produk?

---

#### **Cell: Product-Level Forecasting**
📌 **Penjelasan**: Forecast demand per produk (top 10)
- Exponential Smoothing untuk setiap produk
- Prediksi 30 hari ke depan
- Handle missing dates (fill dengan 0)

**Output:**
Forecasted quantity untuk setiap produk

---

#### **Cell: Export Product Forecasts**
📌 **Penjelasan**: Export ke CSV
- ProductNo, ProductName
- Date, Forecasted_Quantity, Day
- File: `product_forecasts_30days.csv`

---

### **15. Seasonal Trend Analysis**

#### **Cell: Time-Based Features**
📌 **Penjelasan**: Extract fitur waktu
- DayOfWeek (Monday-Sunday)
- Month (January-December)
- WeekOfYear, DayOfMonth
- IsWeekend (Boolean)

**Kegunaan:**
Analisis pola penjualan berdasarkan waktu

---

#### **📈 Grafik: Seasonal Patterns (4 Panel)**
📌 **Penjelasan**: Visualisasi pola musiman
1. **Revenue by Day of Week**: Hari apa penjualan tertinggi?
2. **Weekday vs Weekend**: Perbandingan weekday/weekend
3. **Revenue by Month**: Pola bulanan (jika data cukup)
4. **Weekly Distribution**: Scatter plot revenue per minggu

**Insight untuk bisnis:**
- Hari mana untuk run promosi?
- Apakah weekend lebih ramai?
- Bulan mana peak season?

---

#### **Cell: Demand Period Analysis**
📌 **Penjelasan**: Identifikasi periode high/low demand
- **High Demand**: Revenue > Mean + 1 SD
- **Low Demand**: Revenue < Mean - 1 SD

**Kegunaan:**
- Persiapan inventory untuk high demand
- Promosi untuk boost low demand periods

---

### **16. Interactive Dashboard (Plotly)**

#### **Cell: Load Plotly**
📌 **Penjelasan**: Import library untuk dashboard interaktif

---

#### **📊 Dashboard 1: Sales Forecasting Dashboard (4 Panel)**
📌 **Penjelasan**: Dashboard interaktif utama
1. **Historical & Forecast Revenue**: Timeline lengkap
2. **Top 10 Products**: Horizontal bar chart
3. **Model Performance**: MAE comparison
4. **Revenue by Day**: Weekly pattern

**Fitur Interaktif:**
- ✓ Hover untuk detail
- ✓ Zoom in/out
- ✓ Pan untuk navigasi
- ✓ Export to PNG

---

#### **📊 Dashboard 2: Product Performance Matrix**
📌 **Penjelasan**: Scatter plot ABC classification
- X-axis: Total Quantity
- Y-axis: Total Revenue
- Color: Category (A/B/C)
- Hover: Product name & details

**Insight:**
Visual product positioning berdasarkan quantity vs revenue

---

#### **📊 Dashboard 3: Revenue Forecast with Confidence**
📌 **Penjelasan**: Interactive time series
- Historical data (full)
- 30-day forecast
- 95% confidence interval (shaded area)
- High/Low demand threshold lines

**Kegunaan:**
Eksplorasi interaktif forecast dengan uncertainty

---

### **17. Business Intelligence Report**

#### **Cell: Comprehensive BI Report**
📌 **Penjelasan**: Laporan lengkap untuk eksekutif
1. **Executive Summary**: Overview proyek
2. **Product Performance**: Top 5 best sellers & slow movers
3. **Seasonal Trends**: Best/worst days, weekday vs weekend
4. **Demand Periods**: High/low demand analysis
5. **Forecast Accuracy**: Model performance
6. **30-Day Forecast**: Revenue projection
7. **Inventory Recommendations**: Stock levels untuk top products
8. **Strategic Recommendations**: Action items
9. **Risk Factors**: Uncertainty & volatility

**Target Audience:**
Management, Operations, Marketing, Finance

---

#### **Cell: Export Reports**
📌 **Penjelasan**: Export 5 CSV files
1. `sales_forecast_30days.csv`: Overall revenue forecast
2. `product_forecasts_30days.csv`: Per-product demand
3. `inventory_recommendations.csv`: Stock recommendations
4. `product_classification_abc.csv`: ABC analysis
5. `seasonal_analysis.csv`: Seasonal patterns

**Kegunaan:**
Integrasi dengan sistem ERP/inventory management

---

## 📊 Output Files

| File | Deskripsi | Kolom Utama |
|------|-----------|-------------|
| `sales_forecast_30days.csv` | Prediksi revenue harian | Date, Forecasted_Revenue, Confidence Bounds |
| `product_forecasts_30days.csv` | Prediksi demand per produk | ProductNo, ProductName, Date, Forecasted_Quantity |
| `inventory_recommendations.csv` | Rekomendasi stok | ProductNo, Total_30Day_Forecast, Recommended_Stock |
| `product_classification_abc.csv` | Klasifikasi produk | ProductNo, ProductName, Category, RevenuePct |
| `seasonal_analysis.csv` | Pola musiman | DayOfWeek, TotalRevenue, AvgRevenue |

---

## 🎯 Hasil yang Diharapkan

### ✅ **Prediksi Penjualan**
- ✓ Prediksi revenue 30 hari ke depan dengan confidence interval
- ✓ Prediksi demand per produk (top 10)
- ✓ Akurasi model >85% (tergantung data)

### ✅ **Identifikasi Produk**
- ✓ Top 10 best selling products
- ✓ Slow movers yang perlu promosi
- ✓ ABC classification untuk prioritas inventory

### ✅ **Tren Seasonal**
- ✓ Hari dengan penjualan tertinggi/terendah
- ✓ Pola weekday vs weekend
- ✓ Periode high/low demand

### ✅ **Dashboard Interaktif**
- ✓ 3 dashboard Plotly untuk eksplorasi data
- ✓ Visualisasi yang dapat di-zoom dan di-export

---

## 💼 Manfaat Bisnis

### **1. Efisiensi Pengelolaan Stok**
- ❌ **Sebelum**: Stock berdasarkan intuisi → overstock/understock
- ✅ **Sesudah**: Stock berdasarkan prediksi → efisiensi 20-30%

**Contoh:**
Product A diprediksi butuh 500 units dalam 30 hari → stock 600 units (dengan 20% buffer)

### **2. Peningkatan Profitabilitas**
- ❌ **Sebelum**: Semua produk diperlakukan sama
- ✅ **Sesudah**: Fokus pada Category A (80% revenue)

**Contoh:**
10 produk Category A menghasilkan 80% revenue → prioritas marketing & inventory

### **3. Perencanaan Strategi Promosi**
- ❌ **Sebelum**: Promosi random tanpa pattern
- ✅ **Sesudah**: Target hari/periode optimal

**Contoh:**
Jika Jumat adalah best day → run flash sale di Kamis untuk boost lebih tinggi

### **4. Data-Driven Decision Making**
- ❌ **Sebelum**: Keputusan based on feeling
- ✅ **Sesudah**: Keputusan based on forecast dengan confidence level

**Contoh:**
Forecast menunjukkan growth 5% → expand inventory & hiring dengan confident

---

## 🚀 Cara Menjalankan

### **Prerequisites:**
```bash
pip install pandas numpy matplotlib seaborn plotly statsmodels scikit-learn scipy
```

### **Langkah-langkah:**

1. **Letakkan file CSV** di folder yang sama dengan notebook
   - `Sales Transaction v.4a.csv`

2. **Buka Jupyter Notebook**
   ```bash
   jupyter notebook mining.ipynb
   ```

3. **Run All Cells** (Cell → Run All)
   - Atau jalankan cell by cell untuk melihat hasil bertahap

4. **Hasil:**
   - Grafik akan muncul inline
   - Dashboard interaktif Plotly
   - 5 CSV files akan di-export

5. **Review Output:**
   - Baca Business Intelligence Report
   - Eksplorasi interactive dashboards
   - Gunakan CSV untuk sistem lain

---

## 📈 Interpretasi Hasil

### **Model Performance:**
- **MAE < 1000**: Model sangat baik ✓
- **MAE 1000-5000**: Model cukup baik ✓
- **MAE > 5000**: Model perlu improvement ⚠️

### **Growth Rate:**
- **> 5%**: Strong growth → expand inventory
- **0-5%**: Moderate growth → maintain
- **< 0%**: Decline → review strategy

### **ABC Classification:**
- **Category A**: 20% products = 80% revenue → Focus here!
- **Category B**: Monitor performance
- **Category C**: Consider discount or discontinue

---

## ⚠️ Limitasi & Asumsi

### **Limitasi:**
1. **Data Quality**: Hasil tergantung kualitas data input
2. **External Factors**: Model tidak memperhitungkan faktor eksternal (promosi, kompetitor, ekonomi)
3. **Historical Pattern**: Asumsi pola masa lalu akan berulang
4. **Seasonality**: Perlu minimal 2 siklus seasonal untuk akurasi maksimal

### **Asumsi:**
1. Transaksi cancelled (dengan "C") sudah dihapus
2. Tidak ada perubahan signifikan dalam bisnis model
3. Data historis representative untuk masa depan
4. Seasonality konsisten (mingguan)

---

## 🔄 Update & Maintenance

### **Recommended:**
- **Update forecast setiap minggu** untuk akurasi terbaik
- **Retrain model setiap bulan** jika ada perubahan pola
- **Monitor actual vs forecast** untuk evaluasi model
- **Adjust parameters** jika akurasi menurun

### **Improvement Ideas:**
1. Tambahkan external variables (holiday, promotion)
2. Ensemble multiple models untuk robust prediction
3. Real-time dashboard dengan automated update
4. Alert system untuk anomaly detection
5. Integration dengan ERP/POS system

---

## 👥 Stakeholder Usage

### **For Management:**
- Review Business Intelligence Report (Section 17)
- Fokus pada Strategic Recommendations
- Monitor 30-day forecast & growth rate

### **For Operations/Inventory:**
- Gunakan `inventory_recommendations.csv`
- Review product forecasts untuk stock planning
- Monitor high/low demand periods

### **For Marketing:**
- Review seasonal patterns (best days)
- Focus promosi pada Category C products
- Target periode low demand untuk campaign

### **For Finance:**
- Gunakan revenue forecast untuk budgeting
- Monitor growth rate untuk projection
- Review confidence intervals untuk risk assessment

---

## 📞 Support & Documentation

### **Troubleshooting:**
- **Error "File not found"**: Pastikan CSV di folder yang sama
- **Model tidak converge**: Kurangi seasonal_periods atau gunakan simple model
- **Dashboard tidak muncul**: Install plotly dan restart kernel
- **Akurasi rendah**: Cek data quality dan coba parameter ARIMA lain

### **Contact:**
Untuk pertanyaan atau issue, silakan hubungi tim data analytics.

---

## 📝 Version History

- **v1.0**: Initial release dengan 3 models forecasting
- **v1.1**: Tambah product-level forecasting
- **v1.2**: Tambah interactive dashboards
- **v1.3**: Tambah Business Intelligence Report
- **v1.4**: Data cleaning untuk cancelled transactions

---

## 📄 License

Proyek ini dibuat untuk keperluan Decision Support System (DSS) - UAS Semester 5

---

**🎓 Dibuat untuk pembelajaran Data Science & Business Analytics**

*Last Updated: December 2025*
